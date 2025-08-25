import strawberry
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, func
from datetime import datetime, timedelta, timezone
from app.core.config import settings
from typing import Optional

from app.graphql.types.auth import (
    AuthType,
    AuthLoginType,
    LoginInput,
    AccountAccessInput,
)
from app.graphql.types.common import DefaultResponse
from app.models.user import User, AccountProvider
from app.models.subscription import PurchasedSubscription
from app.models.pivot import Country

from app.helpers.auth import (
    create_access_token,
    verify_password,
    get_password_hash,
    verify_google_token,
    generate_pin,
)
from app.helpers.subscription import (
    get_or_create_free_subscription,
    add_earned_points_async,
)
from app.helpers.pivot import get_or_create_country_from_object
from app.helpers.user import create_user_profile
from app.helpers.email import (
    send_verification_email,
    send_reset_email,
    send_welcome_email,
)
from app.constants.constant import CONSTANTS, COIN


@strawberry.type
class AuthQuery:
    @strawberry.field
    async def account(self, info) -> AuthType:
        context = info.context
        if not context.current_user:
            return AuthType(success=False, message="User not logged in")

        # Update last login
        context.current_user.last_login_at = func.now()
        await context.db.commit()

        account = await create_user_profile(context.current_user)
        print(account, "-----------account -------")
        return AuthType(
            success=True, message="Account retrieved successfully", account=account
        )


@strawberry.type
class AuthMutation:
    @strawberry.mutation
    async def account_access(
        self,
        info,
        input: AccountAccessInput,
        check_valid: Optional[bool] = False,
    ) -> AuthType:

        context = info.context
        db: AsyncSession = context.db

        if input.provider == "GOOGLE":
            if not input.credential:
                return AuthType(
                    success=False, message="Access token is required for verification."
                )

            # Verify Google token

            payload = await verify_google_token(input.credential)

            if not payload:
                return AuthType(
                    success=False,
                    message="Unable to verify your google account! Please try again.",
                )

            email = payload.get("email")
            if not email:
                return AuthType(success=False, message="Email is required")

            # Check if user exists
            result = await db.execute(
                select(User)
                .options(
                    selectinload(User.purchased_subscription).selectinload(
                        PurchasedSubscription.subscription
                    ),
                    selectinload(User.country),
                )
                .where(User.email == email)
            )
            existing_user = result.scalar_one_or_none()

            if existing_user:
                # Login existing user
                token = create_access_token(data={"sub": existing_user.id})
                account = await create_user_profile(existing_user)
                return AuthType(
                    success=True, message="Welcome back", account=account, token=token
                )

            # Create new user
            free_sub = await get_or_create_free_subscription(db)

            new_user = User(
                email=email,
                first_name=payload.get("given_name", ""),
                last_name=payload.get("family_name", ""),
                photo_url=payload.get("picture"),
                account_provider=AccountProvider.GOOGLE,
                verify_status=payload.get("email_verified", False),
                purchased_subscription_id=free_sub.id,
            )

            db.add(new_user)
            await db.commit()
            await db.refresh(new_user)

            token = create_access_token(data={"sub": new_user.id})

            new_user_result = await db.execute(
                select(User)
                .options(
                    selectinload(User.purchased_subscription).selectinload(
                        PurchasedSubscription.subscription
                    ),
                    selectinload(User.country),
                )
                .where(User.id == new_user.id)
            )
            get_new_user = new_user_result.scalar_one_or_none()

            account = await create_user_profile(get_new_user)
            await send_welcome_email(
                get_new_user.email,
                f"{get_new_user.first_name} {get_new_user.last_name}",
                settings.APP_LINK,
            )

            return AuthType(
                success=True,
                message="Account created successfully",
                account=account,
                token=token,
            )

        # Email registration
        required_fields = [
            input.email,
            input.first_name,
            input.last_name,
            input.phone_number,
            input.education_level,
            input.address,
        ]
        if not check_valid:
            required_fields.append(input.password)

        if not all(required_fields):
            return AuthType(success=False, message="Please fill out all the fields")

        # Check if user exists
        result = await db.execute(
            select(User)
            .options(
                selectinload(User.purchased_subscription).selectinload(
                    PurchasedSubscription.subscription
                ),
                selectinload(User.country),
            )
            .where(
                or_(User.email == input.email, User.phone_number == input.phone_number)
            )
        )
        existing_user = result.scalar_one_or_none()

        if check_valid:

            if existing_user:

                return AuthType(
                    success=False,
                    response_status="account_registered",
                    message="Looks like you already have an account. Try logging in!",
                )
            if not existing_user:

                return AuthType(
                    success=True,
                    response_status="no_account_found",
                    message="No account found with this email. Let’s create one!",
                )

        if existing_user:
            return AuthType(
                success=False,
                message="This email number is already linked to an existing account. Try logging in or using a different email!",
            )

        # Create user
        hashed_password = get_password_hash(input.password)
        free_sub = await get_or_create_free_subscription(db)
        user_country = await get_or_create_country_from_object(
            db,
            country_obj=input.country,
        )

        new_user = User(
            email=input.email,
            first_name=input.first_name,
            last_name=input.last_name,
            password=hashed_password,
            account_provider=AccountProvider.EMAIL,
            education_level=input.education_level,
            phone_number=input.phone_number.replace(" ", ""),
            purchased_subscription_id=free_sub.id,
            country_id=user_country.id,
        )

        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)

        # Send verification email
        # pin = generate_verify_pin()
        # new_user.verify_pin = pin
        # new_user.verify_pin_expire_date = datetime.utcnow() + timedelta(minutes=15)
        await db.commit()

        new_user_result = await db.execute(
            select(User)
            .options(
                selectinload(User.purchased_subscription).selectinload(
                    PurchasedSubscription.subscription
                ),
                selectinload(User.country),
            )
            .where(User.id == new_user.id)
        )
        get_new_user = new_user_result.scalar_one_or_none()

        await send_welcome_email(
            get_new_user.email,
            f"{get_new_user.first_name} {get_new_user.last_name}",
            isByEmail=True,
        )
        token = create_access_token(data={"sub": get_new_user.id})
        account = await create_user_profile(get_new_user)

        return AuthType(
            success=True,
            message="New account created successfully",
            account=account,
            token=token,
        )

    @strawberry.mutation
    async def login(self, info, input: LoginInput) -> AuthLoginType:
        context = info.context
        db: AsyncSession = context.db

        if input.provider == "EMAIL":
            # Validate email input
            if not input.email or not input.password:
                return AuthLoginType(success=False, message=CONSTANTS.NOT_FILLED)

            # Find user by email with subscription data
            result = await db.execute(
                select(User)
                .options(
                    selectinload(User.purchased_subscription).selectinload(
                        PurchasedSubscription.subscription
                    ),
                    selectinload(User.country),
                )
                .where(User.email == input.email)
            )
            user = result.scalar_one_or_none()

            if not user:
                return AuthLoginType(
                    success=False,
                    message=CONSTANTS.NOT_FOUND,
                )

            # Verify password
            if not user.password or not verify_password(input.password, user.password):
                return AuthLoginType(
                    success=False, message=CONSTANTS.INCORRECT_PASSWORD
                )

            # Check email verification status
            # if not user.email_verified:
            #     # Send new verification code
            #     pin = generate_verify_pin()
            #     user.verify_pin = pin
            #     user.verify_pin_expire_date = datetime.utcnow() + timedelta(minutes=15)
            #     await db.commit()

            #     await send_verification_email(
            #         user.email, pin, f"{user.first_name} {user.last_name}"
            #     )

            #     return AuthLoginType(
            #         success=False,
            #         message=f"Your email is not verified. We've sent a 6-digit verification code to {user.email}",
            #         verify=False,
            #         email=user.email,
            #     )

        elif input.provider == "PHONE":
            selected_country = input.country

            phone_number = (input.phone_number.replace(" ", ""),)
            # Validate phone input
            if not phone_number or not input.password:
                return AuthLoginType(success=False, message=CONSTANTS.NOT_FILLED)

            # Find user by phone with subscription data
            result = await db.execute(
                select(User)
                .join(Country, User.country_id == Country.id)
                .options(
                    selectinload(User.purchased_subscription).selectinload(
                        PurchasedSubscription.subscription
                    ),
                    selectinload(User.country),
                )
                .where(
                    User.phone_number == phone_number,
                    Country.country_code == selected_country.country_code,
                )
            )
            user = result.scalar_one_or_none()

            if not user:
                return AuthLoginType(
                    success=False,
                    message=CONSTANTS.INVALID_PHONE,
                )

            # Verify password
            if not user.password or not verify_password(input.password, user.password):
                return AuthLoginType(
                    success=False, message=CONSTANTS.INCORRECT_PASSWORD
                )

            # Check phone verification status
            # if not user.phone_verified:
            #     # Send new verification code via SMS
            #     pin = generate_verify_pin()
            #     user.phone_verify_pin = pin
            #     user.phone_verify_pin_expire_date = datetime.utcnow() + timedelta(minutes=15)
            #     await db.commit()

            #     # Send SMS verification (implement your SMS service)
            #     await send_verification_sms(
            #         user.phone, pin, f"{user.first_name} {user.last_name}"
            #     )

            #     return AuthLoginType(
            #         success=False,
            #         message=f"Your phone number is not verified. We've sent a 6-digit verification code to {user.phone}",
            #         verify=False,
            #         phone=user.phone,
            #     )

        else:
            return AuthLoginType(
                success=False, message="Invalid provider. Please use EMAIL or PHONE"
            )

        # Check if account is active/not banned
        # if hasattr(user, 'is_active') and not user.is_active:
        #     return AuthLoginType(
        #         success=False,
        #         message="Your account has been deactivated. Please contact support."
        #     )

        # Update last login
        user.last_login_at = func.now()
        await db.commit()

        # Create token
        token = create_access_token(data={"sub": user.id})
        account = await create_user_profile(user)

        return AuthLoginType(
            success=True, message="Login successful", account=account, token=token
        )

    @strawberry.mutation
    async def verify(
        self,
        info,
        pin: str,
        email: str,
        verify_reset: Optional[bool] = False,
    ) -> AuthType:
        context = info.context
        db: AsyncSession = context.db

        if not pin or not email:
            return AuthType(success=False, message=CONSTANTS.NOT_FILLED)

        # Find user with valid pin
        result = await db.execute(
            select(User)
            .options(
                selectinload(User.purchased_subscription).selectinload(
                    PurchasedSubscription.subscription
                ),
                selectinload(User.country),
            )
            .where(
                User.email == email,
                User.verify_pin_expire_date > datetime.now(timezone.utc),
            )
        )

        user = result.scalar_one_or_none()

        if not user:
            return AuthType(success=False, message=CONSTANTS.NOT_FOUND)

        if verify_reset:
            if user.reset_password_pin != int(pin):
                return AuthType(
                    success=False,
                    message=CONSTANTS.INVALID_OR_EXPIRE_PIN,
                )

            # Update user
            user.reset_password_pin = None
            user.reset_password_expire_date = None
            await db.commit()

            return AuthType(
                success=True,
                message=CONSTANTS.PIN_SUCCESS,
            )

        if user.verify_pin != int(pin):
            return AuthType(
                success=False,
                message=CONSTANTS.INVALID_OR_EXPIRE_PIN,
            )

        # Update user
        user.verify_pin = None
        user.verify_pin_expire_date = None
        user.verify_status = True
        await db.commit()
        await add_earned_points_async(
            db=db,
            user_id=user.id,
            points=int(COIN.EARN_VERIFY_EMAIL),
            description="Free coins as a welcome gift.",
        )

        # Create token
        token = create_access_token(data={"sub": user.id})
        account = await create_user_profile(user)

        return AuthType(
            success=True,
            message="Your email address has been successfully verified",
            account=account,
            token=token,
        )

    @strawberry.mutation
    async def send_verification(
        self, info, email: str, for_reset: Optional[bool] = False
    ) -> DefaultResponse:
        context = info.context
        db: AsyncSession = context.db

        if not email:
            return DefaultResponse(
                success=False, message="Please fill out all the fields"
            )

        # Find user
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

        if not user:
            return DefaultResponse(
                success=False,
                message=CONSTANTS.NOT_FOUND_2,
            )

        if for_reset:
            pin = generate_pin()
            user.reset_password_pin = pin
            user.reset_password_expire_date = datetime.utcnow() + timedelta(minutes=15)
            await db.commit()
            # Send reset email
            await send_reset_email(
                user.email, pin, f"{user.first_name} {user.last_name}"
            )

            return DefaultResponse(
                success=True, message="OTP has been sent to your email."
            )

        # Generate and send verification code
        pin = generate_pin()
        user.verify_pin = pin
        user.verify_pin_expire_date = datetime.utcnow() + timedelta(minutes=15)
        await db.commit()

        await send_verification_email(
            user.email, pin, f"{user.first_name} {user.last_name}"
        )

        return DefaultResponse(
            success=True,
            message=f"We’ve just sent a 6-digit verification code to {email}. Please check your inbox (and spam/junk folder if needed) to continue.",
        )

    @strawberry.mutation
    async def forgot_password(self, info, email: str) -> DefaultResponse:
        context = info.context
        db: AsyncSession = context.db

        if not email:
            return DefaultResponse(
                success=False, message="Please fill out all the fields"
            )

        # Find user
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

        if not user:
            return DefaultResponse(
                success=False,
                message="We couldn't locate an account under that email address",
            )

        # Generate reset PIN
        pin = generate_pin()
        user.reset_password_pin = pin
        user.reset_password_expire_date = datetime.utcnow() + timedelta(minutes=15)
        await db.commit()

        # Send reset email
        await send_reset_email(user.email, pin, f"{user.first_name} {user.last_name}")

        return DefaultResponse(success=True, message="OTP has been sent to your email.")

    @strawberry.mutation
    async def update_reset_password(
        self, info, password: str, email: str, pin: str
    ) -> DefaultResponse:
        context = info.context
        db: AsyncSession = context.db

        if not all([email, password, pin]):
            return DefaultResponse(success=False, message=CONSTANTS.NOT_FILLED)

        # Find user with valid reset pin
        result = await db.execute(
            select(User).where(
                User.email == email, User.reset_password_expire_date > datetime.utcnow()
            )
        )
        user = result.scalar_one_or_none()

        if not user:
            return DefaultResponse(
                success=False,
                message=CONSTANTS.NOT_FOUND,
            )

        if user.reset_password_pin != int(pin):
            return DefaultResponse(
                success=False, message=CONSTANTS.INVALID_OR_EXPIRE_PIN
            )

        # Update password
        user.password = get_password_hash(password)
        user.reset_password_pin = None
        user.reset_password_expire_date = None
        await db.commit()

        return DefaultResponse(
            success=True,
            message="Password change complete. Your account security has been enhanced.",
        )
