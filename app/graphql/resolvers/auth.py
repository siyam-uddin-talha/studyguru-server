import strawberry
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, func
from datetime import datetime, timedelta
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

from app.helpers.auth import (
    create_access_token,
    verify_password,
    get_password_hash,
    verify_google_token,
    generate_verify_pin,
)
from app.helpers.subscription import get_or_create_free_subscription
from app.helpers.user import create_user_profile
from app.helpers.email import (
    send_verification_email,
    send_reset_email,
    send_welcome_email,
)


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
                    )
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
                    )
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
                )
            )
            .where(
                or_(User.email == input.email, User.phone_number == input.phone_number)
            )
        )
        existing_user = result.scalar_one_or_none()
        print(check_valid)

        if check_valid:

            if existing_user:
                print("found")
                return AuthType(
                    success=False,
                    response_status="account_registered",
                    message="Looks like you already have an account. Try logging in!",
                )
            if not existing_user:
                print("not found")
                return AuthType(
                    success=True,
                    response_status="no_account_found",
                    message="No account found with this email. Let’s create one!",
                )

        if existing_user:
            return AuthType(success=False, message="This account is already registered")

        # Create user
        hashed_password = get_password_hash(input.password)
        free_sub = await get_or_create_free_subscription(db)

        new_user = User(
            email=input.email,
            first_name=input.first_name,
            last_name=input.last_name,
            password=hashed_password,
            account_provider=AccountProvider.EMAIL,
            education_level=input.education_level,
            phone_number=input.phone_number.replace(" ", ""),
            purchased_subscription_id=free_sub.id,
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
                )
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

        if not input.email or not input.password:
            return AuthLoginType(
                success=False, message="Please fill out all the fields"
            )

        # Find user
        result = await db.execute(select(User).where(User.email == input.email))
        user = result.scalar_one_or_none()

        if not user:
            return AuthLoginType(
                success=False,
                message="We couldn't find an account associated with this email address",
            )

        # Verify password
        if not user.password or not verify_password(input.password, user.password):
            return AuthLoginType(
                success=False, message="The password you entered is incorrect"
            )

        # Check verification status
        # if not user.verify_status:
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
        #         message=f"Your account is not verified. We've sent a 6-digit verification code to your registered email {user.email}",
        #         verify=False,
        #         email=user.email,
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
    async def verify(self, info, pin: str, email: str) -> AuthType:
        context = info.context
        db: AsyncSession = context.db

        if not pin or not email:
            return AuthType(success=False, message="Please fill out all the fields")

        # Find user with valid pin
        result = await db.execute(
            select(User).where(
                User.email == email, User.verify_pin_expire_date > datetime.utcnow()
            )
        )
        user = result.scalar_one_or_none()

        if not user:
            return AuthType(success=False, message="Invalid or expired OTP")

        if user.verify_pin != int(pin):
            return AuthType(success=False, message="Invalid verification code")

        # Update user
        user.verify_pin = None
        user.verify_pin_expire_date = None
        user.verify_status = True
        await db.commit()

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
    async def send_verification(self, info, email: str) -> DefaultResponse:
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
                message="We couldn't find an account associated with this email address",
            )

        # Generate and send verification code
        pin = generate_verify_pin()
        user.verify_pin = pin
        user.verify_pin_expire_date = datetime.utcnow() + timedelta(minutes=15)
        await db.commit()

        await send_verification_email(
            user.email, pin, f"{user.first_name} {user.last_name}"
        )

        return DefaultResponse(
            success=True, message=f"A verification code has been sent to {email}"
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

        if not user.verify_status:
            return DefaultResponse(
                success=False,
                message="We're unable to reset your password until your email address is verified",
            )

        # Generate reset PIN
        pin = generate_verify_pin()
        user.reset_password_pin = pin
        user.reset_password_expire_date = datetime.utcnow() + timedelta(minutes=15)
        await db.commit()

        # Send reset email
        await send_reset_email(user.email, pin, f"{user.first_name} {user.last_name}")

        return DefaultResponse(success=True, message="Reset OTP sent successfully")

    @strawberry.mutation
    async def update_reset_password(
        self, info, password: str, email: str, pin: str
    ) -> DefaultResponse:
        context = info.context
        db: AsyncSession = context.db

        if not all([email, password, pin]):
            return DefaultResponse(
                success=False, message="Incorrect information! Try again"
            )

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
                message="The password reset OTP has expired or is no longer valid",
            )

        if user.reset_password_pin != int(pin):
            return DefaultResponse(success=False, message="Invalid reset PIN")

        # Update password
        user.password = get_password_hash(password)
        user.reset_password_pin = None
        user.reset_password_expire_date = None
        await db.commit()

        return DefaultResponse(
            success=True, message="Your password has been updated successfully"
        )
