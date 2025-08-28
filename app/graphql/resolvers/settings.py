import strawberry
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime, timezone
from typing import Optional, Any

from app.graphql.types.settings import (
    UpdatePasswordInput,
    UpdateProfileInput,
    SettingsResponse,
)
from app.models.user import User, AccountProvider
from app.models.subscription import PurchasedSubscription
from app.helpers.auth import verify_password, get_password_hash
from app.helpers.user import create_user_profile
from app.helpers.subscription import (
    add_earned_points_async,
)
from app.helpers.pivot import get_or_create_country_from_object

from app.constants.constant import CONSTANTS, COIN


@strawberry.type
class SettingsQuery:
    @strawberry.field
    async def current_user(self, info) -> SettingsResponse:
        context = info.context

        if not context.current_user:
            return SettingsResponse(
                success=False, message=CONSTANTS.NOT_FOUND, account=None
            )

        try:
            user_result = await context.db.execute(
                select(User)
                .options(
                    selectinload(User.purchased_subscription).selectinload(
                        PurchasedSubscription.subscription
                    ),
                    selectinload(User.country),
                )
                .where(User.id == context.current_user.id)
            )
            new_user = user_result.scalar_one_or_none()

            account = await create_user_profile(new_user)

            return SettingsResponse(
                success=True,
                message="Account information retrieved successfully.",
                account=account,
            )
        except Exception as e:
            return SettingsResponse(success=False, message=str(e), account=None)


@strawberry.type
class SettingsMutation:
    @strawberry.mutation
    async def update_password(
        self, info, current_password: str, new_password: str
    ) -> SettingsResponse:
        context = info.context
        db: AsyncSession = context.db

        if not context.current_user:
            return SettingsResponse(
                success=False, message="Unable to find account, try again"
            )

        try:
            # Find user
            result = await db.execute(
                select(User)
                .options(selectinload(User.purchased_subscription))
                .where(User.id == context.current_user.id)
            )
            user = result.scalar_one_or_none()

            if not user:
                return SettingsResponse(
                    success=False,
                    message="We couldn't find an account associated with this email address.",
                )

            # Handle special cases for social login
            if (
                user.account_provider != AccountProvider.EMAIL
                and new_password
                and not user.password
            ):
                hashed_password = get_password_hash(new_password)
                user.password = hashed_password
                user.reset_password_expire_date = None
                await db.commit()

                return SettingsResponse(
                    success=True,
                    message="Your new password has been successfully updated.",
                )

            # Verify current password
            if not user.password or not verify_password(
                current_password, user.password
            ):
                return SettingsResponse(
                    success=False,
                    message="The current password you provided doesn't match our records.",
                )

            # Update password
            hashed_password = get_password_hash(new_password)
            user.password = hashed_password
            user.reset_password_expire_date = None
            await db.commit()

            return SettingsResponse(
                success=True, message="Your new password has been successfully updated."
            )

        except Exception as e:
            return SettingsResponse(success=False, message=str(e))

    @strawberry.mutation
    async def update_profile(self, info, input: UpdateProfileInput) -> SettingsResponse:
        context = info.context
        db: AsyncSession = context.db

        if not context.current_user:
            return SettingsResponse(success=False, message=CONSTANTS.NOT_FOUND)

        try:
            # Find user
            result = await db.execute(
                select(User)
                .options(selectinload(User.country))
                .where(User.id == context.current_user.id)
            )
            user: User = result.scalar_one_or_none()
            if not user:
                return SettingsResponse(
                    success=False,
                    message=CONSTANTS.NOT_FOUND_2,
                )

            # Check if email is changing and is already in use
            if input.email and input.email != user.email:
                existing_email = await db.execute(
                    select(User).where(User.email == input.email)
                )
                if existing_email.scalar_one_or_none():
                    return SettingsResponse(
                        success=False, message=CONSTANTS.ACCOUNT_FOUND
                    )

            # Define fields to check for updates
            fields_to_check = [
                "first_name",
                "last_name",
                "email",
                "primary_state",
                "primary_city",
                "profession_title",
                "about_description",
                "primary_address",
                "zip_code",
                "photo_url",
                "phone_number",
                "birthday",
                "education_level",
            ]

            # Update user profile
            update_data = {}

            # Check and add fields that have changed
            for field in fields_to_check:
                input_value = getattr(input, field)
                user_value = getattr(user, field)

                # Compare values, handling phone number with space removal
                if field == "phone_number":
                    if input_value is not None:
                        cleaned_input = input_value.replace(" ", "")
                        if cleaned_input != user_value:
                            update_data[field] = cleaned_input
                elif input_value is not None and input_value != user_value:
                    update_data[field] = input_value

            # Handle country separately
            if (
                input.country
                and not user.country
                or (
                    user.country
                    and not user.country.country_code == input.country.country_code
                )
            ):

                # Check if country is different
                user_country = await get_or_create_country_from_object(
                    db, country_obj=input.country
                )

                update_data["country_id"] = user_country.id

            # If no changes, return early
            if not update_data:
                return SettingsResponse(
                    success=True,
                    message="No changes detected. Profile remains the same.",
                    # account=await create_user_profile(user),
                )

            # Reset verification status if email changes
            if "email" in update_data and update_data["email"] != user.email:
                update_data["verify_status"] = False

            # Update user
            for key, value in update_data.items():
                setattr(user, key, value)

            # Bonus points logic for Google account

            if user.account_provider == AccountProvider.GOOGLE and (
                (not user.phone_number and input.phone_number)
                or (not user.country and input.country)
            ):

                await add_earned_points_async(
                    db=db,
                    user_id=user.id,
                    points=int(COIN.EARN_VERIFY_EMAIL),
                    description="Free coins for completing your profile",
                )

            await db.commit()
            await db.refresh(user, ["country"])

            # Fetch updated user with all related data
            updated_user_result = await db.execute(
                select(User)
                .options(
                    selectinload(User.purchased_subscription).selectinload(
                        PurchasedSubscription.subscription
                    ),
                    selectinload(User.country),
                )
                .where(User.id == user.id)
            )
            get_update_user = updated_user_result.scalar_one_or_none()

            updated_account = await create_user_profile(get_update_user)

            return SettingsResponse(
                success=True,
                message="Profile changes saved! Your account information is now up to date.",
                logout="email" in update_data,
                account=updated_account,
            )

        except Exception as e:
            return SettingsResponse(success=False, message=str(e))
