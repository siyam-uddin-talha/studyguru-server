# server/app/graphql/resolvers/settings.py
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
from app.helpers.auth import verify_password, get_password_hash
from app.helpers.user import create_user_profile
from app.helpers.email import send_verification_email
from app.constants.constant import CONSTANTS


@strawberry.type
class SettingsQuery:
    @strawberry.field
    async def current_user(self, info) -> SettingsResponse:
        context = info.context

        if not context.current_user:
            return SettingsResponse(
                success=False, message="Unable to find user!", account=None
            )

        try:
            account = await create_user_profile(context.current_user)
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

            # Check if email is changing and is already in use
            if input.email and input.email != user.email:
                existing_email = await db.execute(
                    select(User).where(User.email == input.email)
                )
                if existing_email.scalar_one_or_none():
                    return SettingsResponse(
                        success=False, message="This account is already registered"
                    )

            # Update user profile
            update_data = {
                key: value for key, value in input.__dict__.items() if value is not None
            }

            # Reset verification status if email changes
            if input.email and input.email != user.email:
                update_data["verify_status"] = False

            # Update user
            for key, value in update_data.items():
                setattr(user, key, value)

            await db.commit()

            return SettingsResponse(
                success=True,
                message="Your profile information has been successfully updated.",
                logout=input.email != user.email,
            )

        except Exception as e:
            return SettingsResponse(success=False, message=str(e))
