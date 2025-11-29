from app.models.user import User, UserAccountType
from app.graphql.types.auth import Account, AccountProviderEnum
from app.graphql.types.subscription import (
    PurchasedSubscriptionType,
    SubscriptionType,
    SubscriptionPlanEnum,
    UsageLimitType,
)
from app.core.config import settings
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update


def parse_photo_url(photo_url: str) -> str:
    """Parse photo URL to include S3 bucket URL if needed"""
    if not photo_url:
        return None

    if photo_url.startswith("https://"):
        return photo_url

    return f"https://{settings.AWS_S3_BUCKET}.s3.amazonaws.com/{photo_url}"


async def create_user_profile(user: User) -> Account:
    """Create user profile from User model"""

    purchased_subscription = None
    if user.purchased_subscription:
        subscription = None
        if user.purchased_subscription.subscription:
            # usage_limit = None
            # if user.purchased_subscription.subscription.usage_limit:
            #     usage_limit = UsageLimitType(
            #         id=user.purchased_subscription.subscription.usage_limit.id,
            #     )
            subscription = SubscriptionType(
                id=user.purchased_subscription.subscription.id,
                subscription_name=user.purchased_subscription.subscription.subscription_name,
                usd_amount=user.purchased_subscription.subscription.usd_amount,
                bdt_amount=user.purchased_subscription.subscription.bdt_amount,
                subscription_plan=SubscriptionPlanEnum(
                    user.purchased_subscription.subscription.subscription_plan.value
                ),
            )

        purchased_subscription = PurchasedSubscriptionType(
            id=user.purchased_subscription.id,
            subscription_id=user.purchased_subscription.subscription_id,
            subscription=subscription,
            # past_due_time=user.purchased_subscription.past_due_time,
            created_at=user.purchased_subscription.created_at,
            updated_at=user.purchased_subscription.updated_at,
        )

    return Account(
        id=user.id,
        first_name=user.first_name,
        last_name=user.last_name,
        email=user.email,
        primary_state=user.primary_state,
        primary_city=user.primary_city,
        profession_title=user.profession_title,
        about_description=user.about_description,
        primary_address=user.primary_address,
        phone_number=user.phone_number,
        zip_code=user.zip_code,
        photo_url=parse_photo_url(user.photo_url),
        created_at=user.created_at,
        # last_login_at=user.last_login_at,
        verify_status=user.verify_status,
        super_admin=user.super_admin,
        account_provider=(
            AccountProviderEnum(user.account_provider.value)
            if user.account_provider
            else None
        ),
        account_type=user.account_type,
        purchased_subscription=purchased_subscription,
        country=user.country,
        current_points=user.current_points,
        total_points_earned=user.total_points_earned,
        total_points_used=user.total_points_used,
        education_level=user.education_level,
        birthday=user.birthday,
    )


async def get_current_user_from_context(context) -> Optional[User]:
    """Extract current user from GraphQL context"""
    if hasattr(context, "current_user"):
        return context.current_user
    return None


async def merge_guest_account_data(
    db: AsyncSession, guest_user: User, real_user: User
) -> None:
    """
    Merge guest account data (interactions, conversations, point transactions, context data)
    into the real user account, then delete the guest account.
    """
    from app.models.interaction import (
        Interaction,
        Conversation,
        InteractionShareVisitor,
    )
    from app.models.subscription import PointTransaction, BillingLog
    from app.models.context import (
        ConversationContext,
        UserLearningProfile,
        DocumentContext,
        ContextUsageLog,
    )
    from sqlalchemy import delete

    # Transfer all interactions (which includes conversations)
    await db.execute(
        update(Interaction)
        .where(Interaction.user_id == guest_user.id)
        .values(user_id=real_user.id)
    )

    # Transfer point transactions
    await db.execute(
        update(PointTransaction)
        .where(PointTransaction.user_id == guest_user.id)
        .values(user_id=real_user.id)
    )

    # Transfer billing logs
    await db.execute(
        update(BillingLog)
        .where(BillingLog.user_id == guest_user.id)
        .values(user_id=real_user.id)
    )

    # Transfer conversation context
    await db.execute(
        update(ConversationContext)
        .where(ConversationContext.user_id == guest_user.id)
        .values(user_id=real_user.id)
    )

    # Transfer document context
    await db.execute(
        update(DocumentContext)
        .where(DocumentContext.user_id == guest_user.id)
        .values(user_id=real_user.id)
    )

    # Transfer context usage logs
    await db.execute(
        update(ContextUsageLog)
        .where(ContextUsageLog.user_id == guest_user.id)
        .values(user_id=real_user.id)
    )

    # Transfer user learning profile (if exists)
    # Check if real user already has a learning profile
    result = await db.execute(
        select(UserLearningProfile).where(UserLearningProfile.user_id == real_user.id)
    )
    real_user_profile = result.scalar_one_or_none()

    result = await db.execute(
        select(UserLearningProfile).where(UserLearningProfile.user_id == guest_user.id)
    )
    guest_user_profile = result.scalar_one_or_none()

    if guest_user_profile:
        if real_user_profile:
            # Real user already has a profile, delete guest's profile
            # (or you could merge the data if needed)
            await db.execute(
                delete(UserLearningProfile).where(
                    UserLearningProfile.user_id == guest_user.id
                )
            )
        else:
            # Real user doesn't have a profile, transfer guest's
            await db.execute(
                update(UserLearningProfile)
                .where(UserLearningProfile.user_id == guest_user.id)
                .values(user_id=real_user.id)
            )

    # Transfer interaction share visitors (if any)
    await db.execute(
        update(InteractionShareVisitor)
        .where(InteractionShareVisitor.visitor_user_id == guest_user.id)
        .values(visitor_user_id=real_user.id)
    )

    # Merge points (add guest points to real user)
    real_user.current_points += guest_user.current_points
    real_user.total_points_earned += guest_user.total_points_earned
    real_user.total_points_used += guest_user.total_points_used

    # Delete the guest account
    # Note: user_module_permission has cascade delete, so it will be handled automatically
    await db.execute(delete(User).where(User.id == guest_user.id))
    await db.commit()
