from app.models.user import User
from app.graphql.types.auth import Account, AccountProviderEnum
from app.graphql.types.subscription import (
    PurchasedSubscriptionType,
    SubscriptionType,
    SubscriptionPlanEnum,
    UsageLimitType,
)
from app.core.config import settings


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
    )
