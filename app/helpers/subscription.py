from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.subscription import (
    Subscription,
    PurchasedSubscription,
    SubscriptionPlan,
)


async def get_or_create_free_subscription(db: AsyncSession) -> PurchasedSubscription:
    """Get or create a free subscription for new users"""

    # Find free subscription plan
    result = await db.execute(
        select(Subscription).where(
            Subscription.subscription_plan == SubscriptionPlan.ESSENTIAL
        )
    )
    free_subscription = result.scalar_one_or_none()

    if not free_subscription:
        raise Exception("No free subscription plan found")

    # Create purchased subscription
    purchased_sub = PurchasedSubscription(
        subscription_id=free_subscription.id, subscription_status="active"
    )

    db.add(purchased_sub)
    await db.commit()
    await db.refresh(purchased_sub)

    return purchased_sub
