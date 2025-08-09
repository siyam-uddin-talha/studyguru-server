import strawberry
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.graphql.types.subscription import (
    SubscriptionsResponse,
    SubscriptionType,
    SubscriptionPlanEnum,
    UsageLimitType,
)
from app.models.subscription import Subscription, SubscriptionPlan


@strawberry.type
class PublicQuery:
    @strawberry.field
    async def subscriptions(self, info) -> SubscriptionsResponse:
        context = info.context
        db: AsyncSession = context.db

        # Get all subscriptions except FREE
        result = await db.execute(
            select(Subscription)
            .where(Subscription.subscription_plan != SubscriptionPlan.FREE)
            .options(selectinload(Subscription.usage_limit))
            .order_by(Subscription.usd_amount.asc())
        )
        subscriptions = result.scalars().all()

        subscription_types = []
        for sub in subscriptions:
            subscription_types.append(
                SubscriptionType(
                    id=sub.id,
                    subscription_name=sub.subscription_name,
                    usd_amount=sub.usd_amount,
                    gbp_amount=sub.gbp_amount,
                    subscription_plan=SubscriptionPlanEnum(sub.subscription_plan.value),
                    usage_limit=(
                        UsageLimitType(
                            id=sub.usage_limit.id,
                        )
                        if sub.usage_limit
                        else None
                    ),
                    created_at=sub.created_at,
                )
            )

        return SubscriptionsResponse(
            success=True,
            message="Getting pricing successfully",
            subscriptions=subscription_types,
        )
