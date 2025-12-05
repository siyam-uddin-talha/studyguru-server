import strawberry
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.graphql.types.subscription import (
    SubscriptionsResponse,
    SubscriptionType,
    SubscriptionPlanEnum,
    UsageLimitType,
    ModelsResponse,
    ModelType,
)
from app.models.subscription import Subscription, SubscriptionPlan, Model


@strawberry.type
class PublicQuery:
    @strawberry.field
    async def subscriptions(self, info) -> SubscriptionsResponse:
        context = info.context
        db: AsyncSession = context.db

        # Get all subscriptions except FREE
        result = await db.execute(
            select(Subscription)
            .where(Subscription.subscription_plan != SubscriptionPlan.ESSENTIAL)
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
                    bdt_amount=sub.bdt_amount,
                    subscription_plan=SubscriptionPlanEnum(sub.subscription_plan.value),
                    points_per_month=sub.points_per_month,
                    points_per_day=sub.points_per_day,
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

    @strawberry.field
    async def models(self, info) -> ModelsResponse:
        context = info.context
        db: AsyncSession = context.db

        # Get all models ordered by display_order
        result = await db.execute(select(Model).order_by(Model.display_order.asc()))
        models = result.scalars().all()

        model_types = []
        for model in models:
            model_types.append(
                ModelType(
                    id=model.id,
                    display_name=model.display_name,
                    use_case=model.use_case.value,
                    llm_model_name=model.llm_model_name,
                    group_category=model.group_category.value,
                    display_order=model.display_order,
                )
            )

        return ModelsResponse(
            success=True,
            message="Getting models successfully",
            models=model_types,
        )
