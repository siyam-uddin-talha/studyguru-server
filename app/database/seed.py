"""
Database seeding script for StudyGuru Pro
"""

import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import AsyncSessionLocal, init_db
from app.models.subscription import (
    Subscription,
    SubscriptionPlan,
    UsageLimit,
    Model,
    UseCase,
)
from app.database.rbac_seed import seed_rbac


async def seed_subscriptions():
    """Seed subscription plans"""
    async with AsyncSessionLocal() as db:
        # Check if subscriptions already exist
        existing = await db.execute(
            select(Subscription).where(
                Subscription.subscription_plan == SubscriptionPlan.ESSENTIAL
            )
        )
        if existing.scalar_one_or_none():
            print("Subscriptions already seeded")
            return

        subscriptions = [
            # Free Plan (ESSENTIAL)
            Subscription(
                subscription_name="Essential",
                usd_amount=0.0,
                bdt_amount=0,
                subscription_plan=SubscriptionPlan.ESSENTIAL,
                points_per_month=3000,  # 100 points/day * 30 days = 3000 points/month
                points_per_day=100,  # 100 points/day = 10k tokens/day
                is_addon=False,
                is_daily_refill=True,  # Daily cron refill
            ),
            # Plus Plan
            Subscription(
                subscription_name="Plus",
                usd_amount=4.99,
                bdt_amount=499,
                subscription_plan=SubscriptionPlan.PLUS,
                points_per_month=9000,  # 900k tokens/month = 9000 points/month
                points_per_day=None,
                is_addon=False,
                is_daily_refill=False,  # Monthly allocation
            ),
            # Elite Plan
            Subscription(
                subscription_name="Elite",
                usd_amount=14.99,
                bdt_amount=1499,
                subscription_plan=SubscriptionPlan.ELITE,
                points_per_month=20000,  # 2M tokens/month = 20000 points/month
                points_per_day=None,
                is_addon=False,
                is_daily_refill=False,  # Monthly allocation
            ),
        ]

        for subscription in subscriptions:
            db.add(subscription)

        await db.commit()
        print("Subscriptions seeded successfully")

        # Get subscription IDs for usage limits and models
        essential_sub = await db.execute(
            select(Subscription).where(
                Subscription.subscription_plan == SubscriptionPlan.ESSENTIAL
            )
        )
        essential = essential_sub.scalar_one()

        plus_sub = await db.execute(
            select(Subscription).where(
                Subscription.subscription_plan == SubscriptionPlan.PLUS
            )
        )
        plus = plus_sub.scalar_one()

        elite_sub = await db.execute(
            select(Subscription).where(
                Subscription.subscription_plan == SubscriptionPlan.ELITE
            )
        )
        elite = elite_sub.scalar_one()

        # Seed Usage Limits
        await seed_usage_limits(db, essential.id, plus.id, elite.id)

        # Seed Models
        await seed_models(db, essential.id, plus.id, elite.id)


async def seed_usage_limits(
    db: AsyncSession, essential_id: str, plus_id: str, elite_id: str
):
    """Seed usage limits for each subscription plan"""
    usage_limits = [
        # ESSENTIAL (FREE) Plan
        UsageLimit(
            subscription_id=essential_id,
            daily_token_limit=10000,  # 10k tokens/day
            monthly_token_limit=-1,  # Unlimited monthly (daily refill)
            daily_ads_limit=10,  # 10 ads per day
            ad_interval_minutes=3,  # 3 minutes between ads
            mini_model_daily_tokens=10000,  # 10k tokens/day for mini-model
        ),
        # PLUS Plan
        UsageLimit(
            subscription_id=plus_id,
            daily_token_limit=-1,  # Unlimited daily
            monthly_token_limit=900000,  # 900k tokens/month
            daily_ads_limit=-1,  # No ads
            ad_interval_minutes=-1,  # No ads
            mini_model_daily_tokens=10000,  # 10k tokens/day for mini-model
        ),
        # ELITE Plan
        UsageLimit(
            subscription_id=elite_id,
            daily_token_limit=-1,  # Unlimited daily
            monthly_token_limit=2000000,  # 2M tokens/month
            daily_ads_limit=-1,  # No ads
            ad_interval_minutes=-1,  # No ads
            mini_model_daily_tokens=10000,  # 10k tokens/day for mini-model
        ),
    ]

    for usage_limit in usage_limits:
        db.add(usage_limit)

    await db.commit()
    print("Usage limits seeded successfully")


async def seed_models(db: AsyncSession, essential_id: str, plus_id: str, elite_id: str):
    """Seed models for each subscription plan"""
    models = [
        # ESSENTIAL (FREE) Plan - row1_models
        Model(
            subscription_id=essential_id,
            display_name="Gemini 3 Pro",
            use_case=UseCase.VISUALIZE,
            llm_model_name="gemini-3-pro-preview",
            daily_calling_limit=None,  # Unlimited until tokens end
            tier="row1",
        ),
        Model(
            subscription_id=essential_id,
            display_name="Gemini 2.5 Flash",
            use_case=UseCase.CHAT,
            llm_model_name="gemini-2.5-flash",
            daily_calling_limit=None,  # Unlimited until tokens end
            tier="row1",
        ),
        # ESSENTIAL - mini model
        Model(
            subscription_id=essential_id,
            display_name="Gemini Flash",
            use_case=UseCase.CHAT,
            llm_model_name="gemini-mini",
            daily_calling_limit=None,  # Unlimited until tokens end
            tier="mini",
        ),
        # PLUS Plan - row2_models
        Model(
            subscription_id=plus_id,
            display_name="Gemini 3 Pro Preview",
            use_case=UseCase.VISUALIZE,
            llm_model_name="gemini-3-pro-preview",
            daily_calling_limit=None,  # Unlimited until tokens end
            tier="row2",
        ),
        Model(
            subscription_id=plus_id,
            display_name="Gemini 2.5 Pro",
            use_case=UseCase.CHAT,
            llm_model_name="gemini-2.5-pro",
            daily_calling_limit=None,  # Unlimited until tokens end
            tier="row2",
        ),
        # PLUS - mini model
        Model(
            subscription_id=plus_id,
            display_name="Gemini Mini",
            use_case=UseCase.CHAT,
            llm_model_name="gemini-mini",
            daily_calling_limit=None,  # Unlimited until tokens end
            tier="mini",
        ),
        # ELITE Plan - row1_models (same as ESSENTIAL)
        Model(
            subscription_id=elite_id,
            display_name="Gemini 3 Pro Preview",
            use_case=UseCase.VISUALIZE,
            llm_model_name="gemini-3-pro-preview",
            daily_calling_limit=None,  # Unlimited until tokens end
            tier="row1",
        ),
        Model(
            subscription_id=elite_id,
            display_name="Gemini 2.5 Flash",
            use_case=UseCase.CHAT,
            llm_model_name="gemini-2.5-flash",
            daily_calling_limit=None,  # Unlimited until tokens end
            tier="row1",
        ),
        # ELITE - mini model
        Model(
            subscription_id=elite_id,
            display_name="Gemini Mini",
            use_case=UseCase.CHAT,
            llm_model_name="gemini-mini",
            daily_calling_limit=None,  # Unlimited until tokens end
            tier="mini",
        ),
    ]

    for model in models:
        db.add(model)

    await db.commit()
    print("Models seeded successfully")


async def main():
    """Run all seeding functions"""
    print("Starting database seeding...")

    # Initialize database connection
    # await init_db()

    # # Seed subscriptions
    # await seed_subscriptions()

    # # Seed RBAC system
    # await seed_rbac()

    print("\n🎉 Database seeding completed!")


if __name__ == "__main__":
    asyncio.run(main())
