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
    ModelGroup,
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

        # Seed Models (no subscription link - available to all plans)
        await seed_models(db)


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


async def seed_models(db: AsyncSession):
    """Seed models available to all subscription plans"""
    # Check if models already exist
    existing = await db.execute(select(Model))
    if existing.scalars().first():
        print("Models already seeded")
        return

    models = [
        # Order 1: Gemini 2.5 Flash
        Model(
            display_name="Gemini 2.5 Flash",
            use_case=UseCase.CHAT,
            llm_model_name="gemini-2.5-flash",
            group_category=ModelGroup.GEMINI,
            display_order=1,
            daily_calling_limit=None,  # Unlimited until tokens end
            tier=None,
        ),
        # Order 2: GPT 5 Mini
        Model(
            display_name="GPT 5 Mini",
            use_case=UseCase.CHAT,
            llm_model_name="gpt-5-mini",
            group_category=ModelGroup.GPT,
            display_order=2,
            daily_calling_limit=None,  # Unlimited until tokens end
            tier=None,
        ),
        # Order 3: Gemini 2.5 Pro
        Model(
            display_name="Gemini 2.5 Pro",
            use_case=UseCase.CHAT,
            llm_model_name="gemini-2.5-pro",
            group_category=ModelGroup.GEMINI,
            display_order=3,
            daily_calling_limit=None,  # Unlimited until tokens end
            tier=None,
        ),
        # Order 4: GPT 5
        Model(
            display_name="GPT 5",
            use_case=UseCase.CHAT,
            llm_model_name="gpt-5",
            group_category=ModelGroup.GPT,
            display_order=4,
            daily_calling_limit=None,  # Unlimited until tokens end
            tier=None,
        ),
        # Order 5: GPT 5.1
        Model(
            display_name="GPT 5.1",
            use_case=UseCase.CHAT,
            llm_model_name="gpt-5.1",
            group_category=ModelGroup.GPT,
            display_order=5,
            daily_calling_limit=None,  # Unlimited until tokens end
            tier=None,
        ),
        # Order 6: Kimi K2 Turbo
        Model(
            display_name="Kimi K2 Turbo",
            use_case=UseCase.CHAT,
            llm_model_name="kimi-k2-turbo-preview",
            group_category=ModelGroup.KIMI,
            display_order=6,
            daily_calling_limit=None,  # Unlimited until tokens end
            tier=None,
        ),
        # Order 7: Gemini 3 Pro Preview
        Model(
            display_name="Gemini 3 Pro",
            use_case=UseCase.CHAT,
            llm_model_name="gemini-3-pro-preview",
            group_category=ModelGroup.GEMINI,
            display_order=7,
            daily_calling_limit=None,  # Unlimited until tokens end
            tier=None,
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
