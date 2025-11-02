"""
Database seeding script for StudyGuru Pro
"""

import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import AsyncSessionLocal, init_db
from app.models.subscription import Subscription, SubscriptionPlan
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
            # Free Plan
            Subscription(
                subscription_name="Essential",
                usd_amount=0.0,
                bdt_amount=0,
                subscription_plan=SubscriptionPlan.ESSENTIAL,
                points_per_month=30,  # 30 points on signup
                is_addon=False,
            ),
            # Basic Plan
            Subscription(
                subscription_name="Plus",
                usd_amount=1.0,
                bdt_amount=100,
                subscription_plan=SubscriptionPlan.PLUS,
                points_per_month=100,
                is_addon=False,
            ),
            # Pro Plan
            Subscription(
                subscription_name="Elite",
                usd_amount=5.0,
                bdt_amount=500,
                subscription_plan=SubscriptionPlan.ELITE,
                points_per_month=700,
                is_addon=False,
            ),
        ]

        for subscription in subscriptions:
            db.add(subscription)

        await db.commit()
        print("Subscriptions seeded successfully")


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
