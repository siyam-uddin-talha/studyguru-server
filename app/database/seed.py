"""
Database seeding script for StudyGuru Pro
"""
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import AsyncSessionLocal
from app.models.subscription import Subscription, SubscriptionPlan


async def seed_subscriptions():
    """Seed subscription plans"""
    async with AsyncSessionLocal() as db:
        # Check if subscriptions already exist
        existing = await db.execute(
            select(Subscription).where(Subscription.subscription_plan == SubscriptionPlan.FREE)
        )
        if existing.scalar_one_or_none():
            print("Subscriptions already seeded")
            return
        
        subscriptions = [
            # Free Plan
            Subscription(
                subscription_name="Free Plan",
                usd_amount=0.0,
                subscription_plan=SubscriptionPlan.FREE,
                points_per_month=30,  # 30 points on signup
                is_addon=False
            ),
            
            # Basic Plan
            Subscription(
                subscription_name="Basic Plan",
                usd_amount=1.0,
                subscription_plan=SubscriptionPlan.BASIC,
                points_per_month=100,
                is_addon=False
            ),
            
            # Pro Plan
            Subscription(
                subscription_name="Pro Plan",
                usd_amount=5.0,
                subscription_plan=SubscriptionPlan.PRO,
                points_per_month=700,
                is_addon=False
            ),
            
            # Point Add-on
            Subscription(
                subscription_name="Additional Points",
                usd_amount=0.01,  # $0.01 per point
                subscription_plan=SubscriptionPlan.FREE,  # Not tied to a specific plan
                points_per_month=1,  # 1 point per $0.01
                is_addon=True,
                min_points=100  # Minimum 100 points purchase
            )
        ]
        
        for subscription in subscriptions:
            db.add(subscription)
        
        await db.commit()
        print("Subscriptions seeded successfully")


async def main():
    """Run all seeding functions"""
    print("Starting database seeding...")
    await seed_subscriptions()
    print("Database seeding completed!")


if __name__ == "__main__":
    asyncio.run(main())