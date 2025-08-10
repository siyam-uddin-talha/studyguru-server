import asyncio
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from app.core.database import AsyncSessionLocal
from app.models.user import User
from app.models.subscription import PurchasedSubscription, SubscriptionPlan
from app.helpers.email import send_email
from app.helpers.email_templates import generate_welcome_email


async def cleanup_inactive_users():
    """Clean up inactive users (runs daily)"""
    print("Running daily inactive user cleanup...")

    six_months_ago = datetime.utcnow() - timedelta(days=180)

    async with AsyncSessionLocal() as db:
        try:
            # Find inactive free users
            result = await db.execute(
                select(User)
                .join(PurchasedSubscription)
                .join(PurchasedSubscription.subscription)
                .where(
                    and_(
                        User.last_login_at < six_months_ago,
                        PurchasedSubscription.subscription.subscription_plan
                        == SubscriptionPlan.ESSENTIAL,
                    )
                )
            )
            inactive_users = result.scalars().all()

            for user in inactive_users:
                # Delete user and related data
                await delete_user_data(db, user.id)

                # Send deletion notice email
                await send_email(
                    user.email,
                    "Account Deletion Notice",
                    f"Your account has been deleted due to inactivity.",
                )

            await db.commit()
            print(f"Deleted {len(inactive_users)} inactive users.")

        except Exception as e:
            print(f"Error during inactive user cleanup: {e}")
            await db.rollback()


async def cancel_past_due_subscriptions():
    """Cancel past due subscriptions (runs daily)"""
    print("Running cron job to check past-due subscriptions...")

    one_day_ago = datetime.utcnow() - timedelta(days=1)

    async with AsyncSessionLocal() as db:
        try:
            # Find past due subscriptions
            result = await db.execute(
                select(PurchasedSubscription).where(
                    and_(
                        PurchasedSubscription.subscription_status == "past_due",
                        PurchasedSubscription.updated_at < one_day_ago,
                        PurchasedSubscription.paddle_subscription_id.isnot(None),
                    )
                )
            )
            past_due_subs = result.scalars().all()

            for sub in past_due_subs:
                try:
                    # Cancel in Paddle (or handle according to your payment provider)
                    if sub.paddle_subscription_id:
                        # TODO: Implement Paddle subscription cancellation
                        # For now, just update local status
                        pass

                    # Update local status
                    await update_to_free_subscription(db, sub.id)

                except Exception as e:
                    print(f"Error canceling subscription {sub.id}: {e}")

            await db.commit()
            print(f"Processed {len(past_due_subs)} past due subscriptions.")

        except Exception as e:
            print(f"Error during past due subscription cleanup: {e}")
            await db.rollback()


async def delete_user_data(db: AsyncSession, user_id: str):
    """Delete all user data"""
    # TODO: Implement cascade deletion of user data
    # This should delete client_sessions, image_groups, group_image_deck, billing_logs, etc.
    pass


async def update_to_free_subscription(db: AsyncSession, purchased_sub_id: str):
    """Update subscription to free plan"""
    # TODO: Implement logic to downgrade to free subscription
    pass


def start_scheduler():
    """Start background tasks"""
    print("Starting background scheduler...")

    # In a real implementation, you would use a proper task scheduler like Celery
    # For now, we'll just print that the scheduler is started
    # You could use asyncio.create_task() to run these periodically

    # Example of how you might schedule tasks:
    # asyncio.create_task(schedule_daily_tasks())


async def schedule_daily_tasks():
    """Schedule daily tasks"""
    while True:
        now = datetime.utcnow()
        # Run at midnight UTC
        if now.hour == 0 and now.minute == 0:
            await cleanup_inactive_users()
            await cancel_past_due_subscriptions()

        # Wait 1 minute before checking again
        await asyncio.sleep(60)
