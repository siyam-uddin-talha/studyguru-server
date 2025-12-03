import asyncio
from sqlalchemy import text
from app.core.database import Base, engine  # import your Base and engine


async def reset_db():
    """⚠ Dev only: Drop and recreate all tables."""
    try:
        async with engine.begin() as conn:
            from app.models import (
                user,
                media,
                pivot,
                subscription,
                interaction,
                goal,
                note,
                rbac,
                context,
            )

            # Disable foreign key checks for MySQL/MariaDB
            await conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))

            # Drop all tables
            await conn.run_sync(Base.metadata.drop_all)

            # Re-enable foreign key checks
            await conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))

            # Recreate all tables
            await conn.run_sync(Base.metadata.create_all)
        print("✅ Database reset successfully.")
    except Exception as e:
        print(f"❌ Error resetting database: {e}")


if __name__ == "__main__":
    asyncio.run(reset_db())
