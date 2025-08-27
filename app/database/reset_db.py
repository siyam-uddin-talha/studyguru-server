import asyncio
from app.core.database import Base, engine  # import your Base and engine


async def reset_db():
    """⚠ Dev only: Drop and recreate all tables."""
    try:
        async with engine.begin() as conn:
            from app.models import user, media, pivot, subscription, interaction

            # Drop all tables
            await conn.run_sync(Base.metadata.drop_all)
            # Recreate all tables
            await conn.run_sync(Base.metadata.create_all)
        print("✅ Database reset successfully.")
    except Exception as e:
        print(f"❌ Error resetting database: {e}")


if __name__ == "__main__":
    asyncio.run(reset_db())
