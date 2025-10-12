from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import MetaData
from app.core.config import settings

# Create async engine
# Ensure the URL uses the correct format for asyncmy
db_url = settings.DATABASE_URL
if db_url.startswith("mysql://"):
    db_url = db_url.replace("mysql://", "mysql+asyncmy://", 1)
elif "mysql+asyncpg://" in db_url:
    # Fix common mistake: asyncpg is for PostgreSQL, not MySQL
    db_url = db_url.replace("mysql+asyncpg://", "mysql+asyncmy://", 1)

engine = create_async_engine(db_url, echo=settings.ENVIRONMENT == "development")

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


# Base class for models
class Base(DeclarativeBase):
    metadata = MetaData()


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Initialize database tables"""
    try:
        async with engine.begin() as conn:
            # Import all models here to ensure they are registered
            from app.models import user, subscription, media, pivot, interaction

            # Create tables if they don't exist
            await conn.run_sync(Base.metadata.create_all)
        print("Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {e}")
