from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from sqlalchemy import select
import uuid
from app.models.pivot import Country


async def get_or_create_country(db: AsyncSession, country_code: str, **kwargs):
    """
    Get an existing country by country_code or create a new one if it doesn't exist.

    Args:
        db (AsyncSession): SQLAlchemy async database session
        country_code (str): The country code to search for or use when creating
        **kwargs: Additional fields for country creation (name, currency_code, calling_code)

    Returns:
        Country: Country instance

    Raises:
        ValueError: If required fields are missing when creating a new country
    """
    # First, try to get existing country (including soft-deleted ones if needed)
    stmt = select(Country).where(
        Country.country_code == country_code, Country.deleted == False
    )
    result = await db.execute(stmt)
    country = result.scalar_one_or_none()

    if country:
        return country

    # Country doesn't exist, create a new one
    # Check for required fields
    if "name" not in kwargs or "currency_code" not in kwargs:
        raise ValueError(
            "name and currency_code are required when creating a new country"
        )

    try:
        new_country = Country(
            id=str(uuid.uuid4()),
            country_code=country_code,
            name=kwargs["name"],
            currency_code=kwargs["currency_code"],
            calling_code=kwargs.get("calling_code"),
            deleted=False,
        )

        db.add(new_country)
        await db.commit()
        await db.refresh(new_country)

        return new_country

    except IntegrityError as e:
        await db.rollback()
        # Handle race condition - another process might have created the country
        stmt = select(Country).where(
            Country.country_code == country_code, Country.deleted == False
        )
        result = await db.execute(stmt)
        country = result.scalar_one_or_none()

        if country:
            return country
        else:
            # Re-raise the original error if it's not a race condition
            raise e


async def get_or_create_country_from_object(db: AsyncSession, country_obj):
    """Wrapper that accepts a Country object and calls the main function"""
    return await get_or_create_country(
        db=db,
        country_code=country_obj.country_code,
        name=country_obj.name,
        currency_code=getattr(country_obj, "currency_code", "USD"),
        calling_code=getattr(country_obj, "calling_code", None),
    )
