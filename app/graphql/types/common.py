import strawberry
from datetime import datetime
from typing import Optional

@strawberry.type
class DefaultResponse:
    success: bool
    message: Optional[str] = None

@strawberry.type
class CountryType:
    id: str
    name: str
    currency_code: Optional[str] = None
    country_code: Optional[str] = None
    calling_code: Optional[str] = None
    deleted: Optional[bool] = None

@strawberry.input
class CountryInput:
    name: str
    currency_code: Optional[str] = None
    country_code: Optional[str] = None
    calling_code: Optional[str] = None

@strawberry.scalar
class DateTime:
    serialize = lambda v: v.isoformat() if v else None
    parse_value = lambda v: datetime.fromisoformat(v) if v else None
