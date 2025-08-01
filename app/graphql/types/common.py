import strawberry
from datetime import datetime
from typing import Optional


@strawberry.type
class DefaultResponse:
    success: bool
    message: Optional[str] = None


@strawberry.scalar
class DateTime:
    serialize = lambda v: v.isoformat() if v else None
    parse_value = lambda v: datetime.fromisoformat(v) if v else None