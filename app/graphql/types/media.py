import strawberry
from typing import Optional, List
from datetime import datetime
from app.models.subscription import Versionize


@strawberry.enum
class VersionizeEnum(Versionize):
    FREE = "free"
    PAID = "paid"


@strawberry.type
class MediaType:
    id: str
    url: str
    file_name: str
    size: float
    meme_type: str
    created_at: Optional[datetime] = None


@strawberry.type
class ImageDeckType:
    id: str
    media_id: str
    versionize: VersionizeEnum
    media: Optional[MediaType] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@strawberry.type
class ImageDeckResponse:
    success: bool
    message: Optional[str] = None
    result: Optional[List[ImageDeckType]] = None
    total: Optional[int] = None
    has_next_page: Optional[bool] = None
    cursor: Optional[str] = None