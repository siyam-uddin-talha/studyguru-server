import strawberry
from typing import Optional, List
from datetime import datetime
from app.models.subscription import Versionize

VersionizeEnum = strawberry.enum(Versionize)


@strawberry.type
class MediaType:
    id: str
    url: str
    file_name: str
    size: float
    meme_type: str
    created_at: Optional[datetime] = None
