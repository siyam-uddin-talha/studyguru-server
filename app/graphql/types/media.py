import strawberry
from typing import Optional, List
from datetime import datetime


@strawberry.type
class MediaType:
    id: str
    original_filename: str
    s3_key: str
    file_type: str
    meme_type: Optional[str] = None
    original_size: float
    compressed_size: Optional[float] = None
    compression_ratio: Optional[float] = None
    created_at: Optional[datetime] = None
