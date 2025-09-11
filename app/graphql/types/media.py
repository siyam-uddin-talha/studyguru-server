import strawberry
from typing import Optional, List
from datetime import datetime


@strawberry.type
class MediaType:
    id: str
    url: str
    file_name: str
    size: float
    meme_type: str
    created_at: Optional[datetime] = None
    original_filename: str
    s3_key: str
    file_type: str
    original_size: float
    compressed_size: Optional[float] = None
    compression_ratio: Optional[float] = None
