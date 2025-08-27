import strawberry
from typing import Optional, List, Dict, Any
from datetime import datetime


@strawberry.type
class MediaType:
    id: str
    original_filename: str
    s3_key: str
    file_type: str
    original_size: float
    compressed_size: Optional[float] = None
    compression_ratio: Optional[float] = None
    created_at: Optional[datetime] = None


@strawberry.type
class InteractionType:
    id: str
    user_id: str
    file_id: str
    analysis_response: Optional[Dict[str, Any]] = None
    question_type: Optional[str] = None
    detected_language: Optional[str] = None
    title: Optional[str] = None
    summary_title: Optional[str] = None
    tokens_used: int
    points_cost: int
    status: str
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    file: Optional[MediaType] = None


@strawberry.type
class InteractionResponse:
    success: bool
    message: Optional[str] = None
    result: Optional[InteractionType] = None


@strawberry.type
class InteractionListResponse:
    success: bool
    message: Optional[str] = None
    result: Optional[List[InteractionType]] = None
    total: Optional[int] = None
    has_next_page: Optional[bool] = None


@strawberry.input
class InteractionUploadInput:
    file: str  # This would be handled differently in actual implementation
    max_tokens: Optional[int] = 1000
