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
    title: Optional[str] = None
    summary_title: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@strawberry.type
class ConversationType:
    id: str
    user_id: str
    analysis_response: Optional[Dict[str, Any]] = None
    question_type: Optional[str] = None
    detected_language: Optional[str] = None
    tokens_used: int
    points_cost: int
    status: str
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    files: Optional[List[MediaType]] = None


@strawberry.type
class InteractionResponse:
    success: bool
    message: str
    result: Optional[ConversationType] = None
    interaction_id: Optional[str] = None
    is_new_interaction: Optional[bool] = None
    interaction: Optional[InteractionType] = None
    ai_response: Optional[str] = None  # The actual AI response content


@strawberry.input
class DoConversationInput:
    interaction_id: Optional[str] = None
    message: Optional[str] = ""
    media_files: Optional[List[str]] = None  # List of media IDs
    max_tokens: Optional[int] = 500


@strawberry.type
class InteractionListResponse:
    success: bool
    message: Optional[str] = None
    result: Optional[List[InteractionType]] = None
    total: Optional[int] = None
    has_next_page: Optional[bool] = None
