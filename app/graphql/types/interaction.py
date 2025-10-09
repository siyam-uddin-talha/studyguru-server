import strawberry
from typing import Optional, List, Dict, Any
from datetime import datetime
from strawberry.scalars import JSON
from app.graphql.types.points import PointTransactionType
from app.graphql.types.media import MediaType


@strawberry.type
class ConversationType:
    id: str
    interaction_id: str
    role: str
    content: Optional[JSON] = None
    question_type: Optional[str] = None
    detected_language: Optional[str] = None
    tokens_used: int
    points_cost: int
    status: str
    is_hidden: Optional[bool] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    files: Optional[List[MediaType]] = None
    point_transaction: Optional[PointTransactionType] = None


@strawberry.type
class InteractionType:
    id: str
    user_id: str
    title: Optional[str] = None
    summary_title: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    file: Optional[MediaType] = None
    conversations: Optional[List[ConversationType]] = None


@strawberry.type
class InteractionResponse:
    success: bool
    message: str
    result: Optional[InteractionType] = None
    interaction_id: Optional[str] = None
    is_new_interaction: Optional[bool] = None
    interaction: Optional[InteractionType] = None
    ai_response: Optional[str] = None


@strawberry.input
class MediaFileInput:
    id: str
    url: Optional[str] = None


@strawberry.input
class DoConversationInput:
    interaction_id: Optional[str] = None
    message: Optional[str] = ""
    media_files: Optional[List[MediaFileInput]] = None  # List of {id: str, url?: str}
    max_tokens: Optional[int] = 500


@strawberry.input
class DeleteMediaFileInput:
    media_id: str


@strawberry.input
class UpdateInteractionTitleInput:
    interaction_id: str
    title: str


@strawberry.input
class DeleteInteractionInput:
    interaction_id: str


@strawberry.input
class CancelGenerationInput:
    interaction_id: str
    conversation_id: Optional[str] = None


@strawberry.type
class InteractionListResponse:
    success: bool
    message: Optional[str] = None
    result: Optional[List[InteractionType]] = None
    total: Optional[int] = None
    has_next_page: Optional[bool] = None
