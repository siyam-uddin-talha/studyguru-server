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
    is_liked: Optional[bool] = None
    liked_at: Optional[datetime] = None
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
    is_pinned: Optional[bool] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    file: Optional[MediaType] = None
    conversations: Optional[List[ConversationType]] = None


@strawberry.input
class LikeDislikeInput:
    conversation_id: str
    is_liked: bool  # True for like, False for dislike


@strawberry.type
class InteractionShareType:
    id: str
    original_interaction_id: str
    share_id: str
    is_public: Optional[bool] = None
    visit_count: Optional[int] = None
    last_visited_at: Optional[datetime] = None
    created_at: Optional[datetime] = None


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
    max_tokens: Optional[int] = (
        None  # Will be calculated dynamically based on file count
    )
    # Model selection from frontend (e.g., 'gemini-2.5-pro', 'gpt-4.1', 'gpt-5')
    visualize_model: Optional[str] = None  # Model for image/document analysis
    assistant_model: Optional[str] = None  # Model for text conversation


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
class DeleteInteractionsInput:
    interaction_ids: List[str]


@strawberry.input
class CancelGenerationInput:
    interaction_id: Optional[str] = None
    conversation_id: Optional[str] = None


@strawberry.input
class PinInteractionInput:
    interaction_id: str
    is_pinned: bool


@strawberry.input
class ShareInteractionInput:
    interaction_id: str


@strawberry.input
class GetSharedInteractionInput:
    share_id: str


@strawberry.input
class GetShareStatsInput:
    interaction_id: str


@strawberry.type
class ShareInteractionResponse:
    success: bool
    message: Optional[str] = None
    share_id: Optional[str] = None
    share_url: Optional[str] = None


@strawberry.type
class ShareStatsResponse:
    success: bool
    message: Optional[str] = None
    share_stats: Optional[InteractionShareType] = None
    total_coins_earned: Optional[int] = None


@strawberry.type
class InteractionListResponse:
    success: bool
    message: Optional[str] = None
    result: Optional[List[InteractionType]] = None
    total: Optional[int] = None
    has_next_page: Optional[bool] = None


@strawberry.input
class MessageFilter:
    role: Optional[str] = None  # "user" or "assistant" to filter by role
