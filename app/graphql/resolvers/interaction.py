import strawberry
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func, or_
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import selectinload

from app.graphql.types.interaction import (
    InteractionResponse,
    InteractionListResponse,
    InteractionType,
    InteractionShareType,
    ConversationType,
    MediaType,
    DeleteMediaFileInput,
    UpdateInteractionTitleInput,
    CancelGenerationInput,
    DeleteInteractionInput,
    DeleteInteractionsInput,
    PinInteractionInput,
    ShareInteractionInput,
    GetSharedInteractionInput,
    GetShareStatsInput,
    ShareInteractionResponse,
    ShareStatsResponse,
    LikeDislikeInput,
    MessageFilter,
)
from app.models.interaction import Interaction, InteractionShare
from app.models.media import Media
from app.models.interaction import Conversation, ConversationRole
from app.core.config import settings
from app.models.user import User

from app.services.interaction import process_conversation_message, cancel_ai_generation
from app.services.file_service import FileService
from app.helpers.user import get_current_user_from_context
from app.helpers.subscription import award_share_visit_reward
from app.constants.constant import CONSTANTS
from app.graphql.types.common import DefaultResponse


@strawberry.type
class BackgroundTaskStatus:
    task_id: str
    task_type: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0


@strawberry.type
class BackgroundMessageTaskStatus:
    task_id: str
    user_id: str
    interaction_id: Optional[str]
    message: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    streaming_content: str = ""


@strawberry.type
class InteractionQuery:
    @strawberry.field
    async def interactions(
        self,
        info,
        page: Optional[int] = 1,
        size: Optional[int] = 10,
        query: Optional[str] = None,
    ) -> InteractionListResponse:
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return InteractionListResponse(
                success=False, message="Authentication required"
            )

        db: AsyncSession = context.db

        # Calculate pagination
        offset = (page - 1) * size

        # Get user's interaction materials - pinned first, then by creation date
        # Also load document context to get directly associated files
        from app.models.context import DocumentContext

        # Build base query
        base_query = (
            select(Interaction)
            .options(
                selectinload(Interaction.conversations).selectinload(Conversation.files)
            )
            .where(Interaction.user_id == current_user.id)
        )

        # Add search filter if query is provided
        if query and query.strip():
            search_term = f"%{query.strip()}%"
            base_query = base_query.where(
                or_(
                    Interaction.title.ilike(search_term),
                    Interaction.summary_title.ilike(search_term),
                )
            )

        # Execute query with ordering and pagination
        result = await db.execute(
            base_query.order_by(
                desc(Interaction.is_pinned), desc(Interaction.created_at)
            )
            .offset(offset)
            .limit(size)
        )
        interactions = result.scalars().all()

        # Get total count efficiently with same search filter
        count_query = (
            select(func.count())
            .select_from(Interaction)
            .where(Interaction.user_id == current_user.id)
        )

        # Add search filter to count query if query is provided
        if query and query.strip():
            search_term = f"%{query.strip()}%"
            count_query = count_query.where(
                or_(
                    Interaction.title.ilike(search_term),
                    Interaction.summary_title.ilike(search_term),
                )
            )

        total_result = await db.scalar(count_query)
        total = total_result or 0
        # print("total", total)
        # Convert to response types
        doc_material_types = []
        for interaction in interactions:
            # Get associated media through conversations
            # For now, we'll get the first media file from the first conversation
            # In the future, you might want to return all files or handle multiple files differently
            media = None

            # Get media from conversations
            if interaction.conversations:
                for conv in interaction.conversations:
                    if conv.files:
                        media = conv.files[0]  # Get first file
                        break

            doc_material_types.append(
                InteractionType(
                    id=interaction.id,
                    user_id=interaction.user_id,
                    title=interaction.title,
                    summary_title=interaction.summary_title,
                    is_pinned=interaction.is_pinned,
                    created_at=interaction.created_at,
                    updated_at=interaction.updated_at,
                    file=(
                        MediaType(
                            id=media.id,
                            original_filename=media.original_filename,
                            s3_key=media.s3_key,
                            file_type=media.file_type,
                            meme_type=media.meme_type,
                            original_size=media.original_size,
                            compressed_size=media.compressed_size,
                            compression_ratio=media.compression_ratio,
                            created_at=media.created_at,
                        )
                        if media
                        else None
                    ),
                )
            )

        return InteractionListResponse(
            success=True,
            message="Doc materials retrieved successfully",
            result=doc_material_types,
            total=total,
            has_next_page=(offset + size) < total,
        )

    @strawberry.field
    async def messages(
        self,
        info,
        id: str,
        limit: Optional[int] = 12,
        offset: Optional[int] = 0,
        filter: Optional[MessageFilter] = None,
    ) -> List[ConversationType]:
        """
        Get messages (conversations) for a specific interaction with pagination
        Supports filtering by role (user or assistant)
        """
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return []

        db: AsyncSession = context.db

        # Build the base query
        base_query = (
            select(Conversation)
            .options(selectinload(Conversation.files))
            .join(Interaction)
            .where(
                Interaction.id == id,
                Interaction.user_id == current_user.id,
                Conversation.interaction_id == id,
            )
        )

        # Apply role filter if provided
        if filter and filter.role:
            # Map client role values to enum values
            # Client sends "user" or "assistant", we need to map to ConversationRole enum
            role_mapping = {
                "user": ConversationRole.USER,
                "assistant": ConversationRole.AI,
            }
            mapped_role = role_mapping.get(filter.role.lower())
            if mapped_role:
                base_query = base_query.where(Conversation.role == mapped_role)

        # Get conversations for the interaction with pagination
        result = await db.execute(
            base_query.order_by(desc(Conversation.created_at))
            .offset(offset)
            .limit(limit)
        )
        conversations = result.scalars().all()

        # Convert to ConversationType
        conversation_types = []
        for conv in conversations:
            # Convert files to MediaType
            media_files = []
            if conv.files:
                for file in conv.files:
                    media_files.append(
                        MediaType(
                            id=file.id,
                            original_filename=file.original_filename,
                            s3_key=file.s3_key,
                            file_type=file.file_type,
                            meme_type=file.meme_type,
                            original_size=file.original_size,
                            compressed_size=file.compressed_size,
                            compression_ratio=file.compression_ratio,
                            created_at=file.created_at,
                        )
                    )

            # Process content to resolve media_ids to proper file information
            processed_content = conv.content
            if conv.content and isinstance(conv.content, dict):
                # Check if content has media_ids that need to be resolved
                if (
                    conv.content.get("_result")
                    and conv.content["_result"].get("media_ids")
                    and isinstance(conv.content["_result"]["media_ids"], list)
                ):

                    # Fetch media files for the media_ids
                    media_ids = conv.content["_result"]["media_ids"]
                    media_result = await db.execute(
                        select(Media).where(Media.id.in_(media_ids))
                    )
                    media_files_for_content = media_result.scalars().all()

                    # Create media_urls from the fetched media files
                    media_urls = []
                    for media_file in media_files_for_content:
                        media_urls.append(
                            {
                                "id": media_file.id,
                                "url": f"https://{settings.AWS_S3_BUCKET}.s3.amazonaws.com/{media_file.s3_key}",
                                "type": media_file.file_type,
                                "name": media_file.original_filename,
                                "size": media_file.original_size,
                            }
                        )

                    # Update the content with media_urls
                    processed_content = conv.content.copy()
                    processed_content["_result"]["media_urls"] = media_urls

            conversation_types.append(
                ConversationType(
                    id=conv.id,
                    interaction_id=conv.interaction_id,
                    role=conv.role.value if conv.role else "USER",
                    content=processed_content,
                    question_type=conv.question_type,
                    detected_language=conv.detected_language,
                    tokens_used=conv.tokens_used or 0,
                    points_cost=conv.points_cost or 0,
                    status=conv.status,
                    is_hidden=conv.is_hidden,
                    error_message=conv.error_message,
                    is_liked=conv.is_liked,
                    liked_at=conv.liked_at,
                    created_at=conv.created_at,
                    updated_at=conv.updated_at,
                    files=media_files if media_files else None,
                    point_transaction=None,  # TODO: Add point transaction if needed
                )
            )

        return conversation_types

    @strawberry.field
    async def interaction(self, info, id: str) -> InteractionResponse:
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return InteractionResponse(success=False, message=CONSTANTS.NOT_FOUND)

        db: AsyncSession = context.db

        # Get interaction material
        result = await db.execute(
            select(Interaction)
            .options(
                selectinload(Interaction.conversations).selectinload(Conversation.files)
            )
            .where(Interaction.id == id, Interaction.user_id == current_user.id)
        )
        interaction = result.scalar_one_or_none()

        if not interaction:
            return InteractionResponse(success=False, message="Document not found")

        # Get associated media through conversations
        media = None
        conversation_types = []
        if interaction.conversations:
            for conv in interaction.conversations:
                if conv.files and not media:
                    media = conv.files[0]  # Get first file

                # Convert files to MediaType
                media_files = []
                if conv.files:
                    for file in conv.files:
                        media_files.append(
                            MediaType(
                                id=file.id,
                                original_filename=file.original_filename,
                                s3_key=file.s3_key,
                                file_type=file.file_type,
                                meme_type=file.meme_type,
                                original_size=file.original_size,
                                compressed_size=file.compressed_size,
                                compression_ratio=file.compression_ratio,
                                created_at=file.created_at,
                            )
                        )

                conversation_types.append(
                    ConversationType(
                        id=conv.id,
                        interaction_id=conv.interaction_id,
                        role=conv.role.value if conv.role else "USER",
                        content=conv.content,
                        question_type=conv.question_type,
                        detected_language=conv.detected_language,
                        tokens_used=conv.tokens_used or 0,
                        points_cost=conv.points_cost or 0,
                        status=conv.status,
                        is_hidden=conv.is_hidden,
                        error_message=conv.error_message,
                        is_liked=conv.is_liked,
                        liked_at=conv.liked_at,
                        created_at=conv.created_at,
                        updated_at=conv.updated_at,
                        files=media_files if media_files else None,
                        point_transaction=None,  # TODO: Add point transaction if needed
                    )
                )

        return InteractionResponse(
            success=True,
            message="Document retrieved successfully",
            result=InteractionType(
                id=interaction.id,
                user_id=interaction.user_id,
                title=interaction.title,
                summary_title=interaction.summary_title,
                is_pinned=interaction.is_pinned,
                created_at=interaction.created_at,
                updated_at=interaction.updated_at,
                file=(
                    MediaType(
                        id=media.id,
                        original_filename=media.original_filename,
                        s3_key=media.s3_key,
                        file_type=media.file_type,
                        meme_type=media.meme_type,
                        original_size=media.original_size,
                        compressed_size=media.compressed_size,
                        compression_ratio=media.compression_ratio,
                        created_at=media.created_at,
                    )
                    if media
                    else None
                ),
                conversations=conversation_types if conversation_types else None,
            ),
        )

    @strawberry.field
    async def background_task_status(
        self, info, task_id: str
    ) -> Optional[BackgroundTaskStatus]:
        """Get the status of a background task"""
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return None

        from app.services.background_task_service import background_task_service

        task = background_task_service.get_task_status(task_id)
        if not task or task.user_id != current_user.id:
            return None

        return BackgroundTaskStatus(
            task_id=task.id,
            task_type=task.task_type,
            status=task.status.value,
            created_at=task.created_at.isoformat(),
            started_at=task.started_at.isoformat() if task.started_at else None,
            completed_at=task.completed_at.isoformat() if task.completed_at else None,
            error_message=task.error_message,
            retry_count=task.retry_count,
        )

    @strawberry.field
    async def user_background_tasks(self, info) -> List[BackgroundTaskStatus]:
        """Get all background tasks for the current user"""
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return []

        from app.services.background_task_service import background_task_service

        user_tasks = background_task_service.get_user_tasks(current_user.id)

        return [
            BackgroundTaskStatus(
                task_id=task.id,
                task_type=task.task_type,
                status=task.status.value,
                created_at=task.created_at.isoformat(),
                started_at=task.started_at.isoformat() if task.started_at else None,
                completed_at=(
                    task.completed_at.isoformat() if task.completed_at else None
                ),
                error_message=task.error_message,
                retry_count=task.retry_count,
            )
            for task in user_tasks
        ]

    @strawberry.field
    async def background_message_task_status(
        self, info, task_id: str
    ) -> Optional[BackgroundMessageTaskStatus]:
        """Get the status of a background message task"""
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return None

        from app.services.background_message_service import background_message_service

        task = background_message_service.get_task_status(task_id)
        if not task or task.user_id != current_user.id:
            return None

        return BackgroundMessageTaskStatus(
            task_id=task.id,
            user_id=task.user_id,
            interaction_id=task.interaction_id,
            message=task.message,
            status=task.status.value,
            created_at=task.created_at.isoformat(),
            started_at=task.started_at.isoformat() if task.started_at else None,
            completed_at=task.completed_at.isoformat() if task.completed_at else None,
            error_message=task.error_message,
            retry_count=task.retry_count,
            streaming_content=task.streaming_content,
        )

    @strawberry.field
    async def user_background_message_tasks(
        self, info
    ) -> List[BackgroundMessageTaskStatus]:
        """Get all background message tasks for the current user"""
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return []

        from app.services.background_message_service import background_message_service

        user_tasks = background_message_service.get_user_tasks(current_user.id)

        return [
            BackgroundMessageTaskStatus(
                task_id=task.id,
                user_id=task.user_id,
                interaction_id=task.interaction_id,
                message=task.message,
                status=task.status.value,
                created_at=task.created_at.isoformat(),
                started_at=task.started_at.isoformat() if task.started_at else None,
                completed_at=(
                    task.completed_at.isoformat() if task.completed_at else None
                ),
                error_message=task.error_message,
                retry_count=task.retry_count,
                streaming_content=task.streaming_content,
            )
            for task in user_tasks
        ]

    @strawberry.field
    async def interaction_background_message_tasks(
        self, info, interaction_id: str
    ) -> List[BackgroundMessageTaskStatus]:
        """Get all background message tasks for a specific interaction"""
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return []

        from app.services.background_message_service import background_message_service

        interaction_tasks = background_message_service.get_interaction_tasks(
            interaction_id
        )

        # Filter tasks that belong to the current user
        user_tasks = [
            task for task in interaction_tasks if task.user_id == current_user.id
        ]

        return [
            BackgroundMessageTaskStatus(
                task_id=task.id,
                user_id=task.user_id,
                interaction_id=task.interaction_id,
                message=task.message,
                status=task.status.value,
                created_at=task.created_at.isoformat(),
                started_at=task.started_at.isoformat() if task.started_at else None,
                completed_at=(
                    task.completed_at.isoformat() if task.completed_at else None
                ),
                error_message=task.error_message,
                retry_count=task.retry_count,
                streaming_content=task.streaming_content,
            )
            for task in user_tasks
        ]

    @strawberry.field
    async def get_shared_interaction(
        self,
        info,
        input: GetSharedInteractionInput,
    ) -> InteractionResponse:
        """
        Get a shared interaction by share_id (returns original interaction data for temporary view)
        Awards 5 coins to the owner when someone visits the shared link (only once per unique visitor)
        """
        context = info.context
        db: AsyncSession = context.db
        current_user = await get_current_user_from_context(context)

        try:
            # Get the share record
            share_result = await db.execute(
                select(InteractionShare).where(
                    InteractionShare.share_id == input.share_id,
                    InteractionShare.is_public == True,
                )
            )
            interaction_share = share_result.scalar_one_or_none()

            if not interaction_share:
                return InteractionResponse(
                    success=False, message="Shared interaction not found"
                )

            # Award coins to the owner for the visit (5 coins per unique visitor)
            # Only reward if visitor is not the owner and hasn't been rewarded before
            visitor_user_id = current_user.id if current_user else None
            # Note: In a real implementation, you'd get visitor_ip from request headers
            # and visitor_fingerprint from client-side device fingerprinting
            await award_share_visit_reward(
                db,
                interaction_share,
                visitor_user_id=visitor_user_id,
                visitor_ip=None,  # Would be extracted from request in real implementation
                visitor_fingerprint=None,  # Would be provided by client in real implementation
                reward_amount=5,
            )

            # Get the original interaction with conversations
            result = await db.execute(
                select(Interaction)
                .options(
                    selectinload(Interaction.conversations).selectinload(
                        Conversation.files
                    )
                )
                .where(Interaction.id == interaction_share.original_interaction_id)
            )
            original_interaction = result.scalar_one_or_none()

            if not original_interaction:
                return InteractionResponse(
                    success=False, message="Original interaction not found"
                )

            # Convert conversations to the expected format
            conversations = []
            for conv in original_interaction.conversations:
                # Get media files for this conversation
                media_files = []
                if conv.files:
                    for media_file in conv.files:
                        media_files.append(
                            MediaType(
                                id=media_file.id,
                                original_filename=media_file.original_filename,
                                s3_key=media_file.s3_key,
                                file_type=media_file.file_type,
                                meme_type=media_file.meme_type,
                                original_size=media_file.original_size,
                                compressed_size=media_file.compressed_size,
                                compression_ratio=media_file.compression_ratio,
                                created_at=media_file.created_at,
                            )
                        )

                conversation = ConversationType(
                    id=conv.id,
                    interaction_id=original_interaction.id,
                    role=conv.role.value,
                    content=conv.content,
                    question_type=conv.question_type,
                    detected_language=conv.detected_language,
                    tokens_used=conv.tokens_used or 0,
                    points_cost=conv.points_cost or 0,
                    status=conv.status,
                    is_hidden=conv.is_hidden,
                    is_liked=conv.is_liked,
                    liked_at=conv.liked_at,
                    error_message=conv.error_message,
                    created_at=conv.created_at,
                    updated_at=conv.updated_at,
                    files=media_files if media_files else None,
                    point_transaction=None,
                )
                conversations.append(conversation)

            # Get associated media through conversations
            media = None
            if original_interaction.conversations:
                for conv in original_interaction.conversations:
                    if conv.files:
                        media = conv.files[0]  # Get first file
                        break

            # Create the interaction type (returning original data for temporary view)
            interaction_type = InteractionType(
                id=original_interaction.id,
                user_id=original_interaction.user_id,
                title=original_interaction.title,
                summary_title=original_interaction.summary_title,
                is_pinned=original_interaction.is_pinned,
                created_at=original_interaction.created_at,
                updated_at=original_interaction.updated_at,
                file=(
                    MediaType(
                        id=media.id,
                        original_filename=media.original_filename,
                        s3_key=media.s3_key,
                        file_type=media.file_type,
                        meme_type=media.meme_type,
                        original_size=media.original_size,
                        compressed_size=media.compressed_size,
                        compression_ratio=media.compression_ratio,
                        created_at=media.created_at,
                    )
                    if media
                    else None
                ),
                conversations=conversations,
            )

            return InteractionResponse(
                success=True,
                message="Shared interaction retrieved successfully",
                result=interaction_type,
                interaction_id=original_interaction.id,
                is_new_interaction=False,
            )

        except Exception as e:
            return InteractionResponse(
                success=False,
                message=f"Failed to retrieve shared interaction: {str(e)}",
            )

    @strawberry.field
    async def get_share_stats(
        self,
        info,
        input: GetShareStatsInput,
    ) -> ShareStatsResponse:
        """
        Get share statistics for an interaction (visits, coins earned, etc.)
        """
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return ShareStatsResponse(success=False, message="Authentication required")

        db: AsyncSession = context.db

        try:
            # Get the share record for this interaction
            share_result = await db.execute(
                select(InteractionShare).where(
                    InteractionShare.original_interaction_id == input.interaction_id
                )
            )
            interaction_share = share_result.scalar_one_or_none()

            if not interaction_share:
                return ShareStatsResponse(
                    success=False, message="No share record found for this interaction"
                )

            # Verify the user owns this interaction
            interaction_result = await db.execute(
                select(Interaction).where(
                    Interaction.id == input.interaction_id,
                    Interaction.user_id == current_user.id,
                )
            )
            interaction = interaction_result.scalar_one_or_none()

            if not interaction:
                return ShareStatsResponse(
                    success=False, message="Interaction not found or access denied"
                )

            # Calculate total coins earned from visits (5 coins per visit)
            total_coins_earned = interaction_share.visit_count * 5

            # Create share stats response
            share_stats = InteractionShareType(
                id=interaction_share.id,
                original_interaction_id=interaction_share.original_interaction_id,
                share_id=interaction_share.share_id,
                is_public=interaction_share.is_public,
                visit_count=interaction_share.visit_count,
                last_visited_at=interaction_share.last_visited_at,
                created_at=interaction_share.created_at,
            )

            return ShareStatsResponse(
                success=True,
                message="Share statistics retrieved successfully",
                share_stats=share_stats,
                total_coins_earned=total_coins_earned,
            )

        except Exception as e:
            return ShareStatsResponse(
                success=False,
                message=f"Failed to retrieve share statistics: {str(e)}",
            )


@strawberry.type
class InteractionMutation:
    @strawberry.mutation
    async def delete_document(self, info, id: str) -> DefaultResponse:
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return DefaultResponse(success=False, message="Authentication required")

        db: AsyncSession = context.db

        # Get interaction material
        result = await db.execute(
            select(Interaction).where(
                Interaction.id == id, Interaction.user_id == current_user.id
            )
        )
        interaction = result.scalar_one_or_none()

        if not interaction:
            return DefaultResponse(success=False, message="Document not found")

        # Delete the document
        await db.delete(interaction)
        await db.commit()

        return DefaultResponse(success=True, message="Document deleted successfully")

    # @strawberry.mutation
    # async def do_conversation(
    #     self,
    #     info,
    #     input: DoConversationInput,
    # ) -> InteractionResponse:
    #     context = info.context
    #     current_user = await get_current_user_from_context(context)

    #     if not current_user:
    #         return InteractionResponse(success=False, message=CONSTANTS.NOT_FOUND)

    #     db: AsyncSession = context.db

    #     interaction = None
    #     is_fresh_interaction = False

    #     # If interaction_id provided validate; else create a new one

    #     if input.interaction_id:
    #         result = await db.execute(
    #             select(Interaction).where(
    #                 Interaction.id == input.interaction_id,
    #                 Interaction.user_id == current_user.id,
    #             )
    #         )
    #         interaction = result.scalar_one_or_none()
    #         if not interaction:
    #             return InteractionResponse(
    #                 success=False, message="No conversation found!"
    #             )

    #         # Check if this is a fresh interaction by querying conversations
    #         conv_result = await db.execute(
    #             select(Conversation).where(
    #                 Conversation.interaction_id == interaction.id
    #             )
    #         )
    #         existing_conversations = conv_result.scalars().all()
    #         is_fresh_interaction = len(existing_conversations) == 0
    #     else:
    #         interaction = Interaction(
    #             user_id=str(current_user.id),
    #             title=None,
    #             summary_title=None,
    #         )
    #         db.add(interaction)
    #         await db.commit()  # Commit the transaction so the interaction is persisted
    #         is_fresh_interaction = True  # New interaction is always fresh

    #     # Convert GraphQL input to service format
    #     media_files_dict = None
    #     if input.media_files:
    #         media_files_dict = [
    #             {"id": media_file.id, "url": media_file.url}
    #             for media_file in input.media_files
    #         ]

    #     # Delegate to service function with dynamic token calculation
    #     print(f"🔧 RESOLVER: Calling process_conversation_message")
    #     result = await process_conversation_message(
    #         user=current_user,
    #         interaction=interaction,
    #         message=input.message,
    #         media_files=media_files_dict,
    #         max_tokens=None,  # Will be calculated dynamically based on file count
    #         db=db,  # Pass the database session
    #         visualize_model=input.visualize_model,
    #         assistant_model=input.assistant_model,
    #     )
    #     print(f"🔧 RESOLVER: Got result from service: {result}")

    #     # Get the updated interaction to return current state
    #     # Note: The service now handles database operations, so we need to refresh from our session
    #     await db.refresh(interaction)

    #     # Log the complete AI response from resolver
    #     ai_response = result.get("ai_response")
    #     print(f"🔧 RESOLVER: Extracted ai_response: {ai_response}")
    #     print(f"🔧 RESOLVER: ai_response type: {type(ai_response)}")
    #     print(
    #         f"🔧 RESOLVER: ai_response length: {len(str(ai_response)) if ai_response else 0}"
    #     )

    #     if ai_response:
    #         # Truncate long responses for readability
    #         if len(str(ai_response)) > 500:
    #             print(f"{str(ai_response)}")

    #         else:
    #             print(ai_response)
    #     else:
    #         print("No AI response content")
    #     print("=" * 80)

    #     return InteractionResponse(
    #         success=bool(result.get("success")),
    #         message=result.get("message"),
    #         interaction_id=result.get("interaction_id"),
    #         is_new_interaction=is_fresh_interaction,
    #         interaction=(
    #             None
    #             if not is_fresh_interaction
    #             else InteractionType(
    #                 id=interaction.id,
    #                 user_id=interaction.user_id,
    #                 title=interaction.title,
    #                 summary_title=interaction.summary_title,
    #                 is_pinned=interaction.is_pinned,
    #                 created_at=interaction.created_at,
    #                 updated_at=interaction.updated_at,
    #             )
    #         ),
    #         ai_response=result.get("ai_response"),  # The actual AI response content
    #     )

    @strawberry.mutation
    async def delete_media_file(
        self,
        info,
        input: DeleteMediaFileInput,
    ) -> DefaultResponse:
        """
        Delete a media file from S3 and database
        This is called when user removes a file from the chat interface
        """
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return DefaultResponse(success=False, message="Authentication required")

        db: AsyncSession = context.db

        # Get the media file
        result = await db.execute(select(Media).where(Media.id == input.media_id))
        media = result.scalar_one_or_none()

        if not media:
            return DefaultResponse(success=False, message="Media file not found")

        # Check if the media file belongs to the current user
        # We need to check through conversations to ensure user ownership
        # conv_result = await db.execute(
        #     select(Conversation)
        #     .join(Conversation.files)
        #     .where(Media.id == input.media_id)
        #     .join(Interaction)
        #     .where(Interaction.user_id == current_user.id)
        # )
        # conversation = conv_result.scalar_one_or_none()

        # if not conversation:
        #     return DefaultResponse(
        #         success=False, message="You don't have permission to delete this file"
        #     )

        try:
            # Delete from S3
            from app.services.file_service import FileService

            s3_deleted = await FileService.delete_file_from_s3(media.s3_key)

            if not s3_deleted:
                return DefaultResponse(
                    success=False, message="Failed to delete file from storage"
                )

            # Remove the file from the conversation
            # conversation.files.remove(media)

            # Delete the media record from database
            await db.delete(media)
            await db.commit()

            return DefaultResponse(
                success=True, message="Media file deleted successfully"
            )

        except Exception as e:
            await db.rollback()
            return DefaultResponse(
                success=False, message=f"Failed to delete media file: {str(e)}"
            )

    @strawberry.mutation
    async def update_interaction_title(
        self,
        info,
        input: UpdateInteractionTitleInput,
    ) -> DefaultResponse:
        """
        Update the title of an interaction
        """
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return DefaultResponse(success=False, message="Authentication required")

        db: AsyncSession = context.db

        # Get the interaction
        result = await db.execute(
            select(Interaction).where(
                Interaction.id == input.interaction_id,
                Interaction.user_id == current_user.id,
            )
        )
        interaction = result.scalar_one_or_none()

        if not interaction:
            return DefaultResponse(success=False, message="Interaction not found")

        try:
            # Update the title
            interaction.title = input.title
            await db.commit()

            return DefaultResponse(
                success=True, message="Interaction title updated successfully"
            )

        except Exception as e:
            await db.rollback()
            return DefaultResponse(
                success=False, message=f"Failed to update title: {str(e)}"
            )

    @strawberry.mutation
    async def cancel_generation(
        self,
        info,
        input: CancelGenerationInput,
    ) -> DefaultResponse:
        """
        Cancel ongoing AI response generation
        """
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return DefaultResponse(success=False, message="Authentication required")

        db: AsyncSession = context.db

        try:
            # Determine interaction_id from either direct input or conversation_id
            interaction_id = input.interaction_id

            if not interaction_id and input.conversation_id:
                # Get interaction_id from conversation
                from app.models.interaction import Conversation

                conv_result = await db.execute(
                    select(Conversation).where(Conversation.id == input.conversation_id)
                )
                conversation = conv_result.scalar_one_or_none()

                if not conversation:
                    return DefaultResponse(
                        success=False, message="Conversation not found"
                    )

                interaction_id = conversation.interaction_id

            if not interaction_id:
                return DefaultResponse(
                    success=False,
                    message="Either interaction_id or conversation_id must be provided",
                )

            result = await cancel_ai_generation(
                user_id=str(current_user.id), interaction_id=interaction_id, db=db
            )

            return DefaultResponse(success=result["success"], message=result["message"])

        except Exception as e:
            return DefaultResponse(
                success=False, message=f"Failed to cancel generation: {str(e)}"
            )

    @strawberry.mutation
    async def delete_interaction(
        self,
        info,
        input: DeleteInteractionInput,
    ) -> DefaultResponse:
        """
        Delete an interaction and all associated data (conversations, media files, embeddings)
        Now runs in background to avoid blocking other API requests
        """
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return DefaultResponse(success=False, message="Authentication required")

        db: AsyncSession = context.db

        try:
            # First, verify the interaction exists and belongs to the user
            result = await db.execute(
                select(Interaction).where(
                    Interaction.id == input.interaction_id,
                    Interaction.user_id == current_user.id,
                )
            )
            interaction = result.scalar_one_or_none()

            if not interaction:
                return DefaultResponse(success=False, message="Interaction not found")

            # Submit the deletion task to background processing
            from app.services.background_task_service import (
                background_task_service,
                TaskPriority,
            )

            task_id = await background_task_service.submit_task(
                task_type="delete_interaction",
                user_id=current_user.id,
                payload={
                    "interaction_id": input.interaction_id,
                    "interaction_title": interaction.title or "Untitled",
                },
                priority=TaskPriority.HIGH,  # High priority for user-initiated deletions
            )

            return DefaultResponse(
                success=True,
                message=f"Conversation '{interaction.title or 'Untitled'}' deletion has been started. This will be processed in the background.",
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            return DefaultResponse(
                success=False,
                message=f"Failed to initiate conversation deletion: {str(e)}",
            )

    @strawberry.mutation
    async def delete_interactions(
        self,
        info,
        input: DeleteInteractionsInput,
    ) -> DefaultResponse:
        """
        Delete multiple interactions and all associated data (conversations, media files, embeddings)
        Now runs in background to avoid blocking other API requests
        """
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return DefaultResponse(success=False, message="Authentication required")

        if not input.interaction_ids or len(input.interaction_ids) == 0:
            return DefaultResponse(success=False, message="No interactions to delete")

        db: AsyncSession = context.db

        try:
            # Verify that all interactions exist and belong to the user
            valid_interaction_ids = []
            for interaction_id in input.interaction_ids:
                result = await db.execute(
                    select(Interaction).where(
                        Interaction.id == interaction_id,
                        Interaction.user_id == current_user.id,
                    )
                )
                interaction = result.scalar_one_or_none()
                if interaction:
                    valid_interaction_ids.append(interaction_id)

            if not valid_interaction_ids:
                return DefaultResponse(
                    success=False,
                    message="No valid interactions found to delete. Please check if you have permission to delete these conversations.",
                )

            # Submit the bulk deletion task to background processing
            from app.services.background_task_service import (
                background_task_service,
                TaskPriority,
            )

            task_id = await background_task_service.submit_task(
                task_type="delete_interactions",
                user_id=current_user.id,
                payload={"interaction_ids": valid_interaction_ids},
                priority=TaskPriority.HIGH,  # High priority for user-initiated deletions
            )

            return DefaultResponse(
                success=True,
                message=f"Bulk deletion of {len(valid_interaction_ids)} conversation(s) has been started. This will be processed in the background.",
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            return DefaultResponse(
                success=False,
                message=f"Failed to initiate bulk conversation deletion: {str(e)}",
            )

    @strawberry.mutation
    async def cancel_background_task(self, info, task_id: str) -> DefaultResponse:
        """Cancel a background task"""
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return DefaultResponse(success=False, message="Authentication required")

        try:
            from app.services.background_task_service import background_task_service

            # Check if task exists and belongs to user
            task = background_task_service.get_task_status(task_id)
            if not task or task.user_id != current_user.id:
                return DefaultResponse(
                    success=False, message="Task not found or access denied"
                )

            # Cancel the task
            success = await background_task_service.cancel_task(task_id)

            if success:
                return DefaultResponse(
                    success=True, message="Background task has been cancelled"
                )
            else:
                return DefaultResponse(
                    success=False, message="Failed to cancel background task"
                )

        except Exception as e:
            return DefaultResponse(
                success=False, message=f"Failed to cancel background task: {str(e)}"
            )

    @strawberry.mutation
    async def submit_background_message(
        self,
        info,
        message: str,
        interaction_id: Optional[str] = None,
        media_files: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        visualize_model: Optional[str] = None,
        assistant_model: Optional[str] = None,
    ) -> DefaultResponse:
        """Submit a message for background processing"""
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return DefaultResponse(success=False, message="Authentication required")

        if not message.strip():
            return DefaultResponse(success=False, message="Message cannot be empty")

        try:
            from app.services.background_message_service import (
                background_message_service,
                MessageTaskPriority,
            )

            # Convert media files to the expected format
            media_files_list = []
            if media_files:
                for media_id in media_files:
                    media_files_list.append({"id": media_id})

            # Set default max tokens if not provided
            if max_tokens is None:
                max_tokens = 2000  # Default value

            task_id = await background_message_service.submit_message_task(
                user_id=current_user.id,
                interaction_id=interaction_id,
                message=message,
                media_files=media_files_list,
                max_tokens=max_tokens,
                visualize_model=visualize_model,
                assistant_model=assistant_model,
                priority=MessageTaskPriority.HIGH,  # High priority for user-initiated messages
            )

            return DefaultResponse(
                success=True,
                message=f"Message submitted for background processing. Task ID: {task_id}",
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            return DefaultResponse(
                success=False,
                message=f"Failed to submit message for processing: {str(e)}",
            )

    @strawberry.mutation
    async def cancel_background_message_task(
        self, info, task_id: str
    ) -> DefaultResponse:
        """Cancel a background message task"""
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return DefaultResponse(success=False, message="Authentication required")

        try:
            from app.services.background_message_service import (
                background_message_service,
            )

            # Check if task exists and belongs to user
            task = background_message_service.get_task_status(task_id)
            if not task or task.user_id != current_user.id:
                return DefaultResponse(
                    success=False, message="Task not found or access denied"
                )

            # Cancel the task
            success = await background_message_service.cancel_task(task_id)

            if success:
                return DefaultResponse(
                    success=True, message="Background message task has been cancelled"
                )
            else:
                return DefaultResponse(
                    success=False, message="Failed to cancel background message task"
                )

        except Exception as e:
            return DefaultResponse(
                success=False,
                message=f"Failed to cancel background message task: {str(e)}",
            )

    @strawberry.mutation
    async def retry_background_message_task(
        self, info, task_id: str
    ) -> DefaultResponse:
        """Retry a failed background message task"""
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return DefaultResponse(success=False, message="Authentication required")

        try:
            from app.services.background_message_service import (
                background_message_service,
            )

            # Check if task exists and belongs to user
            task = background_message_service.get_task_status(task_id)
            if not task or task.user_id != current_user.id:
                return DefaultResponse(
                    success=False, message="Task not found or access denied"
                )

            # Retry the task
            success = await background_message_service.retry_task(task_id)

            if success:
                return DefaultResponse(
                    success=True,
                    message="Background message task has been queued for retry",
                )
            else:
                return DefaultResponse(
                    success=False, message="Failed to retry background message task"
                )

        except Exception as e:
            return DefaultResponse(
                success=False,
                message=f"Failed to retry background message task: {str(e)}",
            )

    @strawberry.mutation
    async def pin_interaction(
        self,
        info,
        input: PinInteractionInput,
    ) -> DefaultResponse:
        """
        Pin or unpin an interaction
        """
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return DefaultResponse(success=False, message="Authentication required")

        db: AsyncSession = context.db

        try:
            # Get the interaction
            result = await db.execute(
                select(Interaction).where(
                    Interaction.id == input.interaction_id,
                    Interaction.user_id == current_user.id,
                )
            )
            interaction = result.scalar_one_or_none()

            if not interaction:
                return DefaultResponse(success=False, message="Interaction not found")

            # Update the pin status
            interaction.is_pinned = input.is_pinned
            await db.commit()

            action = "pinned" if input.is_pinned else "unpinned"
            return DefaultResponse(
                success=True, message=f"Interaction {action} successfully"
            )

        except Exception as e:
            await db.rollback()
            return DefaultResponse(
                success=False, message=f"Failed to update pin status: {str(e)}"
            )

    @strawberry.mutation
    async def share_interaction(
        self,
        info,
        input: ShareInteractionInput,
    ) -> ShareInteractionResponse:
        """
        Create a shareable link for an interaction (temporary view)
        """
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return ShareInteractionResponse(
                success=False, message="Authentication required"
            )

        db: AsyncSession = context.db

        try:
            # Get the original interaction
            result = await db.execute(
                select(Interaction).where(
                    Interaction.id == input.interaction_id,
                    Interaction.user_id == current_user.id,
                )
            )
            original_interaction = result.scalar_one_or_none()

            if not original_interaction:
                return ShareInteractionResponse(
                    success=False, message="Interaction not found"
                )

            # Generate a unique share ID
            import uuid

            share_id = str(uuid.uuid4())[:8]  # Short ID for sharing

            # Create the interaction share record
            interaction_share = InteractionShare(
                original_interaction_id=original_interaction.id,
                share_id=share_id,
                is_public=True,
            )
            db.add(interaction_share)
            await db.commit()

            # Generate share URL
            share_url = f"https://www.studyguru.pro/share/{share_id}"

            return ShareInteractionResponse(
                success=True,
                message="Interaction shared successfully",
                share_id=share_id,
                share_url=share_url,
            )

        except Exception as e:
            await db.rollback()
            return ShareInteractionResponse(
                success=False, message=f"Failed to share interaction: {str(e)}"
            )

    @strawberry.mutation
    async def like_dislike_message(
        self,
        info,
        input: LikeDislikeInput,
    ) -> DefaultResponse:
        """
        Like or dislike a conversation message
        """
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return DefaultResponse(success=False, message="Authentication required")

        db: AsyncSession = context.db

        try:
            # Get the conversation
            result = await db.execute(
                select(Conversation).where(Conversation.id == input.conversation_id)
            )
            conversation = result.scalar_one_or_none()

            if not conversation:
                return DefaultResponse(success=False, message="Message not found")

            # Verify the user owns this conversation through the interaction
            interaction_result = await db.execute(
                select(Interaction).where(
                    Interaction.id == conversation.interaction_id,
                    Interaction.user_id == current_user.id,
                )
            )
            interaction = interaction_result.scalar_one_or_none()

            if not interaction:
                return DefaultResponse(
                    success=False,
                    message="You don't have permission to rate this message",
                )

            # Update the like/dislike status
            conversation.is_liked = input.is_liked
            conversation.liked_at = func.now()

            await db.commit()

            action = "liked" if input.is_liked else "disliked"
            return DefaultResponse(
                success=True, message=f"Message {action} successfully"
            )

        except Exception as e:
            await db.rollback()
            return DefaultResponse(
                success=False, message=f"Failed to rate message: {str(e)}"
            )
