import strawberry
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import selectinload

from app.graphql.types.interaction import (
    InteractionResponse,
    InteractionListResponse,
    InteractionType,
    MediaType,
    DoConversationInput,
)
from app.models.interaction import Interaction
from app.models.media import Media
from app.models.interaction import Conversation, ConversationRole
from app.models.user import User
from app.models.subscription import PointTransaction
from app.services.openai_service import OpenAIService
from app.services.interaction import process_conversation_message
from app.services.file_service import FileService
from app.helpers.user import get_current_user_from_context
from app.constants.constant import CONSTANTS
from app.graphql.types.common import DefaultResponse


@strawberry.type
class InteractionQuery:
    @strawberry.field
    async def interactions(
        self, info, page: Optional[int] = 1, size: Optional[int] = 10
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

        # Get user's interaction materials
        result = await db.execute(
            select(Interaction)
            .options(
                selectinload(Interaction.conversations).selectinload(Conversation.files)
            )
            .where(Interaction.user_id == current_user.id)
            .order_by(desc(Interaction.created_at))
            .offset(offset)
            .limit(size)
        )
        interactions = result.scalars().all()

        # Get total count
        count_result = await db.execute(
            select(Interaction).where(Interaction.user_id == current_user.id)
        )
        total = len(count_result.scalars().all())

        # Convert to response types
        doc_material_types = []
        for interaction in interactions:
            # Get associated media through conversations
            # For now, we'll get the first media file from the first conversation
            # In the future, you might want to return all files or handle multiple files differently
            media = None
            if interaction.conversations:
                for conv in interaction.conversations:
                    if conv.files:
                        media = conv.files[0]  # Get first file
                        break

            doc_material_types.append(
                InteractionType(
                    id=interaction.id,
                    user_id=interaction.user_id,
                    file_id=media.id if media else None,
                    analysis_response=interaction.analysis_response,
                    question_type=interaction.question_type,
                    detected_language=interaction.detected_language,
                    title=interaction.title,
                    summary_title=interaction.summary_title,
                    tokens_used=interaction.tokens_used,
                    points_cost=interaction.points_cost,
                    status=interaction.status,
                    error_message=interaction.error_message,
                    created_at=interaction.created_at,
                    updated_at=interaction.updated_at,
                    file=(
                        MediaType(
                            id=media.id,
                            original_filename=media.original_filename,
                            s3_key=media.s3_key,
                            file_type=media.file_type,
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
        if interaction.conversations:
            for conv in interaction.conversations:
                if conv.files:
                    media = conv.files[0]  # Get first file
                    break

        return InteractionResponse(
            success=True,
            message="Document retrieved successfully",
            result=InteractionType(
                id=interaction.id,
                user_id=interaction.user_id,
                file_id=media.id if media else None,
                analysis_response=interaction.analysis_response,
                question_type=interaction.question_type,
                detected_language=interaction.detected_language,
                title=interaction.title,
                summary_title=interaction.summary_title,
                tokens_used=interaction.tokens_used,
                points_cost=interaction.points_cost,
                status=interaction.status,
                error_message=interaction.error_message,
                created_at=interaction.created_at,
                updated_at=interaction.updated_at,
                file=(
                    MediaType(
                        id=media.id,
                        original_filename=media.original_filename,
                        s3_key=media.s3_key,
                        file_type=media.file_type,
                        original_size=media.original_size,
                        compressed_size=media.compressed_size,
                        compression_ratio=media.compression_ratio,
                        created_at=media.created_at,
                    )
                    if media
                    else None
                ),
            ),
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

    @strawberry.mutation
    async def do_conversation(
        self,
        info,
        input: DoConversationInput,
    ) -> InteractionResponse:
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return InteractionResponse(success=False, message=CONSTANTS.NOT_FOUND)

        db: AsyncSession = context.db

        interaction = None
        is_fresh_interaction = False

        # If interaction_id provided validate; else create a new one
        if input.interaction_id:
            result = await db.execute(
                select(Interaction).where(
                    Interaction.id == input.interaction_id,
                    Interaction.user_id == current_user.id,
                )
            )
            interaction = result.scalar_one_or_none()
            if not interaction:
                return InteractionResponse(
                    success=False, message="No chat conversation found!"
                )

            # Check if this is a fresh interaction by querying conversations
            conv_result = await db.execute(
                select(Conversation).where(
                    Conversation.interaction_id == interaction.id
                )
            )
            existing_conversations = conv_result.scalars().all()
            is_fresh_interaction = len(existing_conversations) == 0
        else:
            interaction = Interaction(
                user_id=str(current_user.id),
                title=None,
                summary_title=None,
            )
            db.add(interaction)
            await db.commit()  # Commit the transaction so the interaction is persisted
            is_fresh_interaction = True  # New interaction is always fresh

        # Convert GraphQL input to service format
        media_files_dict = None
        if input.media_files:
            media_files_dict = [
                {"id": media_file.id, "url": media_file.url}
                for media_file in input.media_files
            ]

        # Delegate to service function
        result = await process_conversation_message(
            user_id=str(current_user.id),
            interaction=interaction,
            message=input.message,
            media_files=media_files_dict,
            max_tokens=int(input.max_tokens),
        )

        # Get the updated interaction to return current state
        await db.refresh(interaction)

        return InteractionResponse(
            success=bool(result.get("success")),
            message=result.get("message"),
            interaction_id=result.get("interaction_id"),
            is_new_interaction=is_fresh_interaction,
            interaction=(
                None
                if not is_fresh_interaction
                else InteractionType(
                    id=interaction.id,
                    user_id=interaction.user_id,
                    title=interaction.title,
                    summary_title=interaction.summary_title,
                    created_at=interaction.created_at,
                    updated_at=interaction.updated_at,
                )
            ),
            ai_response=result.get("ai_response"),  # The actual AI response content
        )
