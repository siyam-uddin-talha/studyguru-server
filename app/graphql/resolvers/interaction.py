import strawberry
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from fastapi import UploadFile, HTTPException

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
            # Get associated media
            media_result = await db.execute(
                select(Media).where(Media.id == interaction.file_id)
            )
            media = media_result.scalar_one_or_none()

            doc_material_types.append(
                InteractionType(
                    id=interaction.id,
                    user_id=interaction.user_id,
                    file_id=interaction.file_id,
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
            return InteractionResponse(success=False, message="Authentication required")

        db: AsyncSession = context.db

        # Get interaction material
        result = await db.execute(
            select(Interaction).where(
                Interaction.id == id, Interaction.user_id == current_user.id
            )
        )
        interaction = result.scalar_one_or_none()

        if not interaction:
            return InteractionResponse(success=False, message="Document not found")

        # Get associated media
        media_result = await db.execute(
            select(Media).where(Media.id == interaction.file_id)
        )
        media = media_result.scalar_one_or_none()

        return InteractionResponse(
            success=True,
            message="Document retrieved successfully",
            result=InteractionType(
                id=interaction.id,
                user_id=interaction.user_id,
                file_id=interaction.file_id,
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
    @strawberry.field
    async def upload_document(
        self, info, max_tokens: Optional[int] = 1000
    ) -> InteractionResponse:
        """
        Note: This is a simplified version. In practice, file upload would be handled
        via REST endpoint and then processed asynchronously.
        """
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return InteractionResponse(success=False, message="Authentication required")

        # Check if user is free and limit tokens
        if (
            current_user.purchased_subscription.subscription.subscription_plan.value
            == "FREE"
        ):
            max_tokens = min(max_tokens, 1000)

        # Check if user has enough points
        estimated_points = max(1, max_tokens // 100)
        if current_user.current_points < estimated_points:
            return InteractionResponse(
                success=False,
                message="Insufficient points. Please purchase more points or upgrade your plan.",
            )

        return InteractionResponse(
            success=True,
            message="Document upload endpoint should be implemented via REST API",
        )

    @strawberry.field
    async def delete_document(self, info, id: str) -> InteractionResponse:
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return InteractionResponse(success=False, message="Authentication required")

        db: AsyncSession = context.db

        # Get interaction material
        result = await db.execute(
            select(Interaction).where(
                Interaction.id == id, Interaction.user_id == current_user.id
            )
        )
        interaction = result.scalar_one_or_none()

        if not interaction:
            return InteractionResponse(success=False, message="Document not found")

        # Delete the document
        await db.delete(interaction)
        await db.commit()

        return InteractionResponse(
            success=True, message="Document deleted successfully"
        )

    @strawberry.field
    async def do_conversation(
        self,
        info,
        input: DoConversationInput,
    ) -> InteractionResponse:
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return InteractionResponse(success=False, message="Authentication required")

        db: AsyncSession = context.db

        interaction = None
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
                    success=False, message="Interaction not found"
                )
        else:
            interaction = Interaction(
                user_id=str(current_user.id),
                title=None,
                summary_title=None,
            )
            db.add(interaction)
            await db.flush()

        # Delegate to service function
        result = await process_conversation_message(
            user_id=str(current_user.id),
            interaction_id=input.interaction_id,
            message=input.message,
            image_urls=input.image_urls,
            max_tokens=int(input.max_tokens or 500),
        )

        return InteractionResponse(
            success=bool(result.get("success")),
            message=result.get("message"),
            interaction_id=result.get("interaction_id"),
        )
