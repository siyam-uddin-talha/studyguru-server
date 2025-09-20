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
    ConversationType,
    MediaType,
    DoConversationInput,
    DeleteMediaFileInput,
    UpdateInteractionTitleInput,
)
from app.models.interaction import Interaction
from app.models.media import Media
from app.models.interaction import Conversation, ConversationRole
from app.models.user import User

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
                    title=interaction.title,
                    summary_title=interaction.summary_title,
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
        self, info, id: str, limit: Optional[int] = 12, offset: Optional[int] = 0
    ) -> List[ConversationType]:
        """
        Get messages (conversations) for a specific interaction with pagination
        """
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return []

        db: AsyncSession = context.db

        # Get conversations for the interaction with pagination
        result = await db.execute(
            select(Conversation)
            .options(selectinload(Conversation.files))
            .join(Interaction)
            .where(
                Interaction.id == id,
                Interaction.user_id == current_user.id,
                Conversation.interaction_id == id,
            )
            .order_by(desc(Conversation.created_at))
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
            user=current_user,
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
