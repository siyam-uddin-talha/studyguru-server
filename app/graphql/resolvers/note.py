import strawberry
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from datetime import datetime

from app.graphql.types.note import (
    NoteType,
    NoteResponse,
    NoteListResponse,
    CreateNoteInput,
    UpdateNoteInput,
    DeleteNoteInput,
    GetNotesInput,
)
from app.models.note import Note
from app.helpers.user import get_current_user_from_context
from app.graphql.types.common import DefaultResponse


@strawberry.type
class NoteQuery:
    @strawberry.field
    async def notes(
        self,
        info,
        input: Optional[GetNotesInput] = None,
    ) -> NoteListResponse:
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return NoteListResponse(success=False, message="Authentication required")

        db: AsyncSession = context.db

        try:
            # Build base query
            base_query = select(Note).where(Note.user_id == current_user.id)

            # Get pagination params
            page = input.page if input and input.page else 1
            size = input.size if input and input.size else 100
            offset = (page - 1) * size

            # Execute query with ordering and pagination
            result = await db.execute(
                base_query.order_by(desc(Note.updated_at)).offset(offset).limit(size)
            )
            notes = result.scalars().all()

            # Get total count
            count_query = (
                select(func.count())
                .select_from(Note)
                .where(Note.user_id == current_user.id)
            )
            total_result = await db.scalar(count_query)
            total = total_result or 0

            # Convert to response types
            note_types = []
            for note in notes:
                note_types.append(
                    NoteType(
                        id=note.id,
                        user_id=note.user_id,
                        title=note.title,
                        content=note.content,
                        preview=note.preview,
                        color=note.color,
                        text_color=note.text_color,
                        created_at=note.created_at,
                        updated_at=note.updated_at,
                    )
                )

            has_next_page = total > offset + len(notes)

            return NoteListResponse(
                success=True,
                message="Notes retrieved successfully",
                notes=note_types,
                total=total,
                has_next_page=has_next_page,
            )
        except Exception as e:
            return NoteListResponse(
                success=False, message=f"Error retrieving notes: {str(e)}"
            )

    @strawberry.field
    async def note(self, info, note_id: str) -> NoteResponse:
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return NoteResponse(success=False, message="Authentication required")

        db: AsyncSession = context.db

        try:
            result = await db.execute(
                select(Note).where(Note.id == note_id, Note.user_id == current_user.id)
            )
            note = result.scalar_one_or_none()

            if not note:
                return NoteResponse(success=False, message="Note not found")

            note_type = NoteType(
                id=note.id,
                user_id=note.user_id,
                title=note.title,
                content=note.content,
                preview=note.preview,
                color=note.color,
                text_color=note.text_color,
                created_at=note.created_at,
                updated_at=note.updated_at,
            )

            return NoteResponse(
                success=True,
                message="Note retrieved successfully",
                note=note_type,
            )
        except Exception as e:
            return NoteResponse(
                success=False, message=f"Error retrieving note: {str(e)}"
            )


@strawberry.type
class NoteMutation:
    @strawberry.mutation
    async def create_note(
        self,
        info,
        input: CreateNoteInput,
    ) -> NoteResponse:
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return NoteResponse(success=False, message="Authentication required")

        db: AsyncSession = context.db

        try:
            # Create note
            note = Note(
                user_id=current_user.id,
                title=input.title or "Untitled",
                content=input.content,
                preview=input.preview,
                color=input.color or "#FDF2F8",
                text_color=input.text_color,
            )

            db.add(note)
            await db.commit()
            await db.refresh(note)

            # Convert to response type
            note_type = NoteType(
                id=note.id,
                user_id=note.user_id,
                title=note.title,
                content=note.content,
                preview=note.preview,
                color=note.color,
                text_color=note.text_color,
                created_at=note.created_at,
                updated_at=note.updated_at,
            )

            return NoteResponse(
                success=True,
                message="Note created successfully",
                note=note_type,
            )
        except Exception as e:
            await db.rollback()
            return NoteResponse(success=False, message=f"Error creating note: {str(e)}")

    @strawberry.mutation
    async def update_note(
        self,
        info,
        input: UpdateNoteInput,
    ) -> NoteResponse:
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return NoteResponse(success=False, message="Authentication required")

        db: AsyncSession = context.db

        try:
            result = await db.execute(
                select(Note).where(
                    Note.id == input.note_id, Note.user_id == current_user.id
                )
            )
            note = result.scalar_one_or_none()

            if not note:
                return NoteResponse(success=False, message="Note not found")

            # Update fields
            if input.title is not None:
                note.title = input.title
            if input.content is not None:
                note.content = input.content
            if input.preview is not None:
                note.preview = input.preview
            if input.color is not None:
                note.color = input.color
            if input.text_color is not None:
                note.text_color = input.text_color

            await db.commit()
            await db.refresh(note)

            note_type = NoteType(
                id=note.id,
                user_id=note.user_id,
                title=note.title,
                content=note.content,
                preview=note.preview,
                color=note.color,
                text_color=note.text_color,
                created_at=note.created_at,
                updated_at=note.updated_at,
            )

            return NoteResponse(
                success=True,
                message="Note updated successfully",
                note=note_type,
            )
        except Exception as e:
            await db.rollback()
            return NoteResponse(success=False, message=f"Error updating note: {str(e)}")

    @strawberry.mutation
    async def delete_note(
        self,
        info,
        input: DeleteNoteInput,
    ) -> DefaultResponse:
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return DefaultResponse(success=False, message="Authentication required")

        db: AsyncSession = context.db

        try:
            result = await db.execute(
                select(Note).where(
                    Note.id == input.note_id, Note.user_id == current_user.id
                )
            )
            note = result.scalar_one_or_none()

            if not note:
                return DefaultResponse(success=False, message="Note not found")

            await db.delete(note)
            await db.commit()

            return DefaultResponse(success=True, message="Note deleted successfully")
        except Exception as e:
            await db.rollback()
            return DefaultResponse(
                success=False, message=f"Error deleting note: {str(e)}"
            )
