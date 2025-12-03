import strawberry
from typing import Optional, List
from datetime import datetime


@strawberry.type
class NoteType:
    id: str
    user_id: str
    title: str
    content: str  # HTML content
    preview: Optional[str] = None  # Plain text preview
    color: str
    text_color: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@strawberry.input
class CreateNoteInput:
    title: str
    content: str  # HTML content
    preview: Optional[str] = None  # Plain text preview
    color: Optional[str] = "#FDF2F8"
    text_color: Optional[str] = None


@strawberry.input
class UpdateNoteInput:
    note_id: str
    title: Optional[str] = None
    content: Optional[str] = None
    preview: Optional[str] = None
    color: Optional[str] = None
    text_color: Optional[str] = None


@strawberry.input
class DeleteNoteInput:
    note_id: str


@strawberry.input
class GetNotesInput:
    page: Optional[int] = 1
    size: Optional[int] = 100


@strawberry.type
class NoteResponse:
    success: bool
    message: Optional[str] = None
    note: Optional[NoteType] = None


@strawberry.type
class NoteListResponse:
    success: bool
    message: Optional[str] = None
    notes: Optional[List[NoteType]] = None
    total: Optional[int] = None
    has_next_page: Optional[bool] = None
