from sqlalchemy import (
    Column,
    String,
    DateTime,
    Text,
    ForeignKey,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
import uuid


class Note(Base):
    __tablename__ = "note"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(191), ForeignKey("user.id"), nullable=False)

    title = Column(String(500), nullable=False, default="Untitled")
    content = Column(Text, nullable=False)  # HTML content
    preview = Column(Text, nullable=True)  # Plain text preview
    color = Column(String(50), nullable=False, default="#FDF2F8")
    text_color = Column(String(50), nullable=True)  # Optional text color

    created_at = Column(DateTime, default=func.now(), nullable=True)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=True
    )

    # Relationships
    user = relationship("User", back_populates="notes")
