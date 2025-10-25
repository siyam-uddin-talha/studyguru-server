from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base

import uuid


class Media(Base):
    __tablename__ = "media"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    original_filename = Column(String(500), nullable=False)
    s3_key = Column(String(500), nullable=False)
    file_type = Column(String(100), nullable=False)
    meme_type = Column(String(191), nullable=True)
    original_size = Column(Float, nullable=False)  # in bytes
    compressed_size = Column(Float, nullable=True)  # in bytes after compression
    compression_ratio = Column(Float, nullable=True)  # original/compressed
    created_at = Column(DateTime, default=func.now(), nullable=True)

    # Relationships
    conversations = relationship(
        "Conversation", secondary="conversation_files", back_populates="files"
    )
