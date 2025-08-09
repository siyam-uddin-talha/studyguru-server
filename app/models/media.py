from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base

import uuid


class ImageMedia(Base):
    __tablename__ = "image_media"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    url = Column(String(191), nullable=False)
    file_name = Column(String(191), nullable=False)
    size = Column(Float, nullable=False)
    meme_type = Column(String(191), nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=True)
