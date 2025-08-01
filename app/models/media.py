from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
from app.models.subscription import Versionize
import uuid


class Media(Base):
    __tablename__ = "media"
    
    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    url = Column(String(191), nullable=False)
    file_name = Column(String(191), nullable=False)
    size = Column(Float, nullable=False)
    meme_type = Column(String(191), nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=True)
    
    # Relationships
    image_decks = relationship("ImageDeck", back_populates="media")


class ImageDeck(Base):
    __tablename__ = "image_deck"
    
    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    versionize = Column(Enum(Versionize), default=Versionize.PAID, nullable=False)
    media_id = Column(String(191), ForeignKey("media.id"), nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=True)
    
    # Relationships
    media = relationship("Media", back_populates="image_decks")
    group_images = relationship("GroupImageDeck", back_populates="image_deck")