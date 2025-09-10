from sqlalchemy import (
    Column,
    String,
    DateTime,
    Integer,
    ForeignKey,
    Text,
    JSON,
    Float,
    Enum,
    Boolean,
    Table,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
import uuid
import enum


class ConversationRole(enum.Enum):
    USER = "USER"
    AI = "AI"


# Junction table for conversation-file relationships
conversation_files = Table(
    "conversation_files",
    Base.metadata,
    Column(
        "conversation_id", String(191), ForeignKey("conversation.id"), primary_key=True
    ),
    Column("media_id", String(191), ForeignKey("media.id"), primary_key=True),
    Column("created_at", DateTime, default=func.now(), nullable=True),
)


class Interaction(Base):
    __tablename__ = "interaction"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(191), ForeignKey("user.id"), nullable=False)

    title = Column(String(500), nullable=True)
    summary_title = Column(String(500), nullable=True)

    created_at = Column(DateTime, default=func.now(), nullable=True)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=True
    )

    # Relationships
    user = relationship("User", back_populates="interactions")
    conversations = relationship("Conversation", back_populates="interaction")


class Conversation(Base):
    __tablename__ = "conversation"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    interaction_id = Column(String(191), ForeignKey("user.id"), nullable=False)
    role = Column(Enum(ConversationRole), nullable=False)
    # OpenAI Analysis Results
    content = Column(JSON, nullable=True)  # Store the full OpenAI response
    question_type = Column(String(50), nullable=True)  # "mcq", "written", "other"
    detected_language = Column(String(50), nullable=True)

    # Token usage
    input_tokens = Column(Integer, nullable=True, default=0)
    output_tokens = Column(Integer, nullable=True, default=0)
    tokens_used = Column(Integer, nullable=True, default=0)

    points_cost = Column(Integer, nullable=True, default=0)

    # for visua
    is_hidden = Column(Boolean, nullable=True, default=False)

    # Processing status
    status = Column(
        String(50), default="processing", nullable=False
    )  # processing, completed, failed
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=func.now(), nullable=True)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=True
    )

    # Relationships
    interaction = relationship("Interaction", back_populates="conversations")
    files = relationship(
        "Media", secondary=conversation_files, back_populates="conversations"
    )
    point_transaction = relationship(
        "PointTransaction", back_populates="conversation", uselist=False
    )
