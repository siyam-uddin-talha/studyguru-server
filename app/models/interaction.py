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
    UniqueConstraint,
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
    is_pinned = Column(Boolean, nullable=False, default=False)

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
    interaction_id = Column(String(191), ForeignKey("interaction.id"), nullable=False)
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


# Share table for temporary clone access
class InteractionShare(Base):
    __tablename__ = "interaction_share"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    original_interaction_id = Column(
        String(191), ForeignKey("interaction.id"), nullable=False
    )
    share_id = Column(String(191), unique=True, nullable=False)  # Public ID for sharing
    is_public = Column(
        Boolean, nullable=False, default=True
    )  # Whether it can be accessed publicly
    visit_count = Column(Integer, nullable=False, default=0)  # Track number of visits
    last_visited_at = Column(DateTime, nullable=True)  # Last visit timestamp
    created_at = Column(DateTime, default=func.now(), nullable=True)

    # Relationships
    original_interaction = relationship(
        "Interaction", foreign_keys=[original_interaction_id]
    )


# Track unique visitors to shared interactions
class InteractionShareVisitor(Base):
    __tablename__ = "interaction_share_visitor"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    interaction_share_id = Column(
        String(191), ForeignKey("interaction_share.id"), nullable=False
    )
    visitor_user_id = Column(
        String(191), ForeignKey("user.id"), nullable=True
    )  # Null for anonymous visitors
    visitor_ip = Column(String(45), nullable=True)  # Store IP for anonymous tracking
    visitor_fingerprint = Column(
        String(191), nullable=True
    )  # Device/browser fingerprint
    reward_given = Column(
        Boolean, nullable=False, default=False
    )  # Track if reward was given
    visited_at = Column(DateTime, default=func.now(), nullable=True)

    # Relationships
    interaction_share = relationship("InteractionShare")
    visitor_user = relationship("User", foreign_keys=[visitor_user_id])

    # Unique constraint to prevent duplicate rewards
    __table_args__ = (
        UniqueConstraint(
            "interaction_share_id", "visitor_user_id", name="unique_user_visit"
        ),
        UniqueConstraint("interaction_share_id", "visitor_ip", name="unique_ip_visit"),
        UniqueConstraint(
            "interaction_share_id",
            "visitor_fingerprint",
            name="unique_fingerprint_visit",
        ),
    )
