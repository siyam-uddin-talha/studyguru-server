"""
Context-related database models for enhanced RAG system
"""

from sqlalchemy import (
    Column,
    String,
    DateTime,
    Integer,
    ForeignKey,
    Text,
    JSON,
    Float,
    Boolean,
    Index,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
import uuid


class ConversationContext(Base):
    """
    Table to store preprocessed context snapshots for faster retrieval
    """

    __tablename__ = "conversation_context"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    interaction_id = Column(String(191), ForeignKey("interaction.id"), nullable=False)
    user_id = Column(String(191), ForeignKey("user.id"), nullable=False)

    # Context snapshot data
    context_type = Column(
        String(50), nullable=False
    )  # 'semantic_summary', 'vector_search', 'document_content', 'cross_interaction'
    context_data = Column(JSON, nullable=False)  # The actual context content
    context_hash = Column(
        String(64), nullable=False
    )  # Hash of the context for deduplication

    # Metadata for context ranking and relevance
    relevance_score = Column(Float, nullable=True, default=0.0)
    recency_score = Column(Float, nullable=True, default=0.0)
    importance_score = Column(Float, nullable=True, default=0.0)

    # Content metadata
    content_length = Column(Integer, nullable=True)
    topic_tags = Column(JSON, nullable=True)  # List of topic tags
    question_numbers = Column(
        JSON, nullable=True
    )  # List of question numbers referenced

    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=True)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=True
    )
    expires_at = Column(DateTime, nullable=True)  # For temporary context

    # Relationships
    interaction = relationship("Interaction")
    user = relationship("User")

    # Indexes for faster queries
    __table_args__ = (
        Index("idx_context_user_interaction", "user_id", "interaction_id"),
        Index("idx_context_type_hash", "context_type", "context_hash"),
        Index("idx_context_relevance", "relevance_score", "recency_score"),
        Index("idx_context_created", "created_at"),
    )


class UserLearningProfile(Base):
    """
    Table to track cross-interaction learning patterns and preferences
    """

    __tablename__ = "user_learning_profile"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(191), ForeignKey("user.id"), nullable=False, unique=True)

    # Learning patterns
    preferred_topics = Column(
        JSON, nullable=True
    )  # List of frequently discussed topics
    learning_style = Column(
        String(50), nullable=True
    )  # 'visual', 'analytical', 'step_by_step', etc.
    difficulty_preference = Column(
        String(20), nullable=True
    )  # 'beginner', 'intermediate', 'advanced'

    # Knowledge tracking
    mastered_concepts = Column(
        JSON, nullable=True
    )  # List of concepts the user has mastered
    struggling_areas = Column(JSON, nullable=True)  # Areas where user needs more help
    learning_gaps = Column(JSON, nullable=True)  # Identified knowledge gaps

    # Interaction patterns
    average_session_length = Column(Float, nullable=True)
    preferred_question_types = Column(
        JSON, nullable=True
    )  # ['mcq', 'written', 'problem_solving']
    most_productive_times = Column(
        JSON, nullable=True
    )  # Time patterns when user is most active

    # Cross-interaction references
    related_interactions = Column(JSON, nullable=True)  # IDs of related interactions
    topic_clusters = Column(JSON, nullable=True)  # Clusters of related topics

    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=True)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=True
    )
    last_activity = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User")

    # Indexes
    __table_args__ = (
        Index("idx_learning_profile_user", "user_id"),
        Index("idx_learning_profile_activity", "last_activity"),
    )


class DocumentContext(Base):
    """
    Table to store document-specific context and metadata
    """

    __tablename__ = "document_context"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    media_id = Column(String(191), ForeignKey("media.id"), nullable=False)
    interaction_id = Column(String(191), ForeignKey("interaction.id"), nullable=False)
    user_id = Column(String(191), ForeignKey("user.id"), nullable=False)

    # Document structure
    document_type = Column(
        String(50), nullable=False
    )  # 'mcq', 'written', 'mixed', 'other'
    total_questions = Column(Integer, nullable=True)
    question_structure = Column(
        JSON, nullable=True
    )  # Structure of questions in the document

    # Content analysis
    main_topics = Column(JSON, nullable=True)  # Main topics covered in the document
    difficulty_level = Column(
        String(20), nullable=True
    )  # 'beginner', 'intermediate', 'advanced'
    subject_area = Column(String(100), nullable=True)  # Math, Science, etc.

    # Question mapping
    question_mapping = Column(JSON, nullable=True)  # Maps question numbers to content
    answer_key = Column(JSON, nullable=True)  # Stores answers if available

    # Context for retrieval
    full_content = Column(Text, nullable=True)  # Full document content for reference
    content_summary = Column(Text, nullable=True)  # AI-generated summary
    key_concepts = Column(JSON, nullable=True)  # Key concepts extracted

    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=True)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=True
    )

    # Relationships
    media = relationship("Media")
    interaction = relationship("Interaction")
    user = relationship("User")

    # Indexes
    __table_args__ = (
        Index("idx_document_media", "media_id"),
        Index("idx_document_interaction", "interaction_id"),
        Index("idx_document_user", "user_id"),
        Index("idx_document_type", "document_type"),
        # Note: JSON columns cannot be directly indexed in MySQL, using generated columns instead
    )


class ContextUsageLog(Base):
    """
    Table to track how context is being used for monitoring and optimization
    """

    __tablename__ = "context_usage_log"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    interaction_id = Column(String(191), ForeignKey("interaction.id"), nullable=False)
    user_id = Column(String(191), ForeignKey("user.id"), nullable=False)
    conversation_id = Column(String(191), ForeignKey("conversation.id"), nullable=True)

    # Context usage details
    context_sources_used = Column(
        JSON, nullable=False
    )  # List of context sources that were retrieved
    context_sources_ignored = Column(
        JSON, nullable=True
    )  # Context sources that were available but not used
    context_retrieval_time = Column(
        Float, nullable=True
    )  # Time taken to retrieve context
    context_processing_time = Column(
        Float, nullable=True
    )  # Time taken to process context

    # Usage metrics
    context_relevance_score = Column(
        Float, nullable=True
    )  # How relevant the context was
    user_satisfaction_score = Column(
        Float, nullable=True
    )  # User feedback on response quality
    context_utilization_rate = Column(
        Float, nullable=True
    )  # Percentage of retrieved context actually used

    # Query details
    user_query = Column(Text, nullable=True)  # The user's query
    query_type = Column(
        String(50), nullable=True
    )  # 'specific_question', 'general_inquiry', 'follow_up', etc.
    response_quality = Column(
        String(20), nullable=True
    )  # 'excellent', 'good', 'fair', 'poor'

    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=True)

    # Relationships
    interaction = relationship("Interaction")
    user = relationship("User")
    conversation = relationship("Conversation")

    # Indexes
    __table_args__ = (
        Index("idx_usage_log_interaction", "interaction_id"),
        Index("idx_usage_log_user", "user_id"),
        Index("idx_usage_log_created", "created_at"),
        Index("idx_usage_log_quality", "response_quality"),
    )
