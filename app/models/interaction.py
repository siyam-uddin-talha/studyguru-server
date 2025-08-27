from sqlalchemy import Column, String, DateTime, Integer, ForeignKey, Text, JSON, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
import uuid


class Interaction(Base):
    __tablename__ = "interaction"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(191), ForeignKey("user.id"), nullable=False)
    file_id = Column(String(191), ForeignKey("media.id"), nullable=False)

    # OpenAI Analysis Results
    analysis_response = Column(JSON, nullable=True)  # Store the full OpenAI response
    question_type = Column(String(50), nullable=True)  # "mcq", "written", "other"
    detected_language = Column(String(50), nullable=True)
    title = Column(String(500), nullable=True)
    summary_title = Column(String(500), nullable=True)

    # Token usage
    tokens_used = Column(Integer, nullable=False, default=0)
    points_cost = Column(Integer, nullable=False, default=0)

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
    user = relationship("User", back_populates="interactions")
    file = relationship("Media", back_populates="interactions")
    point_transaction = relationship(
        "PointTransaction", back_populates="interaction", uselist=False
    )
