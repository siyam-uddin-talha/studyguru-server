from sqlalchemy import (
    Column,
    String,
    DateTime,
    Integer,
    ForeignKey,
    Text,
    JSON,
    Boolean,
    Enum,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
import uuid
import enum


class GoalType(enum.Enum):
    TARGET = "Target"
    BUILD = "Build"
    QUIT = "Quit"


class GoalPeriod(enum.Enum):
    DAILY = "daily"
    WEEKLY = "weekly"


class ChartType(enum.Enum):
    LINE = "line"
    BAR = "bar"
    PIE = "pie"


class Goal(Base):
    __tablename__ = "goal"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(191), ForeignKey("user.id"), nullable=False)

    name = Column(String(500), nullable=False)
    icon = Column(
        JSON, nullable=False
    )  # Store icon as JSON: {type: 'emoji'|'svg'|'vector', value: str, library?: str}
    color = Column(String(50), nullable=False)
    type = Column(Enum(GoalType), nullable=False, default=GoalType.TARGET)
    period = Column(Enum(GoalPeriod), nullable=False, default=GoalPeriod.DAILY)
    target_value = Column(Integer, nullable=False, default=1)
    unit = Column(String(50), nullable=False, default="times")
    task_days = Column(JSON, nullable=False)  # Array of integers [0-6] for days of week

    # Reminder settings
    reminder_enabled = Column(Boolean, nullable=False, default=False)
    reminder_time = Column(String(10), nullable=True)  # HH:mm format
    reminder_message = Column(Text, nullable=True)

    # Data preferences
    show_memo = Column(Boolean, nullable=False, default=False)
    chart_type = Column(Enum(ChartType), nullable=False, default=ChartType.LINE)

    # Duration
    start_date = Column(DateTime, nullable=False, default=func.now())
    end_date = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=func.now(), nullable=True)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=True
    )

    # Relationships
    user = relationship("User", back_populates="goals")
    progress = relationship(
        "GoalProgress", back_populates="goal", cascade="all, delete-orphan"
    )


class GoalProgress(Base):
    __tablename__ = "goal_progress"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    goal_id = Column(String(191), ForeignKey("goal.id"), nullable=False)
    date = Column(String(10), nullable=False)  # YYYY-MM-DD format
    value = Column(Integer, nullable=False, default=0)
    completed = Column(Boolean, nullable=False, default=False)
    memo = Column(Text, nullable=True)

    created_at = Column(DateTime, default=func.now(), nullable=True)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=True
    )

    # Relationships
    goal = relationship("Goal", back_populates="progress")

    # Unique constraint: one progress entry per goal per date
    __table_args__ = ({"sqlite_autoincrement": True},)
