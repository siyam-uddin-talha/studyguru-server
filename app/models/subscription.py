from sqlalchemy import (
    Column,
    String,
    Float,
    Boolean,
    DateTime,
    Integer,
    ForeignKey,
    Enum,
    Text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
import enum
import uuid


class SubscriptionPlan(enum.Enum):
    ESSENTIAL = "ESSENTIAL"
    PLUS = "PLUS"
    ELITE = "ELITE"


class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    subscription_name = Column(String(191), nullable=False)
    usd_amount = Column(Float, nullable=False)
    bdt_amount = Column(Float, nullable=False)
    subscription_plan = Column(Enum(SubscriptionPlan), nullable=False)
    points_per_month = Column(Integer, nullable=False, default=0)
    is_addon = Column(Boolean, default=False, nullable=False)  # For point add-ons
    min_points = Column(Integer, nullable=True)  # Minimum points for add-ons
    created_at = Column(DateTime, default=func.now(), nullable=True)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=True
    )

    # Relationships
    purchased_subscriptions = relationship(
        "PurchasedSubscription", back_populates="subscription"
    )
    billing_logs = relationship("BillingLog", back_populates="subscription")


class PurchasedSubscription(Base):
    __tablename__ = "purchased_subscription"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    subscription_id = Column(
        String(191), ForeignKey("subscriptions.id"), nullable=False
    )
    paddle_subscription_id = Column(String(191), nullable=True)
    subscription_status = Column(
        String(191), nullable=True
    )  # active, past_due, canceled, etc.
    next_billing_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=True)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=True
    )

    # Relationships
    subscription = relationship(
        "Subscription", back_populates="purchased_subscriptions"
    )
    users = relationship("User", back_populates="purchased_subscription")


class PointTransaction(Base):
    __tablename__ = "point_transaction"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(191), ForeignKey("user.id"), nullable=False)
    transaction_type = Column(
        String(50), nullable=False
    )  # "earned", "used", "purchased"
    points = Column(Integer, nullable=False)
    description = Column(Text, nullable=True)
    conversation_id = Column(String(191), ForeignKey("conversation.id"), nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=True)

    # Relationships
    user = relationship("User", back_populates="point_transactions")
    conversation = relationship("Conversation", back_populates="point_transaction")


class BillingLog(Base):
    __tablename__ = "billing_log"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(191), ForeignKey("user.id"), nullable=False)
    subscription_id = Column(
        String(191), ForeignKey("subscriptions.id"), nullable=False
    )
    amount = Column(Float, nullable=False)
    currency = Column(String(10), default="USD", nullable=False)
    paddle_transaction_id = Column(String(191), nullable=True)
    status = Column(String(50), nullable=True)
    points_added = Column(Integer, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=True)

    # Relationships
    user = relationship("User", back_populates="billing_logs")
    subscription = relationship("Subscription", back_populates="billing_logs")
