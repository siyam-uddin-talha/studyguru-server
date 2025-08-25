from sqlalchemy import (
    Column,
    String,
    Boolean,
    DateTime,
    Integer,
    Text,
    ForeignKey,
    Enum,
    Float,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
import enum
import uuid


class AccountProvider(enum.Enum):
    EMAIL = "EMAIL"
    GOOGLE = "GOOGLE"


class UserAccountType(enum.Enum):
    SUPER_ADMIN = "SUPER_ADMIN"
    ADMIN = "ADMIN"
    USER = "USER"


class User(Base):
    __tablename__ = "user"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    first_name = Column(String(191), nullable=False)
    last_name = Column(String(191), nullable=False)
    password = Column(String(191), nullable=True)
    email = Column(String(191), unique=True, nullable=False)
    account_provider = Column(Enum(AccountProvider), nullable=True)
    account_type = Column(
        Enum(UserAccountType), default=UserAccountType.USER, nullable=False
    )
    photo_url = Column(String(191), nullable=True)
    birthday = Column(DateTime, nullable=True)
    profession_title = Column(Text, nullable=True)
    education_level = Column(Text, nullable=True)
    primary_city = Column(Text, nullable=True)
    primary_state = Column(Text, nullable=True)
    primary_address = Column(Text, nullable=True)
    about_description = Column(Text, nullable=True)
    phone_number = Column(Text, nullable=True)
    zip_code = Column(Text, nullable=True)
    reset_password_pin = Column(Integer, nullable=True)
    reset_password_expire_date = Column(DateTime, nullable=True)
    verify_pin = Column(Integer, nullable=True)
    verify_pin_expire_date = Column(DateTime, nullable=True)
    verify_status = Column(Boolean, default=False, nullable=False)

    # Points system
    current_points = Column(
        Integer, default=0, nullable=False
    )  # Free users get 30 points
    total_points_earned = Column(Integer, default=0, nullable=False)
    total_points_used = Column(Integer, default=0, nullable=False)

    # Subscription
    purchased_subscription_id = Column(
        String(191), ForeignKey("purchased_subscription.id"), nullable=True
    )
    country_id = Column(String(191), ForeignKey("country.id"), nullable=True)
    super_admin = Column(Boolean, default=False, nullable=True)
    last_login_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=True)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=True
    )

    # Relationships
    purchased_subscription = relationship(
        "PurchasedSubscription", back_populates="users"
    )
    country = relationship("Country", back_populates="users")
    doc_materials = relationship("DocMaterial", back_populates="user")
    point_transactions = relationship("PointTransaction", back_populates="user")
    billing_logs = relationship("BillingLog", back_populates="user")
