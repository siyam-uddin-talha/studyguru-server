from sqlalchemy import (
    Column,
    String,
    Boolean,
    DateTime,
    Integer,
    ForeignKey,
    Text,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
import uuid


class Module(Base):
    """
    Dynamic modules table - modules can be added via dashboard
    Examples: user_module, support_module, etc.
    """

    __tablename__ = "module"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(
        String(191), unique=True, nullable=False
    )  # e.g., "User Module", "Support Module"
    slug = Column(
        String(191), unique=True, nullable=False
    )  # e.g., "user_module", "support_module"
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=True)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=True
    )

    # Relationships
    user_permissions = relationship(
        "UserModulePermission", back_populates="module", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_module_slug", "slug"),
        Index("idx_module_active", "is_active"),
    )


class UserModulePermission(Base):
    """
    Pivot table linking users to modules with specific permissions
    A user can be assigned to multiple modules with different permissions
    """

    __tablename__ = "user_module_permission"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(191), ForeignKey("user.id"), nullable=False)
    module_id = Column(String(191), ForeignKey("module.id"), nullable=False)

    # Granular permissions
    can_read = Column(Boolean, default=False, nullable=False)
    can_insert = Column(Boolean, default=False, nullable=False)
    can_update = Column(Boolean, default=False, nullable=False)
    can_delete = Column(Boolean, default=False, nullable=False)

    created_at = Column(DateTime, default=func.now(), nullable=True)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=True
    )

    # Relationships
    user = relationship("User", back_populates="module_permissions")
    module = relationship("Module", back_populates="user_permissions")

    __table_args__ = (
        UniqueConstraint("user_id", "module_id", name="uq_user_module"),
        Index("idx_user_module_user", "user_id"),
        Index("idx_user_module_module", "module_id"),
    )
