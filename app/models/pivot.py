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
    Table,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from app.core.database import Base
import enum
import uuid


class Country(Base):
    __tablename__ = "country"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(191), unique=True, nullable=False)
    currency_code = Column(Text, nullable=False)
    country_code = Column(Text, nullable=False)
    calling_code = Column(Text, nullable=True)
    deleted = Column(Boolean, default=False)

    # Relationships
    users = relationship("User", back_populates="country")


# # User-Role Pivot Model
# class UserRolePivot(Base):
#     __tablename__ = "user_role_pivot"

#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
#     role_id = Column(Integer, ForeignKey("roles.id"))

#     is_active = Column(Boolean, default=True)
#     assigned_at = Column(DateTime(timezone=True), server_default=func.now())

#     user = relationship("User", back_populates="role_pivot")
#     role = relationship("Role", back_populates="user_pivot")

#     # Ensure unique combination of user and role
#     __table_args__ = (UniqueConstraint("user_id", "role_id", name="uq_user_role"),)


# # Enum for Permission Types
# class PermissionType(enum.Enum):
#     READ = "read"
#     WRITE = "write"
#     DELETE = "delete"
#     ADMIN = "admin"


# # Role-Permission Pivot Model
# class RolePermissionPivot(Base):
#     __tablename__ = "role_permission_pivot"

#     id = Column(Integer, primary_key=True, index=True)
#     role_id = Column(Integer, ForeignKey("roles.id"))
#     permission_type = Column(Enum(PermissionType))

#     resource_type = Column(String, nullable=False)  # e.g., 'user', 'company', 'project'

#     role = relationship("Role", back_populates="permission_pivot")

#     # Ensure unique combination of role and permission
#     __table_args__ = (
#         UniqueConstraint(
#             "role_id", "permission_type", "resource_type", name="uq_role_permission"
#         ),
#     )
