"""
Permission checking helpers for RBAC system
"""

from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from sqlalchemy.orm import selectinload

from app.models.user import User, UserAccountType
from app.models.rbac import Module, UserModulePermission


async def check_permission(
    db: AsyncSession,
    user: User,
    module_slug: str,
    permission: str,  # 'read', 'insert', 'update', 'delete'
) -> bool:
    """
    Check if user has specific permission for a module

    Args:
        db: Database session
        user: User object
        module_slug: Module slug (e.g., 'user_module', 'support_module')
        permission: Permission type ('read', 'insert', 'update', 'delete')

    Returns:
        True if user has permission, False otherwise
    """
    # EXECUTIVE can do anything
    if user.account_type == UserAccountType.EXICUTIVE:
        return True

    # Only EXECUTIVE, SUPER_ADMIN, and ADMIN can have permissions
    if user.account_type not in [
        UserAccountType.EXICUTIVE,
        UserAccountType.SUPER_ADMIN,
        UserAccountType.ADMIN,
    ]:
        return False

    # Get module
    module_result = await db.execute(
        select(Module).where(Module.slug == module_slug, Module.is_active == True)
    )
    module = module_result.scalar_one_or_none()

    if not module:
        return False

    # Get user permission for this module
    permission_result = await db.execute(
        select(UserModulePermission).where(
            UserModulePermission.user_id == user.id,
            UserModulePermission.module_id == module.id,
        )
    )
    user_permission = permission_result.scalar_one_or_none()

    if not user_permission:
        return False

    # Check specific permission
    permission_map = {
        "read": user_permission.can_read,
        "insert": user_permission.can_insert,
        "update": user_permission.can_update,
        "delete": user_permission.can_delete,
    }

    return permission_map.get(permission, False)


async def require_permission(
    db: AsyncSession,
    user: Optional[User],
    module_slug: str,
    permission: str,
    raise_exception: bool = True,
) -> bool:
    """
    Require permission - raises exception if user doesn't have permission

    Args:
        db: Database session
        user: User object (can be None)
        module_slug: Module slug
        permission: Permission type
        raise_exception: Whether to raise exception or return False

    Returns:
        True if user has permission

    Raises:
        PermissionError if user doesn't have permission and raise_exception is True
    """
    if not user:
        if raise_exception:
            raise PermissionError("Authentication required")
        return False

    has_permission = await check_permission(db, user, module_slug, permission)

    if not has_permission:
        if raise_exception:
            raise PermissionError(
                f"You don't have {permission} permission for {module_slug}"
            )
        return False

    return True


async def get_user_modules(
    db: AsyncSession,
    user: User,
) -> list[dict]:
    """
    Get all modules and permissions for a user

    Args:
        db: Database session
        user: User object

    Returns:
        List of dictionaries with module and permission info
    """
    # EXECUTIVE has access to all modules
    if user.account_type == UserAccountType.EXICUTIVE:
        # Get all active modules
        modules_result = await db.execute(
            select(Module).where(Module.is_active == True)
        )
        modules = modules_result.scalars().all()

        return [
            {
                "module_id": module.id,
                "module_name": module.name,
                "module_slug": module.slug,
                "can_read": True,
                "can_insert": True,
                "can_update": True,
                "can_delete": True,
            }
            for module in modules
        ]

    # Get user permissions with module details
    permissions_result = await db.execute(
        select(UserModulePermission)
        .options(selectinload(UserModulePermission.module))
        .where(UserModulePermission.user_id == user.id)
    )
    user_permissions = permissions_result.scalars().all()

    return [
        {
            "module_id": perm.module.id,
            "module_name": perm.module.name,
            "module_slug": perm.module.slug,
            "can_read": perm.can_read,
            "can_insert": perm.can_insert,
            "can_update": perm.can_update,
            "can_delete": perm.can_delete,
        }
        for perm in user_permissions
        if perm.module.is_active
    ]


async def is_admin_user(user: Optional[User]) -> bool:
    """
    Check if user is an admin user (EXECUTIVE, SUPER_ADMIN, or ADMIN)

    Args:
        user: User object (can be None)

    Returns:
        True if user is admin, False otherwise
    """
    if not user:
        return False

    return user.account_type in [
        UserAccountType.EXICUTIVE,
        UserAccountType.SUPER_ADMIN,
        UserAccountType.ADMIN,
    ]
