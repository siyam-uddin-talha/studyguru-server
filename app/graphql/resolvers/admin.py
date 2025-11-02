import strawberry
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from typing import Optional, List

from app.graphql.types.admin import (
    ModuleType,
    ModuleInput,
    UpdateModuleInput,
    UserModulePermissionType,
    UserModulePermissionInput,
    UpdateUserModulePermissionInput,
    UserModuleInfoType,
    ModuleListResponse,
    ModuleResponse,
    UserModulePermissionListResponse,
    UserModulePermissionResponse,
    UserModulesResponse,
)
from app.graphql.types.common import DefaultResponse
from app.models.user import User
from app.models.rbac import Module, UserModulePermission
from app.helpers.permissions import (
    require_permission,
    check_permission,
    get_user_modules,
    is_admin_user,
)
from app.constants.constant import CONSTANTS


@strawberry.type
class AdminQuery:
    @strawberry.field
    async def modules(self, info) -> ModuleListResponse:
        """Get all modules - requires read permission on user_module"""
        context = info.context
        db: AsyncSession = context.db

        if not context.current_user:
            return ModuleListResponse(
                success=False,
                message="Authentication required",
                modules=None,
            )

        # Check if user is admin
        if not is_admin_user(context.current_user):
            return ModuleListResponse(
                success=False,
                message="Admin access required",
                modules=None,
            )

        try:
            # EXECUTIVE can see all modules, others need read permission
            if context.current_user.account_type.value == "EXICUTIVE":
                # Get all modules
                result = await db.execute(
                    select(Module).order_by(Module.created_at.desc())
                )
                modules = result.scalars().all()
            else:
                # Check read permission for user_module (admin module)
                has_permission = await check_permission(
                    db, context.current_user, "user_module", "read"
                )
                if not has_permission:
                    return ModuleListResponse(
                        success=False,
                        message="You don't have permission to view modules",
                        modules=None,
                    )
                result = await db.execute(
                    select(Module).order_by(Module.created_at.desc())
                )
                modules = result.scalars().all()

            module_types = [
                ModuleType(
                    id=module.id,
                    name=module.name,
                    slug=module.slug,
                    description=module.description,
                    is_active=module.is_active,
                    created_at=module.created_at,
                    updated_at=module.updated_at,
                )
                for module in modules
            ]

            return ModuleListResponse(
                success=True,
                message="Modules retrieved successfully",
                modules=module_types,
            )
        except Exception as e:
            return ModuleListResponse(
                success=False,
                message=str(e),
                modules=None,
            )

    @strawberry.field
    async def module(self, info, module_id: str) -> ModuleResponse:
        """Get a single module by ID"""
        context = info.context
        db: AsyncSession = context.db

        if not context.current_user:
            return ModuleResponse(
                success=False,
                message="Authentication required",
                module=None,
            )

        if not is_admin_user(context.current_user):
            return ModuleResponse(
                success=False,
                message="Admin access required",
                module=None,
            )

        try:
            # Check permission
            if context.current_user.account_type.value != "EXICUTIVE":
                has_permission = await check_permission(
                    db, context.current_user, "user_module", "read"
                )
                if not has_permission:
                    return ModuleResponse(
                        success=False,
                        message="You don't have permission to view modules",
                        module=None,
                    )

            result = await db.execute(select(Module).where(Module.id == module_id))
            module = result.scalar_one_or_none()

            if not module:
                return ModuleResponse(
                    success=False,
                    message="Module not found",
                    module=None,
                )

            module_type = ModuleType(
                id=module.id,
                name=module.name,
                slug=module.slug,
                description=module.description,
                is_active=module.is_active,
                created_at=module.created_at,
                updated_at=module.updated_at,
            )

            return ModuleResponse(
                success=True,
                message="Module retrieved successfully",
                module=module_type,
            )
        except Exception as e:
            return ModuleResponse(
                success=False,
                message=str(e),
                module=None,
            )

    @strawberry.field
    async def my_modules(self, info) -> UserModulesResponse:
        """Get current user's modules and permissions"""
        context = info.context
        db: AsyncSession = context.db

        if not context.current_user:
            return UserModulesResponse(
                success=False,
                message="Authentication required",
                modules=None,
            )

        try:
            modules_data = await get_user_modules(db, context.current_user)

            module_info_types = [
                UserModuleInfoType(
                    module_id=mod["module_id"],
                    module_name=mod["module_name"],
                    module_slug=mod["module_slug"],
                    can_read=mod["can_read"],
                    can_insert=mod["can_insert"],
                    can_update=mod["can_update"],
                    can_delete=mod["can_delete"],
                )
                for mod in modules_data
            ]

            return UserModulesResponse(
                success=True,
                message="Modules retrieved successfully",
                modules=module_info_types,
            )
        except Exception as e:
            return UserModulesResponse(
                success=False,
                message=str(e),
                modules=None,
            )

    @strawberry.field
    async def user_module_permissions(
        self,
        info,
        user_id: Optional[str] = None,
        module_id: Optional[str] = None,
    ) -> UserModulePermissionListResponse:
        """Get user module permissions - requires read permission on user_module"""
        context = info.context
        db: AsyncSession = context.db

        if not context.current_user:
            return UserModulePermissionListResponse(
                success=False,
                message="Authentication required",
                permissions=None,
            )

        if not is_admin_user(context.current_user):
            return UserModulePermissionListResponse(
                success=False,
                message="Admin access required",
                permissions=None,
            )

        try:
            # Check permission
            if context.current_user.account_type.value != "EXICUTIVE":
                await require_permission(
                    db, context.current_user, "user_module", "read"
                )

            # Build query
            query = select(UserModulePermission).options(
                selectinload(UserModulePermission.module)
            )

            if user_id:
                query = query.where(UserModulePermission.user_id == user_id)
            if module_id:
                query = query.where(UserModulePermission.module_id == module_id)

            result = await db.execute(query)
            permissions = result.scalars().all()

            permission_types = [
                UserModulePermissionType(
                    id=perm.id,
                    user_id=perm.user_id,
                    module_id=perm.module_id,
                    module=(
                        ModuleType(
                            id=perm.module.id,
                            name=perm.module.name,
                            slug=perm.module.slug,
                            description=perm.module.description,
                            is_active=perm.module.is_active,
                            created_at=perm.module.created_at,
                            updated_at=perm.module.updated_at,
                        )
                        if perm.module
                        else None
                    ),
                    can_read=perm.can_read,
                    can_insert=perm.can_insert,
                    can_update=perm.can_update,
                    can_delete=perm.can_delete,
                    created_at=perm.created_at,
                    updated_at=perm.updated_at,
                )
                for perm in permissions
            ]

            return UserModulePermissionListResponse(
                success=True,
                message="Permissions retrieved successfully",
                permissions=permission_types,
            )
        except PermissionError as e:
            return UserModulePermissionListResponse(
                success=False,
                message=str(e),
                permissions=None,
            )
        except Exception as e:
            return UserModulePermissionListResponse(
                success=False,
                message=str(e),
                permissions=None,
            )


@strawberry.type
class AdminMutation:
    @strawberry.mutation
    async def create_module(self, info, input: ModuleInput) -> ModuleResponse:
        """Create a new module - requires insert permission on user_module"""
        context = info.context
        db: AsyncSession = context.db

        if not context.current_user:
            return ModuleResponse(
                success=False,
                message="Authentication required",
                module=None,
            )

        if not is_admin_user(context.current_user):
            return ModuleResponse(
                success=False,
                message="Admin access required",
                module=None,
            )

        try:
            # Check permission
            if context.current_user.account_type.value != "EXICUTIVE":
                await require_permission(
                    db, context.current_user, "user_module", "insert"
                )

            # Check if module with same slug exists
            existing_result = await db.execute(
                select(Module).where(
                    or_(Module.slug == input.slug, Module.name == input.name)
                )
            )
            existing = existing_result.scalar_one_or_none()

            if existing:
                return ModuleResponse(
                    success=False,
                    message="Module with this slug or name already exists",
                    module=None,
                )

            # Create module
            new_module = Module(
                name=input.name,
                slug=input.slug,
                description=input.description,
                is_active=input.is_active if input.is_active is not None else True,
            )

            db.add(new_module)
            await db.commit()
            await db.refresh(new_module)

            module_type = ModuleType(
                id=new_module.id,
                name=new_module.name,
                slug=new_module.slug,
                description=new_module.description,
                is_active=new_module.is_active,
                created_at=new_module.created_at,
                updated_at=new_module.updated_at,
            )

            return ModuleResponse(
                success=True,
                message="Module created successfully",
                module=module_type,
            )
        except PermissionError as e:
            return ModuleResponse(
                success=False,
                message=str(e),
                module=None,
            )
        except Exception as e:
            await db.rollback()
            return ModuleResponse(
                success=False,
                message=str(e),
                module=None,
            )

    @strawberry.mutation
    async def update_module(
        self, info, module_id: str, input: UpdateModuleInput
    ) -> ModuleResponse:
        """Update a module - requires update permission on user_module"""
        context = info.context
        db: AsyncSession = context.db

        if not context.current_user:
            return ModuleResponse(
                success=False,
                message="Authentication required",
                module=None,
            )

        if not is_admin_user(context.current_user):
            return ModuleResponse(
                success=False,
                message="Admin access required",
                module=None,
            )

        try:
            # Check permission
            if context.current_user.account_type.value != "EXICUTIVE":
                await require_permission(
                    db, context.current_user, "user_module", "update"
                )

            # Get module
            result = await db.execute(select(Module).where(Module.id == module_id))
            module = result.scalar_one_or_none()

            if not module:
                return ModuleResponse(
                    success=False,
                    message="Module not found",
                    module=None,
                )

            # Check if slug/name conflicts (if being updated)
            if input.slug and input.slug != module.slug:
                existing_result = await db.execute(
                    select(Module).where(
                        Module.slug == input.slug,
                        Module.id != module_id,
                    )
                )
                if existing_result.scalar_one_or_none():
                    return ModuleResponse(
                        success=False,
                        message="Module with this slug already exists",
                        module=None,
                    )

            if input.name and input.name != module.name:
                existing_result = await db.execute(
                    select(Module).where(
                        Module.name == input.name,
                        Module.id != module_id,
                    )
                )
                if existing_result.scalar_one_or_none():
                    return ModuleResponse(
                        success=False,
                        message="Module with this name already exists",
                        module=None,
                    )

            # Update module
            if input.name is not None:
                module.name = input.name
            if input.slug is not None:
                module.slug = input.slug
            if input.description is not None:
                module.description = input.description
            if input.is_active is not None:
                module.is_active = input.is_active

            await db.commit()
            await db.refresh(module)

            module_type = ModuleType(
                id=module.id,
                name=module.name,
                slug=module.slug,
                description=module.description,
                is_active=module.is_active,
                created_at=module.created_at,
                updated_at=module.updated_at,
            )

            return ModuleResponse(
                success=True,
                message="Module updated successfully",
                module=module_type,
            )
        except PermissionError as e:
            return ModuleResponse(
                success=False,
                message=str(e),
                module=None,
            )
        except Exception as e:
            await db.rollback()
            return ModuleResponse(
                success=False,
                message=str(e),
                module=None,
            )

    @strawberry.mutation
    async def delete_module(self, info, module_id: str) -> DefaultResponse:
        """Delete a module - requires delete permission on user_module"""
        context = info.context
        db: AsyncSession = context.db

        if not context.current_user:
            return DefaultResponse(
                success=False,
                message="Authentication required",
            )

        if not is_admin_user(context.current_user):
            return DefaultResponse(
                success=False,
                message="Admin access required",
            )

        try:
            # Check permission
            if context.current_user.account_type.value != "EXICUTIVE":
                await require_permission(
                    db, context.current_user, "user_module", "delete"
                )

            # Get module
            result = await db.execute(select(Module).where(Module.id == module_id))
            module = result.scalar_one_or_none()

            if not module:
                return DefaultResponse(
                    success=False,
                    message="Module not found",
                )

            # Delete module (cascade will delete permissions)
            await db.delete(module)
            await db.flush()  # Flush before commit to handle cascade
            await db.commit()

            return DefaultResponse(
                success=True,
                message="Module deleted successfully",
            )
        except PermissionError as e:
            return DefaultResponse(
                success=False,
                message=str(e),
            )
        except Exception as e:
            await db.rollback()
            return DefaultResponse(
                success=False,
                message=str(e),
            )

    @strawberry.mutation
    async def assign_module_permission(
        self, info, input: UserModulePermissionInput
    ) -> UserModulePermissionResponse:
        """Assign module permission to user - requires insert permission on user_module"""
        context = info.context
        db: AsyncSession = context.db

        if not context.current_user:
            return UserModulePermissionResponse(
                success=False,
                message="Authentication required",
                permission=None,
            )

        if not is_admin_user(context.current_user):
            return UserModulePermissionResponse(
                success=False,
                message="Admin access required",
                permission=None,
            )

        try:
            # Check permission
            if context.current_user.account_type.value != "EXICUTIVE":
                await require_permission(
                    db, context.current_user, "user_module", "insert"
                )

            # Verify user exists
            user_result = await db.execute(select(User).where(User.id == input.user_id))
            user = user_result.scalar_one_or_none()

            if not user:
                return UserModulePermissionResponse(
                    success=False,
                    message="User not found",
                    permission=None,
                )

            # Verify module exists
            module_result = await db.execute(
                select(Module).where(Module.id == input.module_id)
            )
            module = module_result.scalar_one_or_none()

            if not module:
                return UserModulePermissionResponse(
                    success=False,
                    message="Module not found",
                    permission=None,
                )

            # Check if permission already exists
            existing_result = await db.execute(
                select(UserModulePermission).where(
                    UserModulePermission.user_id == input.user_id,
                    UserModulePermission.module_id == input.module_id,
                )
            )
            existing = existing_result.scalar_one_or_none()

            if existing:
                return UserModulePermissionResponse(
                    success=False,
                    message="Permission already exists. Use update instead.",
                    permission=None,
                )

            # Create permission
            new_permission = UserModulePermission(
                user_id=input.user_id,
                module_id=input.module_id,
                can_read=input.can_read,
                can_insert=input.can_insert,
                can_update=input.can_update,
                can_delete=input.can_delete,
            )

            db.add(new_permission)
            await db.commit()
            await db.refresh(new_permission)

            # Load module relationship
            await db.refresh(new_permission, ["module"])

            permission_type = UserModulePermissionType(
                id=new_permission.id,
                user_id=new_permission.user_id,
                module_id=new_permission.module_id,
                module=(
                    ModuleType(
                        id=new_permission.module.id,
                        name=new_permission.module.name,
                        slug=new_permission.module.slug,
                        description=new_permission.module.description,
                        is_active=new_permission.module.is_active,
                        created_at=new_permission.module.created_at,
                        updated_at=new_permission.module.updated_at,
                    )
                    if new_permission.module
                    else None
                ),
                can_read=new_permission.can_read,
                can_insert=new_permission.can_insert,
                can_update=new_permission.can_update,
                can_delete=new_permission.can_delete,
                created_at=new_permission.created_at,
                updated_at=new_permission.updated_at,
            )

            return UserModulePermissionResponse(
                success=True,
                message="Permission assigned successfully",
                permission=permission_type,
            )
        except PermissionError as e:
            return UserModulePermissionResponse(
                success=False,
                message=str(e),
                permission=None,
            )
        except Exception as e:
            await db.rollback()
            return UserModulePermissionResponse(
                success=False,
                message=str(e),
                permission=None,
            )

    @strawberry.mutation
    async def update_module_permission(
        self,
        info,
        permission_id: str,
        input: UpdateUserModulePermissionInput,
    ) -> UserModulePermissionResponse:
        """Update module permission - requires update permission on user_module"""
        context = info.context
        db: AsyncSession = context.db

        if not context.current_user:
            return UserModulePermissionResponse(
                success=False,
                message="Authentication required",
                permission=None,
            )

        if not is_admin_user(context.current_user):
            return UserModulePermissionResponse(
                success=False,
                message="Admin access required",
                permission=None,
            )

        try:
            # Check permission
            if context.current_user.account_type.value != "EXICUTIVE":
                await require_permission(
                    db, context.current_user, "user_module", "update"
                )

            # Get permission
            result = await db.execute(
                select(UserModulePermission)
                .options(selectinload(UserModulePermission.module))
                .where(UserModulePermission.id == permission_id)
            )
            permission = result.scalar_one_or_none()

            if not permission:
                return UserModulePermissionResponse(
                    success=False,
                    message="Permission not found",
                    permission=None,
                )

            # Update permission
            if input.can_read is not None:
                permission.can_read = input.can_read
            if input.can_insert is not None:
                permission.can_insert = input.can_insert
            if input.can_update is not None:
                permission.can_update = input.can_update
            if input.can_delete is not None:
                permission.can_delete = input.can_delete

            await db.commit()
            await db.refresh(permission)

            permission_type = UserModulePermissionType(
                id=permission.id,
                user_id=permission.user_id,
                module_id=permission.module_id,
                module=(
                    ModuleType(
                        id=permission.module.id,
                        name=permission.module.name,
                        slug=permission.module.slug,
                        description=permission.module.description,
                        is_active=permission.module.is_active,
                        created_at=permission.module.created_at,
                        updated_at=permission.module.updated_at,
                    )
                    if permission.module
                    else None
                ),
                can_read=permission.can_read,
                can_insert=permission.can_insert,
                can_update=permission.can_update,
                can_delete=permission.can_delete,
                created_at=permission.created_at,
                updated_at=permission.updated_at,
            )

            return UserModulePermissionResponse(
                success=True,
                message="Permission updated successfully",
                permission=permission_type,
            )
        except PermissionError as e:
            return UserModulePermissionResponse(
                success=False,
                message=str(e),
                permission=None,
            )
        except Exception as e:
            await db.rollback()
            return UserModulePermissionResponse(
                success=False,
                message=str(e),
                permission=None,
            )

    @strawberry.mutation
    async def delete_module_permission(
        self, info, permission_id: str
    ) -> DefaultResponse:
        """Delete module permission - requires delete permission on user_module"""
        context = info.context
        db: AsyncSession = context.db

        if not context.current_user:
            return DefaultResponse(
                success=False,
                message="Authentication required",
            )

        if not is_admin_user(context.current_user):
            return DefaultResponse(
                success=False,
                message="Admin access required",
            )

        try:
            # Check permission
            if context.current_user.account_type.value != "EXICUTIVE":
                await require_permission(
                    db, context.current_user, "user_module", "delete"
                )

            # Get permission
            result = await db.execute(
                select(UserModulePermission).where(
                    UserModulePermission.id == permission_id
                )
            )
            permission = result.scalar_one_or_none()

            if not permission:
                return DefaultResponse(
                    success=False,
                    message="Permission not found",
                )

            # Delete permission
            await db.delete(permission)
            await db.commit()

            return DefaultResponse(
                success=True,
                message="Permission deleted successfully",
            )
        except PermissionError as e:
            return DefaultResponse(
                success=False,
                message=str(e),
            )
        except Exception as e:
            await db.rollback()
            return DefaultResponse(
                success=False,
                message=str(e),
            )
