import strawberry
from typing import Optional, List
from datetime import datetime


@strawberry.type
class ModuleType:
    id: str
    name: str
    slug: str
    description: Optional[str] = None
    is_active: bool
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@strawberry.input
class ModuleInput:
    name: str
    slug: str
    description: Optional[str] = None
    is_active: Optional[bool] = True


@strawberry.input
class UpdateModuleInput:
    name: Optional[str] = None
    slug: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


@strawberry.type
class PermissionType:
    can_read: bool
    can_insert: bool
    can_update: bool
    can_delete: bool


@strawberry.type
class UserModulePermissionType:
    id: str
    user_id: str
    module_id: str
    module: Optional[ModuleType] = None
    can_read: bool
    can_insert: bool
    can_update: bool
    can_delete: bool
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@strawberry.input
class PermissionInput:
    can_read: bool = False
    can_insert: bool = False
    can_update: bool = False
    can_delete: bool = False


@strawberry.input
class UserModulePermissionInput:
    user_id: str
    module_id: str
    can_read: bool = False
    can_insert: bool = False
    can_update: bool = False
    can_delete: bool = False


@strawberry.input
class UpdateUserModulePermissionInput:
    can_read: Optional[bool] = None
    can_insert: Optional[bool] = None
    can_update: Optional[bool] = None
    can_delete: Optional[bool] = None


@strawberry.type
class UserModuleInfoType:
    module_id: str
    module_name: str
    module_slug: str
    can_read: bool
    can_insert: bool
    can_update: bool
    can_delete: bool


@strawberry.type
class ModuleListResponse:
    success: bool
    message: Optional[str] = None
    modules: Optional[List[ModuleType]] = None


@strawberry.type
class ModuleResponse:
    success: bool
    message: Optional[str] = None
    module: Optional[ModuleType] = None


@strawberry.type
class UserModulePermissionListResponse:
    success: bool
    message: Optional[str] = None
    permissions: Optional[List[UserModulePermissionType]] = None


@strawberry.type
class UserModulePermissionResponse:
    success: bool
    message: Optional[str] = None
    permission: Optional[UserModulePermissionType] = None


@strawberry.type
class UserModulesResponse:
    success: bool
    message: Optional[str] = None
    modules: Optional[List[UserModuleInfoType]] = None
