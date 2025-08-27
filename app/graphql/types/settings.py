# server/app/graphql/types/settings.py
from typing import Optional
import strawberry
from app.graphql.types.auth import AccountType


@strawberry.input
class UpdatePasswordInput:
    current_password: str
    new_password: str


@strawberry.input
class UpdateProfileInput:
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone_number: Optional[str] = None
    photo_url: Optional[str] = None
    # Add other fields as needed


@strawberry.type
class SettingsResponse:
    success: bool
    message: str
    account: Optional[AccountType] = None
    logout: bool = False
