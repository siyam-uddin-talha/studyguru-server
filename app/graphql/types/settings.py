# server/app/graphql/types/settings.py
from typing import Optional
import strawberry
from app.graphql.types.auth import Account
from .common import CountryInput
from datetime import datetime

@strawberry.input
class UpdatePasswordInput:
    current_password: str
    new_password: str

@strawberry.input
class UpdateProfileInput:
    # id: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    # account_type: UserAccountTypeEnum
    primary_state: Optional[str] = None
    primary_city: Optional[str] = None
    profession_title: Optional[str] = None
    about_description: Optional[str] = None
    primary_address: Optional[str] = None
    phone_number: Optional[str] = None
    zip_code: Optional[str] = None
    photo_url: Optional[str] = None
    education_level: Optional[str] = None
    birthday: Optional[datetime] = None
    # account_provider: Optional[AccountProviderEnum] = None
    # purchased_subscription: Optional[PurchasedSubscriptionType] = None
    country: Optional[CountryInput] = None

@strawberry.type
class SettingsResponse:
    success: bool
    message: str
    account: Optional[Account] = None
    logout: Optional[bool] = False

@strawberry.type
class PasswordConfirmationResponse:
    success: bool
    message: str
    has_password: bool
