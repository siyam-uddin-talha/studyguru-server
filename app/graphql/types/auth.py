import strawberry
from typing import Optional
from datetime import datetime
from app.models.user import AccountProvider, UserAccountType
from .common import CountryType
from .subscription import PurchasedSubscriptionType
from typing import Type
import enum


AccountProviderEnum: Type[AccountProvider] = strawberry.enum(AccountProvider)
UserAccountTypeEnum: Type[UserAccountType] = strawberry.enum(UserAccountType)


@strawberry.type
class Account:
    id: str
    first_name: str
    last_name: str
    email: str
    account_type: UserAccountTypeEnum
    primary_state: Optional[str] = None
    primary_city: Optional[str] = None
    profession_title: Optional[str] = None
    about_description: Optional[str] = None
    primary_address: Optional[str] = None
    phone_number: Optional[str] = None
    zip_code: Optional[str] = None
    photo_url: Optional[str] = None
    created_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None
    verify_status: Optional[bool] = None
    super_admin: Optional[bool] = None
    account_provider: Optional[AccountProviderEnum] = None
    purchased_subscription: Optional[PurchasedSubscriptionType] = None
    country: Optional[CountryType] = None


@strawberry.type
class AuthType:
    success: bool
    message: Optional[str] = None
    account: Optional[Account] = None
    token: Optional[str] = None


@strawberry.type
class AuthLoginType:
    success: bool
    message: Optional[str] = None
    verify: Optional[bool] = None
    email: Optional[str] = None
    token: Optional[str] = None
    account: Optional[Account] = None


@strawberry.input
class LoginInput:
    email: str
    password: str


@strawberry.input
class CountryInput:
    id: str
    name: str
    currency_code: Optional[str] = None
    country_code: Optional[str] = None
    calling_code: Optional[str] = None


@strawberry.input
class AccountAccessInput:
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone_number: Optional[str] = None
    password: Optional[str] = None
    provider: str
    credential: Optional[str] = None
    country: Optional[CountryInput] = None
