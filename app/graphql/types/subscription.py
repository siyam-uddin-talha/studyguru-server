import strawberry
from typing import Optional, List
from datetime import datetime
from app.models.subscription import SubscriptionPlan


@strawberry.enum
class SubscriptionPlanEnum(SubscriptionPlan):
    FREE = "FREE"
    STARTER = "STARTER" 
    BASIC = "BASIC"
    PREMIUM = "PREMIUM"
    LIFE_TIME = "LIFE_TIME"


@strawberry.type
class UsageLimitType:
    id: str
    total_user_track: int
    image_deck: int
    custom_deck: bool
    advanced_analytics: bool
    export_feature: bool
    note: bool


@strawberry.type
class SubscriptionType:
    id: str
    subscription_name: str
    usd_amount: float
    gbp_amount: float
    subscription_plan: SubscriptionPlanEnum
    usage_limit: Optional[UsageLimitType] = None
    created_at: Optional[datetime] = None


@strawberry.type
class PurchasedSubscriptionType:
    id: str
    subscription_id: str
    subscription: Optional[SubscriptionType] = None
    past_due_time: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@strawberry.type
class SubscriptionsResponse:
    success: bool
    message: Optional[str] = None
    subscriptions: Optional[List[SubscriptionType]] = None


@strawberry.type
class UsageCount:
    total_user_track: int
    image_deck: int


@strawberry.type
class AccountSubscriptionDetails:
    success: bool
    message: Optional[str] = None
    purchased_subscription: Optional[PurchasedSubscriptionType] = None
    free_subscription: Optional[SubscriptionType] = None
    usage_count: Optional[UsageCount] = None


@strawberry.type
class SubscriptionHistory:
    id: str
    amount: float
    billing_month: Optional[datetime] = None
    created_at: Optional[datetime] = None
    invoice_number: Optional[str] = None
    summery: Optional[str] = None
    invoice_url: Optional[str] = None
    currency: Optional[str] = None
    status: Optional[str] = None


@strawberry.type
class SubscriptionHistoryDetails:
    success: bool
    message: Optional[str] = None
    result: Optional[List[SubscriptionHistory]] = None