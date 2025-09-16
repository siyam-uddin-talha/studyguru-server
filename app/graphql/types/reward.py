import strawberry
from typing import Optional
from datetime import datetime
from app.graphql.types.common import DefaultResponse


@strawberry.input
class AddRewardPointsInput:
    # No amount parameter - this is calculated server-side for security
    reward_type: str = "ad_reward"  # Type of reward (ad_reward, referral, etc.)
    source: str = "mobile_app"  # Source of the reward
    interaction_id: Optional[str] = None


@strawberry.type
class RewardPointsResponse:
    success: bool
    message: str
    points_earned: Optional[int] = None
    new_balance: Optional[int] = None
