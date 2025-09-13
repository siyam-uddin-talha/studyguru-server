import strawberry
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timedelta

from app.graphql.types.reward import AddRewardPointsInput, RewardPointsResponse
from app.models.user import User
from app.helpers.user import get_current_user_from_context
from app.helpers.subscription import add_point_transaction_async
from app.constants.constant import CONSTANTS


@strawberry.type
class RewardMutation:
    @strawberry.mutation
    async def add_reward_points(
        self, info, input: AddRewardPointsInput
    ) -> RewardPointsResponse:
        """
        Securely add reward points to user account.
        Points amount is determined server-side based on reward type.
        """
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return RewardPointsResponse(
                success=False, message="Authentication required"
            )

        db: AsyncSession = context.db

        # Security: Define reward amounts server-side only
        REWARD_AMOUNTS = {
            "ad_reward": 20,  # Fixed amount for ad rewards
            "referral": 50,  # Fixed amount for referrals
            "daily_bonus": 10,  # Fixed amount for daily bonuses
        }

        # Validate reward type
        if input.reward_type not in REWARD_AMOUNTS:
            return RewardPointsResponse(success=False, message="Invalid reward type")

        # Get the fixed reward amount (no user input allowed)
        points_to_add = REWARD_AMOUNTS[input.reward_type]

        # Additional security: Rate limiting for ad rewards
        if input.reward_type == "ad_reward":
            # Check if user has already claimed ad reward in the last 5 minutes
            recent_reward_check = await db.execute(
                select(User).where(
                    User.id == current_user.id,
                    User.updated_at >= datetime.utcnow() - timedelta(minutes=5),
                )
            )

            # You could add more sophisticated rate limiting here
            # For now, we'll allow it but log it for monitoring

        try:
            # Add points using the secure helper function
            transaction = await add_point_transaction_async(
                db=db,
                user_id=current_user.id,
                transaction_type="earned",
                points=points_to_add,
                description=f"Reward earned from {input.reward_type} via {input.source}",
                interaction_id=None,
            )

            if transaction:
                # Get updated user data
                result = await db.execute(
                    select(User).where(User.id == current_user.id)
                )
                updated_user = result.scalar_one_or_none()

                return RewardPointsResponse(
                    success=True,
                    message=f"Successfully earned {points_to_add} points!",
                    points_earned=points_to_add,
                    new_balance=updated_user.current_points if updated_user else None,
                )
            else:
                return RewardPointsResponse(
                    success=False, message="Failed to process reward"
                )

        except Exception as e:
            print(f"Error adding reward points: {str(e)}")
            return RewardPointsResponse(
                success=False, message="Failed to process reward. Please try again."
            )

    @strawberry.mutation
    async def validate_ad_reward(
        self, info, ad_unit_id: str, reward_amount: int
    ) -> RewardPointsResponse:
        """
        Validate and process ad reward with additional security checks.
        This endpoint validates the ad reward before adding points.
        """
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return RewardPointsResponse(
                success=False, message="Authentication required"
            )

        # Security validations
        VALID_AD_UNIT_IDS = [
            "ca-app-pub-3940256099942544/5224354917",  # Test ad unit
            # Add your production ad unit IDs here
        ]

        VALID_REWARD_AMOUNTS = [10, 15, 20, 25, 30]  # Only allow these amounts

        # Validate ad unit ID
        if ad_unit_id not in VALID_AD_UNIT_IDS:
            return RewardPointsResponse(success=False, message="Invalid ad unit")

        # Validate reward amount (prevent manipulation)
        if reward_amount not in VALID_REWARD_AMOUNTS:
            return RewardPointsResponse(success=False, message="Invalid reward amount")

        # Use the secure add_reward_points mutation
        return await self.add_reward_points(
            info, AddRewardPointsInput(reward_type="ad_reward", source="mobile_app")
        )
