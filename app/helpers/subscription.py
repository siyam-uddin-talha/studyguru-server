from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional
import uuid
from datetime import datetime

from app.models.subscription import (
    Subscription,
    PurchasedSubscription,
    SubscriptionPlan,
    PointTransaction,
)
from app.models.user import User
from app.models.interaction import (
    Interaction,
    InteractionShare,
    InteractionShareVisitor,
)
from sqlalchemy.exc import SQLAlchemyError


async def get_or_create_free_subscription(db: AsyncSession) -> PurchasedSubscription:
    """Get or create a free subscription for new users"""

    # Find free subscription plan
    result = await db.execute(
        select(Subscription).where(
            Subscription.subscription_plan == SubscriptionPlan.ESSENTIAL
        )
    )
    free_subscription = result.scalar_one_or_none()

    if not free_subscription:
        raise Exception("No free subscription plan found")

    # Create purchased subscription
    purchased_sub = PurchasedSubscription(
        subscription_id=free_subscription.id, subscription_status="active"
    )

    db.add(purchased_sub)
    await db.commit()
    await db.refresh(purchased_sub)

    return purchased_sub


async def add_point_transaction_async(
    db: AsyncSession,
    user_id: str,
    transaction_type: str,
    points: int,
    description: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> Optional[PointTransaction]:
    """
    Add a point transaction for a user and update their point balances (Async version).

    Args:
        db: Async database session
        user_id: ID of the user
        transaction_type: Type of transaction ("earned", "used", "purchased")
        points: Number of points (positive for earned/purchased, positive for used)
        description: Optional description of the transaction
        conversation_id: Optional ID of related document material

    Returns:
        PointTransaction: Created transaction object or None if failed

    Raises:
        ValueError: If invalid parameters are provided
        SQLAlchemyError: If database operation fails
    """

    # Validate transaction type
    valid_types = ["earned", "used", "purchased"]
    if transaction_type not in valid_types:
        raise ValueError(f"Invalid transaction_type. Must be one of: {valid_types}")

    # Validate points
    if not isinstance(points, int) or points <= 0:
        raise ValueError("Points must be a positive integer")

    try:
        # Get the user (async query)
        result = await db.execute(select(User).filter(User.id == user_id))
        user = result.scalar_one_or_none()

        if not user:
            raise ValueError(f"User with ID {user_id} not found")

        # For "used" transactions, check if user has enough points
        if transaction_type == "used" and user.current_points < points:
            raise ValueError(
                f"Insufficient points. User has {user.current_points}, trying to use {points}"
            )

        # Create the transaction
        transaction = PointTransaction(
            id=str(uuid.uuid4()),
            user_id=user_id,
            transaction_type=transaction_type,
            points=points,
            description=description,
            conversation_id=conversation_id,
            created_at=datetime.utcnow(),
        )

        # Update user's point balances
        if transaction_type in ["earned", "purchased"]:
            user.current_points += points
            user.total_points_earned += points
        elif transaction_type == "used":
            user.current_points -= points
            user.total_points_used += points

        # Add transaction to database
        db.add(transaction)
        await db.commit()
        await db.refresh(transaction)

        return transaction

    except SQLAlchemyError as e:
        await db.rollback()
        raise SQLAlchemyError(f"Database error occurred: {str(e)}")
    except Exception as e:
        await db.rollback()
        raise Exception(f"Unexpected error: {str(e)}")


async def add_earned_points_async(
    db: AsyncSession,
    user_id: str,
    points: int,
    description: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> Optional[PointTransaction]:
    """
    Convenience function to add earned points (Async version).
    """
    return await add_point_transaction_async(
        db=db,
        user_id=user_id,
        transaction_type="earned",
        points=points,
        description=description,
        conversation_id=conversation_id,
    )


async def add_used_points_async(
    db: AsyncSession,
    user_id: str,
    points: int,
    description: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> Optional[PointTransaction]:
    """
    Convenience function to deduct used points (Async version).
    """
    return await add_point_transaction_async(
        db=db,
        user_id=user_id,
        transaction_type="used",
        points=points,
        description=description,
        conversation_id=conversation_id,
    )


async def add_purchased_points_async(
    db: AsyncSession, user_id: str, points: int, description: Optional[str] = None
) -> Optional[PointTransaction]:
    """
    Convenience function to add purchased points (Async version).
    """
    return await add_point_transaction_async(
        db=db,
        user_id=user_id,
        transaction_type="purchased",
        points=points,
        description=description,
    )


async def add_purchased_points_async(
    db: AsyncSession, user_id: str, points: int, description: Optional[str] = None
) -> Optional[PointTransaction]:
    """
    Convenience function to add purchased points (Async version).
    """
    return await add_point_transaction_async(
        db=db,
        user_id=user_id,
        transaction_type="purchased",
        points=points,
        description=description,
    )


# Batch operations for better performance
async def add_multiple_transactions_async(
    db: AsyncSession, transactions_data: list[dict]
) -> list[PointTransaction]:
    """
    Add multiple point transactions in a single database transaction.

    Args:
        db: Async database session
        transactions_data: List of transaction dictionaries with keys:
            - user_id, transaction_type, points, description, conversation_id

    Returns:
        List of created PointTransaction objects
    """
    created_transactions = []
    user_updates = {}  # Cache user objects to avoid multiple queries

    try:
        for tx_data in transactions_data:
            user_id = tx_data["user_id"]
            transaction_type = tx_data["transaction_type"]
            points = tx_data["points"]

            # Get user if not already cached
            if user_id not in user_updates:
                result = await db.execute(select(User).filter(User.id == user_id))
                user = result.scalar_one_or_none()
                if not user:
                    raise ValueError(f"User with ID {user_id} not found")
                user_updates[user_id] = user

            user = user_updates[user_id]

            # Validate sufficient points for "used" transactions
            if transaction_type == "used" and user.current_points < points:
                raise ValueError(f"Insufficient points for user {user_id}")

            # Create transaction
            transaction = PointTransaction(
                id=str(uuid.uuid4()),
                user_id=user_id,
                transaction_type=transaction_type,
                points=points,
                description=tx_data.get("description"),
                conversation_id=tx_data.get("conversation_id"),
                created_at=datetime.utcnow(),
            )

            # Update user balances
            if transaction_type in ["earned", "purchased"]:
                user.current_points += points
                user.total_points_earned += points
            elif transaction_type == "used":
                user.current_points -= points
                user.total_points_used += points

            db.add(transaction)
            created_transactions.append(transaction)

        await db.commit()

        # Refresh all transactions
        for transaction in created_transactions:
            await db.refresh(transaction)

        return created_transactions

    except Exception as e:
        await db.rollback()
        raise Exception(f"Batch transaction failed: {str(e)}")


async def award_share_visit_reward(
    db: AsyncSession,
    interaction_share: InteractionShare,
    visitor_user_id: Optional[str] = None,
    visitor_ip: Optional[str] = None,
    visitor_fingerprint: Optional[str] = None,
    reward_amount: int = 5,
) -> bool:
    """
    Award coins to the owner of a shared interaction when someone visits it.
    Only rewards once per unique visitor and never rewards the owner themselves.
    """
    try:
        # Get the original interaction and its owner
        result = await db.execute(
            select(Interaction).where(
                Interaction.id == interaction_share.original_interaction_id
            )
        )
        original_interaction = result.scalar_one_or_none()

        if not original_interaction:
            return False

        # Get the owner user
        user_result = await db.execute(
            select(User).where(User.id == original_interaction.user_id)
        )
        owner_user = user_result.scalar_one_or_none()

        if not owner_user:
            return False

        # Don't reward if the visitor is the owner themselves
        if visitor_user_id and visitor_user_id == owner_user.id:
            print(f"🚫 Owner visited their own shared interaction - no reward given")
            return False

        # Check if this visitor has already been rewarded
        visitor_query = select(InteractionShareVisitor).where(
            InteractionShareVisitor.interaction_share_id == interaction_share.id
        )

        # Check by user ID if available
        if visitor_user_id:
            visitor_query = visitor_query.where(
                InteractionShareVisitor.visitor_user_id == visitor_user_id
            )
        # Check by IP if no user ID
        elif visitor_ip:
            visitor_query = visitor_query.where(
                InteractionShareVisitor.visitor_ip == visitor_ip
            )
        # Check by fingerprint if no user ID or IP
        elif visitor_fingerprint:
            visitor_query = visitor_query.where(
                InteractionShareVisitor.visitor_fingerprint == visitor_fingerprint
            )
        else:
            # No way to identify visitor uniquely
            print(f"⚠️ Cannot identify visitor uniquely - no reward given")
            return False

        existing_visitor_result = await db.execute(visitor_query)
        existing_visitor = existing_visitor_result.scalar_one_or_none()

        if existing_visitor:
            if existing_visitor.reward_given:
                print(f"🎯 Visitor already rewarded - no duplicate reward")
                return False
            else:
                # Update existing visitor record
                existing_visitor.reward_given = True
                existing_visitor.visited_at = datetime.utcnow()
        else:
            # Create new visitor record
            new_visitor = InteractionShareVisitor(
                interaction_share_id=interaction_share.id,
                visitor_user_id=visitor_user_id,
                visitor_ip=visitor_ip,
                visitor_fingerprint=visitor_fingerprint,
                reward_given=True,
                visited_at=datetime.utcnow(),
            )
            db.add(new_visitor)

        # Award coins to the owner
        owner_user.current_points += reward_amount
        owner_user.total_points_earned += reward_amount

        # Create a point transaction record
        point_transaction = PointTransaction(
            user_id=owner_user.id,
            transaction_type="earned",
            points=reward_amount,
            description=f"Reward for shared interaction visit (Share ID: {interaction_share.share_id})",
        )
        db.add(point_transaction)

        # Update visit count and timestamp
        interaction_share.visit_count += 1
        interaction_share.last_visited_at = datetime.utcnow()

        await db.commit()

        print(
            f"🎉 Awarded {reward_amount} coins to user {owner_user.id} for unique share visit"
        )
        return True

    except Exception as e:
        await db.rollback()
        print(f"❌ Failed to award share visit reward: {e}")
        return False
