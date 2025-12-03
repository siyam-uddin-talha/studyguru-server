import strawberry
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func, and_, or_
from sqlalchemy.orm import selectinload
from datetime import datetime
import json

from app.graphql.types.goal import (
    GoalType,
    GoalProgressType,
    GoalResponse,
    GoalListResponse,
    GoalProgressResponse,
    CreateGoalInput,
    UpdateGoalInput,
    DeleteGoalInput,
    AddGoalProgressInput,
    UpdateGoalProgressInput,
    DeleteGoalProgressInput,
    GetGoalsInput,
)
from app.models.goal import (
    Goal,
    GoalProgress,
    GoalType as GoalTypeEnum,
    GoalPeriod,
    ChartType,
)
from app.helpers.user import get_current_user_from_context
from app.graphql.types.common import DefaultResponse


@strawberry.type
class GoalQuery:
    @strawberry.field
    async def goals(
        self,
        info,
        input: Optional[GetGoalsInput] = None,
    ) -> GoalListResponse:
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return GoalListResponse(success=False, message="Authentication required")

        db: AsyncSession = context.db

        try:
            # Build base query
            base_query = (
                select(Goal)
                .options(selectinload(Goal.progress))
                .where(Goal.user_id == current_user.id)
            )

            # If date is provided, filter goals that are active on that date
            if input and input.date:
                date_obj = datetime.strptime(input.date, "%Y-%m-%d").date()
                day_of_week = date_obj.weekday()  # 0 = Monday, 6 = Sunday
                # Convert to Sunday-based (0 = Sunday, 6 = Saturday)
                day_of_week = (day_of_week + 1) % 7

                base_query = base_query.where(
                    and_(
                        Goal.start_date <= date_obj,
                        or_(Goal.end_date.is_(None), Goal.end_date >= date_obj),
                        # Check if day_of_week is in task_days JSON array
                        func.json_contains(
                            func.cast(Goal.task_days, "JSON"), str(day_of_week)
                        ),
                    )
                )

            # Get pagination params
            page = input.page if input and input.page else 1
            size = input.size if input and input.size else 100
            offset = (page - 1) * size

            # Execute query with ordering and pagination
            result = await db.execute(
                base_query.order_by(desc(Goal.created_at)).offset(offset).limit(size)
            )
            goals = result.scalars().all()

            # Get total count
            count_query = (
                select(func.count())
                .select_from(Goal)
                .where(Goal.user_id == current_user.id)
            )
            if input and input.date:
                date_obj = datetime.strptime(input.date, "%Y-%m-%d").date()
                day_of_week = (date_obj.weekday() + 1) % 7
                count_query = count_query.where(
                    and_(
                        Goal.start_date <= date_obj,
                        or_(Goal.end_date.is_(None), Goal.end_date >= date_obj),
                        func.json_contains(
                            func.cast(Goal.task_days, "JSON"), str(day_of_week)
                        ),
                    )
                )

            total_result = await db.scalar(count_query)
            total = total_result or 0

            # Convert to response types
            goal_types = []
            for goal in goals:
                # Convert progress
                progress_list = []
                if goal.progress:
                    for prog in goal.progress:
                        progress_list.append(
                            GoalProgressType(
                                id=prog.id,
                                goal_id=prog.goal_id,
                                date=prog.date,
                                value=prog.value,
                                completed=prog.completed,
                                memo=prog.memo,
                                created_at=prog.created_at,
                                updated_at=prog.updated_at,
                            )
                        )

                goal_types.append(
                    GoalType(
                        id=goal.id,
                        user_id=goal.user_id,
                        name=goal.name,
                        icon=(
                            goal.icon
                            if isinstance(goal.icon, dict)
                            else json.loads(goal.icon) if goal.icon else {}
                        ),
                        color=goal.color,
                        type=(
                            goal.type.value
                            if isinstance(goal.type, GoalTypeEnum)
                            else goal.type
                        ),
                        period=(
                            goal.period.value
                            if isinstance(goal.period, GoalPeriod)
                            else goal.period
                        ),
                        target_value=goal.target_value,
                        unit=goal.unit,
                        task_days=(
                            goal.task_days
                            if isinstance(goal.task_days, list)
                            else json.loads(goal.task_days) if goal.task_days else []
                        ),
                        reminder={
                            "enabled": goal.reminder_enabled,
                            "time": goal.reminder_time,
                            "message": goal.reminder_message,
                        },
                        show_memo=goal.show_memo,
                        chart_type=(
                            goal.chart_type.value
                            if isinstance(goal.chart_type, ChartType)
                            else goal.chart_type
                        ),
                        start_date=(
                            goal.start_date.strftime("%Y-%m-%d")
                            if isinstance(goal.start_date, datetime)
                            else goal.start_date
                        ),
                        end_date=(
                            goal.end_date.strftime("%Y-%m-%d")
                            if goal.end_date and isinstance(goal.end_date, datetime)
                            else (goal.end_date if goal.end_date else None)
                        ),
                        created_at=goal.created_at,
                        updated_at=goal.updated_at,
                        progress=progress_list,
                    )
                )

            has_next_page = total > offset + len(goals)

            return GoalListResponse(
                success=True,
                message="Goals retrieved successfully",
                goals=goal_types,
                total=total,
                has_next_page=has_next_page,
            )
        except Exception as e:
            return GoalListResponse(
                success=False, message=f"Error retrieving goals: {str(e)}"
            )

    @strawberry.field
    async def goal(self, info, goal_id: str) -> GoalResponse:
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return GoalResponse(success=False, message="Authentication required")

        db: AsyncSession = context.db

        try:
            result = await db.execute(
                select(Goal)
                .options(selectinload(Goal.progress))
                .where(Goal.id == goal_id, Goal.user_id == current_user.id)
            )
            goal = result.scalar_one_or_none()

            if not goal:
                return GoalResponse(success=False, message="Goal not found")

            # Convert progress
            progress_list = []
            if goal.progress:
                for prog in goal.progress:
                    progress_list.append(
                        GoalProgressType(
                            id=prog.id,
                            goal_id=prog.goal_id,
                            date=prog.date,
                            value=prog.value,
                            completed=prog.completed,
                            memo=prog.memo,
                            created_at=prog.created_at,
                            updated_at=prog.updated_at,
                        )
                    )

            goal_type = GoalType(
                id=goal.id,
                user_id=goal.user_id,
                name=goal.name,
                icon=(
                    goal.icon
                    if isinstance(goal.icon, dict)
                    else json.loads(goal.icon) if goal.icon else {}
                ),
                color=goal.color,
                type=(
                    goal.type.value
                    if isinstance(goal.type, GoalTypeEnum)
                    else goal.type
                ),
                period=(
                    goal.period.value
                    if isinstance(goal.period, GoalPeriod)
                    else goal.period
                ),
                target_value=goal.target_value,
                unit=goal.unit,
                task_days=(
                    goal.task_days
                    if isinstance(goal.task_days, list)
                    else json.loads(goal.task_days) if goal.task_days else []
                ),
                reminder={
                    "enabled": goal.reminder_enabled,
                    "time": goal.reminder_time,
                    "message": goal.reminder_message,
                },
                show_memo=goal.show_memo,
                chart_type=(
                    goal.chart_type.value
                    if isinstance(goal.chart_type, ChartType)
                    else goal.chart_type
                ),
                start_date=(
                    goal.start_date.strftime("%Y-%m-%d")
                    if isinstance(goal.start_date, datetime)
                    else goal.start_date
                ),
                end_date=(
                    goal.end_date.strftime("%Y-%m-%d")
                    if goal.end_date and isinstance(goal.end_date, datetime)
                    else (goal.end_date if goal.end_date else None)
                ),
                created_at=goal.created_at,
                updated_at=goal.updated_at,
                progress=progress_list,
            )

            return GoalResponse(
                success=True,
                message="Goal retrieved successfully",
                goal=goal_type,
            )
        except Exception as e:
            return GoalResponse(
                success=False, message=f"Error retrieving goal: {str(e)}"
            )


@strawberry.type
class GoalMutation:
    @strawberry.mutation
    async def create_goal(
        self,
        info,
        input: CreateGoalInput,
    ) -> GoalResponse:
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return GoalResponse(success=False, message="Authentication required")

        db: AsyncSession = context.db

        try:
            # Parse reminder
            reminder_data = (
                input.reminder
                if isinstance(input.reminder, dict)
                else json.loads(input.reminder) if input.reminder else {}
            )

            # Create goal
            goal = Goal(
                user_id=current_user.id,
                name=input.name,
                icon=(
                    input.icon
                    if isinstance(input.icon, dict)
                    else json.loads(input.icon) if input.icon else {}
                ),
                color=input.color,
                type=(
                    GoalTypeEnum[input.type.upper()]
                    if hasattr(GoalTypeEnum, input.type.upper())
                    else GoalTypeEnum.TARGET
                ),
                period=(
                    GoalPeriod[input.period.upper()]
                    if hasattr(GoalPeriod, input.period.upper())
                    else GoalPeriod.DAILY
                ),
                target_value=input.target_value,
                unit=input.unit,
                task_days=(
                    input.task_days
                    if isinstance(input.task_days, list)
                    else json.loads(input.task_days) if input.task_days else []
                ),
                reminder_enabled=reminder_data.get("enabled", False),
                reminder_time=reminder_data.get("time"),
                reminder_message=reminder_data.get("message"),
                show_memo=input.show_memo,
                chart_type=(
                    ChartType[input.chart_type.upper()]
                    if hasattr(ChartType, input.chart_type.upper())
                    else ChartType.LINE
                ),
                start_date=datetime.strptime(input.start_date, "%Y-%m-%d").date(),
                end_date=(
                    datetime.strptime(input.end_date, "%Y-%m-%d").date()
                    if input.end_date
                    else None
                ),
            )

            db.add(goal)
            await db.commit()
            await db.refresh(goal)

            # Convert to response type
            goal_type = GoalType(
                id=goal.id,
                user_id=goal.user_id,
                name=goal.name,
                icon=(
                    goal.icon
                    if isinstance(goal.icon, dict)
                    else json.loads(goal.icon) if goal.icon else {}
                ),
                color=goal.color,
                type=goal.type.value,
                period=goal.period.value,
                target_value=goal.target_value,
                unit=goal.unit,
                task_days=(
                    goal.task_days
                    if isinstance(goal.task_days, list)
                    else json.loads(goal.task_days) if goal.task_days else []
                ),
                reminder={
                    "enabled": goal.reminder_enabled,
                    "time": goal.reminder_time,
                    "message": goal.reminder_message,
                },
                show_memo=goal.show_memo,
                chart_type=goal.chart_type.value,
                start_date=goal.start_date.strftime("%Y-%m-%d"),
                end_date=goal.end_date.strftime("%Y-%m-%d") if goal.end_date else None,
                created_at=goal.created_at,
                updated_at=goal.updated_at,
                progress=[],
            )

            return GoalResponse(
                success=True,
                message="Goal created successfully",
                goal=goal_type,
            )
        except Exception as e:
            await db.rollback()
            return GoalResponse(success=False, message=f"Error creating goal: {str(e)}")

    @strawberry.mutation
    async def update_goal(
        self,
        info,
        input: UpdateGoalInput,
    ) -> GoalResponse:
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return GoalResponse(success=False, message="Authentication required")

        db: AsyncSession = context.db

        try:
            result = await db.execute(
                select(Goal).where(
                    Goal.id == input.goal_id, Goal.user_id == current_user.id
                )
            )
            goal = result.scalar_one_or_none()

            if not goal:
                return GoalResponse(success=False, message="Goal not found")

            # Update fields
            if input.name is not None:
                goal.name = input.name
            if input.icon is not None:
                goal.icon = (
                    input.icon
                    if isinstance(input.icon, dict)
                    else json.loads(input.icon) if input.icon else {}
                )
            if input.color is not None:
                goal.color = input.color
            if input.type is not None:
                goal.type = (
                    GoalTypeEnum[input.type.upper()]
                    if hasattr(GoalTypeEnum, input.type.upper())
                    else goal.type
                )
            if input.period is not None:
                goal.period = (
                    GoalPeriod[input.period.upper()]
                    if hasattr(GoalPeriod, input.period.upper())
                    else goal.period
                )
            if input.target_value is not None:
                goal.target_value = input.target_value
            if input.unit is not None:
                goal.unit = input.unit
            if input.task_days is not None:
                goal.task_days = (
                    input.task_days
                    if isinstance(input.task_days, list)
                    else json.loads(input.task_days) if input.task_days else []
                )
            if input.reminder is not None:
                reminder_data = (
                    input.reminder
                    if isinstance(input.reminder, dict)
                    else json.loads(input.reminder) if input.reminder else {}
                )
                goal.reminder_enabled = reminder_data.get(
                    "enabled", goal.reminder_enabled
                )
                goal.reminder_time = reminder_data.get("time", goal.reminder_time)
                goal.reminder_message = reminder_data.get(
                    "message", goal.reminder_message
                )
            if input.show_memo is not None:
                goal.show_memo = input.show_memo
            if input.chart_type is not None:
                goal.chart_type = (
                    ChartType[input.chart_type.upper()]
                    if hasattr(ChartType, input.chart_type.upper())
                    else goal.chart_type
                )
            if input.start_date is not None:
                goal.start_date = datetime.strptime(input.start_date, "%Y-%m-%d").date()
            if input.end_date is not None:
                goal.end_date = (
                    datetime.strptime(input.end_date, "%Y-%m-%d").date()
                    if input.end_date
                    else None
                )

            await db.commit()
            await db.refresh(goal)

            # Load progress
            await db.refresh(goal, ["progress"])

            # Convert progress
            progress_list = []
            if goal.progress:
                for prog in goal.progress:
                    progress_list.append(
                        GoalProgressType(
                            id=prog.id,
                            goal_id=prog.goal_id,
                            date=prog.date,
                            value=prog.value,
                            completed=prog.completed,
                            memo=prog.memo,
                            created_at=prog.created_at,
                            updated_at=prog.updated_at,
                        )
                    )

            goal_type = GoalType(
                id=goal.id,
                user_id=goal.user_id,
                name=goal.name,
                icon=(
                    goal.icon
                    if isinstance(goal.icon, dict)
                    else json.loads(goal.icon) if goal.icon else {}
                ),
                color=goal.color,
                type=goal.type.value,
                period=goal.period.value,
                target_value=goal.target_value,
                unit=goal.unit,
                task_days=(
                    goal.task_days
                    if isinstance(goal.task_days, list)
                    else json.loads(goal.task_days) if goal.task_days else []
                ),
                reminder={
                    "enabled": goal.reminder_enabled,
                    "time": goal.reminder_time,
                    "message": goal.reminder_message,
                },
                show_memo=goal.show_memo,
                chart_type=goal.chart_type.value,
                start_date=goal.start_date.strftime("%Y-%m-%d"),
                end_date=goal.end_date.strftime("%Y-%m-%d") if goal.end_date else None,
                created_at=goal.created_at,
                updated_at=goal.updated_at,
                progress=progress_list,
            )

            return GoalResponse(
                success=True,
                message="Goal updated successfully",
                goal=goal_type,
            )
        except Exception as e:
            await db.rollback()
            return GoalResponse(success=False, message=f"Error updating goal: {str(e)}")

    @strawberry.mutation
    async def delete_goal(
        self,
        info,
        input: DeleteGoalInput,
    ) -> DefaultResponse:
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return DefaultResponse(success=False, message="Authentication required")

        db: AsyncSession = context.db

        try:
            result = await db.execute(
                select(Goal).where(
                    Goal.id == input.goal_id, Goal.user_id == current_user.id
                )
            )
            goal = result.scalar_one_or_none()

            if not goal:
                return DefaultResponse(success=False, message="Goal not found")

            await db.delete(goal)
            await db.commit()

            return DefaultResponse(success=True, message="Goal deleted successfully")
        except Exception as e:
            await db.rollback()
            return DefaultResponse(
                success=False, message=f"Error deleting goal: {str(e)}"
            )

    @strawberry.mutation
    async def add_goal_progress(
        self,
        info,
        input: AddGoalProgressInput,
    ) -> GoalProgressResponse:
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return GoalProgressResponse(
                success=False, message="Authentication required"
            )

        db: AsyncSession = context.db

        try:
            # Verify goal belongs to user
            goal_result = await db.execute(
                select(Goal).where(
                    Goal.id == input.goal_id, Goal.user_id == current_user.id
                )
            )
            goal = goal_result.scalar_one_or_none()

            if not goal:
                return GoalProgressResponse(success=False, message="Goal not found")

            # Check if progress already exists for this date
            existing_result = await db.execute(
                select(GoalProgress).where(
                    GoalProgress.goal_id == input.goal_id,
                    GoalProgress.date == input.date,
                )
            )
            existing = existing_result.scalar_one_or_none()

            if existing:
                # Update existing progress
                existing.value = input.value
                existing.completed = input.completed
                if input.memo is not None:
                    existing.memo = input.memo
                progress = existing
            else:
                # Create new progress
                progress = GoalProgress(
                    goal_id=input.goal_id,
                    date=input.date,
                    value=input.value,
                    completed=input.completed,
                    memo=input.memo,
                )
                db.add(progress)

            await db.commit()
            await db.refresh(progress)

            progress_type = GoalProgressType(
                id=progress.id,
                goal_id=progress.goal_id,
                date=progress.date,
                value=progress.value,
                completed=progress.completed,
                memo=progress.memo,
                created_at=progress.created_at,
                updated_at=progress.updated_at,
            )

            return GoalProgressResponse(
                success=True,
                message="Progress added successfully",
                progress=progress_type,
            )
        except Exception as e:
            await db.rollback()
            return GoalProgressResponse(
                success=False, message=f"Error adding progress: {str(e)}"
            )

    @strawberry.mutation
    async def update_goal_progress(
        self,
        info,
        input: UpdateGoalProgressInput,
    ) -> GoalProgressResponse:
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return GoalProgressResponse(
                success=False, message="Authentication required"
            )

        db: AsyncSession = context.db

        try:
            result = await db.execute(
                select(GoalProgress)
                .join(Goal)
                .where(
                    GoalProgress.id == input.progress_id,
                    Goal.user_id == current_user.id,
                )
            )
            progress = result.scalar_one_or_none()

            if not progress:
                return GoalProgressResponse(success=False, message="Progress not found")

            if input.value is not None:
                progress.value = input.value
            if input.completed is not None:
                progress.completed = input.completed
            if input.memo is not None:
                progress.memo = input.memo

            await db.commit()
            await db.refresh(progress)

            progress_type = GoalProgressType(
                id=progress.id,
                goal_id=progress.goal_id,
                date=progress.date,
                value=progress.value,
                completed=progress.completed,
                memo=progress.memo,
                created_at=progress.created_at,
                updated_at=progress.updated_at,
            )

            return GoalProgressResponse(
                success=True,
                message="Progress updated successfully",
                progress=progress_type,
            )
        except Exception as e:
            await db.rollback()
            return GoalProgressResponse(
                success=False, message=f"Error updating progress: {str(e)}"
            )

    @strawberry.mutation
    async def delete_goal_progress(
        self,
        info,
        input: DeleteGoalProgressInput,
    ) -> DefaultResponse:
        context = info.context
        current_user = await get_current_user_from_context(context)

        if not current_user:
            return DefaultResponse(success=False, message="Authentication required")

        db: AsyncSession = context.db

        try:
            result = await db.execute(
                select(GoalProgress)
                .join(Goal)
                .where(
                    GoalProgress.id == input.progress_id,
                    Goal.user_id == current_user.id,
                )
            )
            progress = result.scalar_one_or_none()

            if not progress:
                return DefaultResponse(success=False, message="Progress not found")

            await db.delete(progress)
            await db.commit()

            return DefaultResponse(
                success=True, message="Progress deleted successfully"
            )
        except Exception as e:
            await db.rollback()
            return DefaultResponse(
                success=False, message=f"Error deleting progress: {str(e)}"
            )
