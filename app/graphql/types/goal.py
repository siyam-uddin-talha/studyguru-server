import strawberry
from typing import Optional, List
from datetime import datetime
from strawberry.scalars import JSON


@strawberry.type
class GoalIconType:
    type: str  # 'emoji', 'svg', or 'vector'
    value: str
    library: Optional[str] = None  # For vector icons


@strawberry.type
class GoalReminderType:
    enabled: bool
    time: Optional[str] = None  # HH:mm format
    message: Optional[str] = None


@strawberry.type
class GoalProgressType:
    id: str
    goal_id: str
    date: str  # YYYY-MM-DD format
    value: int
    completed: bool
    memo: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@strawberry.type
class GoalType:
    id: str
    user_id: str
    name: str
    icon: JSON  # GoalIconType as JSON
    color: str
    type: str  # 'Target', 'Build', 'Quit'
    period: str  # 'daily' or 'weekly'
    target_value: int
    unit: str
    task_days: JSON  # Array of integers [0-6]
    reminder: JSON  # GoalReminderType as JSON
    show_memo: bool
    chart_type: str  # 'line', 'bar', 'pie'
    start_date: str  # YYYY-MM-DD format
    end_date: Optional[str] = None  # YYYY-MM-DD format
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    progress: Optional[List[GoalProgressType]] = None


@strawberry.input
class GoalIconInput:
    type: str  # 'emoji', 'svg', or 'vector'
    value: str
    library: Optional[str] = None


@strawberry.input
class GoalReminderInput:
    enabled: bool
    time: Optional[str] = None  # HH:mm format
    message: Optional[str] = None


@strawberry.input
class CreateGoalInput:
    name: str
    icon: JSON  # GoalIconInput as JSON
    color: str
    type: str  # 'Target', 'Build', 'Quit'
    period: str  # 'daily' or 'weekly'
    target_value: int
    unit: str
    task_days: JSON  # Array of integers [0-6]
    reminder: JSON  # GoalReminderInput as JSON
    show_memo: bool = False
    chart_type: str = "line"  # 'line', 'bar', 'pie'
    start_date: str  # YYYY-MM-DD format
    end_date: Optional[str] = None  # YYYY-MM-DD format


@strawberry.input
class UpdateGoalInput:
    goal_id: str
    name: Optional[str] = None
    icon: Optional[JSON] = None
    color: Optional[str] = None
    type: Optional[str] = None
    period: Optional[str] = None
    target_value: Optional[int] = None
    unit: Optional[str] = None
    task_days: Optional[JSON] = None
    reminder: Optional[JSON] = None
    show_memo: Optional[bool] = None
    chart_type: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@strawberry.input
class DeleteGoalInput:
    goal_id: str


@strawberry.input
class AddGoalProgressInput:
    goal_id: str
    date: str  # YYYY-MM-DD format
    value: int
    completed: bool = False
    memo: Optional[str] = None


@strawberry.input
class UpdateGoalProgressInput:
    progress_id: str
    value: Optional[int] = None
    completed: Optional[bool] = None
    memo: Optional[str] = None


@strawberry.input
class DeleteGoalProgressInput:
    progress_id: str


@strawberry.input
class GetGoalsInput:
    date: Optional[str] = None  # YYYY-MM-DD format, if provided, filter by date
    page: Optional[int] = 1
    size: Optional[int] = 100


@strawberry.type
class GoalResponse:
    success: bool
    message: Optional[str] = None
    goal: Optional[GoalType] = None


@strawberry.type
class GoalListResponse:
    success: bool
    message: Optional[str] = None
    goals: Optional[List[GoalType]] = None
    total: Optional[int] = None
    has_next_page: Optional[bool] = None


@strawberry.type
class GoalProgressResponse:
    success: bool
    message: Optional[str] = None
    progress: Optional[GoalProgressType] = None
