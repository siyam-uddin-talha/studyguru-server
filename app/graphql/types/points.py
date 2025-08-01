import strawberry
from typing import Optional, List
from datetime import datetime


@strawberry.type
class PointTransactionType:
    id: str
    user_id: str
    transaction_type: str
    points: int
    description: Optional[str] = None
    doc_material_id: Optional[str] = None
    created_at: Optional[datetime] = None


@strawberry.type
class PointsHistoryResponse:
    success: bool
    message: Optional[str] = None
    result: Optional[List[PointTransactionType]] = None
    total: Optional[int] = None


@strawberry.type
class UserPointsInfo:
    current_points: int
    total_points_earned: int
    total_points_used: int


@strawberry.type
class PointsInfoResponse:
    success: bool
    message: Optional[str] = None
    result: Optional[UserPointsInfo] = None