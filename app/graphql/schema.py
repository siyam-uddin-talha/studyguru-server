import strawberry
from strawberry.fastapi import BaseContext
from fastapi import Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.graphql.resolvers.auth import AuthQuery, AuthMutation
from app.graphql.resolvers.public import PublicQuery
from app.graphql.resolvers.payment import PaymentQuery, PaymentMutation
from app.graphql.resolvers.subscription import SubscriptionQuery, SubscriptionMutation
from app.graphql.resolvers.admin import AdminQuery, AdminMutation
from app.graphql.resolvers.swipe import SwipeQuery, SwipeMutation
from app.graphql.resolvers.settings import SettingsQuery, SettingsMutation
from app.graphql.resolvers.dashboard import DashboardQuery
from app.helpers.auth import get_current_user_optional
from app.models.user import User


@strawberry.type
class Context(BaseContext):
    request: Request
    db: AsyncSession
    current_user: User | None = None


@strawberry.type
class Query(
    AuthQuery,
    PublicQuery, 
    PaymentQuery,
    SubscriptionQuery,
    AdminQuery,
    SwipeQuery,
    SettingsQuery,
    DashboardQuery
):
    pass


@strawberry.type  
class Mutation(
    AuthMutation,
    PaymentMutation,
    SubscriptionMutation,
    AdminMutation,
    SwipeMutation,
    SettingsMutation
):
    pass


schema = strawberry.Schema(query=Query, mutation=Mutation)


async def get_context(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> Context:
    current_user = await get_current_user_optional(request, db)
    return Context(
        request=request,
        db=db,
        current_user=current_user
    )