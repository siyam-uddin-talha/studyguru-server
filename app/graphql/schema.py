import strawberry
from strawberry.fastapi import BaseContext
from fastapi import Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.graphql.resolvers.auth import AuthQuery, AuthMutation
from app.graphql.resolvers.public import PublicQuery
from app.graphql.resolvers.settings import SettingsQuery, SettingsMutation
from app.graphql.resolvers.interaction import InteractionQuery, InteractionMutation
from app.graphql.resolvers.reward import RewardMutation
from app.graphql.resolvers.admin import AdminQuery, AdminMutation
from app.graphql.resolvers.goal import GoalQuery, GoalMutation
from app.graphql.resolvers.note import NoteQuery, NoteMutation

from app.helpers.auth import get_current_user_optional
from app.models.user import User
from strawberry.schema.config import StrawberryConfig


@strawberry.type
class Context(BaseContext):
    request: Request
    db: AsyncSession
    current_user: User | None = None


@strawberry.type
class Query(
    AuthQuery,
    PublicQuery,
    SettingsQuery,
    InteractionQuery,
    AdminQuery,
    GoalQuery,
    NoteQuery,
):
    pass


@strawberry.type
class Mutation(
    AuthMutation,
    SettingsMutation,
    InteractionMutation,
    RewardMutation,
    AdminMutation,
    GoalMutation,
    NoteMutation,
):
    pass


schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    config=StrawberryConfig(auto_camel_case=False),
)


async def get_context(request: Request, db: AsyncSession = Depends(get_db)) -> Context:
    current_user = await get_current_user_optional(request, db)
    return Context(request=request, db=db, current_user=current_user)
