from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from strawberry.fastapi import GraphQLRouter

from contextlib import asynccontextmanager

from app.core.config import settings

from app.graphql.schema import schema, get_context
from app.api.webhook import webhook_router
from app.api.app_routes import account_router
from app.api.websocket_routes import router as websocket_router
from app.api.sse_routes import router as sse_router

from app.core.database import init_db
from app.workers.scheduler import start_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    start_scheduler()
    yield
    # Shutdown
    pass


app = FastAPI(
    title="StudyGuru Pro API",
    description="FastAPI GraphQL API for StudyGuru Pro",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
if settings.ENVIRONMENT == "development":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        # allow_origins=[settings.CLIENT_ORIGIN],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# GraphQL router (temporarily disabled due to dependency issues)
graphql_app = GraphQLRouter(schema, context_getter=get_context)
app.include_router(graphql_app, prefix="/graphql")

# REST routes
app.include_router(webhook_router, prefix="/webhook")
app.include_router(account_router, prefix="/api/app")

# WebSocket routes
app.include_router(websocket_router, prefix="/ws")

# Server-Sent Events routes
app.include_router(sse_router, prefix="/api/sse")


@app.get("/")
async def root():
    return {"message": "StudyGuru Pro API"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(settings.PORT),
        reload=settings.ENVIRONMENT == "development",
    )
