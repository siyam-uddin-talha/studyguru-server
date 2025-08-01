from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter
import strawberry
from contextlib import asynccontextmanager

from app.core.config import settings
from app.graphql.schema import schema
from app.api.routes import webhook_router, admin_router, settings_router
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
    title="Inner States Therapy API",
    description="FastAPI GraphQL API for Inner States Therapy",
    version="1.0.0",
    lifespan=lifespan
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
        allow_origins=[settings.CLIENT_ORIGIN],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# GraphQL router
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")

# REST routes
app.include_router(webhook_router, prefix="/webhook")
app.include_router(admin_router, prefix="/admin")
app.include_router(settings_router, prefix="/settings")


@app.get("/")
async def root():
    return {"message": "Inner States Therapy API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(settings.PORT),
        reload=settings.ENVIRONMENT == "development"
    )