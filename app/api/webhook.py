from fastapi import (
    APIRouter,
    Request,
    HTTPException,
    Depends,
)
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

try:
    import paddle
except ImportError:
    paddle = None
import json
import asyncio

from sqlalchemy import select
from app.core.config import settings
from app.core.database import get_db
from app.helpers.auth import get_current_user
from app.models.user import User
from app.models.interaction import Interaction
from app.models.subscription import PointTransaction
from app.services.openai_service import OpenAIService
from app.services.file_service import FileService

# Initialize Paddle
if paddle:
    paddle.api_key = settings.PADDLE_API_KEY
    paddle.environment = settings.PADDLE_ENVIRONMENT

# Routers
webhook_router = APIRouter()


@webhook_router.post("/paddle")
async def paddle_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    """Handle Paddle webhooks"""
    payload = await request.body()
    signature = request.headers.get("paddle-signature")

    try:
        # Verify webhook signature
        event = paddle.Webhook.verify(
            payload, signature, settings.PADDLE_WEBHOOK_SECRET
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    event_type = event.get("event_type")

    if event_type == "subscription.created":
        # Handle new subscription
        await handle_subscription_created(event.get("data"), db)

    elif event_type == "subscription.updated":
        # Handle subscription update
        await handle_subscription_updated(event.get("data"), db)

    elif event_type == "subscription.cancelled":
        # Handle subscription cancellation
        await handle_subscription_cancelled(event.get("data"), db)

    elif event_type == "transaction.completed":
        # Handle successful payment
        await handle_transaction_completed(event.get("data"), db)

    return JSONResponse(content={"received": True})

    # Temporarily disabled due to dependency issues
    # @doc_material_router.post("/upload")
    # async def upload_document(
    #     background_tasks: BackgroundTasks,
    #     file: UploadFile = File(...),
    #     max_tokens: int = 1000,
    #     current_user: User = Depends(get_current_user),
    #     db: AsyncSession = Depends(get_db),
    # ):
    """Upload and analyze document"""
    pass
    # Temporarily disabled - function body commented out


# Paddle webhook handlers
async def handle_subscription_created(data: dict, db: AsyncSession):
    """Handle new subscription creation"""
    # Implementation for subscription creation
    pass


async def handle_subscription_updated(data: dict, db: AsyncSession):
    """Handle subscription updates"""
    # Implementation for subscription updates
    pass


async def handle_subscription_cancelled(data: dict, db: AsyncSession):
    """Handle subscription cancellation"""
    # Implementation for subscription cancellation
    pass


async def handle_transaction_completed(data: dict, db: AsyncSession):
    """Handle completed transactions (points purchase)"""
    # Implementation for transaction completion
    pass
