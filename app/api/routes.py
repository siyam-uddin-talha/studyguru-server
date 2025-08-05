from fastapi import (
    APIRouter,
    Request,
    HTTPException,
    Depends,
    UploadFile,
    File,
    BackgroundTasks,
)
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

try:
    import paddle
except ImportError:
    paddle = None
import json
import asyncio

from app.core.config import settings
from app.core.database import get_db
from app.helpers.auth import get_current_user
from app.models.user import User
from app.models.doc_material import DocMaterial, Media
from app.models.subscription import PointTransaction
from app.services.openai_service import OpenAIService
from app.services.file_service import FileService

# Initialize Paddle
if paddle:
    paddle.api_key = settings.PADDLE_API_KEY
    paddle.environment = settings.PADDLE_ENVIRONMENT

# Routers
webhook_router = APIRouter()
doc_material_router = APIRouter()


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


async def process_document_analysis(
    doc_material_id: str, file_content: bytes, file_type: str, max_tokens: int
):
    """Background task to analyze document with OpenAI"""
    from app.core.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        try:
            # Get doc material
            result = await db.execute(
                select(DocMaterial).where(DocMaterial.id == doc_material_id)
            )
            doc_material = result.scalar_one()

            # Get user
            user_result = await db.execute(
                select(User).where(User.id == doc_material.user_id)
            )
            user = user_result.scalar_one()

            # Analyze with OpenAI
            analysis_result = await OpenAIService.analyze_document(
                file_content, file_type, max_tokens
            )

            tokens_used = analysis_result.get("token", 0)
            points_cost = OpenAIService.calculate_points_cost(tokens_used)

            # Check if user still has enough points
            if user.current_points < points_cost:
                doc_material.status = "failed"
                doc_material.error_message = "Insufficient points"
                await db.commit()
                return

            # Deduct points from user
            user.current_points -= points_cost
            user.total_points_used += points_cost

            # Create point transaction
            point_transaction = PointTransaction(
                user_id=user.id,
                transaction_type="used",
                points=points_cost,
                description=f"Document analysis - {tokens_used} tokens",
                doc_material_id=doc_material.id,
            )
            db.add(point_transaction)

            # Update doc material
            doc_material.analysis_response = analysis_result
            doc_material.question_type = analysis_result.get("type")
            doc_material.detected_language = analysis_result.get("language")
            doc_material.title = analysis_result.get("title")
            doc_material.summary_title = analysis_result.get("summary_title")
            doc_material.tokens_used = tokens_used
            doc_material.points_cost = points_cost
            doc_material.status = "completed"

            await db.commit()

        except Exception as e:
            # Update status to failed
            doc_material.status = "failed"
            doc_material.error_message = str(e)
            await db.commit()


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
