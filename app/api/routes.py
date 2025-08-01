from fastapi import APIRouter, Request, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
import paddle
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
paddle.api_key = settings.PADDLE_API_KEY
paddle.environment = settings.PADDLE_ENVIRONMENT

# Routers
webhook_router = APIRouter()
doc_material_router = APIRouter()


@webhook_router.post("/paddle")
async def paddle_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    """Handle Paddle webhooks"""
    payload = await request.body()
    signature = request.headers.get('paddle-signature')
    
    try:
        # Verify webhook signature
        event = paddle.Webhook.verify(payload, signature, settings.PADDLE_WEBHOOK_SECRET)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid webhook signature")
    
    event_type = event.get('event_type')
    
    if event_type == 'subscription.created':
        # Handle new subscription
        await handle_subscription_created(event.get('data'), db)
    
    elif event_type == 'subscription.updated':
        # Handle subscription update
        await handle_subscription_updated(event.get('data'), db)
    
    elif event_type == 'subscription.cancelled':
        # Handle subscription cancellation
        await handle_subscription_cancelled(event.get('data'), db)
    
    elif event_type == 'transaction.completed':
        # Handle successful payment
        await handle_transaction_completed(event.get('data'), db)
    
    return JSONResponse(content={"received": True})


@doc_material_router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    max_tokens: int = 1000,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Upload and analyze document"""
    
    # Validate file type
    allowed_types = ['image/jpeg', 'image/png', 'image/gif', 'application/pdf']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file type. Please upload images or PDF files."
        )
    
    # Check file size (max 10MB)
    file_content = await file.read()
    if len(file_content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")
    
    # Check if user is free and limit tokens
    if current_user.purchased_subscription.subscription.subscription_plan.value == "FREE":
        max_tokens = min(max_tokens, 1000)
    
    # Estimate points cost
    estimated_points = OpenAIService.calculate_points_cost(max_tokens)
    
    # Check if user has enough points
    if current_user.current_points < estimated_points:
        raise HTTPException(
            status_code=400, 
            detail="Insufficient points. Please purchase more points or upgrade your plan."
        )
    
    try:
        # Upload and compress file
        s3_key, original_size, compressed_size = await FileService.compress_and_upload(
            file_content, file.filename, file.content_type, current_user.id
        )
        
        # Create media record
        media = Media(
            original_filename=file.filename,
            s3_key=s3_key,
            file_type=file.content_type,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=original_size / compressed_size if compressed_size else None
        )
        db.add(media)
        await db.flush()
        
        # Create doc material record
        doc_material = DocMaterial(
            user_id=current_user.id,
            file_id=media.id,
            status="processing"
        )
        db.add(doc_material)
        await db.commit()
        
        # Process document in background
        background_tasks.add_task(
            process_document_analysis,
            doc_material.id,
            file_content,
            file.content_type,
            max_tokens
        )
        
        return JSONResponse(content={
            "success": True,
            "message": "Document uploaded successfully. Analysis in progress.",
            "doc_material_id": doc_material.id
        })
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


async def process_document_analysis(
    doc_material_id: str,
    file_content: bytes,
    file_type: str,
    max_tokens: int
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
            
            tokens_used = analysis_result.get('token', 0)
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
                doc_material_id=doc_material.id
            )
            db.add(point_transaction)
            
            # Update doc material
            doc_material.analysis_response = analysis_result
            doc_material.question_type = analysis_result.get('type')
            doc_material.detected_language = analysis_result.get('language')
            doc_material.title = analysis_result.get('title')
            doc_material.summary_title = analysis_result.get('summary_title')
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