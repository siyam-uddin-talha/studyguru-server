from fastapi import (
    APIRouter,
    Request,
    HTTPException,
    Depends,
)
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.user import User
from app.models.interaction import Interaction
from app.models.subscription import PointTransaction
from app.services.openai_service import OpenAIService


async def process_document_analysis(
    doc_material_id: str, file_content: bytes, file_type: str, max_tokens: int
):
    """Background task to analyze document with OpenAI"""
    from app.core.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        try:
            # Get doc material
            result = await db.execute(
                select(Interaction).where(Interaction.id == doc_material_id)
            )
            interaction = result.scalar_one()

            # Get user
            user_result = await db.execute(
                select(User).where(User.id == interaction.user_id)
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
                interaction.status = "failed"
                interaction.error_message = "Insufficient points"
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
                doc_material_id=interaction.id,
            )
            db.add(point_transaction)

            # Update doc material
            interaction.analysis_response = analysis_result
            interaction.question_type = analysis_result.get("type")
            interaction.detected_language = analysis_result.get("language")
            interaction.title = analysis_result.get("title")
            interaction.summary_title = analysis_result.get("summary_title")
            interaction.tokens_used = tokens_used
            interaction.points_cost = points_cost
            interaction.status = "completed"

            await db.commit()

        except Exception as e:
            # Update status to failed
            interaction.status = "failed"
            interaction.error_message = str(e)
            await db.commit()
