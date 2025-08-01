import strawberry
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from fastapi import UploadFile, HTTPException

from app.graphql.types.doc_material import (
    DocMaterialResponse, 
    DocMaterialListResponse, 
    DocMaterialType,
    MediaType
)
from app.models.doc_material import DocMaterial, Media
from app.models.user import User
from app.models.subscription import PointTransaction
from app.services.openai_service import OpenAIService
from app.services.file_service import FileService
from app.helpers.user import get_current_user_from_context


@strawberry.type
class DocMaterialQuery:
    @strawberry.field
    async def doc_materials(
        self, 
        info,
        page: Optional[int] = 1,
        size: Optional[int] = 10
    ) -> DocMaterialListResponse:
        context = info.context
        current_user = await get_current_user_from_context(context)
        
        if not current_user:
            return DocMaterialListResponse(
                success=False,
                message="Authentication required"
            )
        
        db: AsyncSession = context.db
        
        # Calculate pagination
        offset = (page - 1) * size
        
        # Get user's doc materials
        result = await db.execute(
            select(DocMaterial)
            .where(DocMaterial.user_id == current_user.id)
            .order_by(desc(DocMaterial.created_at))
            .offset(offset)
            .limit(size)
        )
        doc_materials = result.scalars().all()
        
        # Get total count
        count_result = await db.execute(
            select(DocMaterial)
            .where(DocMaterial.user_id == current_user.id)
        )
        total = len(count_result.scalars().all())
        
        # Convert to response types
        doc_material_types = []
        for doc in doc_materials:
            # Get associated media
            media_result = await db.execute(
                select(Media).where(Media.id == doc.file_id)
            )
            media = media_result.scalar_one_or_none()
            
            doc_material_types.append(
                DocMaterialType(
                    id=doc.id,
                    user_id=doc.user_id,
                    file_id=doc.file_id,
                    analysis_response=doc.analysis_response,
                    question_type=doc.question_type,
                    detected_language=doc.detected_language,
                    title=doc.title,
                    summary_title=doc.summary_title,
                    tokens_used=doc.tokens_used,
                    points_cost=doc.points_cost,
                    status=doc.status,
                    error_message=doc.error_message,
                    created_at=doc.created_at,
                    updated_at=doc.updated_at,
                    file=MediaType(
                        id=media.id,
                        original_filename=media.original_filename,
                        s3_key=media.s3_key,
                        file_type=media.file_type,
                        original_size=media.original_size,
                        compressed_size=media.compressed_size,
                        compression_ratio=media.compression_ratio,
                        created_at=media.created_at
                    ) if media else None
                )
            )
        
        return DocMaterialListResponse(
            success=True,
            message="Doc materials retrieved successfully",
            result=doc_material_types,
            total=total,
            has_next_page=(offset + size) < total
        )

    @strawberry.field
    async def doc_material(self, info, id: str) -> DocMaterialResponse:
        context = info.context
        current_user = await get_current_user_from_context(context)
        
        if not current_user:
            return DocMaterialResponse(
                success=False,
                message="Authentication required"
            )
        
        db: AsyncSession = context.db
        
        # Get doc material
        result = await db.execute(
            select(DocMaterial)
            .where(
                DocMaterial.id == id,
                DocMaterial.user_id == current_user.id
            )
        )
        doc_material = result.scalar_one_or_none()
        
        if not doc_material:
            return DocMaterialResponse(
                success=False,
                message="Document not found"
            )
        
        # Get associated media
        media_result = await db.execute(
            select(Media).where(Media.id == doc_material.file_id)
        )
        media = media_result.scalar_one_or_none()
        
        return DocMaterialResponse(
            success=True,
            message="Document retrieved successfully",
            result=DocMaterialType(
                id=doc_material.id,
                user_id=doc_material.user_id,
                file_id=doc_material.file_id,
                analysis_response=doc_material.analysis_response,
                question_type=doc_material.question_type,
                detected_language=doc_material.detected_language,
                title=doc_material.title,
                summary_title=doc_material.summary_title,
                tokens_used=doc_material.tokens_used,
                points_cost=doc_material.points_cost,
                status=doc_material.status,
                error_message=doc_material.error_message,
                created_at=doc_material.created_at,
                updated_at=doc_material.updated_at,
                file=MediaType(
                    id=media.id,
                    original_filename=media.original_filename,
                    s3_key=media.s3_key,
                    file_type=media.file_type,
                    original_size=media.original_size,
                    compressed_size=media.compressed_size,
                    compression_ratio=media.compression_ratio,
                    created_at=media.created_at
                ) if media else None
            )
        )


@strawberry.type
class DocMaterialMutation:
    @strawberry.field
    async def upload_document(
        self, 
        info,
        max_tokens: Optional[int] = 1000
    ) -> DocMaterialResponse:
        """
        Note: This is a simplified version. In practice, file upload would be handled
        via REST endpoint and then processed asynchronously.
        """
        context = info.context
        current_user = await get_current_user_from_context(context)
        
        if not current_user:
            return DocMaterialResponse(
                success=False,
                message="Authentication required"
            )
        
        # Check if user is free and limit tokens
        if current_user.purchased_subscription.subscription.subscription_plan.value == "FREE":
            max_tokens = min(max_tokens, 1000)
        
        # Check if user has enough points
        estimated_points = max(1, max_tokens // 100)
        if current_user.current_points < estimated_points:
            return DocMaterialResponse(
                success=False,
                message="Insufficient points. Please purchase more points or upgrade your plan."
            )
        
        return DocMaterialResponse(
            success=True,
            message="Document upload endpoint should be implemented via REST API"
        )

    @strawberry.field
    async def delete_document(self, info, id: str) -> DocMaterialResponse:
        context = info.context
        current_user = await get_current_user_from_context(context)
        
        if not current_user:
            return DocMaterialResponse(
                success=False,
                message="Authentication required"
            )
        
        db: AsyncSession = context.db
        
        # Get doc material
        result = await db.execute(
            select(DocMaterial)
            .where(
                DocMaterial.id == id,
                DocMaterial.user_id == current_user.id
            )
        )
        doc_material = result.scalar_one_or_none()
        
        if not doc_material:
            return DocMaterialResponse(
                success=False,
                message="Document not found"
            )
        
        # Delete the document
        await db.delete(doc_material)
        await db.commit()
        
        return DocMaterialResponse(
            success=True,
            message="Document deleted successfully"
        )