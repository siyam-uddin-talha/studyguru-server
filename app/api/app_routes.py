# server/app/api/app_routes.py
from fastapi import (
    APIRouter,
    Request,
    HTTPException,
    Depends,
    UploadFile,
    File,
    BackgroundTasks,
    Form,
    Header,
    Security,
)
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, Dict, Any, Annotated
import uuid
import os
import boto3
from botocore.exceptions import ClientError

import json
import asyncio

from app.core.config import settings
from app.core.database import get_db
from app.helpers.auth import verify_token
from app.models.user import User
from app.models.interaction import Interaction, Conversation, ConversationRole
from app.models.media import Media
from app.models.subscription import PointTransaction
from app.services.file_service import FileService
from app.services.langchain_service import langchain_service
from pydantic import BaseModel, Field


# Define a Pydantic model for the file info
class FileInfoResponse(BaseModel):
    original_size: int
    compressed_size: int
    compression_ratio: str


# Define a Pydantic model for the upload response
class UploadResponse(BaseModel):
    success: bool
    url: Optional[str] = None
    message: str
    file_info: Optional[FileInfoResponse] = None


# Routers
interaction_router = APIRouter()
account_router = APIRouter()

# Storage folder keys
STORAGE_KEYS = {"user_profile": "portraits", "interaction": "content"}


# Custom dependency to extract and verify user from token
async def get_user_from_token(
    authorization: Annotated[Optional[str], Header()] = None,
    db: AsyncSession = Depends(get_db),
):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is missing")

    try:
        # Split the authorization header
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authorization scheme")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    # Verify the token
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # Get user ID from token
    user_id: str = payload.get("sub")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    # Fetch user from database
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    return user


@account_router.post("/upload-profile", response_model=UploadResponse)
async def upload_profile(
    file: Annotated[UploadFile, File(...)],
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_user_from_token)],
) -> UploadResponse:
    """
    Upload a profile photo for the authenticated user.
    The image will be compressed and stored in S3.
    """
    try:
        # Validate file exists
        if not file:
            return UploadResponse(
                success=False,
                message="File is required.",
            )

        # Read file content
        file_content = await file.read()
        if not file_content:
            return UploadResponse(
                success=False,
                message="Empty file.",
            )

        # Generate a unique file name(fallback)
        unique_id = str(uuid.uuid4())

        # Upload file to S3
        s3_key, original_size, compressed_size = await FileService.compress_and_upload(
            file_content=file_content,
            original_filename=file.filename or unique_id,
            file_type=file.content_type,
            user_id=str(current_user.id),
            folder_name=STORAGE_KEYS["user_profile"],
        )

        # Generate full URL for the uploaded file
        file_url = f"https://{settings.AWS_S3_BUCKET}.s3.amazonaws.com/{s3_key}"

        # If user already has a profile photo, delete the old one
        if current_user.photo_url:
            try:
                # Extract S3 key from the stored URL if necessary
                old_key = current_user.photo_url
                if "s3.amazonaws.com" in old_key:
                    # Extract key from full URL if stored as URL
                    old_key = old_key.split("s3.amazonaws.com/")[1]

                # Delete old file from S3
                s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=settings.AWS_ACCESS_KEY,
                    aws_secret_access_key=settings.SECRET_ACCESS_KEY,
                    region_name=settings.AWS_ORIGIN,
                )
                s3_client.delete_object(Bucket=settings.AWS_S3_BUCKET, Key=old_key)
            except Exception as e:
                # Log error but continue with the update
                print(f"Error deleting old profile photo: {str(e)}")

        # Update user profile with new photo URL
        current_user.photo_url = s3_key
        await db.commit()

        # Prepare file info
        file_info = FileInfoResponse(
            original_size=original_size,
            compressed_size=compressed_size if compressed_size else original_size,
            compression_ratio=(
                f"{(1 - (compressed_size / original_size)) * 100:.2f}%"
                if compressed_size
                else "0%"
            ),
        )

        # Return success response with file URL
        return UploadResponse(
            success=True,
            url=file_url,
            message="Profile photo uploaded successfully",
            file_info=file_info,
        )

    except HTTPException as http_ex:
        return UploadResponse(
            success=False,
            message=http_ex.detail,
        )
    except Exception as e:
        return UploadResponse(
            success=False,
            message=f"An error occurred: {str(e)}",
        )


class InteractionUploadResponse(BaseModel):
    success: bool
    message: str
    interaction_id: Optional[str] = None
    media_id: Optional[str] = None
    file_url: Optional[str] = None


@interaction_router.post("/upload-file", response_model=InteractionUploadResponse)
async def upload_interaction(
    file: Annotated[UploadFile, File(...)],
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_user_from_token)],
    interaction_id: Annotated[Optional[str], Form()] = None,
):
    """
    Upload a document to start an interaction. The file is uploaded to S3, a Media row
    is created, then an Interaction and initial Conversations (USER upload note and AI analysis)
    are created. The analysis is also embedded into the vector DB if configured.
    """
    try:
        if not file:
            raise HTTPException(status_code=400, detail="File is required")

        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        unique_id = str(uuid.uuid4())

        # Upload to S3 via FileService (handles compression if applicable)
        s3_key, original_size, compressed_size = await FileService.compress_and_upload(
            file_content=file_bytes,
            original_filename=file.filename or unique_id,
            file_type=file.content_type or "application/octet-stream",
            user_id=str(current_user.id),
            folder_name=STORAGE_KEYS["interaction"],
        )

        file_url = f"https://{settings.AWS_S3_BUCKET}.s3.amazonaws.com/{s3_key}"

        # Create Media row
        media = Media(
            original_filename=file.filename or unique_id,
            s3_key=s3_key,
            file_type=file.content_type or "application/octet-stream",
            original_size=float(len(file_bytes)),
            compressed_size=float(compressed_size) if compressed_size else None,
            compression_ratio=(
                float(original_size) / float(compressed_size)
                if compressed_size and compressed_size > 0
                else None
            ),
        )
        db.add(media)
        await db.flush()

        # If interaction_id provided, validate and create a conversation entry tying this media
        if interaction_id:
            result = await db.execute(
                select(Interaction).where(
                    Interaction.id == interaction_id,
                    Interaction.user_id == current_user.id,
                )
            )
            interaction = result.scalar_one_or_none()
            if not interaction:
                raise HTTPException(status_code=404, detail="Interaction not found")

            # conv = Conversation(
            #     interaction_id=str(current_user.id),
            #     role=ConversationRole.USER,
            #     content={
            #         "type": "upload",
            #         "_result": {
            #             "note": "User attached a document",
            #             "file_url": file_url,
            #         },
            #     },
            #     status="completed",
            # )
            # db.add(conv)
            # await db.flush()

            # Associate the media with the conversation using the junction table
            # conv.files.append(media)
            # await db.commit()

            return InteractionUploadResponse(
                success=True,
                message="Media uploaded and attached to interaction",
                interaction_id=str(interaction.id),
                media_id=str(media.id),
                file_url=file_url,
            )

        # Otherwise, only return the media details; user will start conversation later
        await db.commit()
        return InteractionUploadResponse(
            success=True,
            message="Media uploaded",
            media_id=str(media.id),
            file_url=file_url,
        )

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
