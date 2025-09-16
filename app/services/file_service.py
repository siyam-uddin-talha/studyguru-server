import boto3
from PIL import Image
import io
import os
import uuid
from typing import Tuple, Optional
from app.core.config import settings

s3_client = boto3.client(
    "s3",
    aws_access_key_id=settings.AWS_ACCESS_KEY,
    aws_secret_access_key=settings.SECRET_ACCESS_KEY,
    region_name=settings.AWS_ORIGIN,
)


class FileService:
    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to remove problematic characters and ensure safety
        """
        # Remove or replace characters that might cause issues in S3 keys
        # Keep only alphanumeric, periods, hyphens, and underscores
        import re

        # Remove any non-safe characters, replace with underscores
        safe_filename = re.sub(r"[^\w\-_\.]", "_", filename)

        # Ensure the filename doesn't start with a period or special character
        safe_filename = safe_filename.lstrip(".-_")

        # Truncate filename if it's too long
        max_filename_length = 255
        if len(safe_filename) > max_filename_length:
            # Keep the file extension
            name, ext = os.path.splitext(safe_filename)
            safe_filename = f"{name[:max_filename_length - len(ext)]}{ext}"

        # Ensure we have a filename
        if not safe_filename:
            safe_filename = "unnamed_file"

        return safe_filename

    @staticmethod
    def _get_file_extension(filename: str, file_type: str) -> str:
        """
        Determine the most appropriate file extension
        """
        # First, try to get extension from filename
        ext = os.path.splitext(filename)[1]

        # If no extension in filename, use mime type
        if not ext:
            mime_to_ext = {
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "image/gif": ".gif",
                "image/webp": ".webp",
                "application/pdf": ".pdf",
                "text/plain": ".txt",
                "application/msword": ".doc",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            }
            ext = mime_to_ext.get(file_type, "")

        # Ensure extension starts with a dot
        return ext if ext.startswith(".") else f".{ext}" if ext else ""

    @staticmethod
    async def compress_and_upload(
        file_content: bytes,
        original_filename: str,
        file_type: str,
        user_id: str,
        folder_name: str,
    ) -> Tuple[str, float, Optional[float]]:
        """
        Compress file (if image) and upload to S3
        Returns: (s3_key, original_size, compressed_size)
        """
        original_size = len(file_content)
        compressed_content = file_content
        compressed_size = None

        # Sanitize filename
        safe_filename = FileService._sanitize_filename(original_filename)

        # Determine file extension
        file_extension = FileService._get_file_extension(safe_filename, file_type)

        # Compress if it's an image
        if file_type.startswith("image/"):
            try:
                compressed_content, compressed_size = await FileService._compress_image(
                    file_content
                )
            except Exception:
                # If compression fails, use original content
                pass

        # Generate unique S3 key
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        s3_key = f"{folder_name}/{user_id}/{unique_filename}"

        # Upload to S3
        s3_client.put_object(
            Bucket=settings.AWS_S3_BUCKET,
            Key=s3_key,
            Body=compressed_content,
            ContentType=file_type,
            Metadata={
                "original_filename": safe_filename  # Store original filename in metadata
            },
        )

        return s3_key, original_size, compressed_size

    @staticmethod
    async def _compress_image(image_content: bytes) -> Tuple[bytes, float]:
        """
        Compress image while maintaining quality
        """
        try:
            # Open image
            image = Image.open(io.BytesIO(image_content))

            # Convert to RGB if necessary
            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGB")

            # Compress
            output = io.BytesIO()

            # Determine optimal quality based on file size
            quality = 85
            if len(image_content) > 5 * 1024 * 1024:  # > 5MB
                quality = 70
            elif len(image_content) > 2 * 1024 * 1024:  # > 2MB
                quality = 80

            image.save(output, format="JPEG", quality=quality, optimize=True)
            compressed_content = output.getvalue()

            return compressed_content, len(compressed_content)

        except Exception:
            # If compression fails, return original
            return image_content, len(image_content)

    @staticmethod
    async def get_file_from_s3(s3_key: str) -> bytes:
        """
        Retrieve file content from S3
        """
        response = s3_client.get_object(Bucket=settings.AWS_S3_BUCKET, Key=s3_key)
        return response["Body"].read()

    @staticmethod
    async def delete_file_from_s3(s3_key: str) -> bool:
        """
        Delete file from S3
        Returns True if successful, False otherwise
        """
        try:
            s3_client.delete_object(Bucket=settings.AWS_S3_BUCKET, Key=s3_key)
            return True
        except Exception as e:
            print(f"Error deleting file from S3: {e}")
            return False
