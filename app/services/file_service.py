import boto3
from PIL import Image
import io
import os
from typing import Tuple, Optional
from app.core.config import settings

s3_client = boto3.client(
    's3',
    aws_access_key_id=settings.AWS_ACCESS_KEY,
    aws_secret_access_key=settings.SECRET_ACCESS_KEY,
    region_name='us-east-1'  # Adjust as needed
)


class FileService:
    @staticmethod
    async def compress_and_upload(
        file_content: bytes, 
        filename: str, 
        file_type: str,
        user_id: str
    ) -> Tuple[str, float, Optional[float]]:
        """
        Compress file and upload to S3
        Returns: (s3_key, original_size, compressed_size)
        """
        original_size = len(file_content)
        compressed_content = file_content
        compressed_size = None
        
        # Compress if it's an image
        if file_type.startswith('image/'):
            compressed_content, compressed_size = await FileService._compress_image(file_content)
        
        # Generate S3 key
        file_extension = os.path.splitext(filename)[1]
        s3_key = f"doc-materials/{user_id}/{filename}_{hash(file_content)}{file_extension}"
        
        # Upload to S3
        s3_client.put_object(
            Bucket=settings.AWS_S3_BUCKET,
            Key=s3_key,
            Body=compressed_content,
            ContentType=file_type
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
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            
            # Compress
            output = io.BytesIO()
            
            # Determine optimal quality based on file size
            quality = 85
            if len(image_content) > 5 * 1024 * 1024:  # > 5MB
                quality = 70
            elif len(image_content) > 2 * 1024 * 1024:  # > 2MB
                quality = 80
            
            image.save(output, format='JPEG', quality=quality, optimize=True)
            compressed_content = output.getvalue()
            
            return compressed_content, len(compressed_content)
            
        except Exception as e:
            # If compression fails, return original
            return image_content, len(image_content)

    @staticmethod
    async def get_file_from_s3(s3_key: str) -> bytes:
        """
        Retrieve file content from S3
        """
        response = s3_client.get_object(Bucket=settings.AWS_S3_BUCKET, Key=s3_key)
        return response['Body'].read()