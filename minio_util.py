from concurrent import futures
from io import BytesIO
import json
import os
import re
import subprocess
import uuid
import asyncio

import httpx
from minio import Minio
from minio.error import S3Error

# S3
MINIO_ADDRESS = os.environ.get('MINIO_ADDRESS', '')
BUCKET_NAME = os.environ.get('BUCKET_NAME', 'cache')
MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY', None)
MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY', None)

def generate_presigned_download_url(bucket_name, file_name, expires_hours=24):
    """
    Generate a presigned URL for direct download from MinIO.

    Args:
        bucket_name: Name of the bucket
        file_name: Name of the file in the bucket
        expires_hours: URL validity period in hours

    Returns:
        Presigned URL as string or None if error occurs
    """
    try:
        # Initialize MinIO client inside the function
        minio_client = Minio(
            MINIO_ADDRESS,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=True
        )

        # Generate presigned URL (valid for specified hours)
        url = minio_client.presigned_get_object(
            bucket_name,
            file_name,
            expires=timedelta(hours=expires_hours)
        )
        return url
    except S3Error as e:
        # logger.error("Error generating presigned URL: %s", str(e))
        return None
    except Exception as e:
        # logger.error("Unexpected error generating presigned URL: %s", str(e))
        return None
