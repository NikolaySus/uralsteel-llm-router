from datetime import timedelta
import os

import diskcache
from minio import Minio

# S3
MINIO_ADDRESS = os.environ.get('MINIO_ADDRESS', '')
BUCKET_NAME = os.environ.get('BUCKET_NAME', 'cache')
MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY', None)
MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY', None)

# Cache for presigned URLs using diskcache for thread-safe persistence
_presigned_url_cache = diskcache.Cache('.presigned_url_cache')

def generate_presigned_download_url(bucket_name, file_name, expires_hours=None):
    """
    Generate a presigned URL for direct download from MinIO with infinite TTL.
    URLs are cached - if a URL has already been generated for a file, the cached
    version is returned.

    Args:
        bucket_name: Name of the bucket
        file_name: Name of the file in the bucket
        expires_hours: Deprecated parameter (ignored), kept for backward compatibility

    Returns:
        Presigned URL as string or None if error occurs
    """
    # Create cache key combining bucket and file
    cache_key = f"{bucket_name}:{file_name}"
    
    # Return cached URL if it exists
    if cache_key in _presigned_url_cache:
        return _presigned_url_cache[cache_key]
    
    # Initialize MinIO client inside the function
    minio_client = Minio(
        MINIO_ADDRESS,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=True
    )

    # Generate presigned URL with very long expiration (effectively infinite - 100 years)
    url = minio_client.presigned_get_object(
        bucket_name,
        file_name,
        expires=timedelta(days=365 * 100)
    )
    
    # Cache the URL for future use (diskcache handles persistence automatically)
    _presigned_url_cache[cache_key] = url
    
    return url
