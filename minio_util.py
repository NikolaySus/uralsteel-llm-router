from datetime import timedelta
import os
import json

import diskcache
from minio import Minio

# S3
MINIO_ADDRESS = os.environ.get('MINIO_ADDRESS', '')
BUCKET_NAME = os.environ.get('BUCKET_NAME', 'cache')
MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY', None)
MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY', None)

# Cache for presigned URLs using diskcache for thread-safe persistence
_presigned_url_cache = diskcache.Cache('.presigned_url_cache')

def _set_bucket_policy_public_read(bucket_name):
    """
    Set MinIO bucket policy to allow public read-only access.
    This enables permanent, infinite-TTL URLs for all objects.
    """
    minio_client = Minio(
        MINIO_ADDRESS,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=True
    )
    
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": "*"},
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{bucket_name}/*"]
            }
        ]
    }
    
    minio_client.set_bucket_policy(bucket_name, json.dumps(policy))
    print(f"Bucket '{bucket_name}' is now public for read-only access")

def generate_public_download_url(bucket_name, file_name):
    """
    Generate a public read-only download URL for MinIO object.
    The URL has infinite validity since the bucket is public.
    
    Args:
        bucket_name: Name of the bucket
        file_name: Name of the file in the bucket
        
    Returns:
        Public URL as string
    """
    # Create cache key combining bucket and file
    cache_key = f"{bucket_name}:{file_name}"
    
    # Return cached URL if it exists
    if cache_key in _presigned_url_cache:
        return _presigned_url_cache[cache_key]
    
    # Generate public URL
    url = f"http://{MINIO_ADDRESS}/{bucket_name}/{file_name}"
    
    # Cache the URL for future use (diskcache handles persistence automatically)
    _presigned_url_cache[cache_key] = url
    
    return url
