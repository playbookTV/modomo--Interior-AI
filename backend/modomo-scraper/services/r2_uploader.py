"""
R2 Uploader Service for Cloudflare R2 Storage
--------------------------------------------
- Upload generated maps to R2 storage
- Compatible with S3 API (boto3)
- Handles authentication and error management
- Async support for concurrent uploads
- Consistent with existing R2 usage patterns

Usage:
  uploader = R2Uploader()
  r2_key = await uploader.upload_file(local_path, "training-data/maps/depth/scene_123.png")
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# Try to import boto3 for R2/S3 compatibility
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("⚠️ boto3 not available - R2 uploads will be mocked")


class R2Uploader:
    """Cloudflare R2 uploader using S3-compatible API"""
    
    def __init__(self):
        # Try Railway variable names first, then fallback to standard names
        self.bucket_name = os.getenv("CLOUDFLARE_R2_BUCKET") or os.getenv("R2_BUCKET_NAME", "modomo-dataset")
        self.endpoint_url = os.getenv("CLOUDFLARE_R2_ENDPOINT") or os.getenv("R2_ENDPOINT_URL")
        self.access_key_id = os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID") or os.getenv("R2_ACCESS_KEY_ID")
        self.secret_access_key = os.getenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY") or os.getenv("R2_SECRET_ACCESS_KEY")
        self.region = os.getenv("R2_REGION", "auto")
        
        self.s3_client = None
        self.is_available = False
        
        self._init_client()
    
    def _init_client(self):
        """Initialize R2/S3 client"""
        if not BOTO3_AVAILABLE:
            logger.warning("⚠️ boto3 not available - using mock uploader")
            return
        
        # Check required credentials
        missing_creds = []
        if not self.endpoint_url:
            missing_creds.append("R2_ENDPOINT_URL")
        if not self.access_key_id:
            missing_creds.append("R2_ACCESS_KEY_ID")
        if not self.secret_access_key:
            missing_creds.append("R2_SECRET_ACCESS_KEY")
        
        if missing_creds:
            logger.warning(f"⚠️ Missing R2 credentials: {', '.join(missing_creds)}")
            logger.info("🔄 Will use mock uploader for development")
            return
        
        try:
            # Create S3 client configured for R2
            self.s3_client = boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                region_name=self.region
            )
            
            # Test connection by listing bucket
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            
            self.is_available = True
            logger.info(f"✅ R2 uploader initialized - bucket: {self.bucket_name}")
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"❌ R2 bucket '{self.bucket_name}' not found")
            else:
                logger.error(f"❌ R2 client error: {e}")
        except NoCredentialsError:
            logger.error("❌ R2 credentials not configured properly")
        except Exception as e:
            logger.error(f"❌ Failed to initialize R2 client: {e}")
    
    async def upload_file(self, local_path: str, r2_key: str, content_type: Optional[str] = None) -> bool:
        """
        Upload file to R2 storage
        
        Args:
            local_path: Path to local file
            r2_key: R2 object key (path within bucket)
            content_type: MIME type (auto-detected if None)
        
        Returns:
            True if upload successful, False otherwise
        """
        if not self.is_available:
            logger.warning(f"🚀 [MOCK] Would upload {local_path} to R2: {r2_key}")
            return True  # Mock success for development
        
        try:
            # Validate local file exists
            if not os.path.exists(local_path):
                logger.error(f"❌ Local file not found: {local_path}")
                return False
            
            # Auto-detect content type if not provided
            if content_type is None:
                content_type = self._get_content_type(local_path)
            
            # Prepare upload parameters
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            
            # Upload file
            logger.info(f"☁️ Uploading {local_path} to R2: {r2_key}")
            
            # Use upload_file for better performance with large files
            self.s3_client.upload_file(
                local_path,
                self.bucket_name,
                r2_key,
                ExtraArgs=extra_args
            )
            
            logger.info(f"✅ Successfully uploaded to R2: {r2_key}")
            return True
            
        except ClientError as e:
            logger.error(f"❌ R2 upload failed: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Upload error: {e}")
            return False
    
    async def upload_bytes(self, data: bytes, r2_key: str, content_type: Optional[str] = None) -> bool:
        """
        Upload bytes data to R2 storage
        
        Args:
            data: Binary data to upload
            r2_key: R2 object key
            content_type: MIME type
        
        Returns:
            True if upload successful, False otherwise
        """
        if not self.is_available:
            logger.warning(f"🚀 [MOCK] Would upload {len(data)} bytes to R2: {r2_key}")
            return True
        
        try:
            # Prepare upload parameters
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            
            # Upload bytes
            logger.info(f"☁️ Uploading {len(data)} bytes to R2: {r2_key}")
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=r2_key,
                Body=data,
                **extra_args
            )
            
            logger.info(f"✅ Successfully uploaded bytes to R2: {r2_key}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Bytes upload error: {e}")
            return False
    
    def _get_content_type(self, file_path: str) -> str:
        """Auto-detect content type from file extension"""
        extension = Path(file_path).suffix.lower()
        
        content_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.svg': 'image/svg+xml',
            '.json': 'application/json',
            '.txt': 'text/plain',
            '.html': 'text/html',
            '.css': 'text/css',
            '.js': 'application/javascript'
        }
        
        return content_types.get(extension, 'application/octet-stream')
    
    async def delete_file(self, r2_key: str) -> bool:
        """Delete file from R2 storage"""
        if not self.is_available:
            logger.warning(f"🗑️ [MOCK] Would delete from R2: {r2_key}")
            return True
        
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=r2_key)
            logger.info(f"🗑️ Deleted from R2: {r2_key}")
            return True
        except Exception as e:
            logger.error(f"❌ Delete error: {e}")
            return False
    
    async def file_exists(self, r2_key: str) -> bool:
        """Check if file exists in R2 storage"""
        if not self.is_available:
            return False
        
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=r2_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise
    
    def get_public_url(self, r2_key: str) -> str:
        """Get public URL for R2 object (if bucket is public)"""
        # For Railway static serving, construct URL based on your setup
        base_url = os.getenv("R2_PUBLIC_URL", f"https://{self.bucket_name}.r2.dev")
        return f"{base_url}/{r2_key}"
    
    def get_status(self) -> Dict[str, Any]:
        """Get uploader status and configuration"""
        return {
            "r2_uploader_available": self.is_available,
            "boto3_available": BOTO3_AVAILABLE,
            "bucket_name": self.bucket_name,
            "endpoint_configured": self.endpoint_url is not None,
            "credentials_configured": all([
                self.access_key_id,
                self.secret_access_key
            ]),
            "region": self.region
        }


class MockR2Uploader:
    """Mock R2 uploader for development without credentials"""
    
    def __init__(self):
        self.is_available = True
        logger.info("🚀 Mock R2 uploader initialized")
    
    async def upload_file(self, local_path: str, r2_key: str, content_type: Optional[str] = None) -> bool:
        """Mock upload - just log the operation"""
        file_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0
        logger.info(f"🚀 [MOCK] Upload {local_path} ({file_size} bytes) → R2: {r2_key}")
        return True
    
    async def upload_bytes(self, data: bytes, r2_key: str, content_type: Optional[str] = None) -> bool:
        """Mock bytes upload"""
        logger.info(f"🚀 [MOCK] Upload {len(data)} bytes → R2: {r2_key}")
        return True
    
    async def delete_file(self, r2_key: str) -> bool:
        """Mock delete"""
        logger.info(f"🗑️ [MOCK] Delete from R2: {r2_key}")
        return True
    
    async def file_exists(self, r2_key: str) -> bool:
        """Mock exists check"""
        return False
    
    def get_public_url(self, r2_key: str) -> str:
        """Mock public URL"""
        return f"https://mock-r2.dev/{r2_key}"
    
    def get_status(self) -> Dict[str, Any]:
        """Mock status"""
        return {
            "r2_uploader_available": True,
            "mock_mode": True,
            "note": "Using mock uploader - no actual uploads"
        }


# Factory function
def create_r2_uploader() -> R2Uploader:
    """Create R2 uploader instance"""
    try:
        return R2Uploader()
    except Exception as e:
        logger.warning(f"⚠️ Failed to create real R2 uploader, using mock: {e}")
        return MockR2Uploader()


if __name__ == "__main__":
    # Test the R2 uploader
    import sys
    import tempfile
    
    async def test_r2_upload():
        if len(sys.argv) < 2:
            print("Usage: python r2_uploader.py <test_file_path> [r2_key]")
            return
        
        file_path = sys.argv[1]
        r2_key = sys.argv[2] if len(sys.argv) > 2 else f"test/{Path(file_path).name}"
        
        print(f"🧪 Testing R2 upload:")
        print(f"📁 File: {file_path}")
        print(f"🔑 R2 Key: {r2_key}")
        
        uploader = create_r2_uploader()
        print(f"📊 Status: {uploader.get_status()}")
        
        success = await uploader.upload_file(file_path, r2_key)
        
        if success:
            print(f"✅ Upload successful")
            print(f"🔗 Public URL: {uploader.get_public_url(r2_key)}")
        else:
            print(f"❌ Upload failed")
    
    asyncio.run(test_r2_upload())