"""
Mask serving endpoints with R2 integration (from main_full.py)
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, Response
from core.dependencies import get_database_service
from utils.logging import get_logger

logger = get_logger(__name__)

# Global R2 client - will be set by app factory
r2_client = None
r2_bucket_name = None


def set_r2_client(client, bucket):
    """Set R2 client for mask serving"""
    global r2_client, r2_bucket_name
    r2_client = client
    r2_bucket_name = bucket


def register_mask_routes(app: FastAPI):
    """Register mask serving endpoints to the app"""
    
    @app.get("/masks/{filename}")
    async def serve_mask(filename: str):
        """Serve mask files by redirecting to R2 public URL"""
        try:
            # Construct R2 key
            r2_key = f"masks/{filename}"
            
            # Use R2 client if available
            if r2_client and r2_bucket_name:
                try:
                    # Try to generate presigned URL as fallback
                    presigned_url = r2_client.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': r2_bucket_name, 'Key': r2_key},
                        ExpiresIn=3600  # 1 hour
                    )
                    
                    # Redirect to the presigned URL
                    return RedirectResponse(
                        url=presigned_url,
                        status_code=302,
                        headers={
                            "Access-Control-Allow-Origin": "*",
                            "Access-Control-Allow-Methods": "GET, OPTIONS",
                            "Access-Control-Allow-Headers": "*",
                            "Cache-Control": "public, max-age=3600"
                        }
                    )
                except Exception as e:
                    logger.error(f"❌ Failed to generate presigned URL for {r2_key}: {e}")
                    
                    # Try direct public URL as fallback
                    # This works if the R2 bucket has public read access
                    base_url = os.getenv("CLOUDFLARE_R2_PUBLIC_URL", "https://pub-fa2319b55e064be087da337e9655b9de.r2.dev")
                    public_url = f"{base_url}/{r2_key}"
                    return RedirectResponse(
                        url=public_url,
                        status_code=302,
                        headers={
                            "Access-Control-Allow-Origin": "*",
                            "Access-Control-Allow-Methods": "GET, OPTIONS", 
                            "Access-Control-Allow-Headers": "*",
                            "Cache-Control": "public, max-age=3600"
                        }
                    )
            else:
                logger.error("❌ R2 client not available")
                raise HTTPException(status_code=503, detail="R2 storage not available")
                
        except Exception as e:
            logger.error(f"❌ Error serving mask {filename}: {e}")
            raise HTTPException(status_code=404, detail="Mask file not found")

    @app.options("/masks/{filename}")
    async def mask_options(filename: str):
        """Handle CORS preflight requests for mask files"""
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )