"""
Static file serving endpoints including SAM2 masks
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from core.dependencies import get_r2_client
from utils.logging import get_logger

logger = get_logger(__name__)


def register_static_routes(app: FastAPI):
    """Register static file serving endpoints"""
    
    # Create masks directory
    masks_dir = "/app/cache_volume/masks"
    os.makedirs(masks_dir, exist_ok=True)
    
    @app.get("/masks/{filename}")
    async def serve_mask(filename: str):
        """Serve SAM2 mask files from R2 storage or local volume"""
        try:
            # First, try R2 storage
            r2_client = get_r2_client()
            if r2_client:
                try:
                    r2_key = f"masks/{filename}"
                    
                    # Generate presigned URL for R2 access
                    presigned_url = r2_client.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': 'reroom', 'Key': r2_key},  # TODO: Get bucket from settings
                        ExpiresIn=3600  # 1 hour
                    )
                    
                    logger.info(f"Serving mask {filename} from R2: {presigned_url[:50]}...")
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
                except Exception as r2_error:
                    logger.warning(f"R2 mask serving failed for {filename}: {r2_error}")
            
            # Fallback to local volume serving
            local_path = os.path.join(masks_dir, filename)
            if os.path.exists(local_path):
                logger.info(f"Serving mask {filename} from local volume")
                from fastapi.responses import FileResponse
                return FileResponse(
                    path=local_path,
                    headers={
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Methods": "GET, OPTIONS",
                        "Access-Control-Allow-Headers": "*",
                        "Cache-Control": "public, max-age=3600"
                    }
                )
            
            # Not found in R2 or local
            logger.error(f"Mask not found: {filename}")
            raise HTTPException(status_code=404, detail="Mask file not found")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error serving mask {filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to serve mask: {str(e)}")
    
    @app.options("/masks/{filename}")
    async def mask_options(filename: str):
        """Handle CORS preflight requests for mask files"""
        from fastapi import Response
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )