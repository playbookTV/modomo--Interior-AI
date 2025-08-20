"""
Admin API routes for database management and system testing
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import structlog

from core.dependencies import get_database_service
from core.dependencies import get_database_service

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/test-supabase", response_model=None)
async def test_supabase(database_service = Depends(get_database_service)):
    """Test Supabase connection and permissions"""
    if not database_service:
        raise HTTPException(status_code=503, detail="Database service not available")
    
    result = await database_service.test_connection()
    
    if result["status"] == "error":
        raise HTTPException(status_code=503, detail=result["message"])
    
    return result


@router.post("/init-database", response_model=None)
async def init_database():
    """Initialize database tables (admin only)"""
    # Note: This endpoint would need database pool connection
    # For now, return a message about Supabase-managed schema
    return {
        "status": "info", 
        "message": "Database schema is managed by Supabase migrations. Use Supabase dashboard for schema changes."
    }