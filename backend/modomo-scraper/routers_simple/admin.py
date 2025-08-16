"""
Admin API routes for database management and system testing
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import structlog

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/test-supabase")
async def test_supabase():
    """Test Supabase connection and permissions"""
    # Import here to avoid circular dependency
    try:
        from main_refactored import database_service
        
        if not database_service:
            raise HTTPException(status_code=503, detail="Database service not available")
        
        result = await database_service.test_connection()
        
        if result["status"] == "error":
            raise HTTPException(status_code=503, detail=result["message"])
        
        return result
    except ImportError:
        raise HTTPException(status_code=503, detail="Database service not available")


@router.post("/init-database")
async def init_database():
    """Initialize database tables (admin only)"""
    return {
        "status": "info", 
        "message": "Database schema is managed by Supabase migrations. Use Supabase dashboard for schema changes."
    }