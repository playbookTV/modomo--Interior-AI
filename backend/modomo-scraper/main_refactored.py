"""
Modomo Dataset Scraping System - Refactored Main Application
Clean modular architecture with proper separation of concerns and dependency injection
"""
from contextlib import asynccontextmanager
from core.app_factory import create_complete_app
from core.dependencies import check_services_ready
from utils.logging import get_logger

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app):
    """Application lifespan handler"""
    # Startup
    services = check_services_ready()
    logger.info("🚀 Modomo Dataset Scraping System - Refactored Architecture")
    logger.info(f"📊 Services Status: {services}")
    
    ready_services = sum(1 for status in services.values() if status)
    total_services = len(services)
    logger.info(f"✅ Application started successfully ({ready_services}/{total_services} services ready)")
    
    yield
    
    # Shutdown
    logger.info("🛑 Application shutting down")

# Create the application instance with lifespan
app = create_complete_app(lifespan=lifespan)


if __name__ == "__main__":
    import uvicorn
    import os
    
    # Use Railway's PORT env var or fallback to 8000
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"🚀 Starting Modomo Dataset Scraping System on port {port}")
    uvicorn.run(
        "main_refactored:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        log_level="info"
    )