"""
Modomo Dataset Scraping System - Clean Refactored Main Application
Modular architecture with proper separation of concerns and dependency injection
"""
from core.app_factory import create_complete_app
from core.dependencies import check_services_ready
from utils.logging import get_logger

logger = get_logger(__name__)

# Create the application instance
app = create_complete_app()

# Log final application status
@app.on_event("startup")
async def startup_event():
    """Log application startup status"""
    services = check_services_ready()
    logger.info("🚀 Modomo Dataset Scraping System - Clean Architecture")
    logger.info(f"📊 Services Status: {services}")
    
    ready_services = sum(1 for status in services.values() if status)
    total_services = len(services)
    logger.info(f"✅ Application started successfully ({ready_services}/{total_services} services ready)")


if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Modomo Dataset Scraping System - Clean Architecture")
    uvicorn.run(
        "main_clean:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )