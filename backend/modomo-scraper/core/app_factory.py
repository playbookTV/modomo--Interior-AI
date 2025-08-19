"""
Application factory for creating FastAPI app with proper service initialization
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from typing import Optional

from config.settings import settings
from utils.logging import configure_logging, get_logger
from core.dependencies import (
    set_database_service, 
    set_job_service, 
    set_detection_service,
    set_r2_client,
    check_services_ready
)

# Configure logging first
configure_logging()
logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    app = FastAPI(
        title=settings.APP_TITLE,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )
    
    return app


def initialize_services() -> dict:
    """Initialize all application services"""
    services_status = {}
    
    # Initialize Database Service
    try:
        from supabase import create_client
        from services.database_service import DatabaseService
        
        if settings.SUPABASE_URL and settings.SUPABASE_SERVICE_ROLE_KEY:
            supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
            database_service = DatabaseService(supabase)
            set_database_service(database_service)
            services_status["database"] = "initialized"
            logger.info("✅ Database service initialized")
        else:
            services_status["database"] = "missing_config"
            logger.warning("⚠️ Database service not initialized - missing config")
    except ImportError as e:
        services_status["database"] = f"import_error: {e}"
        logger.error(f"❌ Database service import failed: {e}")
    except Exception as e:
        services_status["database"] = f"error: {e}"
        logger.error(f"❌ Database service initialization failed: {e}")
    
    # Initialize Job Service
    try:
        import redis
        from services.job_service import JobService
        
        if hasattr(settings, 'REDIS_URL') and settings.REDIS_URL:
            job_service = JobService(settings.REDIS_URL)
            set_job_service(job_service)
            services_status["job_service"] = "initialized"
            logger.info("✅ Job service initialized")
        else:
            services_status["job_service"] = "missing_redis_config"
            logger.warning("⚠️ Job service not initialized - Redis config missing")
    except ImportError:
        services_status["job_service"] = "redis_not_available"
        logger.warning("⚠️ Job service not initialized - Redis not available")
    except Exception as e:
        services_status["job_service"] = f"error: {e}"
        logger.error(f"❌ Job service initialization failed: {e}")
    
    # Initialize Detection Service
    try:
        from services.detection_service import DetectionService
        detection_service = DetectionService()
        set_detection_service(detection_service)
        services_status["detection_service"] = "initialized"
        logger.info("✅ Detection service initialized")
    except Exception as e:
        services_status["detection_service"] = f"error: {e}"
        logger.error(f"❌ Detection service initialization failed: {e}")
    
    # Initialize R2 Client for mask storage
    try:
        import boto3
        
        if (hasattr(settings, 'R2_ENDPOINT') and settings.R2_ENDPOINT and
            hasattr(settings, 'R2_ACCESS_KEY_ID') and settings.R2_ACCESS_KEY_ID and
            hasattr(settings, 'R2_SECRET_ACCESS_KEY') and settings.R2_SECRET_ACCESS_KEY):
            
            r2_client = boto3.client(
                's3',
                endpoint_url=settings.R2_ENDPOINT,
                aws_access_key_id=settings.R2_ACCESS_KEY_ID,
                aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
                region_name='auto'
            )
            
            # Test R2 connection
            bucket_name = getattr(settings, 'R2_BUCKET_NAME', 'reroom')
            r2_client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
            
            set_r2_client(r2_client, bucket_name)
            
            # Also set R2 client for mask and admin endpoints
            try:
                from routers.mask_endpoints import set_r2_client as set_mask_r2_client
                from routers.admin_utilities import set_r2_client as set_admin_r2_client
                set_mask_r2_client(r2_client, bucket_name)
                set_admin_r2_client(r2_client, bucket_name)
            except ImportError:
                # Endpoints not yet imported, will be set later
                pass
            services_status["r2_storage"] = "initialized"
            logger.info(f"✅ R2 client initialized (bucket: {bucket_name})")
        else:
            services_status["r2_storage"] = "missing_config"
            logger.warning("⚠️ R2 client not initialized - missing configuration")
    except Exception as e:
        services_status["r2_storage"] = f"error: {e}"
        logger.error(f"❌ R2 client initialization failed: {e}")
    
    return services_status


def register_routers(app: FastAPI):
    """Register all application routers"""
    try:
        # Add response_model=None to endpoints that might return service objects
        from fastapi.responses import JSONResponse
        from fastapi import HTTPException
        # Import routers with fallback to simple versions
        try:
            from routers.jobs import router as jobs_router
            from routers.detection import router as detection_router
            from routers.scraping import router as scraping_router
            from routers.classification import router as classification_router
            from routers.export import router as export_router
            from routers.analytics import router as analytics_router
            from routers.admin import router as admin_router
            from routers.sync_monitor import sync_router
            
            # Register full routers
            app.include_router(jobs_router)
            app.include_router(detection_router)
            app.include_router(scraping_router)
            app.include_router(classification_router)
            app.include_router(export_router)
            app.include_router(analytics_router)
            app.include_router(admin_router)
            app.include_router(sync_router)
            
            logger.info("✅ Full routers registered")
            
        except ImportError:
            # Fallback to simplified routers
            logger.warning("⚠️ Full routers not available, using simplified versions")
            
            try:
                from routers_simple.admin import router as admin_router
                from routers_simple.analytics import router as analytics_router
                app.include_router(admin_router)
                app.include_router(analytics_router)
                logger.info("✅ Simplified routers registered")
            except ImportError:
                logger.error("❌ Even simplified routers not available")
        
        # Register additional endpoints via separate modules
        from routers.color_endpoints import register_color_routes
        from routers.review_endpoints import register_review_routes
        from routers.dataset_endpoints import register_dataset_routes
        from routers.mask_endpoints import register_mask_routes
        from routers.advanced_ai_endpoints import register_advanced_ai_routes
        from routers.admin_utilities import register_admin_utilities
        
        register_color_routes(app)
        register_review_routes(app)
        register_dataset_routes(app)
        register_mask_routes(app)
        register_advanced_ai_routes(app)
        register_admin_utilities(app)
        
        # Add debug endpoint with explicit response model to prevent Supabase Client serialization error
        @app.get("/debug/database-status", response_model=None)
        async def debug_database_status():
            from core.dependencies import get_database_service
            """Debug database connection status without returning client object"""
            try:
                database_service = get_database_service()
                if not database_service:
                    return {"status": "unavailable", "error": "Database service not initialized"}
                
                # Test connection without returning the client
                result = database_service.supabase.table("scenes").select("scene_id", count="exact").limit(1).execute()
                
                return {
                    "status": "connected",
                    "can_query": True,
                    "total_scenes": result.count or 0,
                    "supabase_url": hasattr(database_service, 'supabase') and database_service.supabase.url is not None
                }
            except Exception as e:
                return {"status": "error", "error": str(e)}
        
        # Set R2 clients for endpoints that need them
        try:
            from core.dependencies import get_r2_client, get_r2_bucket_name
            r2_client = get_r2_client()
            r2_bucket = get_r2_bucket_name()
            
            if r2_client:
                from routers.mask_endpoints import set_r2_client as set_mask_r2_client
                from routers.admin_utilities import set_r2_client as set_admin_r2_client
                set_mask_r2_client(r2_client, r2_bucket)
                set_admin_r2_client(r2_client, r2_bucket)
                logger.info("✅ R2 client configured for mask and admin endpoints")
        except ImportError as e:
            logger.warning(f"⚠️ Could not configure R2 for endpoints: {e}")
        
    except Exception as e:
        logger.error(f"❌ Router registration failed: {e}")
        
        # If router registration fails due to response model issues, try to continue
        # This prevents the entire app from failing due to a single problematic endpoint
        try:
            # Register only essential endpoints as fallback
            @app.get("/health", response_model=None)
            async def emergency_health():
                return {"status": "emergency_mode", "error": "Router registration partially failed"}
        except Exception as fallback_error:
            logger.error(f"❌ Even fallback endpoint registration failed: {fallback_error}")


def add_health_endpoints(app: FastAPI):
    """Add basic health check and status endpoints"""
    
    @app.get("/health")
    async def health_check():
        """Basic health check endpoint"""
        services = check_services_ready()
        return {
            "status": "healthy",
            "services": services,
            "version": settings.APP_VERSION
        }
    
    @app.get("/status")
    async def app_status():
        """Detailed application status"""
        services = check_services_ready()
        return {
            "application": "Modomo Dataset Scraping System",
            "version": settings.APP_VERSION,
            "services": services,
            "environment": getattr(settings, 'ENVIRONMENT', 'unknown')
        }


def create_complete_app() -> FastAPI:
    """Create complete application with services and routers"""
    logger.info("🚀 Creating Modomo Dataset Scraping Application")
    
    # Create app
    app = create_app()
    
    # Initialize services
    services_status = initialize_services()
    logger.info(f"📊 Services status: {services_status}")
    
    # Add health endpoints
    add_health_endpoints(app)
    
    # Register routers
    register_routers(app)
    
    # Add static file serving if directory exists
    static_dir = "static"
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logger.info("✅ Static files mounted")
    
    logger.info("✅ Application created successfully")
    return app