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
        # Import AI models locally to avoid circular dependency and allow graceful fallback
        detector = None
        segmenter = None
        embedder = None
        color_extractor = None

        try:
            from models.grounding_dino import GroundingDINODetector
            from models.sam2_segmenter import SAM2Segmenter
            from models.clip_embedder import CLIPEmbedder
            from models.color_extractor import ColorExtractor

            # Force eager loading for production deployment (Railway)
            detector = GroundingDINODetector()
            segmenter = SAM2Segmenter(eager_load=True)  # Force immediate model loading
            embedder = CLIPEmbedder()
            color_extractor = ColorExtractor()
            logger.info("✅ All AI models loaded eagerly for DetectionService")
        except ImportError as e:
            logger.warning(f"⚠️ Could not load all AI models for DetectionService: {e}")
        except Exception as e:
            logger.error(f"❌ Error loading AI models for DetectionService: {e}")

        detection_service = DetectionService(
            detector=detector,
            segmenter=segmenter,
            embedder=embedder,
            color_extractor=color_extractor
        )
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


# OLD register_routers function - REPLACED with register_all_routers_post_init
def register_routers_DEPRECATED(app: FastAPI):
    """DEPRECATED: Use register_all_routers_post_init instead"""
    try:
        # Core routers (no circular dependencies)
        from routers.jobs import router as jobs_router
        from routers.detection import router as detection_router
        from routers.scraping import router as scraping_router
        from routers.classification import router as classification_router
        from routers.export import router as export_router
        from routers.analytics import router as analytics_router
        from routers.admin import router as admin_router
        
        # Register core routers
        app.include_router(jobs_router)
        app.include_router(detection_router)
        app.include_router(scraping_router)
        app.include_router(classification_router)
        app.include_router(export_router)
        app.include_router(analytics_router)
        app.include_router(admin_router)
        
        logger.info("✅ Core routers registered successfully")
        
        # Register modular endpoints that don't have circular imports
        try:
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
            
            logger.info("✅ Modular endpoints registered successfully")
        except Exception as e:
            logger.warning(f"⚠️ Some modular endpoints failed to register: {e}")
        
        logger.info("✅ Main router registration completed")
        
        # Register ALL routers
        app.include_router(jobs_router)
        app.include_router(detection_router)
        app.include_router(scraping_router)
        app.include_router(classification_router)
        app.include_router(export_router)
        app.include_router(analytics_router)
        app.include_router(admin_router)
        # sync_router excluded due to circular import
        
        logger.info("✅ ALL comprehensive routers registered")
        
        # Register modular endpoints
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
        
        logger.info("✅ ALL routers registered successfully")
        
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
    
    @app.get("/")
    async def root():
        """Root endpoint for Railway deployment verification"""
        return {
            "service": "Modomo AI Dataset Scraping System",
            "version": settings.APP_VERSION,
            "status": "running",
            "health_check": "/health"
        }
    
    @app.get("/health")
    async def health_check():
        """Railway-compatible health check endpoint"""
        try:
            services = check_services_ready()
            
            # Count ready services
            ready_count = sum(1 for status in services.values() if status and status != "missing_config")
            total_count = len(services)
            
            # Application is healthy if core services are ready or at least partially functional
            is_healthy = ready_count > 0 or any(
                status == "initialized" for status in services.values()
            )
            
            response = {
                "status": "healthy" if is_healthy else "starting",
                "services": services,
                "ready": f"{ready_count}/{total_count}",
                "version": settings.APP_VERSION,
                "timestamp": __import__('datetime').datetime.utcnow().isoformat()
            }
            
            # Return 200 even if some services aren't ready (Railway needs 200 for health check)
            return response
            
        except Exception as e:
            # Return minimal healthy response even on error to keep deployment alive
            logger.error(f"Health check error: {e}")
            return {
                "status": "minimal",
                "error": str(e),
                "version": settings.APP_VERSION,
                "timestamp": __import__('datetime').datetime.utcnow().isoformat()
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


def create_app_without_routers() -> FastAPI:
    """Create FastAPI application with services but WITHOUT routers (to avoid circular imports)"""
    logger.info("🚀 Creating Modomo Dataset Scraping Application (Phase 1: No routers)")
    
    # Create base app
    app = create_app()
    
    # Initialize services
    services_status = initialize_services()
    logger.info(f"📊 Services status: {services_status}")
    
    # Add health endpoints
    add_health_endpoints(app)
    
    # Mark app as ready for router registration
    app.state.services_initialized = True
    app.state.services_status = services_status
    
    # Add static file serving if directory exists
    static_dir = "static"
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logger.info("✅ Static files mounted")
    
    logger.info("✅ Application created successfully (Phase 1 complete - no routers yet)")
    return app


def register_all_routers_post_init(app: FastAPI):
    """Register ALL routers AFTER services are initialized (avoids circular imports)"""
    if not getattr(app.state, 'services_initialized', False):
        logger.error("❌ Cannot register routers - services not initialized")
        return False
        
    logger.info("🔄 Phase 2: Registering all routers post-initialization")
    
    # Now safely import and register routers
    try:
        # Import core routers
        from routers.jobs import router as jobs_router
        from routers.detection import router as detection_router  
        from routers.scraping import router as scraping_router
        from routers.classification import router as classification_router
        from routers.export import router as export_router
        from routers.analytics import router as analytics_router
        from routers.admin import router as admin_router
        
        # Register all routers
        routers = [
            (jobs_router, "jobs"),
            (detection_router, "detection"),
            (scraping_router, "scraping"), 
            (classification_router, "classification"),
            (export_router, "export"),
            (analytics_router, "analytics"),
            (admin_router, "admin")
        ]
        
        for router, name in routers:
            app.include_router(router)
            logger.info(f"✅ Registered {name} router")
            
        # Register modular endpoints
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
        
        logger.info("✅ All modular endpoints registered")
        
        # Register clean sync router (no circular imports)
        try:
            from routers.sync_clean import sync_router
            app.include_router(sync_router)
            logger.info("✅ Clean sync monitoring router registered")
        except Exception as sync_error:
            logger.warning(f"⚠️ Clean sync router registration failed: {sync_error}")
        
        logger.info("🎉 Phase 2 complete - All routers registered successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Router registration failed: {e}")
        return False


def create_complete_app() -> FastAPI:
    """Create complete application with post-initialization router registration"""
    # Phase 1: Create app with services but no routers
    app = create_app_without_routers()
    
    # Phase 2: Register routers after services are ready  
    success = register_all_routers_post_init(app)
    
    if success:
        logger.info("✅ Complete application created successfully")
    else:
        logger.warning("⚠️ Application created but some routers failed to register")
        
    return app