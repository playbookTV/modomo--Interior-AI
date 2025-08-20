"""
Dependency injection for services
"""
from typing import Optional
from services.database_service import DatabaseService
from services.job_service import JobService  
from services.detection_service import DetectionService
from utils.logging import get_logger

logger = get_logger(__name__)

# Global service instances
_database_service: Optional[DatabaseService] = None
_job_service: Optional[JobService] = None
_detection_service: Optional[DetectionService] = None
_r2_client = None
_r2_bucket_name = None


def set_database_service(service: DatabaseService):
    """Set the global database service instance"""
    global _database_service
    _database_service = service
    logger.info("✅ Database service registered")


def set_job_service(service: JobService):
    """Set the global job service instance"""
    global _job_service
    _job_service = service
    logger.info("✅ Job service registered")


def set_detection_service(service: DetectionService):
    """Set the global detection service instance"""
    global _detection_service
    _detection_service = service
    logger.info("✅ Detection service registered")


def set_r2_client(client, bucket_name: str = None):
    """Set the global R2 client instance"""
    global _r2_client, _r2_bucket_name
    _r2_client = client
    _r2_bucket_name = bucket_name or "reroom"
    logger.info(f"✅ R2 client registered (bucket: {_r2_bucket_name})")


# FastAPI Dependency functions (for use with Depends())
def get_database_service() -> Optional[DatabaseService]:
    """Get database service instance for FastAPI dependency injection"""
    return _database_service


def get_job_service() -> Optional[JobService]:
    """Get job service instance for FastAPI dependency injection"""
    return _job_service


def get_detection_service() -> Optional[DetectionService]:
    """Get detection service instance for FastAPI dependency injection"""
    return _detection_service


def get_r2_client():
    """Get R2 client instance for dependency injection"""
    return _r2_client


def get_r2_bucket_name() -> str:
    """Get R2 bucket name"""
    return _r2_bucket_name


def check_services_ready() -> dict:
    """Check which services are ready"""
    return {
        "database_service": _database_service is not None,
        "job_service": _job_service is not None and (_job_service.is_available() if hasattr(_job_service, 'is_available') else True),
        "detection_service": _detection_service is not None,
        "r2_storage": _r2_client is not None
    }