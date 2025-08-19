"""
Configuration settings for Modomo Scraper
"""
import os
from typing import Optional


class Settings:
    """Application settings loaded from environment variables"""
    
    # Database Configuration
    SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY: Optional[str] = os.getenv("SUPABASE_ANON_KEY")
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    # Redis Configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # R2 Storage Configuration
    R2_ENDPOINT: Optional[str] = os.getenv("CLOUDFLARE_R2_ENDPOINT")
    R2_ACCESS_KEY_ID: Optional[str] = os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID")  
    R2_SECRET_ACCESS_KEY: Optional[str] = os.getenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY")
    R2_BUCKET_NAME: str = os.getenv("CLOUDFLARE_R2_BUCKET", "reroom")
    
    # AI Model Configuration
    FORCE_COLOR_EXTRACTOR: bool = os.getenv("FORCE_COLOR_EXTRACTOR", "false").lower() == "true"
    
    # File paths
    MASKS_DIR: str = "/app/cache_volume/masks"
    
    # Application Info
    APP_TITLE: str = "Modomo Scraper API (Full AI)"
    APP_DESCRIPTION: str = "Complete dataset creation system with AI processing"
    APP_VERSION: str = "1.0.2-full"
    
    # CORS Configuration
    CORS_ORIGINS: list = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list = ["*"]
    CORS_ALLOW_HEADERS: list = ["*"]
    
    @classmethod
    def validate_required_settings(cls) -> dict:
        """Validate that required settings are present"""
        missing = []
        r2_missing = []
        
        if not cls.SUPABASE_URL:
            missing.append("SUPABASE_URL")
        if not cls.SUPABASE_ANON_KEY:
            missing.append("SUPABASE_ANON_KEY")
        
        # R2 settings validation
        if not cls.R2_ENDPOINT:
            r2_missing.append("CLOUDFLARE_R2_ENDPOINT")
        if not cls.R2_ACCESS_KEY_ID:
            r2_missing.append("CLOUDFLARE_R2_ACCESS_KEY_ID")
        if not cls.R2_SECRET_ACCESS_KEY:
            r2_missing.append("CLOUDFLARE_R2_SECRET_ACCESS_KEY")
        
        return {
            "valid": len(missing) == 0,
            "missing": missing,
            "supabase_configured": bool(cls.SUPABASE_URL and cls.SUPABASE_ANON_KEY),
            "r2_configured": len(r2_missing) == 0,
            "r2_missing": r2_missing
        }


# Global settings instance
settings = Settings()