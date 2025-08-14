#!/usr/bin/env python3
"""
Model Initialization Script for Railway Deployment
Downloads models only if they don't exist on the Railway Volume
"""

import os
import sys
import requests
import hashlib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIG = {
    "sam2_hiera_large": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
        "path": "/app/model_cache/sam2/sam2_hiera_large.pt",
        "size_mb": 223,
        "sha256": None  # Add if available
    },
    "sam2_hiera_base_plus": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt", 
        "path": "/app/model_cache/sam2/sam2_hiera_base_plus.pt",
        "size_mb": 142,
        "sha256": None
    }
}

def check_file_exists_and_valid(file_path: str, expected_size_mb: int = None) -> bool:
    """Check if file exists and has reasonable size"""
    if not os.path.exists(file_path):
        return False
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if expected_size_mb and file_size_mb < (expected_size_mb * 0.9):  # Allow 10% variance
        logger.warning(f"File {file_path} exists but size {file_size_mb:.1f}MB is less than expected {expected_size_mb}MB")
        return False
    
    return True

def download_with_progress(url: str, dest_path: str, expected_size_mb: int = None) -> bool:
    """Download file with progress reporting"""
    try:
        logger.info(f"📦 Downloading {os.path.basename(dest_path)} ({expected_size_mb}MB)...")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Progress reporting every 10MB
                    if downloaded % (10 * 1024 * 1024) == 0:
                        progress = (downloaded / total_size * 100) if total_size > 0 else 0
                        logger.info(f"  Downloaded {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({progress:.1f}%)")
        
        # Verify download
        final_size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        logger.info(f"✅ Downloaded {os.path.basename(dest_path)} ({final_size_mb:.1f}MB)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download {url}: {e}")
        # Clean up partial download
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def init_models() -> bool:
    """Initialize all required models"""
    logger.info("🚀 Initializing AI models...")
    
    # Check Railway Volume mount
    volume_path = "/app/model_cache"
    if not os.path.exists(volume_path):
        logger.info(f"📁 Creating models directory: {volume_path}")
        os.makedirs(volume_path, exist_ok=True)
    
    # Check existing models
    models_to_download = []
    for model_name, config in MODEL_CONFIG.items():
        if check_file_exists_and_valid(config["path"], config["size_mb"]):
            logger.info(f"✅ {model_name} already exists and valid")
        else:
            models_to_download.append((model_name, config))
    
    if not models_to_download:
        logger.info("🎉 All models already downloaded and valid!")
        return True
    
    # Download missing models
    success_count = 0
    for model_name, config in models_to_download:
        logger.info(f"📦 Model {model_name} needs downloading...")
        if download_with_progress(config["url"], config["path"], config["size_mb"]):
            success_count += 1
        else:
            logger.error(f"❌ Failed to download {model_name}")
    
    if success_count == len(models_to_download):
        logger.info(f"🎉 Successfully downloaded {success_count} models!")
        return True
    else:
        logger.warning(f"⚠️ Downloaded {success_count}/{len(models_to_download)} models")
        return success_count > 0  # Partial success is OK

def check_model_availability() -> dict:
    """Check which models are available for the application"""
    status = {}
    for model_name, config in MODEL_CONFIG.items():
        status[model_name] = {
            "available": check_file_exists_and_valid(config["path"], config["size_mb"]),
            "path": config["path"]
        }
    return status

if __name__ == "__main__":
    logger.info("🤖 Modomo AI Model Initialization")
    
    # Check if we're in a Railway environment
    if os.getenv("RAILWAY_ENVIRONMENT"):
        logger.info("🚂 Running in Railway environment")
    
    # Initialize models
    success = init_models()
    
    # Report status
    status = check_model_availability()
    logger.info("📊 Model Availability Status:")
    for model_name, info in status.items():
        status_icon = "✅" if info["available"] else "❌"
        logger.info(f"  {status_icon} {model_name}: {info['available']}")
    
    # Exit with appropriate code
    if success:
        logger.info("🎉 Model initialization completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Model initialization failed!")
        sys.exit(1)