#!/usr/bin/env python3
"""
Test script to validate the refactored module structure
"""
import sys
import os

def test_imports():
    """Test all module imports"""
    print("🧪 Testing refactored module structure...")
    
    # Test configuration imports
    try:
        from config.settings import settings
        from config.taxonomy import MODOMO_TAXONOMY
        print("✅ Config modules import successfully")
        print(f"   - Settings: {settings.APP_TITLE}")
        print(f"   - Taxonomy categories: {len(MODOMO_TAXONOMY)}")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    # Test utilities
    try:
        from utils.logging import configure_logging, get_logger
        from utils.serialization import make_json_serializable
        print("✅ Utils modules import successfully")
        
        # Test serialization
        import numpy as np
        test_data = {"array": np.array([1, 2, 3]), "scalar": np.int64(42)}
        serialized = make_json_serializable(test_data)
        print(f"   - Serialization test: {type(serialized['array'])}, {type(serialized['scalar'])}")
    except ImportError as e:
        print(f"❌ Utils import failed: {e}")
        return False
    
    # Test services (without external dependencies)
    try:
        from services.database_service import DatabaseService
        from services.job_service import JobService
        print("✅ Service modules import successfully")
    except ImportError as e:
        print(f"❌ Services import failed: {e}")
        return False
    
    # Test routers
    try:
        from routers.admin import router as admin_router
        from routers.analytics import router as analytics_router
        from routers.jobs import router as jobs_router
        print("✅ Router modules import successfully")
        print(f"   - Admin routes: {len(admin_router.routes)}")
        print(f"   - Analytics routes: {len(analytics_router.routes)}")
        print(f"   - Jobs routes: {len(jobs_router.routes)}")
    except ImportError as e:
        print(f"❌ Routers import failed: {e}")
        return False
    
    # Test background tasks
    try:
        from tasks.classification_tasks import classify_image_type, get_comprehensive_keywords
        print("✅ Task modules import successfully")
        
        # Test classification function
        keywords = get_comprehensive_keywords()
        print(f"   - Classification keywords: {len(keywords)} categories")
    except ImportError as e:
        print(f"❌ Tasks import failed: {e}")
        return False
    
    return True

def test_structure():
    """Test directory structure"""
    print("📁 Testing directory structure...")
    
    required_dirs = [
        "config",
        "utils", 
        "services",
        "routers",
        "tasks"
    ]
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"   ✅ {dir_name}/ exists")
            
            # Check for __init__.py
            init_file = os.path.join(dir_name, "__init__.py")
            if os.path.exists(init_file):
                print(f"      ✅ {dir_name}/__init__.py exists")
            else:
                print(f"      ❌ {dir_name}/__init__.py missing")
        else:
            print(f"   ❌ {dir_name}/ missing")
            return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 Starting refactored structure validation...")
    print("=" * 50)
    
    # Change to the correct directory
    os.chdir("/Users/leslieisah/modomo/backend/modomo-scraper")
    
    structure_ok = test_structure()
    imports_ok = test_imports()
    
    print("=" * 50)
    if structure_ok and imports_ok:
        print("🎉 All tests passed! Refactored structure is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)