"""
Railway-optimized entry point for Modomo Scraper
Handles AI dependency loading gracefully for Railway deployment
"""

import os
import sys
import importlib

# Ensure current directory is in Python path for models package
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# Also ensure /app is in path for Railway deployment
app_dir = "/app"
if app_dir not in sys.path and os.path.exists(app_dir):
    sys.path.insert(0, app_dir)

def check_ai_dependencies():
    """Check if AI dependencies are available and working"""
    try:
        # Test numpy first (common issue)
        import numpy as np
        np.array([1, 2, 3])
        
        # Test torch carefully
        import torch
        # Don't initialize CUDA or complex operations
        torch.manual_seed(42)
        
        # Test transformers import
        import transformers
        
        # Test structlog
        import structlog
        
        return True
        
    except ImportError as e:
        print(f"💡 Missing AI dependency: {e}")
        return False
    except Exception as e:
        print(f"⚠️ AI dependency error: {e}")
        return False

def get_app():
    """Get the appropriate app - always full AI mode for production"""
    
    # Restore model Python files after Railway Volume mount
    print("🔧 Restoring model Python files after Railway Volume mount...")
    try:
        from restore_models import restore_model_files
        restore_success = restore_model_files()
        if restore_success:
            print("✅ Model Python files restored successfully")
        else:
            print("❌ Failed to restore model Python files")
    except Exception as e:
        print(f"❌ Model restoration failed: {e}")
    
    # Comprehensive fix for models import issues
    print("🔧 Running comprehensive models import fix...")
    try:
        from fix_models_import import fix_models_import
        fix_success = fix_models_import()
        if fix_success:
            print("✅ Models import fix completed successfully")
        else:
            print("⚠️ Models import fix had issues - will use fallbacks")
    except Exception as e:
        print(f"⚠️ Failed to run models import fix: {e}")

    # Initialize models from Railway Volume on startup
    print("🤖 Initializing AI models from Railway Volume...")
    try:
        from init_models import init_models, check_model_availability
        
        # Initialize models (downloads only if missing)
        model_init_success = init_models()
        
        # Check availability
        model_status = check_model_availability()
        available_models = [name for name, info in model_status.items() if info["available"]]
        
        if available_models:
            print(f"✅ Available models: {', '.join(available_models)}")
        else:
            print("⚠️ No models available - will use fallback mode")
            
    except Exception as e:
        print(f"⚠️ Model initialization warning: {e}")
        print("🔄 Continuing with existing models if available...")
        
    # Initialize Phase 2 caches (Hugging Face, Playwright, etc.)
    print("🚀 Initializing Phase 2 caches...")
    try:
        from init_caches import CacheInitializer
        cache_init = CacheInitializer()
        
        # Only initialize what's missing (smart caching)
        cache_stats = cache_init.initialize_all()
        total_cached = sum(s.get("size_mb", 0) for s in cache_stats.values())
        print(f"📦 Phase 2 cache ready: {total_cached:.1f} MB total")
        
    except Exception as e:
        print(f"⚠️ Cache initialization warning: {e}")
        print("🔄 Will initialize caches on first use...")
    
    # Force full AI mode for production
    ai_mode = os.getenv("AI_MODE", "full").lower()
    print(f"🔍 AI_MODE environment variable: {ai_mode}")
    
    # Check if AI dependencies are available
    ai_available = check_ai_dependencies()
    print(f"🤖 AI dependencies available: {ai_available}")
    
    # Try to load refactored architecture first, then fallback to original
    try:
        print("🚀 Loading refactored architecture...")
        from main_refactored import app
        print("✅ Refactored architecture loaded successfully")
        return app
    except Exception as e:
        print(f"❌ Failed to load refactored architecture: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to original main_full
        try:
            print("🔄 Falling back to original full AI mode...")
            from main_full import app
            print("✅ Original full AI mode loaded successfully")
            return app
        except Exception as full_error:
            print(f"❌ Failed to load original full AI mode: {full_error}")
            
            # If AI dependencies are missing, this is a build error
            if not ai_available:
                print("💥 CRITICAL: AI dependencies missing in production build!")
                print("🔧 Check Dockerfile and requirements-ai-stable.txt")
                # Still try basic mode but warn loudly
                
            print("🔄 Final fallback to basic mode (NOT RECOMMENDED FOR PRODUCTION)")
            try:
                from main_basic import app
                print("⚠️ Basic mode loaded - LIMITED FUNCTIONALITY")
                return app
            except Exception as basic_error:
                print(f"💥 FATAL: Even basic mode failed: {basic_error}")
                raise

# Get the app instance
app = get_app()

if __name__ == "__main__":
    import uvicorn
    
    # Railway provides PORT environment variable
    port = int(os.getenv("PORT", 8001))
    
    print(f"🌐 Starting Modomo Scraper on port {port}")
    print(f"📊 Mode: {'AI' if 'full' in str(app.title).lower() else 'Basic'}")
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")