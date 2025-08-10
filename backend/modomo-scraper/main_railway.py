"""
Railway-optimized entry point for Modomo Scraper
Handles AI dependency loading gracefully for Railway deployment
"""

import os
import sys
import importlib

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
    
    # Force full AI mode for production
    ai_mode = os.getenv("AI_MODE", "full").lower()
    print(f"🔍 AI_MODE environment variable: {ai_mode}")
    
    # Check if AI dependencies are available
    ai_available = check_ai_dependencies()
    print(f"🤖 AI dependencies available: {ai_available}")
    
    # Always try to load full AI mode first
    try:
        print("🚀 Loading full AI mode...")
        from main_full import app
        print("✅ Full AI mode loaded successfully")
        return app
    except Exception as e:
        print(f"❌ Failed to load full AI mode: {e}")
        import traceback
        traceback.print_exc()
        
        # If AI dependencies are missing, this is a build error
        if not ai_available:
            print("💥 CRITICAL: AI dependencies missing in production build!")
            print("🔧 Check Dockerfile and requirements-ai-stable.txt")
            # Still try basic mode but warn loudly
            
        print("🔄 Falling back to basic mode (NOT RECOMMENDED FOR PRODUCTION)")
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