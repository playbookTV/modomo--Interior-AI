"""
Modomo Dataset Scraper - Production Entry Point  
This file automatically detects if running with minimal or full dependencies
Updated with real Houzz scraping functionality + Playwright browsers
"""

import os
import sys

# Check if we have AI dependencies available
ai_mode_enabled = False

try:
    # Test critical AI imports
    import torch
    torch.zeros(1)  # Test torch initialization
    
    from transformers import __version__ as transformers_version
    import structlog
    
    # If we get here, AI dependencies are working
    ai_mode_enabled = True
    print(f"🤖 AI dependencies detected (torch: {torch.__version__}, transformers: {transformers_version}) - starting full mode")
    
except ImportError as e:
    print(f"💡 AI dependencies not available ({e}) - starting basic mode")
    print("   Fixed huggingface_hub compatibility - should work on next deploy")
except Exception as e:
    print(f"⚠️ AI dependency issue ({e}) - falling back to basic mode")

# Import appropriate app
if ai_mode_enabled:
    try:
        from main_full import app
        print("✅ Full AI mode active")
    except Exception as e:
        print(f"❌ Failed to load AI mode ({e}) - using basic mode")
        from main_basic import app
else:
    from main_basic import app
    print("✅ Basic mode active")

# This allows Railway to import the app
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)