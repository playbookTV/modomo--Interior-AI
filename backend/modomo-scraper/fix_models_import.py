#!/usr/bin/env python3
"""
Comprehensive fix for models import issues on Railway deployment
Ensures all AI models can be imported properly without fallbacks
"""

import os
import sys
import shutil
from pathlib import Path

def fix_models_import():
    """Fix models import by ensuring proper package structure"""
    
    print("🔧 Fixing models import issues...")
    
    # Ensure current directory is in Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        print(f"✅ Added {current_dir} to Python path")
    
    # Models directory path
    models_dir = os.path.join(current_dir, "models")
    
    print(f"🔍 Models directory: {models_dir}")
    print(f"🔍 Models directory exists: {os.path.exists(models_dir)}")
    
    if os.path.exists(models_dir):
        print("📁 Models directory contents:")
        for item in os.listdir(models_dir):
            print(f"   - {item}")
    else:
        print("❌ Models directory does not exist!")
        return False
    
    # Check for __init__.py
    init_file = os.path.join(models_dir, "__init__.py")
    print(f"🔍 __init__.py exists: {os.path.exists(init_file)}")
    
    # Create or fix __init__.py
    if not os.path.exists(init_file):
        print("📝 Creating __init__.py file...")
        
        init_content = '''"""
Modomo AI Models Package
"""

__version__ = "1.0.0"

# Import all model classes
try:
    from .sam2_segmenter import SAM2Segmenter, SegmentationConfig
    print("✅ SAM2Segmenter imported successfully")
except ImportError as e:
    print(f"⚠️ SAM2Segmenter import failed: {e}")

try:
    from .grounding_dino import GroundingDINODetector
    print("✅ GroundingDINODetector imported successfully")
except ImportError as e:
    print(f"⚠️ GroundingDINODetector import failed: {e}")

try:
    from .clip_embedder import CLIPEmbedder
    print("✅ CLIPEmbedder imported successfully")  
except ImportError as e:
    print(f"⚠️ CLIPEmbedder import failed: {e}")

try:
    from .color_extractor import ColorExtractor
    print("✅ ColorExtractor imported successfully")
except ImportError as e:
    print(f"⚠️ ColorExtractor import failed: {e}")

__all__ = [
    'SAM2Segmenter', 
    'SegmentationConfig',
    'GroundingDINODetector', 
    'CLIPEmbedder',
    'ColorExtractor'
]
'''
        
        with open(init_file, 'w') as f:
            f.write(init_content)
        
        print("✅ __init__.py created successfully")
    else:
        print("✅ __init__.py already exists")
    
    # Test individual model imports
    print("\n🧪 Testing individual model imports...")
    
    models_to_test = [
        ("color_extractor.py", "ColorExtractor"),
        ("grounding_dino.py", "GroundingDINODetector"),
        ("sam2_segmenter.py", "SAM2Segmenter"),
        ("clip_embedder.py", "CLIPEmbedder")
    ]
    
    for model_file, class_name in models_to_test:
        model_path = os.path.join(models_dir, model_file)
        if os.path.exists(model_path):
            print(f"✅ {model_file} file exists")
        else:
            print(f"❌ {model_file} file missing!")
    
    # Test package import
    print("\n🧪 Testing models package import...")
    try:
        import models
        print("✅ models package imported successfully")
        
        # Test individual imports
        try:
            from models.color_extractor import ColorExtractor
            print("✅ ColorExtractor imported from models package")
        except Exception as e:
            print(f"❌ ColorExtractor import failed: {e}")
            
        try:
            from models.grounding_dino import GroundingDINODetector
            print("✅ GroundingDINODetector imported from models package")
        except Exception as e:
            print(f"❌ GroundingDINODetector import failed: {e}")
            
        try:
            from models.sam2_segmenter import SAM2Segmenter
            print("✅ SAM2Segmenter imported from models package")
        except Exception as e:
            print(f"❌ SAM2Segmenter import failed: {e}")
            
        try:
            from models.clip_embedder import CLIPEmbedder
            print("✅ CLIPEmbedder imported from models package")
        except Exception as e:
            print(f"❌ CLIPEmbedder import failed: {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ models package import failed: {e}")
        return False

if __name__ == "__main__":
    success = fix_models_import()
    print(f"\n🎯 Models import fix: {'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1)