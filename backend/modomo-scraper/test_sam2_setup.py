#!/usr/bin/env python3
"""
Test script to verify SAM2 setup works correctly
"""
import sys
import os

def test_sam2_imports():
    """Test if SAM2 can be imported"""
    print("🧪 Testing SAM2 imports...")
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("✅ SAM2 imports successful")
        return True
    except ImportError as e:
        print(f"❌ SAM2 import failed: {e}")
        return False

def test_checkpoint_availability():
    """Test if SAM2 checkpoints are available"""
    print("🧪 Testing checkpoint availability...")
    
    checkpoint_paths = [
        "/app/checkpoints/sam2_hiera_large.pt",
        "/app/checkpoints/sam2_hiera_base_plus.pt", 
        "checkpoints/sam2_hiera_large.pt",
    ]
    
    for path in checkpoint_paths:
        if os.path.exists(path):
            size = os.path.getsize(path) // (1024 * 1024)  # MB
            print(f"✅ Found checkpoint: {path} ({size}MB)")
            return True
    
    print("❌ No checkpoints found")
    return False

def test_sam2_segmenter():
    """Test SAM2Segmenter initialization"""
    print("🧪 Testing SAM2Segmenter...")
    
    try:
        sys.path.append('.')
        from models.sam2_segmenter import SAM2Segmenter, SegmentationConfig
        
        config = SegmentationConfig(device='cpu')
        segmenter = SAM2Segmenter(config=config)
        
        info = segmenter.get_model_info()
        print(f"📊 Segmenter info: {info}")
        
        if info.get('sam2_available') and info.get('sam2_loaded'):
            print("🔥 SAM2 segmenter working with REAL SAM2!")
            return True
        else:
            print("⚠️ SAM2 segmenter using fallback mode")
            return False
            
    except Exception as e:
        print(f"❌ SAM2Segmenter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Testing SAM2 setup for Railway deployment...")
    print("=" * 50)
    
    tests = [
        test_sam2_imports,
        test_checkpoint_availability, 
        test_sam2_segmenter
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    print("=" * 50)
    if all(results):
        print("🎉 ALL TESTS PASSED - SAM2 is ready!")
        return 0
    else:
        print("❌ Some tests failed - check logs above")
        return 1

if __name__ == "__main__":
    sys.exit(main())