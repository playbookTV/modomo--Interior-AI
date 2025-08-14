#!/usr/bin/env python3
"""
Test script to debug module import issues
"""

import sys
import os

print("=== Python Import Debug ===")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")
print()

# Test basic imports
print("=== Testing Basic Imports ===")
try:
    import numpy as np
    print("✅ numpy imported successfully")
except ImportError as e:
    print(f"❌ numpy import failed: {e}")

try:
    import torch
    print("✅ torch imported successfully")
except ImportError as e:
    print(f"❌ torch import failed: {e}")

print()

# Test models directory structure
print("=== Testing Models Directory ===")
models_dir = os.path.join(os.getcwd(), "models")
print(f"Models directory: {models_dir}")
print(f"Models directory exists: {os.path.exists(models_dir)}")

if os.path.exists(models_dir):
    print("Models directory contents:")
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        print(f"  - {item} ({'file' if os.path.isfile(item_path) else 'directory'})")

print()

# Test __init__.py
init_file = os.path.join(models_dir, "__init__.py")
print(f"__init__.py exists: {os.path.exists(init_file)}")

print()

# Test individual model imports
print("=== Testing Model Imports ===")
models_to_test = [
    "models.color_extractor",
    "models.grounding_dino", 
    "models.sam2_segmenter",
    "models.clip_embedder"
]

for model_name in models_to_test:
    try:
        module = __import__(model_name, fromlist=[''])
        print(f"✅ {model_name} imported successfully")
    except ImportError as e:
        print(f"❌ {model_name} import failed: {e}")
    except Exception as e:
        print(f"⚠️ {model_name} import error: {e}")

print()
print("=== Import Test Complete ===")