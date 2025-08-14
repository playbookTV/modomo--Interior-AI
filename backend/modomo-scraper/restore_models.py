#!/usr/bin/env python3
"""
Restore models Python files after Railway Volume mount
This ensures the Python source code is available even if the volume overwrites it
"""

import os
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def restore_model_files():
    """Restore model Python files from backup location"""
    
    source_dir = "/app/models_src"
    target_dir = "/app/models"
    
    if not os.path.exists(source_dir):
        logger.error(f"❌ Source models directory not found: {source_dir}")
        return False
    
    logger.info("🔧 Restoring models Python files after Railway Volume mount...")
    
    try:
        # List what's currently in the models directory (should be volume content)
        if os.path.exists(target_dir):
            current_files = os.listdir(target_dir)
            logger.info(f"📁 Current models directory contains: {current_files}")
        
        # Copy Python files from source to target
        source_files = os.listdir(source_dir)
        logger.info(f"📁 Restoring files: {source_files}")
        
        for file_name in source_files:
            source_file = os.path.join(source_dir, file_name)
            target_file = os.path.join(target_dir, file_name)
            
            if os.path.isfile(source_file) and file_name.endswith('.py'):
                shutil.copy2(source_file, target_file)
                logger.info(f"✅ Restored: {file_name}")
            elif os.path.isdir(source_file):
                if not os.path.exists(target_file):
                    shutil.copytree(source_file, target_file)
                    logger.info(f"✅ Restored directory: {file_name}")
        
        # Verify restoration
        final_files = os.listdir(target_dir)
        logger.info(f"📁 Final models directory contains: {final_files}")
        
        # Check for key files
        key_files = ['color_extractor.py', 'grounding_dino.py', 'sam2_segmenter.py', 'clip_embedder.py']
        missing_files = [f for f in key_files if not os.path.exists(os.path.join(target_dir, f))]
        
        if missing_files:
            logger.error(f"❌ Still missing files: {missing_files}")
            return False
        else:
            logger.info("✅ All model Python files restored successfully!")
            return True
            
    except Exception as e:
        logger.error(f"❌ Failed to restore model files: {e}")
        return False

if __name__ == "__main__":
    success = restore_model_files()
    exit(0 if success else 1)