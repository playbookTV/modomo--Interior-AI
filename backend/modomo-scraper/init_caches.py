#!/usr/bin/env python3
"""
Railway Volume Cache Initialization - Phase 2
Manages Hugging Face models, Playwright browsers, and other cached components
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger('init_caches')

class CacheInitializer:
    """Initialize and manage cached components on Railway Volume"""
    
    def __init__(self):
        self.volume_base = "/app/cache_volume"  # Railway Volume mount point
        self.cache_dirs = {
            "hf_cache": f"{self.volume_base}/huggingface",
            "spacy_data": f"{self.volume_base}/spacy",
            "playwright": f"{self.volume_base}/playwright", 
            "pip_cache": f"{self.volume_base}/pip",
            "nltk_data": f"{self.volume_base}/nltk",
            "matplotlib": f"{self.volume_base}/matplotlib"
        }
        
        # Set environment variables
        os.environ["TRANSFORMERS_CACHE"] = self.cache_dirs["hf_cache"]
        os.environ["HF_HOME"] = self.cache_dirs["hf_cache"]
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = f"{self.cache_dirs['hf_cache']}/sentence_transformers"
        os.environ["SPACY_DATA"] = self.cache_dirs["spacy_data"]
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = self.cache_dirs["playwright"]
        os.environ["PIP_CACHE_DIR"] = self.cache_dirs["pip_cache"]
        os.environ["NLTK_DATA"] = self.cache_dirs["nltk_data"]
        os.environ["MPLCONFIGDIR"] = self.cache_dirs["matplotlib"]
        
    def create_directories(self):
        """Create cache directories if they don't exist"""
        logger.info("🏗️ Creating cache directories...")
        
        for name, path in self.cache_dirs.items():
            Path(path).mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ {name}: {path}")
            
    def init_huggingface_cache(self):
        """Pre-download commonly used Hugging Face models"""
        logger.info("🤗 Initializing Hugging Face model cache...")
        
        models_to_cache = [
            # CLIP models
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14",
            
            # Sentence transformers
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            
            # DETR for object detection 
            "facebook/detr-resnet-50",
            
            # Additional transformers models
            "microsoft/DialoGPT-medium",
        ]
        
        try:
            from transformers import AutoModel, AutoTokenizer
            from sentence_transformers import SentenceTransformer
            
            for model_name in models_to_cache:
                if self._model_cached(model_name):
                    logger.info(f"✅ {model_name} already cached")
                    continue
                    
                logger.info(f"📦 Downloading {model_name}...")
                
                if "sentence-transformers" in model_name:
                    # Sentence transformer model
                    SentenceTransformer(model_name)
                else:
                    # Regular transformers model
                    AutoModel.from_pretrained(model_name)
                    AutoTokenizer.from_pretrained(model_name)
                    
                logger.info(f"✅ {model_name} cached successfully")
                
        except Exception as e:
            logger.warning(f"⚠️ Hugging Face cache initialization failed: {e}")
            logger.info("🔄 Will download models on first use")
            
    def _model_cached(self, model_name: str) -> bool:
        """Check if a model is already cached"""
        # Check if model directory exists in cache
        model_path = Path(self.cache_dirs["hf_cache"]) / "models--" / model_name.replace("/", "--")
        return model_path.exists() and any(model_path.iterdir())
        
    def init_playwright_cache(self):
        """Pre-install Playwright browsers"""
        logger.info("🎭 Initializing Playwright browser cache...")
        
        try:
            # Check if browsers are already installed
            browser_path = Path(self.cache_dirs["playwright"])
            if browser_path.exists() and any(browser_path.iterdir()):
                logger.info("✅ Playwright browsers already cached")
                return
                
            logger.info("📦 Installing Playwright browsers...")
            
            # Install browsers to volume
            result = subprocess.run([
                sys.executable, "-m", "playwright", "install", "chromium"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✅ Playwright browsers installed successfully")
            else:
                logger.warning(f"⚠️ Playwright browser installation warning: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.warning("⚠️ Playwright installation timed out - will retry on next deploy")
        except Exception as e:
            logger.warning(f"⚠️ Playwright cache initialization failed: {e}")
            
    def init_spacy_cache(self):
        """Pre-download spaCy language models"""
        logger.info("📝 Initializing spaCy language model cache...")
        
        try:
            import spacy
            
            # Check if model is already downloaded
            if self._spacy_model_exists("en_core_web_sm"):
                logger.info("✅ spaCy en_core_web_sm already cached")
                return
                
            logger.info("📦 Downloading spaCy language model...")
            
            result = subprocess.run([
                sys.executable, "-m", "spacy", "download", "en_core_web_sm", "--quiet"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                logger.info("✅ spaCy language model cached successfully")
            else:
                logger.warning(f"⚠️ spaCy model download warning: {result.stderr}")
                
        except Exception as e:
            logger.warning(f"⚠️ spaCy cache initialization failed: {e}")
            
    def _spacy_model_exists(self, model_name: str) -> bool:
        """Check if spaCy model exists"""
        try:
            import spacy
            spacy.load(model_name)
            return True
        except OSError:
            return False
            
    def init_nltk_cache(self):
        """Pre-download NLTK data"""
        logger.info("🔬 Initializing NLTK data cache...")
        
        try:
            import nltk
            
            # Download essential NLTK data
            nltk_downloads = [
                'punkt',
                'stopwords', 
                'wordnet',
                'averaged_perceptron_tagger'
            ]
            
            for dataset in nltk_downloads:
                try:
                    nltk.download(dataset, quiet=True, download_dir=self.cache_dirs["nltk_data"])
                    logger.info(f"✅ NLTK {dataset} cached")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to cache NLTK {dataset}: {e}")
                    
        except Exception as e:
            logger.warning(f"⚠️ NLTK cache initialization failed: {e}")
            
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {}
        
        for name, path in self.cache_dirs.items():
            path_obj = Path(path)
            if path_obj.exists():
                # Calculate directory size
                size = sum(f.stat().st_size for f in path_obj.rglob('*') if f.is_file())
                files = sum(1 for f in path_obj.rglob('*') if f.is_file())
                
                stats[name] = {
                    "exists": True,
                    "size_mb": round(size / (1024 * 1024), 2),
                    "files": files
                }
            else:
                stats[name] = {"exists": False, "size_mb": 0, "files": 0}
                
        return stats
        
    def initialize_all(self):
        """Initialize all caches"""
        logger.info("🚀 Starting Railway Volume cache initialization...")
        
        # Create directories
        self.create_directories()
        
        # Initialize caches in order of importance
        self.init_huggingface_cache()  # Highest impact
        self.init_playwright_cache()   # High impact  
        self.init_spacy_cache()        # Medium impact
        self.init_nltk_cache()         # Low impact
        
        # Print final stats
        stats = self.get_cache_stats()
        total_size = sum(s.get("size_mb", 0) for s in stats.values())
        
        logger.info("📊 Cache initialization complete!")
        logger.info(f"📦 Total cache size: {total_size:.1f} MB")
        
        for name, stat in stats.items():
            if stat["exists"]:
                logger.info(f"   {name}: {stat['size_mb']} MB ({stat['files']} files)")
                
        return stats

def main():
    """Main entry point"""
    initializer = CacheInitializer()
    return initializer.initialize_all()

if __name__ == "__main__":
    main()