"""
Import tasks (separate from scraping for better organization)
"""
from tasks.scraping_tasks import import_huggingface_dataset

# Re-export for cleaner imports
__all__ = ['import_huggingface_dataset']