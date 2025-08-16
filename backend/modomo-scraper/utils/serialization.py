"""
JSON serialization utilities for handling NumPy types and other non-serializable objects
"""
try:
    import numpy as np
except ImportError:
    np = None
from typing import Any


def make_json_serializable(obj: Any) -> Any:
    """Convert NumPy types and other non-serializable types to JSON serializable types"""
    
    if np is None:
        # No numpy available, handle basic types
        if isinstance(obj, dict):
            return {key: make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(item) for item in obj]
        return obj
    
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, 'item') and callable(obj.item):  # Handle numpy scalars
        return obj.item()
    return obj