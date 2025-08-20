# Router Directory Cleanup

## Problem
We had both `routers/` and `routers_simple/` directories, which was confusing and redundant.

## Root Cause
The `routers_simple/` was created as a fallback mechanism for when full routers weren't available. However, this created:
- **Confusion** - which routers are actually used?
- **Maintenance burden** - two sets of routers to maintain
- **Circular dependencies** - simple routers had import issues
- **Inconsistent functionality** - simple routers were minimal/incomplete

## Solution
✅ **Removed `routers_simple/` directory completely**
✅ **Updated app factory to use only comprehensive routers**
✅ **Simplified router registration logic**

## Current Structure
```
backend/modomo-scraper/
├── routers/                    # ✅ ONLY comprehensive routers
│   ├── jobs.py                # Job management
│   ├── detection.py           # AI detection  
│   ├── scraping.py            # Scene scraping
│   ├── classification.py      # Classification
│   ├── export.py              # Dataset export
│   ├── analytics.py           # Analytics
│   ├── admin.py               # Admin operations
│   ├── sync_monitor.py        # Sync monitoring
│   ├── color_endpoints.py     # Color processing
│   ├── review_endpoints.py    # Review queue
│   ├── dataset_endpoints.py   # Dataset operations
│   ├── mask_endpoints.py      # R2 mask serving
│   ├── advanced_ai_endpoints.py  # Full AI pipeline
│   └── admin_utilities.py     # Admin utilities
└── core/
    └── app_factory.py         # ✅ Clean router registration
```

## Benefits
- ✅ **Single source of truth** - only one set of routers
- ✅ **No confusion** - clear which routers are used
- ✅ **Easier maintenance** - one codebase to maintain
- ✅ **Consistent functionality** - all routers are comprehensive
- ✅ **No circular dependencies** - proper dependency injection

## Usage
```python
# App factory now simply registers all routers
from routers.jobs import router as jobs_router
from routers.detection import router as detection_router
# ... etc

app.include_router(jobs_router)
app.include_router(detection_router)
# ... etc
```

The system is now cleaner and easier to understand!