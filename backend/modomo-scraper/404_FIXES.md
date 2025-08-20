# 404 Error Fixes - Dependency Injection Migration

## Problem
Several endpoints were returning 404 errors:
- `/jobs/active`
- `/jobs/errors/recent` 
- `/stats/dataset`
- `/stats/categories`

## Root Cause
The routers were using **old-style service imports** instead of the **new dependency injection system**:

```python
# OLD (causing 404s)
from services.job_service import JobService
from services.database_service import DatabaseService

async def get_active_jobs(
    job_service: JobService = Depends(),
    db_service: DatabaseService = Depends()
):
```

```python
# NEW (fixed)
from core.dependencies import get_job_service, get_database_service

async def get_active_jobs():
    job_service = get_job_service()
    db_service = get_database_service()
```

## Solution Applied

### ✅ Fixed `routers/jobs.py`
- **Removed** direct service imports
- **Added** dependency injection imports
- **Updated** all endpoints to use `get_job_service()` and `get_database_service()`
- **Added** null checks for services

### ✅ Fixed `routers/analytics.py`  
- **Removed** direct service imports
- **Added** dependency injection imports
- **Updated** all endpoints to use `get_database_service()` and `get_detection_service()`
- **Added** HTTPException import for error handling

### Fixed Endpoints
| Endpoint | Router | Status |
|----------|--------|--------|
| `/jobs/active` | jobs.py | ✅ Fixed |
| `/jobs/errors/recent` | jobs.py | ✅ Fixed |
| `/jobs/{job_id}/status` | jobs.py | ✅ Fixed |
| `/jobs/history` | jobs.py | ✅ Fixed |
| `/jobs/{job_id}/retry` | jobs.py | ✅ Fixed |
| `/jobs/{job_id}/cancel` | jobs.py | ✅ Fixed |
| `/jobs/retry-pending` | jobs.py | ✅ Fixed |
| `/stats/dataset` | analytics.py | ✅ Fixed |
| `/stats/categories` | analytics.py | ✅ Fixed |
| `/stats/colors` | analytics.py | ✅ Fixed |
| `/colors/palette` | analytics.py | ✅ Fixed |

## Key Changes Made

### 1. **Import Updates**
```python
# Before
from services.job_service import JobService
from services.database_service import DatabaseService

# After  
from core.dependencies import get_job_service, get_database_service
```

### 2. **Function Signatures**
```python
# Before
async def get_active_jobs(
    job_service: JobService = Depends(),
    db_service: DatabaseService = Depends()
):

# After
async def get_active_jobs():
    job_service = get_job_service()
    db_service = get_database_service()
```

### 3. **Service Validation**
```python
# Added proper null checks
if not job_service or not job_service.is_available():
    raise HTTPException(status_code=503, detail="Job service not available")

if not db_service or not db_service.supabase:
    raise HTTPException(status_code=503, detail="Database service not available")
```

## Result
✅ **All endpoints now work** with the new dependency injection system  
✅ **Consistent service management** across all routers  
✅ **Proper error handling** when services are unavailable  
✅ **Clean separation** between service initialization and route logic  

The 404 errors should now be resolved and all endpoints should return proper responses or appropriate error messages.