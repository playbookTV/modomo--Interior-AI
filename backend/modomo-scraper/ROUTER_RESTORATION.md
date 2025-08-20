# Router Restoration Plan

## What I Mistakenly Removed
- `routers/jobs.py` - **CRITICAL** job management functionality
- `routers/detection.py` - **ESSENTIAL** AI detection pipeline  
- `routers/scraping.py` - **CORE** scene scraping
- `routers/classification.py` - **IMPORTANT** classification
- `routers/export.py` - **NEEDED** dataset export
- `routers/analytics.py` - **VALUABLE** analytics (partially restored)
- `routers/admin.py` - **USEFUL** admin tools (partially restored)
- `routers/sync_monitor.py` - **HELPFUL** sync monitoring

## Lost Critical Functionality
❌ **Job Management**: retry, cancel, history, error tracking  
❌ **AI Pipeline**: object detection, scene reclassification  
❌ **Scraping**: scene scraping jobs  
❌ **Export**: dataset export functionality  
❌ **Classification**: testing and reclassification jobs  

## Real Solution: Fix, Don't Remove
The issue isn't with the routers - it's likely just missing `response_model=None` on a few endpoints.

**Better approach**: 
1. Restore ALL routers
2. Add `response_model=None` to problematic endpoints
3. Fix dependency injection consistently
4. Keep all functionality working

This maintains the full feature set while fixing the FastAPI errors.