#!/usr/bin/env python3
"""
Script to resubmit pending jobs from database to Celery workers
"""
import requests
import json

def resubmit_pending_jobs():
    """Fetch pending jobs and resubmit them to Celery"""
    
    # Get pending jobs from API
    print("🔍 Fetching pending jobs from API...")
    response = requests.get("https://ovalay-recruitment-production.up.railway.app/jobs/active")
    
    if response.status_code != 200:
        print(f"❌ Failed to fetch jobs: {response.status_code}")
        return
    
    jobs = response.json()
    pending_jobs = [job for job in jobs if job["status"] == "pending"]
    
    print(f"📋 Found {len(pending_jobs)} pending jobs to resubmit")
    
    if not pending_jobs:
        print("✅ No pending jobs to resubmit")
        return
    
    # Resubmit each job
    success_count = 0
    failed_count = 0
    
    for job in pending_jobs:
        job_id = job["job_id"]
        job_type = job["job_type"]
        
        print(f"🔄 Resubmitting {job_type} job {job_id[:8]}...")
        
        # Determine the correct endpoint based on job type
        if job_type == "import":
            dataset = job.get("dataset", "unknown")
            offset = job.get("offset", "0")
            include_detection = job.get("include_detection", "False")
            total = job.get("total", "50")
            
            endpoint = f"https://ovalay-recruitment-production.up.railway.app/import/huggingface/{dataset}?limit={total}&offset={offset}&include_detection={include_detection}&job_id={job_id}"
        
        elif job_type == "scenes":
            total = job.get("total", "100")
            force_refresh = job.get("force_refresh", "False")
            
            endpoint = f"https://ovalay-recruitment-production.up.railway.app/scenes/scrape?count={total}&force_refresh={force_refresh}&job_id={job_id}"
        
        else:
            print(f"⚠️  Unknown job type: {job_type}, skipping...")
            failed_count += 1
            continue
        
        try:
            # Submit the job
            resubmit_response = requests.post(endpoint)
            
            if resubmit_response.status_code in [200, 202]:
                print(f"✅ Successfully resubmitted {job_id[:8]}")
                success_count += 1
            else:
                print(f"❌ Failed to resubmit {job_id[:8]}: {resubmit_response.status_code}")
                failed_count += 1
                
        except Exception as e:
            print(f"❌ Error resubmitting {job_id[:8]}: {e}")
            failed_count += 1
    
    print(f"\n📊 Resubmission Summary:")
    print(f"   ✅ Success: {success_count}")
    print(f"   ❌ Failed: {failed_count}")
    print(f"   📋 Total: {len(pending_jobs)}")

if __name__ == "__main__":
    resubmit_pending_jobs()