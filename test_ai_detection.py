#!/usr/bin/env python3
"""
Test script to verify Heroku → Railway AI detection flow
"""
import redis
import json
import time
import requests
from celery import Celery

# Configure Redis connection (same as Heroku worker)
REDIS_URL = "redis://default:3JhC7XW2s68Ol4RbT1EO@shuttle.proxy.rlwy.net:11071"

# Set up Celery app
celery_app = Celery(
    "modomo_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL
)

# Test image URL (a simple room image)
TEST_IMAGE_URL = "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1500&q=80"

def test_railway_health():
    """Test if Railway service is healthy"""
    try:
        response = requests.get("https://ovalay-recruitment-production.up.railway.app/health", timeout=10)
        print(f"✅ Railway health check: {response.status_code}")
        print(f"📊 Services: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Railway health check failed: {e}")
        return False

def test_ai_detection_task():
    """Queue an AI detection task and monitor results"""
    print("🚀 Testing AI detection flow...")
    
    # First check Railway health
    if not test_railway_health():
        print("❌ Railway service not available, cannot proceed")
        return False
    
    # Queue AI detection task
    try:
        from tasks.detection_tasks import run_detection_pipeline
        
        # Queue the task
        result = run_detection_pipeline.delay(
            job_id="test_job_001",
            image_url=TEST_IMAGE_URL,
            scene_id="test_scene_001"
        )
        
        print(f"✅ Task queued with ID: {result.id}")
        print(f"📍 Task state: {result.state}")
        
        # Poll for results
        print("⏳ Waiting for task completion...")
        timeout = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if result.ready():
                if result.successful():
                    task_result = result.get()
                    print(f"✅ Task completed successfully!")
                    print(f"📊 Result: {json.dumps(task_result, indent=2)}")
                    return True
                else:
                    print(f"❌ Task failed: {result.result}")
                    return False
            else:
                print(f"⏳ Still processing... ({result.state})")
                time.sleep(10)
        
        print("⏰ Task timed out after 5 minutes")
        return False
        
    except Exception as e:
        print(f"❌ Failed to queue task: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Heroku → Railway AI detection flow")
    print("=" * 50)
    
    success = test_ai_detection_task()
    
    if success:
        print("\n🎉 AI detection flow working correctly!")
    else:
        print("\n💥 AI detection flow has issues")