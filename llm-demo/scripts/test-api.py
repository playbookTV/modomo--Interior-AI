#!/usr/bin/env python3
"""
Test script for the ReRoom AI service API endpoints
"""

import requests
import json
import time
import base64
import os
from pathlib import Path

# Configuration
API_BASE_URL = "https://reroom-production-dcb0.up.railway.app:6969"
TEST_IMAGES_DIR = Path("../examples")
STYLES = ["Modern", "Scandinavian", "Industrial", "Bohemian"]

class AIServiceTester:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        
    def check_health(self):
        """Check if the AI service is running"""
        try:
            response = self.session.get(f"{self.base_url}/health/detailed")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Health check failed: {e}")
            return None
    
    def test_single_makeover(self, image_path: str, style: str = "Modern"):
        """Test single room makeover endpoint"""
        print(f"Testing single makeover: {image_path} -> {style}")
        
        # For demo, we'll use a mock URL since we can't actually upload
        photo_url = f"https://demo.reroom.app/test/{Path(image_path).name}"
        
        payload = {
            "photo_url": photo_url,
            "photo_id": f"test_{int(time.time())}",
            "style_preference": style,
            "budget_range": "medium"
        }
        
        try:
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/makeover", json=payload)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success in {duration:.2f}s")
                print(f"   Objects detected: {len(result.get('transformation', {}).get('detected_objects', []))}")
                print(f"   Products suggested: {len(result.get('transformation', {}).get('suggested_products', []))}")
                print(f"   Processing time: {result.get('processing_time_ms', 0)}ms")
                return result
            else:
                print(f"❌ Failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def test_enhanced_makeover(self, image_path: str, style: str = "Modern"):
        """Test enhanced makeover with custom parameters"""
        print(f"Testing enhanced makeover: {image_path} -> {style}")
        
        photo_url = f"https://demo.reroom.app/test/{Path(image_path).name}"
        
        payload = {
            "photo_url": photo_url,
            "photo_id": f"enhanced_test_{int(time.time())}",
            "style_preference": style,
            "budget_range": "high",
            "use_multi_controlnet": True,
            "quality_level": "high",
            "strength": 0.8,
            "guidance_scale": 8.0,
            "num_inference_steps": 25
        }
        
        try:
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/makeover/enhanced", json=payload)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Enhanced success in {duration:.2f}s")
                return result
            else:
                print(f"❌ Enhanced failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Enhanced error: {e}")
            return None
    
    def test_style_comparison(self, image_path: str):
        """Test style comparison endpoint"""
        print(f"Testing style comparison: {image_path}")
        
        # Convert image to base64 (mock)
        # In real implementation, you'd read the actual file
        mock_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        payload = {
            "image_data": mock_base64,
            "styles": STYLES[:3],  # Test with 3 styles
            "include_original": True
        }
        
        try:
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/analyze/style-comparison", json=payload)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Style comparison success in {duration:.2f}s")
                print(f"   Styles processed: {result.get('styles_processed', 0)}")
                return result
            else:
                print(f"❌ Style comparison failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Style comparison error: {e}")
            return None
    
    def test_monitoring_endpoints(self):
        """Test monitoring and metrics endpoints"""
        print("Testing monitoring endpoints...")
        
        endpoints = [
            "/monitoring/status",
            "/monitoring/metrics", 
            "/metrics/performance",
            "/analytics/usage",
            "/cache/redis",
            "/models/cache"
        ]
        
        results = {}
        for endpoint in endpoints:
            try:
                response = self.session.get(f"{self.base_url}{endpoint}")
                if response.status_code == 200:
                    results[endpoint] = "✅ OK"
                else:
                    results[endpoint] = f"❌ {response.status_code}"
            except Exception as e:
                results[endpoint] = f"❌ Error: {e}"
        
        for endpoint, status in results.items():
            print(f"  {endpoint}: {status}")
        
        return results
    
    def run_comprehensive_test(self):
        """Run all tests"""
        print("🧪 Starting comprehensive AI service tests...\n")
        
        # 1. Health check
        print("1. Health Check")
        health = self.check_health()
        if not health:
            print("❌ Service is not running. Please start the AI service first.")
            return
        
        print(f"✅ Service is {health.get('status', 'unknown')}")
        print()
        
        # 2. Test single makeover
        print("2. Single Makeover Test")
        makeover_result = self.test_single_makeover("sample_room.jpg", "Modern")
        print()
        
        # 3. Test enhanced makeover
        print("3. Enhanced Makeover Test") 
        enhanced_result = self.test_enhanced_makeover("sample_room.jpg", "Scandinavian")
        print()
        
        # 4. Test style comparison
        print("4. Style Comparison Test")
        comparison_result = self.test_style_comparison("sample_room.jpg")
        print()
        
        # 5. Test monitoring endpoints
        print("5. Monitoring Endpoints Test")
        monitoring_results = self.test_monitoring_endpoints()
        print()
        
        # Summary
        print("🎯 Test Summary:")
        print(f"   Single makeover: {'✅' if makeover_result else '❌'}")
        print(f"   Enhanced makeover: {'✅' if enhanced_result else '❌'}")
        print(f"   Style comparison: {'✅' if comparison_result else '❌'}")
        print(f"   Monitoring endpoints: {sum(1 for v in monitoring_results.values() if '✅' in v)}/{len(monitoring_results)} working")

def main():
    print("ReRoom AI Service API Tester")
    print("=" * 40)
    
    tester = AIServiceTester()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main()