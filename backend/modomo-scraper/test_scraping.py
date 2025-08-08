#!/usr/bin/env python3
"""
Test Houzz scraping functionality locally
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crawlers.houzz_crawler import HouzzCrawler

async def test_houzz_scraping():
    """Test the Houzz crawler with a small sample"""
    print("🕷️ Testing Houzz scraping...")
    
    try:
        crawler = HouzzCrawler()
        
        # Test scraping 2 living room scenes
        scenes = await crawler.scrape_scenes(limit=2, room_types=['living-room'])
        
        print(f"✅ Scraped {len(scenes)} scenes:")
        
        for i, scene in enumerate(scenes, 1):
            print(f"\n{i}. Scene ID: {scene.houzz_id}")
            print(f"   Room Type: {scene.room_type}")
            print(f"   Image URL: {scene.image_url[:100]}...")
            print(f"   Style Tags: {scene.style_tags}")
            print(f"   Color Tags: {scene.color_tags}")
            if scene.project_url:
                print(f"   Project: {scene.project_url[:60]}...")
        
        await crawler.close()
        
        return len(scenes) > 0
        
    except Exception as e:
        print(f"❌ Scraping test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_houzz_scraping())
    
    if success:
        print("\n🎉 Houzz scraping test PASSED!")
        print("The scraping functionality is working correctly.")
    else:
        print("\n❌ Houzz scraping test FAILED!")
        print("Check dependencies and network connectivity.")
        
    sys.exit(0 if success else 1)