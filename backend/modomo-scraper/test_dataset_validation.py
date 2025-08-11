#!/usr/bin/env python3
"""
Test script to validate HuggingFace dataset boundaries
"""
import asyncio
import aiohttp
import urllib.parse

async def test_dataset_validation(dataset: str, offset: int, limit: int):
    """Test the improved dataset validation logic"""
    print(f"\n🧪 Testing dataset: {dataset} (offset={offset}, limit={limit})")
    
    dataset_encoded = urllib.parse.quote(dataset, safe='')
    
    async with aiohttp.ClientSession() as session:
        # Step 1: Get dataset info
        info_url = f"https://datasets-server.huggingface.co/info?dataset={dataset_encoded}"
        print(f"📋 Checking dataset info: {info_url}")
        
        try:
            async with session.get(info_url) as info_response:
                if info_response.status == 200:
                    info_data = await info_response.json()
                    total_rows = info_data.get("dataset_info", {}).get("splits", {}).get("train", {}).get("num_examples", 0)
                    print(f"📊 Dataset has {total_rows} total rows")
                    
                    # Validate boundaries
                    max_available = max(0, total_rows - offset)
                    adjusted_limit = min(limit, max_available)
                    
                    if adjusted_limit <= 0:
                        print(f"❌ ERROR: Offset {offset} exceeds dataset size {total_rows}")
                        return False
                    
                    if adjusted_limit < limit:
                        print(f"⚠️  Adjusted limit from {limit} to {adjusted_limit} (max available from offset)")
                    
                    # Test actual data request
                    dataset_url = f"https://datasets-server.huggingface.co/rows?dataset={dataset_encoded}&config=default&split=train&offset={offset}&length={adjusted_limit}"
                    print(f"🔍 Testing data request: offset={offset}, limit={adjusted_limit}")
                    
                    async with session.get(dataset_url) as response:
                        if response.status == 200:
                            data = await response.json()
                            rows = data.get("rows", [])
                            print(f"✅ SUCCESS: Fetched {len(rows)} rows")
                            return True
                        else:
                            print(f"❌ ERROR: Data request failed with status {response.status}")
                            return False
                else:
                    print(f"⚠️  Could not get dataset info (status {info_response.status})")
                    return False
        except Exception as e:
            print(f"❌ ERROR: {e}")
            return False

async def main():
    """Test the problematic cases from the logs"""
    test_cases = [
        # These were failing with 422 errors
        ("sk2003/houzzdata", 300, 150),  # Should work (1600 total rows)
        ("yubinnii/ikea_atmosphere", 0, 150),  # Should adjust limit (20 total rows)
        
        # Edge cases
        ("sk2003/houzzdata", 1500, 200),  # Should adjust limit
        ("sk2003/houzzdata", 1700, 50),   # Should fail (offset too high)
        
        # Valid cases
        ("sk2003/houzzdata", 0, 50),      # Should work fine
        ("yubinnii/ikea_atmosphere", 0, 10),  # Should work fine
    ]
    
    print("🧪 Testing HuggingFace Dataset Validation Logic")
    print("=" * 60)
    
    results = []
    for dataset, offset, limit in test_cases:
        success = await test_dataset_validation(dataset, offset, limit)
        results.append((dataset, offset, limit, success))
        await asyncio.sleep(1)  # Be nice to the API
    
    print("\n📋 Test Results Summary:")
    print("=" * 60)
    for dataset, offset, limit, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {dataset} (offset={offset}, limit={limit})")

if __name__ == "__main__":
    asyncio.run(main())