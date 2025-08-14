#!/usr/bin/env python3
"""
Run the mask_url column migration for detected_objects table
"""

import os
import asyncio
import asyncpg
from pathlib import Path

async def run_migration():
    """Execute the migration script"""
    
    # Get database URL from environment
    database_url = os.getenv("DATABASE_URL_CLOUD") or os.getenv("DATABASE_URL")
    
    if not database_url:
        print("❌ No database URL found in environment variables")
        print("   Set DATABASE_URL_CLOUD or DATABASE_URL")
        return False
    
    try:
        # Connect to database
        print(f"🔗 Connecting to database...")
        conn = await asyncpg.connect(database_url)
        
        # Read migration script
        script_path = Path(__file__).parent / "add_mask_url_column.sql"
        migration_sql = script_path.read_text()
        
        print(f"📜 Executing migration script...")
        
        # Execute migration
        result = await conn.execute(migration_sql)
        print(f"✅ Migration completed successfully")
        
        # Verify the change
        print(f"🔍 Verifying column exists...")
        columns = await conn.fetch("""
            SELECT column_name, data_type, is_nullable 
            FROM information_schema.columns 
            WHERE table_name = 'detected_objects' 
            AND column_name IN ('mask_url', 'mask_r2_key')
            ORDER BY column_name
        """)
        
        print(f"📊 Detected_objects table columns:")
        for col in columns:
            print(f"   - {col['column_name']}: {col['data_type']} (nullable: {col['is_nullable']})")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_migration())
    if success:
        print(f"\n🎉 Migration completed! You can now restart the modomo-scraper service.")
    else:
        print(f"\n💥 Migration failed. Please check the error above.")