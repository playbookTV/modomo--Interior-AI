#!/usr/bin/env python3
"""
Initialize Supabase Database Schema for Modomo
Runs the schema.sql file against the Supabase database
"""

import os
import asyncio
import asyncpg
from pathlib import Path

async def main():
    # Get Supabase connection string from environment
    DATABASE_URL = os.getenv("DATABASE_URL_CLOUD")
    if not DATABASE_URL:
        print("❌ DATABASE_URL_CLOUD not found in environment")
        print("   Make sure to set your Supabase connection string")
        return
    
    # Read schema file
    schema_path = Path(__file__).parent.parent / "backend" / "modomo-scraper" / "database" / "schema.sql"
    if not schema_path.exists():
        print(f"❌ Schema file not found: {schema_path}")
        return
    
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
    
    print("🗄️ Connecting to Supabase database...")
    
    try:
        # Connect to database
        conn = await asyncpg.connect(DATABASE_URL)
        print("✅ Connected successfully!")
        
        # Check if pgvector extension is available
        print("🔍 Checking for pgvector extension...")
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            print("✅ pgvector extension available")
        except Exception as e:
            print(f"⚠️ pgvector extension not available: {e}")
            print("   CLIP embeddings will be stored as JSON")
        
        # Execute schema
        print("📋 Executing database schema...")
        await conn.execute(schema_sql)
        print("✅ Schema created successfully!")
        
        # Verify tables were created
        tables = await conn.fetch("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('scenes', 'products', 'detected_objects', 'scraping_jobs', 'dataset_exports')
            ORDER BY table_name;
        """)
        
        print(f"✅ Created {len(tables)} tables:")
        for table in tables:
            print(f"   - {table['table_name']}")
        
        # Check views
        views = await conn.fetch("""
            SELECT table_name FROM information_schema.views 
            WHERE table_schema = 'public' 
            AND table_name IN ('dataset_stats', 'category_stats')
            ORDER BY table_name;
        """)
        
        print(f"✅ Created {len(views)} views:")
        for view in views:
            print(f"   - {view['table_name']}")
            
        await conn.close()
        print("")
        print("🎉 Modomo database schema initialized successfully!")
        print("🔗 You can now connect your Railway deployment to this database")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return

if __name__ == "__main__":
    asyncio.run(main())