"""
Initialize Modomo Database Schema from Railway Console
Run this script in Railway console after deployment to set up the database
"""

import asyncio
import os
import asyncpg

async def init_database():
    print("🗄️ Initializing Modomo database schema...")
    
    # Get database URL from environment
    db_url = os.getenv("DATABASE_URL_CLOUD")
    if not db_url:
        print("❌ DATABASE_URL_CLOUD not found in environment")
        return False
    
    try:
        # Connect to database
        print("🔗 Connecting to database...")
        conn = await asyncpg.connect(db_url)
        print("✅ Connected successfully!")
        
        # Read and execute schema
        print("📋 Loading schema file...")
        with open('database/schema.sql', 'r') as f:
            schema_sql = f.read()
        
        print("⚡ Executing schema...")
        await conn.execute(schema_sql)
        
        # Verify tables
        tables = await conn.fetch("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('scenes', 'products', 'detected_objects', 'scraping_jobs', 'dataset_exports')
            ORDER BY table_name;
        """)
        
        print(f"✅ Created {len(tables)} tables:")
        for table in tables:
            print(f"   - {table['table_name']}")
        
        await conn.close()
        print("🎉 Database schema initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(init_database())
    if success:
        print("✅ Ready to start processing data!")
    else:
        print("❌ Database initialization failed")