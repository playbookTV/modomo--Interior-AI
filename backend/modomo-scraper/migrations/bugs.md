ERROR:main_full:{"event": "\u274c Failed to create job 5d119b4e-a0db-403f-8d65-2d90dc9d95d4 in database: {'message': 'new row for relation \"scraping_jobs\" violates check constraint \"scraping_jobs_job_type_check\"', 'code': '23514', 'hint': None, 'details': 'Failing row contains (5d119b4e-a0db-403f-8d65-2d90dc9d95d4, processing, pending, 0, 100, 0, {\"limit\": 100, \"operation\": \"scene_reclassification\", \"force_red..., null, {}, 2025-08-16 02:36:54.447201+00, 2025-08-16 02:36:54.447228+00, null).'}", "logger": "main_full", "level": "error", "timestamp": "2025-08-16T02:36:54.587173Z"}

INFO:main_full:{"event": "\ud83d\udd04 Starting scene reclassification job 5d119b4e-a0db-403f-8d65-2d90dc9d95d4 for 100 scenes", "logger": "main_full", "level": "info", "timestamp": "2025-08-16T02:36:54.587637Z"}

INFO:httpx:HTTP Request: GET https://nyeeewcpexqsqfzzmvyu.supabase.co/rest/v1/scenes?select=scene_id%2Chouzz_id%2Cimage_url%2Cimage_type%2Cis_primary_object%2Cprimary_category%2Cmetadata&order=created_at.desc&limit=100 "HTTP/2 400 Bad Request"

ERROR:main_full:{"event": "\u274c Scene reclassification job 5d119b4e-a0db-403f-8d65-2d90dc9d95d4 failed: {'message': 'column scenes.image_type does not exist', 'code': '42703', 'hint': None, 'details': None}", "logger": "main_full", "level": "error", "timestamp": "2025-08-16T02:36:54.646859Z"}

INFO:     100.64.0.9:57560 - "GET /jobs/5d119b4e-a0db-403f-8d65-2d90dc9d95d4/status HTTP/1.1" 500 Internal Server Error

INFO:     100.64.0.9:57560 - "GET /jobs/5d119b4e-a0db-403f-8d65-2d90dc9d95d4/status HTTP/1.1" 500 Internal Server Error


ERROR:main_full:{"event": "\u274c Failed to update job c53da30f-d212-438b-b167-111f22e10ecd in database: {'message': \"Could not find the 'updated_at' column of 'scraping_jobs' in the schema cache\", 'code': 'PGRST204', 'hint': None, 'details': None}", "logger": "main_full", "level": "error", "timestamp": "2025-08-16T02:36:46.013646Z"}


🚀 Loading refactored architecture...

INFO:main_refactored:{"event": "\u2705 Using simplified routers", "logger": "main_refactored", "level": "info", "timestamp": "2025-08-16T02:33:31.330827Z"}

INFO:main_refactored:{"event": "\u2705 AI model classes imported successfully", "logger": "main_refactored", "level": "info", "timestamp": "2025-08-16T02:33:31.331740Z"}

INFO:main_refactored:{"event": "\u2705 Created masks directory: /app/cache_volume/masks", "logger": "main_refactored", "level": "info", "timestamp": "2025-08-16T02:33:31.398303Z"}

❌ Failed to load refactored architecture: name 'Query' is not defined

Traceback (most recent call last):

  File "/app/main_railway.py", line 120, in get_app

    from main_refactored import app

  File "/app/main_refactored.py", line 584, in <module>

    map_types: List[str] = Query(["depth", "edge"], description="Types of maps to generate"),

                           ^^^^^

NameError: name 'Query' is not defined

🔄 Falling back to original full AI mode...