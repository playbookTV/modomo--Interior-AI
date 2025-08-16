# Modomo AI Architecture Status - August 2025

## Current Production Status

### Core Infrastructure ✅
- **Railway Hosting**: Production deployment at `ovalay-recruitment-production.up.railway.app`
- **Database**: Supabase PostgreSQL with proper credentials and connectivity
- **Storage**: Cloudflare R2 for masks, maps, and training data
- **Cache/Queue**: Redis for Celery task queue and caching
- **Monitoring**: Comprehensive health checks and service status tracking

### AI Processing Pipeline ✅
- **Object Detection**: YOLO + GroundingDINO with 200+ item taxonomy
- **Segmentation**: SAM2 with real segmentation masks and R2 storage
- **Depth Estimation**: Depth Anything V2 for monocular depth maps
- **Edge Detection**: CV2 Canny for edge map generation
- **Color Analysis**: Advanced color extraction with hex analysis
- **Embeddings**: CLIP for semantic similarity and search

### Background Processing ✅
- **Celery Framework**: v5.3.4 with Redis broker
- **Task Queues**: Multi-queue routing (ai_processing, scraping, color_processing, classification)
- **Monitoring**: Flower web interface for real-time task tracking
- **Scaling**: Worker specialization for different task types
- **Error Handling**: Automatic retry with exponential backoff

### Frontend Integration ✅
- **Review Dashboard**: React-based UI with real-time monitoring
- **Mask Visualization**: Canvas-based pixel processing for SAM2 overlays
- **Map Display**: Depth and edge map visualization with opacity controls
- **API Connectivity**: Robust error handling and timeout management
- **Proxy Configuration**: CORS resolution for cross-origin requests

## Key Dependencies (Latest)

### Core Framework
```txt
fastapi
uvicorn[standard]
pydantic
```

### Database & Storage
```txt
supabase==2.3.4
websockets==12.0
asyncpg
redis
boto3 (for R2 storage)
psycopg2-binary
```

### AI/ML Stack
```txt
torch==2.1.2+cpu
torchvision==0.16.2+cpu
transformers
sentence-transformers
git+https://github.com/openai/CLIP.git
git+https://github.com/badayvedat/Depth-Anything-V2.git@badayvedat-patch-1
ultralytics (YOLO)
opencv-python-headless
scikit-learn
```

### Task Queue
```txt
celery[redis]==5.3.4
flower==2.0.1
datasets==2.18.0
prometheus-client==0.20.0
```

### Web Scraping
```txt
playwright
aiohttp
beautifulsoup4
requests
```

## Architecture Patterns

### Modular Design
- **Services**: Clean separation (DatabaseService, DetectionService, JobService)
- **Routers**: Organized by domain (detection, jobs, scraping, review)
- **Configuration**: Centralized settings with environment validation
- **Error Handling**: Graceful degradation and comprehensive logging

### Scalability Features
- **Horizontal Scaling**: Multiple Celery workers across different queues
- **Resource Optimization**: CPU/GPU accommodation with dynamic device selection
- **Caching Strategy**: Redis for session management and task results
- **Storage Optimization**: R2 for large assets, database for metadata

### Monitoring & Observability
- **Health Endpoints**: Detailed service status reporting
- **Performance Metrics**: System specs and operation timing
- **Task Tracking**: Real-time progress updates via Celery
- **Error Logging**: Structured logging with contextual information

## Data Flow

### Scene Processing Pipeline
1. **Input**: Image URL from Houzz scraping or manual upload
2. **AI Processing**: 
   - YOLO object detection → bounding boxes
   - SAM2 segmentation → detailed masks
   - Depth Anything V2 → depth maps
   - CV2 Canny → edge maps
   - CLIP → semantic embeddings
   - Color extraction → hex color analysis
3. **Storage**: 
   - Metadata → Supabase PostgreSQL
   - Images/masks/maps → Cloudflare R2
   - Task status → Redis
4. **Frontend**: Real-time visualization with canvas-based overlays

### Background Task Flow
1. **API Request**: User initiates processing job
2. **Celery Queue**: Task routed to appropriate queue based on type
3. **Worker Processing**: Specialized worker picks up task
4. **Progress Updates**: Real-time status updates via Redis
5. **Result Storage**: Processed data saved to Supabase + R2
6. **Completion**: Frontend notified of completion status

## Security & Performance

### Security Measures
- **Environment Variables**: Sensitive credentials via Railway environment
- **CORS Configuration**: Proper cross-origin request handling
- **Input Validation**: Pydantic models for API request validation
- **Error Sanitization**: Sanitized error messages to prevent information leakage

### Performance Optimizations
- **Device Detection**: Automatic CPU/GPU selection based on availability
- **Memory Management**: Worker process recycling and memory limits
- **Caching Strategy**: Redis for frequently accessed data
- **Connection Pooling**: Database connection optimization
- **Asset Optimization**: Compressed storage and efficient retrieval

## Deployment Configuration

### Railway Settings
- **Build**: Dockerfile-based with optimized layer caching
- **Environment**: Production variables for Supabase, R2, Redis
- **Health Checks**: 300s timeout with proper endpoint monitoring
- **Restart Policy**: On failure with max 3 retries
- **Volume**: Persistent model storage at `/app/models`

### Service Dependencies
- **Database**: Supabase PostgreSQL (external)
- **Cache/Queue**: Railway Redis addon
- **Storage**: Cloudflare R2 (external)
- **Monitoring**: Flower UI (deployed alongside main app)

## API Endpoints Overview

### Core Operations
- `GET /health` - System health and service status
- `GET /scenes` - Scene listing with pagination
- `GET /objects` - Detected objects with filtering
- `POST /detect/process` - Trigger AI processing pipeline

### Background Jobs
- `GET /jobs/active` - Active Celery tasks
- `GET /jobs/celery/dashboard` - Comprehensive monitoring
- `POST /jobs/{job_id}/cancel` - Task cancellation

### Map Generation
- `POST /generate-maps/{scene_id}` - Single scene map generation
- `POST /generate-maps/batch` - Batch processing
- `GET /scenes/{scene_id}/maps` - Map availability check

### Review & Management
- `GET /review/queue` - Scenes pending review
- `POST /review/approve/{scene_id}` - Scene approval
- `POST /review/update` - Batch object updates

## Next Steps & Roadmap

### Immediate (Post-Deployment)
1. Verify database connectivity (`"database": true`)
2. Test Celery worker functionality
3. Validate frontend-backend communication
4. Monitor task queue performance

### Short Term
1. Scale Celery workers based on load
2. Implement queue length monitoring
3. Add performance alerts and notifications
4. Optimize batch processing workflows

### Medium Term
1. Enhanced error recovery mechanisms
2. Advanced analytics and reporting
3. ML model performance optimization
4. Automated dataset management

This architecture provides a robust, scalable foundation for AI-powered interior design analysis with comprehensive background processing capabilities.