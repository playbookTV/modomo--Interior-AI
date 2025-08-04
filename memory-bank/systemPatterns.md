# ReRoom: System Patterns & Architecture

## Development Environment Patterns ✅ ACTIVE

### 🔧 Local Development Stack
**Pattern:** Containerized infrastructure with native mobile development
```bash
# Infrastructure Services (Docker)
PostgreSQL (5432) → Database operations
Redis (6379) → Caching and sessions  
Qdrant (6333) → Vector search for AI
MinIO (9000/9001) → S3-compatible storage

# Development Services (Native)
Expo Dev Server → Mobile app with hot reload
TypeScript Compiler → Type checking and builds
```

**Benefits:**
- Consistent environment across team members
- Easy service startup/shutdown
- Production-like local infrastructure
- No external dependencies for core development

### 📱 Mobile Development Patterns
**Pattern:** Expo Router + TypeScript + Simplified Configuration
```typescript
// App Structure
src/app/_layout.tsx    → Root layout with navigation
src/app/index.tsx      → Welcome/home screen  
src/app/camera.tsx     → Photo capture (planned)
src/app/results.tsx    → AI results display (planned)

// Configuration
app.json               → Minimal Expo config for stability
package.json           → React Native 0.73 + modern dependencies
tsconfig.json          → Auto-generated TypeScript config
```

**Key Decisions:**
- Expo Router over React Navigation (simpler file-based routing)
- TypeScript strict mode for type safety
- Minimal plugin configuration to avoid build issues
- Placeholder assets to prevent missing file errors

### 🐳 Docker Infrastructure Patterns
**Pattern:** Service-based containers with health monitoring
```yaml
# docker-compose.yml structure
services:
  postgres: PostgreSQL 15 + health checks
  redis: Redis 7 + persistence  
  qdrant: Vector DB + storage volumes
  minio: S3-compatible + web console

# Environment Configuration
.env → Centralized config for all services
Health Checks → Prevent cascading failures  
Volume Persistence → Data survives container restarts
```

**Established Standards:**
- All services have health check endpoints
- Environment variables for all configuration
- Named volumes for data persistence
- Consistent naming: reroom-{service}

## Dependency Management Patterns ✅ ACTIVE

### 📦 Package Management
**Pattern:** pnpm workspaces with version alignment
```json
// Root package.json
"workspaces": ["mobile", "backend/*", "shared"]
"packageManager": "pnpm@9.12.0"

// Version Alignment Strategy
ESLint: 8.56.0 (downgraded for compatibility)
TypeScript: 5.9.2 (consistent across services)
React Native: 0.73.6 (stable release)
```

**Conflict Resolution:**
- Downgrade to stable versions when peer dependencies conflict
- Use exact versions for critical packages
- Document all version decisions in memory bank

### 🔧 Configuration Management
**Pattern:** Centralized environment with service-specific overrides
```bash
# .env (Development defaults)
NODE_ENV=development
DATABASE_URL=postgresql://reroom:reroom_dev_pass@localhost:5432/reroom_dev
REDIS_URL=redis://localhost:6379

# Service URLs
EXPO_PUBLIC_API_BASE_URL=http://localhost:3000
AWS_ACCESS_KEY_ID=minioadmin (MinIO local)
```

**Benefits:**
- Single source of truth for configuration
- Easy switching between development/production
- No hardcoded values in source code
- Clear documentation of all required variables

## Mobile Application Architecture

### 🗂️ File Organization Pattern
**Pattern:** Feature-based structure with clear separation
```
mobile/src/
├── app/           # Expo Router screens (file-based routing)
├── components/    # Reusable UI components
├── services/      # API calls and business logic
├── stores/        # Zustand state management
├── types/         # TypeScript type definitions
├── utils/         # Helper functions and utilities
└── constants/     # App constants and configuration
```

**Future Expansion:**
```
mobile/src/
├── app/
│   ├── (tabs)/           # Tab navigation group
│   ├── camera/           # Camera flow screens
│   ├── results/          # AI results screens
│   └── settings/         # User settings
├── components/
│   ├── ui/               # Basic UI components
│   ├── camera/           # Camera-specific components
│   └── results/          # Results display components
```

### 🎯 State Management Pattern
**Pattern:** Zustand for global state + React Query for server state
```typescript
// Global App State (Zustand)
interface AppStore {
  user: User | null
  currentRoom: Room | null
  preferences: UserPreferences
  setUser: (user: User) => void
}

// Server State (React Query)
- Photo uploads
- AI processing status  
- Product data
- User authentication
```

**Benefits:**
- Minimal boilerplate compared to Redux
- Automatic server state synchronization
- Offline-first capabilities with MMKV persistence
- Type-safe state access throughout app

## Backend Architecture Patterns

### 🏗️ Microservices Pattern (Planned)
**Pattern:** Domain-driven service separation
```
backend/
├── auth-service/     # User management, JWT tokens
├── photo-service/    # Upload, storage, optimization  
├── ai-service/       # AI processing pipeline
├── product-service/  # E-commerce API integration
├── api-gateway/      # Request routing, rate limiting
└── shared/           # Common types and utilities
```

**Service Responsibilities:**
- **auth-service:** Authentication, user profiles, preferences
- **photo-service:** Photo upload, S3 storage, metadata
- **ai-service:** AI processing, model inference, job queue
- **product-service:** Retailer APIs, price comparison, affiliate tracking
- **api-gateway:** Request routing, authentication middleware, rate limiting

### 🗄️ Database Design Patterns
**Pattern:** Polyglot persistence with clear data ownership
```sql
-- PostgreSQL (Primary Database)
Users → Authentication, profiles, subscription
Rooms → User rooms, designs, saved state  
Photos → Upload metadata, processing status
Products → Cached product data, pricing history
Transactions → Purchases, affiliate tracking

-- Redis (Cache & Sessions)  
Session tokens, API response cache, rate limiting

-- Qdrant (Vector Database)
Product embeddings, visual similarity search

-- MinIO (Object Storage)
Original photos, processed images, AI results
```

**Data Flow Pattern:**
1. Photo uploaded → MinIO storage + PostgreSQL metadata
2. AI processing → Queue in Redis + results in MinIO
3. Product matching → Vector search in Qdrant + cache in Redis
4. User interactions → PostgreSQL transactions + analytics events

## AI Pipeline Architecture (Planned)

### 🤖 Processing Pipeline Pattern
**Pattern:** Async job queue with progress tracking
```python
# AI Service Components
1. Image Preprocessing → Quality check, optimization
2. Depth Estimation → Depth-Anything-V2 model
3. Object Segmentation → SAM2 with furniture focus
4. Style Transfer → Stable Diffusion XL + ControlNet
5. Product Matching → CLIP embeddings + vector search
6. Quality Assurance → Confidence scoring, retry logic
```

**Job Queue Pattern:**
```python
# Redis Queue Structure
ai_jobs:pending → New processing requests
ai_jobs:processing → Currently running jobs
ai_jobs:completed → Finished with results
ai_jobs:failed → Failed jobs for retry

# Progress Tracking
job:{id}:status → Current processing stage
job:{id}:progress → Percentage complete
job:{id}:result → Final output location
```

### 🎨 Style Processing Pattern
**Pattern:** Multi-ControlNet with custom LoRAs
```python
# Style Options (12 planned)
Modern/Contemporary → Clean lines, neutral colors
Scandinavian → Light woods, minimalism  
Bohemian → Eclectic, colorful, textured
Industrial → Metal, concrete, exposed elements
Minimalist → Essential only, maximum space
Traditional → Classic patterns, rich materials

# Processing Flow
Input Photo → Depth Map → Segmentation → Style LoRA → Product Match
```

## Error Handling & Resilience

### 🛡️ Fault Tolerance Patterns
**Pattern:** Graceful degradation with user communication
```typescript
// Mobile Error Handling
try {
  const result = await processPhoto(photo)
  return result
} catch (error) {
  if (error.code === 'NETWORK_ERROR') {
    // Save for retry when online
    await saveForRetry(photo)
    showMessage('Saved for processing when connection returns')
  } else {
    // Show user-friendly error
    showError('Processing failed. Please try a different photo.')
  }
}
```

**Backend Resilience:**
- Circuit breakers for external APIs
- Retry logic with exponential backoff
- Health checks for all services
- Graceful degradation when AI services unavailable

### 📊 Monitoring & Observability
**Pattern:** Comprehensive logging with business context
```typescript
// Structured Logging
logger.info('photo_upload_started', {
  userId: user.id,
  photoSize: photo.size,
  photoFormat: photo.format,
  deviceType: device.type
})

// Business Metrics
- Photo upload success/failure rates
- AI processing times by style
- User conversion from preview to purchase
- Revenue attribution by feature usage
```

## Performance Optimization Patterns

### 🚀 Mobile Performance
**Pattern:** Lazy loading with optimistic UI
```typescript
// Image Loading Strategy
- Thumbnail generation for instant preview
- Progressive JPEG for fast initial display
- Background optimization while user reviews
- Preload common UI assets on app launch

// Memory Management
- Photo compression before upload
- Automatic cleanup of processed images
- Efficient image caching with size limits
- Background processing with foreground updates
```

### ⚡ Backend Performance
**Pattern:** Caching at multiple levels
```typescript
// Caching Strategy
L1: Mobile app cache (MMKV) → Immediate response
L2: Redis cache → Sub-100ms API responses  
L3: Database indexes → Fast complex queries
L4: CDN (CloudFront) → Global image delivery

// AI Processing Optimization
- Model warming to reduce cold start latency
- Batch processing for efficiency
- GPU sharing across multiple requests
- Progressive result delivery (preview → final)
```

## Security Patterns

### 🔐 Authentication & Authorization
**Pattern:** JWT with refresh token rotation
```typescript
// Token Strategy
Access Token: 15 minutes expiry, stateless
Refresh Token: 30 days expiry, database tracked
Device Fingerprinting: Detect token theft
Biometric Gates: Face/Touch ID for sensitive actions
```

**API Security:**
- Rate limiting by user and endpoint
- Input validation with strict schemas
- SQL injection prevention with parameterized queries
- CORS configuration for mobile app domains

### 🛡️ Data Protection
**Pattern:** Privacy by design with minimal retention
```typescript
// Data Handling
- Photos auto-deleted after 30 days
- User data export API for GDPR compliance
- Encryption at rest (AES-256) and in transit (TLS 1.3)
- No tracking without explicit consent
```

## Established Development Practices

### ✅ Proven Patterns (Currently Active)
1. **Environment Setup:** Docker + pnpm + TypeScript = reliable foundation
2. **Configuration:** Centralized .env with service-specific overrides
3. **Error Resolution:** Document all fixes in memory bank for future reference
4. **Version Management:** Conservative versions over cutting-edge for stability
5. **Health Monitoring:** All services have health checks and proper logging

### 🎯 Patterns to Implement Next
1. **Camera Integration:** react-native-vision-camera with permission handling
2. **Photo Upload:** Multipart form data with progress tracking
3. **API Structure:** RESTful endpoints with consistent error responses
4. **State Management:** Zustand stores for app state + React Query for server data
5. **Navigation:** Expo Router with typed routes and deep linking 