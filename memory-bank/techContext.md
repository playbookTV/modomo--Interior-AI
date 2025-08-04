# ReRoom: Technical Context

## Technology Stack

### Frontend (Mobile App)
**React Native 0.73+ (New Architecture)**
- **Framework:** Expo SDK 50+ (Managed workflow for rapid development)
- **Language:** TypeScript (Strict mode for type safety)
- **State Management:** Zustand + React Query (lightweight, performant)
- **Navigation:** React Navigation 6 (native performance)
- **Animations:** Reanimated 3 (60fps animations)
- **Lists:** FlashList (performance for large product catalogs)

**Key Native Libraries:**
```javascript
// Camera & Media
"react-native-vision-camera": "^3.8.0"
"react-native-image-resizer": "^3.0.7"
"@react-native-camera-roll/camera-roll": "^7.4.0"

// Storage & Performance
"react-native-mmkv": "^2.11.0" // Fast storage
"react-native-fast-image": "^8.6.3" // Optimized images

// Payments & Analytics
"react-native-purchases": "^7.17.0" // RevenueCat for subscriptions
"@react-native-firebase/analytics": "^19.0.1"
```

### Backend Infrastructure
**Microservices Architecture (Multi-Cloud)**
- **API Gateway:** AWS API Gateway / Cloudflare
- **Authentication:** Node.js + JWT with refresh token rotation
- **Photo Upload:** Go + AWS S3 (optimized for performance)
- **AI Processing:** Python + FastAPI + GPU clusters
- **Product Matching:** Python + Vector databases (Pinecone/Weaviate)
- **E-commerce API:** Node.js + Redis caching
- **Analytics:** Go + ClickHouse for real-time analytics

**Container Orchestration:**
```yaml
# Kubernetes deployment
- API Gateway (Load balancer)
- Auth Service (3 replicas)
- Photo Service (5 replicas, S3 integration)
- AI Service (2-10 replicas, GPU auto-scaling)
- Product Service (3 replicas, vector DB)
- Analytics Service (2 replicas, ClickHouse)
```

### AI/ML Technology Stack
**Computer Vision Pipeline:**
- **Depth Estimation:** Depth-Anything-V2-Large (state-of-the-art indoor depth)
- **Object Segmentation:** SAM2 with furniture-specific fine-tuning
- **Style Transfer:** Stable Diffusion XL 1.0 + Multi-ControlNet
- **Product Matching:** Custom CLIP model + BLIP-2 for descriptions

**Model Hosting & Inference:**
- **GPU Infrastructure:** NVIDIA A100s on RunPod/Modal for cost optimization
- **Model Serving:** Triton Inference Server for optimized throughput
- **Auto-scaling:** Kubernetes HPA based on queue length + GPU utilization
- **Model Management:** MLflow for versioning and deployment

**Custom Model Training:**
```python
# Style Transfer LoRAs (12 interior styles)
- Modern/Contemporary
- Scandinavian/Japandi  
- Bohemian/Eclectic
- Industrial/Loft
- Minimalist/Japanese
- Traditional/Classic
```

### Database Architecture
**Multi-Database Strategy:**
- **Primary:** PostgreSQL 15 (user data, transactions, product catalog)
- **Vector:** Pinecone/Weaviate (product embeddings, visual search)
- **Cache:** Redis Cluster (sessions, API responses, real-time features)
- **Analytics:** ClickHouse (event tracking, business metrics)
- **File Storage:** AWS S3 + CloudFront (global image CDN)

**Schema Design:**
```sql
-- Core tables
Users (auth, preferences, subscription)
Rooms (user_rooms, designs, saved_state)
Products (catalog, pricing, retailer_info)
Renders (ai_results, processing_metadata)
Transactions (purchases, affiliate_tracking)
Analytics_Events (user_behavior, performance)
```

### External API Integrations
**Retailer APIs:**
- **Amazon Product Advertising API** (primary partner)
- **eBay Browse API** (marketplace integration)
- **Custom scrapers** for Argos, Temu (with rate limiting)
- **IKEA API** (when available, otherwise scraping)
- **Wayfair Partner API** (premium products)

**Third-Party Services:**
- **Payment:** RevenueCat (subscription management)
- **Analytics:** Firebase Analytics + Crashlytics
- **Notifications:** Firebase Cloud Messaging
- **Deep Linking:** Branch.io (attribution, referrals)
- **Error Tracking:** Sentry (backend error monitoring)

## Development Environment

### Local Development Setup
```bash
# Prerequisites
Node.js 18+
Python 3.11+
Docker + Docker Compose
Expo CLI
Xcode (iOS development)
Android Studio (Android development)

# Project structure
modomo/
├── mobile/          # React Native app
├── backend/         # Microservices
├── ai-service/      # Python AI pipeline
├── infrastructure/  # Kubernetes configs
└── docs/           # Documentation
```

**Environment Configuration:**
```javascript
// Development
API_BASE_URL=http://localhost:3000
AI_SERVICE_URL=http://localhost:8000
WEBSOCKET_URL=ws://localhost:3001

// Production
API_BASE_URL=https://api.reroom.app
AI_SERVICE_URL=https://ai.reroom.app
WEBSOCKET_URL=wss://ws.reroom.app
```

### CI/CD Pipeline
**GitHub Actions Workflow:**
1. **Mobile Testing:** Jest + Detox (iOS/Android)
2. **Backend Testing:** Unit + Integration tests
3. **AI Model Validation:** Automated quality checks
4. **Docker Builds:** Multi-arch container images
5. **Kubernetes Deployment:** Rolling updates with health checks
6. **App Store Deployment:** EAS Build + automated submission

## Performance Requirements

### Mobile App Performance
- **App Launch:** <3 seconds cold start, <1 second warm start
- **Photo Capture:** <2 seconds from tap to optimized upload
- **Image Processing:** <1 second for display optimization
- **Navigation:** 60fps animations, <200ms transition times
- **Memory Usage:** <150MB baseline, <300MB peak with AI processing

### Backend Performance
- **API Response:** <200ms for non-AI endpoints
- **AI Processing:** <15 seconds average, <25 seconds 95th percentile
- **Database Queries:** <50ms for user data, <100ms for product search
- **Concurrent Users:** 1,000 simultaneous AI processing requests
- **Uptime:** 99.9% availability target

### Cost Optimization Targets
- **AI Processing:** £0.15-0.25 per render (target <£0.20 by month 6)
- **Infrastructure:** £8,000/month at 50K MAU (£0.16 per user)
- **CDN & Storage:** £2,000/month for global image delivery
- **Total Tech Costs:** <20% of revenue at scale

## Security & Compliance

### Data Protection (GDPR Compliance)
- **Data Minimization:** Only collect necessary user data
- **Encryption:** AES-256 at rest, TLS 1.3 in transit
- **Retention:** Auto-delete user photos after 30 days
- **User Rights:** Data export, deletion, modification APIs
- **Consent Management:** Granular privacy controls

### Mobile Security
- **Certificate Pinning:** Prevent MITM attacks
- **Biometric Authentication:** Touch/Face ID for premium features
- **Jailbreak/Root Detection:** Security warnings for compromised devices
- **Code Obfuscation:** Protect against reverse engineering

### Backend Security
- **API Authentication:** JWT with short expiry + refresh tokens
- **Rate Limiting:** Per-user and per-endpoint limits
- **Input Validation:** Comprehensive sanitization
- **Dependency Scanning:** Automated vulnerability detection

## Technical Constraints & Limitations

### Mobile Platform Constraints
- **iOS:** Minimum iOS 14, optimized for iPhone 12+ screen sizes
- **Android:** Minimum API 26 (Android 8.0), target latest
- **Camera:** Requires rear camera, portrait mode support preferred
- **Storage:** 500MB app size limit, 2GB local cache maximum
- **Network:** Graceful degradation for slow/intermittent connections

### AI Processing Constraints
- **Input Images:** 1-50MB, minimum 512x512px, JPEG/PNG only
- **Processing Time:** Hard timeout at 60 seconds to prevent queue backup
- **Quality Gates:** Minimum 70% confidence score for user delivery
- **GPU Memory:** Optimized for 24GB VRAM (A100), fallback for smaller GPUs
- **Batch Size:** Maximum 4 concurrent renders per GPU to maintain speed

### External API Limitations
- **Rate Limits:** Amazon 100 req/sec, eBay 5000 req/day per token
- **Data Freshness:** Product prices updated every 4 hours
- **Geographic Limits:** UK focus initially, EU/US expansion planned
- **Availability:** Circuit breakers for API failures, graceful degradation

## Monitoring & Observability

### Real-Time Dashboards
- **Business Metrics:** User acquisition, conversion rates, revenue
- **Technical Metrics:** API latency, error rates, AI processing times
- **Infrastructure:** CPU/memory utilization, auto-scaling events
- **User Experience:** App performance, crash rates, session analytics

### Alerting Thresholds
- **Critical:** API downtime, AI processing failures >5%
- **Warning:** Response times >500ms, error rates >1%
- **Info:** Auto-scaling events, deployment completions
- **Custom:** Business metric anomalies, cost threshold breaches 