# ReRoom: Technical Context

## Technology Stack

### Frontend (Mobile App)
**React Native 0.79 (New Architecture)**
- **Framework:** Expo SDK 53 (Managed workflow for rapid development)
- **Language:** TypeScript (Strict mode for type safety)
- **UI Framework:** BNA UI (Ahmedbna) with comprehensive theming system
- **State Management:** Zustand + React Query (lightweight, performant)
- **Navigation:** Expo Router (file-based routing, v5.1.4)
- **Animations:** Reanimated 3 (expo-linear-gradient for gradients)
- **Theme System:** Light/dark mode with ReRoom brand colors (#0066FF)

**Key Native Libraries (Current):**
```javascript
// Camera & Media
"expo-camera": "~16.1.11"
"react-native-vision-camera": "^3.9.2" // Planned upgrade
"@bam.tech/react-native-image-resizer": "^3.0.11"
"@react-native-camera-roll/camera-roll": "^7.4.0"
"expo-image-picker": "~16.1.4"

// Storage & Performance
"react-native-mmkv": "^2.12.2" // Fast storage
"expo-image": "~2.4.0" // Optimized images
"@react-native-async-storage/async-storage": "^2.1.2"

// Payments & Analytics
"react-native-purchases": "^7.17.0" // RevenueCat for subscriptions
"@react-native-firebase/analytics": "^19.0.1"
"@react-native-firebase/app": "^19.0.1"

// Authentication & Backend
"@clerk/clerk-expo": "^2.14.14"
"@supabase/supabase-js": "^2.53.0"
```

### Backend Infrastructure
**Hybrid Architecture (Cloud + Legacy Microservices)**
- **Cloud Backend (v2.0.0):** Railway hosted unified service
  - **Database:** Supabase PostgreSQL
  - **Storage:** Cloudflare R2 (S3-compatible)
  - **Authentication:** Clerk integration
  - **API:** Node.js 18+ + Express + TypeScript
- **Legacy Services (Local Development):**
  - **AI Processing:** Python + FastAPI + GPU clusters
  - **Photo Service:** Node.js + Express + MinIO
  - **Auth Service:** Node.js + Express (being migrated)
- **Analytics:** Firebase Analytics + Crashlytics

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
- **Primary:** PostgreSQL (Supabase - cloud, local Docker for dev)
- **Vector:** Qdrant (local dev), planned Pinecone/Weaviate (production)
- **Cache:** Redis (local Docker, planned cloud Redis)
- **Analytics:** Firebase Analytics (current), planned ClickHouse
- **File Storage:** Cloudflare R2 (production), MinIO (local dev)

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
EXPO_PUBLIC_API_BASE_URL=http://localhost:3000
AI_SERVICE_URL=http://localhost:8000
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_key
CLERK_PUBLISHABLE_KEY=your_clerk_key

// Production
EXPO_PUBLIC_API_BASE_URL=https://your-railway-app.railway.app
AI_SERVICE_URL=https://ai.reroom.app
SUPABASE_URL=your_production_supabase_url
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
- **Android:** Minimum API 26 (Android 8.0), target API 34
- **Camera:** expo-camera integration, react-native-vision-camera upgrade planned
- **Storage:** EAS Build configured, 500MB app size limit
- **Network:** Offline-first with MMKV storage and React Query
- **Expo:** SDK 53 with EAS Build for app store distribution

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