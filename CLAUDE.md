# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ReRoom is an AI-powered interior design app with price discovery, built as a microservices architecture. The monorepo contains a React Native mobile app and Node.js/Python backend services.

## Development Commands

### Setup & Installation
```bash
# Initial setup (installs all dependencies)
pnpm run setup

# Start all services for development
pnpm run dev

# Start only mobile app
pnpm run dev:mobile

# Start only backend services
pnpm run dev:backend
```

### Testing
```bash
# Run all tests (mobile + backend)
pnpm test

# Mobile tests only
pnpm run test:mobile

# Backend tests only  
pnpm run test:backend

# Test with coverage
pnpm run test:coverage

# E2E tests
pnpm run test:e2e
```

### Code Quality
```bash
# Lint all code
pnpm run lint

# Type checking
pnpm run type-check

# Format code (via lint-staged on commits)
```

### Build & Deployment
```bash
# Build all services
pnpm run build

# Build mobile app for production
pnpm run build:mobile

# Build Docker images
pnpm run build:backend
```

### Docker Services
```bash
# Start infrastructure (databases, storage)
pnpm run docker:up

# Stop infrastructure
pnpm run docker:down

# View logs
pnpm run docker:logs
```

### Database Operations
```bash
# Run database migrations
pnpm run db:migrate

# Seed database with test data
pnpm run db:seed
```

## Architecture

### Monorepo Structure
- **mobile/**: React Native app with Expo SDK 53 + BNA UI framework
  - **src/screens/**: Screen components (HomeScreen, CameraScreen, GalleryScreen, MakeoverScreen, ProfileScreen)
  - **src/navigation/**: React Navigation with AppNavigator setup (bottom tabs)
  - **src/components/ui/**: BNA UI components (Button, Card, Text, Input, Loading)
  - **src/theme/**: Complete theme system with colors and typography
  - **src/services/**: API services and business logic (placeholder)
  - **src/stores/**: Zustand state management with MMKV persistence
- **backend/**: Hybrid architecture (Cloud + Legacy microservices)
  - **Cloud Backend (v2.0.0)**: Unified Railway service with Supabase + Cloudflare R2
  - **auth-service/**: Legacy user authentication (Node.js + Express)
  - **photo-service/**: Legacy image upload/storage (Node.js + Express)
  - **ai-service/**: AI processing pipeline (Python + FastAPI + Stable Diffusion)
- **shared/**: Common types and utilities (planned)
- **memory-bank/**: Project context and documentation
- **documentation/**: PRDs and technical specifications

### Key Technologies
- **Mobile**: React Native 0.79, Expo 53, Zustand (state), React Query (data fetching)
- **UI Framework**: BNA UI (Ahmedbna) with comprehensive theming system
- **Backend**: Node.js 18+, Express, TypeScript, Supabase integration
- **Cloud**: Railway hosting, Cloudflare R2 storage, Supabase database
- **AI**: Python 3.11+, FastAPI, Stable Diffusion XL, SAM2 segmentation, Depth-Anything-V2, CLIP embeddings, Playwright web scraping
- **Databases**: PostgreSQL, Redis, Qdrant (vector DB)
- **Storage**: MinIO (local dev), Cloudflare R2 (production)
- **Package Manager**: pnpm 9.12.0 (required)

### Environment Requirements
- Node.js 18+
- pnpm 9+ (not npm/yarn)
- Python 3.11+ for AI service
- Docker & Docker Compose V2

### Development Workflow
1. Use `pnpm run docker:up` to start infrastructure
2. Run individual services with `pnpm run dev:auth`, `pnpm run dev:photo`, etc.
3. Use `pnpm run dev:mobile` for React Native development (Expo Go or device testing)
4. AI service runs with `uvicorn main:app --reload --port 8000`

### Mobile App Development Status
- **✅ Working**: Tab navigation, BNA UI components, theme system, state management
- **✅ Configured**: Camera permissions, MMKV persistence, TypeScript strict mode
- **✅ Compatible**: All dependencies aligned with Expo SDK 53 requirements
- **🚧 Next**: Camera integration, API connections, screen implementations

### Backend AI Service Status
- **✅ Production Deployed**: Full AI pipeline with SAM2 segmentation and advanced processing
- **✅ Enhanced Scraping**: Houzz integration with Playwright automation and dataset validation
- **✅ Advanced Analytics**: Color extraction API with hex color analysis and object categorization
- **✅ Scene Management**: Complete scene vs object distinction with confidence scoring
- **✅ Review Dashboard**: Real-time monitoring with segmentation statistics and UI improvements
- **✅ Dataset Export**: Training dataset functionality with scene/object splits and JSON export
- **✅ Railway Production**: Cloud deployment with health checks and fallback mechanisms
- **✅ Mask Visualization**: Fixed frontend mask overlay system with colored segmentation display
- **✅ API Connectivity**: Resolved frontend-backend communication issues and error handling
- **✅ YOLO Integration**: Object detection working with 200+ item comprehensive taxonomy
- **🔄 Continuous**: Job tracking, error handling improvements, and scraper optimizations

### Service URLs (Development)
- Mobile App: http://localhost:8081 (Expo DevTools) / QR Code for device testing
- Cloud Backend: http://localhost:3000 (Unified Railway service)
- Auth Service: http://localhost:3001 (Legacy - being migrated)
- Photo Service: http://localhost:3002 (Legacy - being migrated)
- AI Service: http://localhost:8000 (Python FastAPI) - **Production AI pipeline with SAM2 segmentation**
- Review Dashboard: https://ovalay-recruitment-production.up.railway.app (Production with advanced monitoring, segmentation statistics, and dataset export)
- MinIO Console: http://localhost:9001 (S3-compatible storage)
- PostgreSQL: localhost:5432
- Redis: localhost:6379
- Qdrant: localhost:6333

### Testing Strategy
- **Mobile**: Jest with React Native Testing Library
- **Backend**: Jest with Supertest for API testing
- **E2E**: Detox for mobile app testing
- **AI**: pytest for Python service testing

### Code Quality Tools
- ESLint + TypeScript ESLint for linting
- Prettier for formatting
- Husky + lint-staged for pre-commit hooks
- TypeScript for type checking

### Database Management
- Prisma for ORM and migrations (auth-service)
- PostgreSQL for primary database
- Redis for caching and sessions
- Qdrant for AI embeddings

### UI Framework (BNA UI by Ahmedbna)
- **Components**: Button, Card, Text, TabBarIcon, CameraView available in `@/components/ui`
- **Theming**: Comprehensive light/dark mode support with ReRoom brand colors
- **Type Safety**: Full TypeScript integration with theme-aware components
- **Path Aliases**: Use `@/components/ui`, `@/theme/*`, `@/hooks/*` imports
- **Theme Provider**: All screens wrapped with ThemeProvider for consistent styling
- **Color System**: Primary (#0066FF), Secondary, Accent colors with semantic variants
- **Theme Files**: `@/theme/colors.ts`, `@/theme/globals.ts`, `@/theme/theme-provider.tsx`
- **Usage**: Import components from `@/components/ui` and use theme-aware styling

### Recent Debug & Fixes (August 2025)
- **✅ SAM2 Mask Visualization System**: Completely resolved mask display issues using canvas-based pixel processing
  - Fixed CORS errors by implementing proxy routing (`/api/masks/...` → `ovalay-recruitment-production.up.railway.app`)
  - Replaced failing CSS mask approach with canvas 2D pixel manipulation for reliable rendering
  - Implemented green/blue color coding (active/inactive objects) with transparent backgrounds
  - Achieved perfect SAM2 segmentation overlay visualization in production review dashboard
- **YOLO Object Detection**: Resolved missing ultralytics dependency → restored object identification capability
- **Database Schema**: Fixed mask_url/mask_r2_key mismatch → preserved mask URLs for frontend while maintaining R2 storage
- **CORS Issues**: Implemented custom static file serving with proper headers for mask image access
- **API Connectivity**: Resolved frontend-backend communication failures → added error handling and timeout management
- **JSX Syntax Errors**: Fixed multiple component export and structure issues in review dashboard
- **Taxonomy Enhancement**: Expanded from 10 to 28 categories with 200+ furniture items for comprehensive object detection
- **Railway URL Alignment**: Synchronized frontend and backend to use correct production endpoints

### Key Development Notes
- Always use pnpm, not npm or yarn (pnpm@9.12.0 required)
- Mobile app uses Expo managed workflow with BNA UI components
- Backend has unified cloud service (v2.0.0) on Railway with legacy microservices
- AI service requires significant memory (8GB+ Docker allocation)
- Pre-commit hooks automatically format and lint code
- Use `pnpm run clean` to reset node_modules if needed
- BNA UI components should be used instead of raw React Native components for consistency
- Mobile app structure: Expo Router with file-based routing in `src/app/`
- Current screens: index (home), camera, gallery, makeover with planned (tabs) navigation
- Architecture is transitioning from microservices to hybrid cloud + legacy approach
- **SAM2 Mask System**: Masks generated via neural segmentation, served via Railway static endpoint, rendered using canvas-based pixel processing with transparent overlays for perfect visualization
- **Frontend Visualization**: Canvas 2D API processes mask pixels to show green (active) and blue (inactive) object boundaries over room images
- **CORS Resolution**: Proxy configuration routes `/api/masks/...` requests through local dev server to bypass cross-origin restrictions
- **Error Handling**: API calls include retry logic and graceful degradation for improved reliability