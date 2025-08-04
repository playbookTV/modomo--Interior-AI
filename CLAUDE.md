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
- **mobile/**: React Native app with Expo
- **backend/**: Microservices architecture
  - **auth-service/**: User authentication (Node.js + Express + Prisma)
  - **photo-service/**: Image upload/storage (Node.js + Express + MinIO)
  - **ai-service/**: AI processing pipeline (Python + FastAPI + Stable Diffusion)
- **shared/**: Common types and utilities (planned)

### Key Technologies
- **Mobile**: React Native 0.73, Expo 50, Zustand (state), React Query (data fetching)
- **UI Framework**: BNA UI (Ahmedbna) with comprehensive theming system
- **Backend**: Node.js 18+, Express, TypeScript, Prisma ORM
- **AI**: Python 3.11+, FastAPI, Stable Diffusion XL, SAM2, Depth-Anything-V2
- **Databases**: PostgreSQL, Redis, Qdrant (vector DB)
- **Storage**: MinIO (S3-compatible)
- **Package Manager**: pnpm 9+ (required)

### Environment Requirements
- Node.js 18+
- pnpm 9+ (not npm/yarn)
- Python 3.11+ for AI service
- Docker & Docker Compose V2

### Development Workflow
1. Use `pnpm run docker:up` to start infrastructure
2. Run individual services with `pnpm run dev:auth`, `pnpm run dev:photo`, etc.
3. Use `pnpm run dev:mobile` for React Native development
4. AI service runs with `uvicorn main:app --reload --port 8000`

### Service URLs (Development)
- Mobile App: http://localhost:19002 (Expo DevTools)
- Auth Service: http://localhost:3001
- Photo Service: http://localhost:3002  
- AI Service: http://localhost:8000
- MinIO Console: http://localhost:9001
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

### UI Framework (BNA UI)
- **Components**: Button, Card, Text, TabBarIcon, CameraView available
- **Theming**: Comprehensive light/dark mode support with ReRoom brand colors
- **Type Safety**: Full TypeScript integration with theme-aware components
- **Path Aliases**: Use `@/components/ui`, `@/theme/*`, `@/hooks/*` imports
- **Theme Provider**: All screens wrapped with ThemeProvider for consistent styling
- **Color System**: Primary (#0066FF), Secondary, Accent colors with semantic variants

### Key Development Notes
- Always use pnpm, not npm or yarn
- Mobile app uses Expo managed workflow with BNA UI components
- Backend services are independently deployable
- AI service requires significant memory (8GB+ Docker allocation)
- Pre-commit hooks automatically format and lint code
- Use `pnpm run clean` to reset node_modules if needed
- BNA UI components should be used instead of raw React Native components for consistency