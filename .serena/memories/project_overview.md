# ReRoom Project Overview

## Purpose
ReRoom is an AI-powered interior design app with price discovery functionality, built as a monorepo with microservices architecture.

## Tech Stack
- **Mobile**: React Native 0.79 + Expo 53 + BNA UI framework
- **Backend**: Node.js 18+ (Express) + Python 3.11+ (FastAPI) 
- **Database**: PostgreSQL, Redis, Qdrant (vector DB)
- **Storage**: MinIO (dev), Cloudflare R2 (prod)
- **Cloud**: Railway hosting, Supabase database
- **Package Manager**: pnpm 9.12.0 (required, not npm/yarn)
- **AI**: Stable Diffusion XL, SAM2, Depth-Anything-V2

## Architecture
- **mobile/**: React Native app with Expo Router + BNA UI
- **backend/**: Hybrid cloud + legacy microservices
  - auth-service (Node.js + Express)
  - photo-service (Node.js + Express) 
  - ai-service (Python + FastAPI)
- **shared/**: Common types/utilities (planned)

## Key Features
- File-based routing with Expo Router
- Theme system with light/dark mode
- Zustand state management
- React Query data fetching
- BNA UI component library
- Docker containerized services