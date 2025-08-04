# ReRoom - AI-Powered Interior Design App

**"Snap. Style. Save."** - Transform any room photo into a professional makeover in under 15 seconds, with every suggested item instantly shoppable at the best prices.

## 🏗️ Architecture Overview

ReRoom is built as a modern microservices application with:

- **Mobile App**: React Native with Expo (iOS/Android)
- **Backend**: Node.js microservices + Python AI service
- **AI Pipeline**: Stable Diffusion XL + SAM2 + Depth-Anything-V2
- **Database**: PostgreSQL + Redis + Qdrant (vector DB)
- **Infrastructure**: Docker + Kubernetes

## 🚀 Quick Start

### Prerequisites

- **Node.js 18+** and **pnpm 9+**
- **Python 3.11+** with pip
- **Docker** and **Docker Compose V2**
- **Expo CLI**: `pnpm install -g @expo/cli`
- **iOS**: Xcode 14+ (macOS only)
- **Android**: Android Studio + SDK

### Install Docker

**macOS:**
```bash
# Install Docker Desktop
brew install --cask docker
# Or download from: https://docker.com/products/docker-desktop
```

**Windows:**
```bash
# Install Docker Desktop
winget install Docker.DockerDesktop
# Or download from: https://docker.com/products/docker-desktop
```

**Linux (Ubuntu/Debian):**
```bash
# Install Docker Engine
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt-get update
sudo apt-get install docker-compose-plugin
```

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd modomo

# Install dependencies
pnpm run setup

# Copy environment variables
cp .env.example .env
# Edit .env with your configuration
```

### 2. Start Development Environment

```bash
# Start all services (backend + databases)
pnpm run docker:up

# Start mobile app (in separate terminal)
pnpm run dev:mobile

# View logs
pnpm run docker:logs
```

### 3. Access Services

- **Mobile App**: Expo DevTools at http://localhost:19002
- **API Gateway**: http://localhost:3000
- **MinIO Console**: http://localhost:9001 (minioadmin/minioadmin)
- **Database**: postgresql://reroom:reroom_dev_pass@localhost:5432/reroom_dev

## 📱 Mobile Development

```bash
# Start Expo development server
cd mobile
pnpm start

# Run on iOS Simulator
pnpm run ios

# Run on Android Emulator  
pnpm run android

# Run tests
pnpm test

# Build for production
pnpm run build
```

## 🔧 Backend Development

```bash
# Start all backend services
pnpm run dev:backend

# Start individual services
pnpm run dev:auth      # Authentication service
pnpm run dev:photo     # Photo upload service
pnpm run dev:ai        # AI processing service

# Run migrations
pnpm run db:migrate

# Seed database
pnpm run db:seed

# Run tests
pnpm run test:backend
```

## 🤖 AI Service Development

```bash
# Navigate to AI service
cd backend/ai-service

# Install Python dependencies
pip install -r requirements.txt

# Start development server
uvicorn main:app --reload --port 8000

# Run AI processing tests
python -m pytest tests/
```

## 📊 Project Structure

```
modomo/
├── mobile/                    # React Native app
│   ├── src/
│   │   ├── components/       # Reusable UI components
│   │   ├── screens/          # Screen components
│   │   ├── services/         # API & business logic
│   │   ├── stores/           # Zustand state management
│   │   └── utils/            # Helper functions
│   ├── app.json              # Expo configuration
│   └── package.json
├── backend/
│   ├── api-gateway/          # Request routing & rate limiting
│   ├── auth-service/         # User authentication & management
│   ├── photo-service/        # Image upload & optimization
│   ├── ai-service/           # AI processing pipeline
│   ├── product-service/      # Multi-retailer integration
│   ├── analytics-service/    # Metrics & business intelligence
│   └── database/             # Schemas & migrations
├── shared/                   # Common types & utilities
├── infrastructure/           # Kubernetes & deployment configs
├── docs/                     # Technical documentation
└── scripts/                  # Development & deployment scripts
```

## 🧪 Testing

```bash
# Run all tests
pnpm test

# Mobile tests
pnpm run test:mobile

# Backend tests  
pnpm run test:backend

# E2E tests
pnpm run test:e2e

# Test coverage
pnpm run test:coverage
```

## 🔍 Code Quality

```bash
# Lint all code
pnpm run lint

# Format code
pnpm run format

# Type checking
pnpm run type-check

# Pre-commit hooks (automatic)
git commit -m "your message"
```

## 📦 Build & Deployment

```bash
# Build all services
pnpm run build

# Build mobile app
pnpm run build:mobile

# Build Docker images
pnpm run build:backend

# Deploy to staging
pnpm run deploy:staging

# Deploy to production
pnpm run deploy:production
```

## 🔧 Development Tools

### Database Management

```bash
# Connect to PostgreSQL
psql postgresql://reroom:reroom_dev_pass@localhost:5432/reroom_dev

# Redis CLI
redis-cli -h localhost -p 6379

# View Qdrant collections
curl http://localhost:6333/collections
```

### API Testing

```bash
# Health checks
curl http://localhost:3000/health
curl http://localhost:3001/health  # Auth service
curl http://localhost:3002/health  # Photo service
curl http://localhost:8000/health  # AI service

# Test authentication
curl -X POST http://localhost:3000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "password"}'
```

## 🐛 Troubleshooting

### Common Issues

**Docker services won't start:**
```bash
# Clean up containers and volumes
docker compose down -v
docker system prune -f
pnpm run docker:up
```

**"docker-compose: command not found":**
```bash
# Install Docker Desktop (includes Compose V2)
# macOS: brew install --cask docker
# Windows: winget install Docker.DockerDesktop

# Or use 'docker compose' instead of 'docker-compose'
docker compose up -d
```

**ESLint peer dependency warnings:**
```bash
# Clean install to resolve peer dependencies
pnpm run clean
pnpm install
```

**Mobile app won't connect to backend:**
- Check your IP address in .env (EXPO_PUBLIC_API_BASE_URL)
- Ensure firewall allows connections on port 3000
- Try `pnpm run dev:mobile -- --tunnel`

**Database connection errors:**
```bash
# Reset database
docker compose down
docker volume rm modomo_postgres_data
pnpm run docker:up
pnpm run db:migrate
```

**AI service memory errors:**
- Ensure Docker has at least 8GB RAM allocated
- Close other applications to free up memory
- Use lighter AI models for development

### Debugging

**Mobile App:**
- Use Flipper for advanced debugging
- Enable remote debugging in Chrome DevTools
- Check Expo logs: `npx expo logs`

**Backend Services:**
```bash
# View specific service logs
docker compose logs -f auth-service
docker compose logs -f ai-service

# Debug Node.js services
node --inspect=0.0.0.0:9229 src/index.js
```

## 📚 Documentation

- [API Documentation](./docs/api.md)
- [Mobile Development Guide](./docs/mobile.md)
- [AI Pipeline Documentation](./docs/ai-pipeline.md)
- [Deployment Guide](./docs/deployment.md)
- [Contributing Guidelines](./docs/contributing.md)

## 🔐 Security

- All API endpoints require authentication (except health checks)
- User photos are encrypted in transit and at rest
- Automatic dependency vulnerability scanning
- Rate limiting on all public endpoints
- Input validation and sanitization

## 📈 Monitoring & Analytics

- **Health Checks**: All services expose `/health` endpoints
- **Metrics**: Prometheus metrics at `/metrics`
- **Logging**: Structured JSON logs with correlation IDs
- **Error Tracking**: Sentry integration for production
- **Analytics**: Firebase Analytics for user behavior

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run quality checks: `pnpm run lint && pnpm test`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📄 License

This project is proprietary software. All rights reserved.

## 🆘 Support

- **Documentation**: Check the docs/ directory
- **Issues**: Create a GitHub issue
- **Emergency**: Contact the development team

---

**ReRoom** - Transforming interior design with AI-powered price discovery 🏠✨ 