# ReRoom Development Commands

## Setup & Installation
```bash
pnpm run setup              # Install all dependencies
pnpm run setup:mobile       # Mobile app setup only
pnpm run setup:backend      # Backend services setup
```

## Development
```bash
pnpm run dev                # Start all services
pnpm run dev:mobile         # Mobile app only (Expo)
pnpm run dev:backend        # Backend services only
pnpm run dev:auth           # Auth service only
pnpm run dev:photo          # Photo service only
pnpm run dev:ai             # AI service only
```

## Building
```bash
pnpm run build              # Build all services
pnpm run build:mobile       # Mobile app for production
pnpm run build:backend      # Docker images
```

## Testing
```bash
pnpm test                   # All tests
pnpm run test:mobile        # Mobile tests only
pnpm run test:backend       # Backend tests only
pnpm run test:coverage      # With coverage
pnpm run test:e2e           # E2E tests
```

## Code Quality
```bash
pnpm run lint               # Lint all code
pnpm run type-check         # TypeScript checking
```

## Docker/Infrastructure
```bash
pnpm run docker:up          # Start databases/storage
pnpm run docker:down        # Stop infrastructure
pnpm run docker:logs        # View logs
```

## Database
```bash
pnpm run db:migrate         # Run migrations
pnpm run db:seed            # Seed test data
```

## Utility
```bash
pnpm run clean              # Clean node_modules
pnpm run clean:mobile       # Clean mobile only
pnpm run clean:backend      # Clean backend only
```

## System Commands (macOS)
- `ls` - List files
- `find` - Search files
- `grep` - Search text
- `git` - Version control
- `docker` - Container management