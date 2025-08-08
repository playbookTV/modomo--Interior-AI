# Task Completion Checklist

When completing development tasks, always run these commands:

## Code Quality Checks
```bash
pnpm run lint          # ESLint + TypeScript ESLint
pnmp run type-check     # TypeScript type checking
```

## Testing
```bash
pnpm test              # All tests (mobile + backend)
pnpm run test:mobile   # Mobile tests only
pnpm run test:backend  # Backend tests only
```

## Build Verification
```bash
pnpm run build         # Build all services
pnpm run build:mobile  # Mobile app build
```

## Pre-commit Process
- Husky + lint-staged automatically runs on commit
- Auto-formats code with Prettier
- Runs ESLint fixes
- TypeScript checking

## Environment Setup
- Ensure Docker services are running: `pnpm run docker:up`
- Check database migrations: `pnpm run db:migrate`
- Verify environment variables are set

## Mobile-Specific Checks
- EAS build configuration is valid
- Firebase configuration is properly set up
- iOS/Android build configurations are correct
- Asset paths and bundle patterns are valid

## Deployment Readiness
- All tests pass
- No linting errors
- TypeScript compilation succeeds
- Docker images build successfully
- Environment variables configured for target environment