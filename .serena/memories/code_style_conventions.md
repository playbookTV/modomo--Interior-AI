# ReRoom Code Style & Conventions

## General Style
- **Package Manager**: pnpm only (not npm/yarn) - version 9.12.0 required
- **TypeScript**: Strict TypeScript settings with type checking
- **Linting**: ESLint + TypeScript ESLint + Prettier
- **Pre-commit**: Husky + lint-staged (auto-format on commit)

## Mobile App Conventions
- **File-based Routing**: Expo Router with screens in `src/app/`
- **Path Aliases**: Use `@/` imports (configured in tsconfig.json)
  - `@/components/*` for components
  - `@/theme/*` for theme system
  - `@/hooks/*`, `@/utils/*`, `@/types/*`, etc.
- **UI Components**: BNA UI framework (import from `@/components/ui`)
- **State Management**: Zustand for global state
- **Data Fetching**: React Query (TanStack Query)
- **Theming**: Light/dark mode support with theme provider

## Backend Conventions
- **Architecture**: Hybrid cloud + legacy microservices
- **Services**: Express.js (Node.js) + FastAPI (Python)
- **Database**: Prisma ORM for PostgreSQL
- **Authentication**: Clerk integration + Supabase

## File Structure Patterns
- **mobile/src/app/**: Expo Router screens
- **mobile/src/components/ui/**: BNA UI components
- **mobile/src/theme/**: Theme system files
- **mobile/src/services/**: API service layer
- **mobile/src/stores/**: Zustand stores

## Testing Conventions
- **Mobile**: Jest + React Native Testing Library
- **Backend**: Jest + Supertest
- **E2E**: Detox for mobile testing
- **AI Service**: pytest for Python testing

## Build Requirements
- Node.js 18+
- pnpm 9+ (enforced in package.json engines)
- Docker for backend services
- Expo SDK 53 + React Native 0.79