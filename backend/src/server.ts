import 'dotenv/config'
import express, { Request, Response, NextFunction } from 'express'
import cors from 'cors'
import helmet from 'helmet'
import compression from 'compression'
import rateLimit from 'express-rate-limit'
import { ClerkExpressWithAuth } from '@clerk/clerk-sdk-node'

// Import route handlers
import cloudPhotosRouter from './routes/cloudPhotos'
import makeoversRouter from './routes/makeovers'

// Import services for health checks
import { cloudflareR2Service } from './services/cloudflareR2'
import { supabaseService } from './services/supabaseService'
import { runpodService } from './services/runpodService'

const app: express.Application = express()
const PORT = process.env.PORT || 6969

// =============================================================================
// MIDDLEWARE CONFIGURATION
// =============================================================================

// Security middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'"],
      imgSrc: ["'self'", "data:", "https:"],
      connectSrc: ["'self'", "https:"],
    },
  },
  crossOriginEmbedderPolicy: false
}))

// CORS configuration for mobile app and web dashboard
app.use(cors({
  origin: [
    'http://localhost:8081', // Expo development
    'https://reroom.app',
    'https://app.reroom.app',
    /\.reroom\.app$/ // All subdomains
  ],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'x-clerk-auth-token']
}))

// Body parsing middleware
app.use(express.json({ limit: '10mb' }))
app.use(express.urlencoded({ extended: true, limit: '10mb' }))

// Compression for better performance
app.use(compression())

// Tiered rate limiting based on endpoint type
const basicLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Reduced from 1000 to 100 requests per IP
  message: {
    success: false,
    error: 'Too many requests, please try again later',
    code: 'RATE_LIMIT_EXCEEDED'
  },
  standardHeaders: true,
  legacyHeaders: false,
})

// More restrictive rate limiting for authenticated users
const authLimiter = rateLimit({
  windowMs: 5 * 60 * 1000, // 5 minutes
  max: 50, // 50 requests per 5 minutes for auth endpoints
  keyGenerator: (req: Request) => {
    // Use user ID if authenticated, otherwise IP
    return (req as any).auth?.userId || req.ip
  },
  message: {
    success: false,
    error: 'Too many authenticated requests, please slow down',
    code: 'AUTH_RATE_LIMIT_EXCEEDED'
  }
})

app.use(basicLimiter)

// Stricter rate limiting for uploads
const uploadLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 50, // Limit each IP to 50 uploads per windowMs
  message: {
    success: false,
    error: 'Upload rate limit exceeded, please try again later',
    code: 'UPLOAD_RATE_LIMIT'
  }
})

// Clerk authentication middleware
app.use(ClerkExpressWithAuth())

// Request logging middleware
app.use((req: Request, res: Response, next: NextFunction) => {
  const start = Date.now()
  
  res.on('finish', () => {
    const duration = Date.now() - start
    console.log(`${req.method} ${req.originalUrl} - ${res.statusCode} (${duration}ms)`)
  })
  
  next()
})

// =============================================================================
// HEALTH CHECK ENDPOINTS
// =============================================================================

/**
 * 🏥 Main health check endpoint
 * GET /health
 */
app.get('/health', async (_req: Request, res: Response) => {
  try {
    const startTime = Date.now()

    // Check all service health in parallel
    const [r2Health, supabaseHealth, runpodHealth] = await Promise.allSettled([
      cloudflareR2Service.healthCheck(),
      supabaseService.healthCheck(),
      runpodService.healthCheck()
    ])

    const healthData = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      environment: process.env.NODE_ENV || 'development',
      version: '2.0.0',
      response_time: Date.now() - startTime,
      services: {
        cloudflare_r2: {
          status: r2Health.status === 'fulfilled' && r2Health.value ? 'healthy' : 'unhealthy',
          error: r2Health.status === 'rejected' ? r2Health.reason?.message : null
        },
        supabase: {
          status: supabaseHealth.status === 'fulfilled' && supabaseHealth.value ? 'healthy' : 'unhealthy',
          error: supabaseHealth.status === 'rejected' ? supabaseHealth.reason?.message : null
        },
        runpod: {
          status: runpodHealth.status === 'fulfilled' && runpodHealth.value ? 'healthy' : 'unhealthy',
          error: runpodHealth.status === 'rejected' ? runpodHealth.reason?.message : null
        }
      }
    }

    // Determine overall health - require only core services (Supabase and R2)
    const coreServicesHealthy = healthData.services.supabase.status === 'healthy' && 
                               healthData.services.cloudflare_r2.status === 'healthy'
    
    if (!coreServicesHealthy) {
      healthData.status = 'unhealthy'
    } else if (healthData.services.runpod.status !== 'healthy') {
      healthData.status = 'degraded' // AI service down but core functions work
    }

    const statusCode = healthData.status === 'unhealthy' ? 503 : 200 // 200 for healthy or degraded
    res.status(statusCode).json(healthData)

  } catch (error) {
    console.error('❌ Health check failed:', error)
    res.status(503).json({
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      error: error instanceof Error ? error.message : String(error)
    })
  }
})

/**
 * 📊 Basic status endpoint for load balancers
 * GET /status
 */
app.get('/status', (_req: Request, res: Response) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString()
  })
})

// =============================================================================
// API ROUTES
// =============================================================================

// Apply upload rate limiter to photo routes
app.use('/api/photos', uploadLimiter, authLimiter, cloudPhotosRouter)
app.use('/api/makeovers', authLimiter, makeoversRouter)

// =============================================================================
// ROOT ENDPOINT
// =============================================================================

app.get('/', (_req: Request, res: Response) => {
  res.json({
    service: 'ReRoom Cloud Backend',
    version: '2.0.0',
    environment: process.env.NODE_ENV || 'development',
    timestamp: new Date().toISOString(),
    endpoints: {
      health: '/health',
      photos: '/api/photos',
      makeovers: '/api/makeovers'
    },
    documentation: 'https://docs.reroom.app'
  })
})

// =============================================================================
// ERROR HANDLING
// =============================================================================

// 404 handler
app.use('*', (req: Request, res: Response) => {
  res.status(404).json({
    success: false,
    error: 'Endpoint not found',
    code: 'NOT_FOUND',
    path: req.originalUrl,
    method: req.method
  })
})

// Global error handler
app.use((error: any, _req: Request, res: Response, _next: NextFunction) => {
  console.error('❌ Unhandled error:', error)

  // Clerk authentication errors
  if (error.name === 'ClerkAPIError') {
    return res.status(401).json({
      success: false,
      error: 'Authentication failed',
      code: 'AUTH_ERROR',
      message: error.message
    })
  }

  // Multer errors (file upload)
  if (error.code === 'LIMIT_FILE_SIZE') {
    return res.status(413).json({
      success: false,
      error: 'File too large',
      code: 'FILE_TOO_LARGE',
      message: 'Maximum file size is 50MB'
    })
  }

  if (error.code === 'LIMIT_UNEXPECTED_FILE') {
    return res.status(400).json({
      success: false,
      error: 'Unexpected file field',
      code: 'INVALID_FILE_FIELD'
    })
  }

  // Generic server error
  return res.status(500).json({
    success: false,
    error: 'Internal server error',
    code: 'INTERNAL_ERROR',
    message: process.env.NODE_ENV === 'development' ? error.message : 'Something went wrong'
  })
})

// =============================================================================
// SERVER STARTUP
// =============================================================================

app.listen(PORT, () => {
  console.log(`🚀 ReRoom Cloud Backend running on port ${PORT}`)
  console.log(`🌍 Environment: ${process.env.NODE_ENV || 'development'}`)
  console.log(`🏥 Health check: http://localhost:${PORT}/health`)
  console.log(`📸 Photos API: http://localhost:${PORT}/api/photos`)
  console.log(`🎨 Makeovers API: http://localhost:${PORT}/api/makeovers`)
  
  // Log service configuration
  console.log('\n🔧 Service Configuration:')
  console.log(`  Supabase: ${process.env.SUPABASE_URL ? '✅ Configured' : '❌ Missing'}`)
  console.log(`  Cloudflare R2: ${process.env.CLOUDFLARE_R2_ENDPOINT ? '✅ Configured' : '❌ Missing'}`)
  console.log(`  RunPod: ${process.env.RUNPOD_ENDPOINT ? '✅ Configured' : '❌ Missing'}`)
  console.log(`  Clerk: ${process.env.CLERK_SECRET_KEY ? '✅ Configured' : '❌ Missing'}`)
})

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('🛑 SIGTERM received, shutting down gracefully')
  process.exit(0)
})

process.on('SIGINT', () => {
  console.log('🛑 SIGINT received, shutting down gracefully')
  process.exit(0)
})

export default app