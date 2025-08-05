import { Router } from 'express'
import { supabaseService } from '../utils/database'
import { redis } from '../utils/redis'

const router = Router()

interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy'
  timestamp: string
  version: string
  environment: string
  services: {
    database: {
      status: 'healthy' | 'unhealthy'
      responseTime?: number
      error?: string
    }
    redis: {
      status: 'healthy' | 'unhealthy'
      responseTime?: number
      error?: string
    }
    clerk: {
      status: 'healthy' | 'unhealthy'
      configured: boolean
    }
  }
  uptime: number
}

// Basic health check endpoint
router.get('/', async (req, res) => {
  const startTime = Date.now()
  
  try {
    const health: HealthStatus = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      version: process.env.npm_package_version || '1.0.0',
      environment: process.env.NODE_ENV || 'development',
      services: {
        database: { status: 'healthy' },
        redis: { status: 'healthy' },
        clerk: { 
          status: 'healthy',
          configured: !!(process.env.CLERK_SECRET_KEY || process.env.CLERK_PUBLISHABLE_KEY)
        }
      },
      uptime: process.uptime()
    }

    // Check Supabase database
    try {
      const dbStart = Date.now()
      const isDbHealthy = await supabaseService.healthCheck()
      health.services.database.responseTime = Date.now() - dbStart
      
      if (!isDbHealthy) {
        health.services.database.status = 'unhealthy'
        health.services.database.error = 'Database connection failed'
        health.status = 'degraded'
      }
    } catch (error) {
      health.services.database.status = 'unhealthy'
      health.services.database.error = error.message
      health.status = 'degraded'
    }

    // Check Redis
    try {
      const redisStart = Date.now()
      await redis.ping()
      health.services.redis.responseTime = Date.now() - redisStart
    } catch (error) {
      health.services.redis.status = 'unhealthy'
      health.services.redis.error = error.message
      health.status = 'degraded'
    }

    // Check Clerk configuration
    if (!health.services.clerk.configured) {
      health.services.clerk.status = 'unhealthy'
      health.status = 'degraded'
    }

    // Overall status determination
    const hasUnhealthyServices = Object.values(health.services).some(service => service.status === 'unhealthy')
    if (hasUnhealthyServices) {
      health.status = 'unhealthy'
    }

    // Return appropriate HTTP status
    const statusCode = health.status === 'healthy' ? 200 : health.status === 'degraded' ? 200 : 503

    res.status(statusCode).json({
      success: health.status !== 'unhealthy',
      data: health
    })

  } catch (error) {
    res.status(503).json({
      success: false,
      error: {
        code: 'HEALTH_CHECK_FAILED',
        message: 'Health check failed',
        details: error.message
      },
      data: {
        status: 'unhealthy',
        timestamp: new Date().toISOString(),
        responseTime: Date.now() - startTime
      }
    })
  }
})

// Detailed service status
router.get('/detailed', async (req, res) => {
  try {
    const checks = await Promise.allSettled([
      // Database detailed check
      supabaseService.supabase.from('users').select('count', { count: 'exact', head: true }),
      
      // Redis detailed check
      redis.info('server'),
      
      // Memory usage
      process.memoryUsage(),
      
      // Environment variables check
      {
        nodeVersion: process.version,
        platform: process.platform,
        arch: process.arch,
        clerkConfigured: !!(process.env.CLERK_SECRET_KEY && process.env.CLERK_PUBLISHABLE_KEY),
        supabaseConfigured: !!(process.env.SUPABASE_URL && process.env.SUPABASE_SERVICE_ROLE_KEY)
      }
    ])

    const [dbCheck, redisCheck, memoryUsage, envCheck] = checks

    res.json({
      success: true,
      data: {
        timestamp: new Date().toISOString(),
        database: {
          status: dbCheck.status === 'fulfilled' ? 'healthy' : 'unhealthy',
          error: dbCheck.status === 'rejected' ? dbCheck.reason.message : null
        },
        redis: {
          status: redisCheck.status === 'fulfilled' ? 'healthy' : 'unhealthy',
          error: redisCheck.status === 'rejected' ? redisCheck.reason.message : null,
          info: redisCheck.status === 'fulfilled' ? redisCheck.value : null
        },
        memory: memoryUsage.status === 'fulfilled' ? memoryUsage.value : null,
        environment: envCheck.status === 'fulfilled' ? envCheck.value : null,
        uptime: process.uptime(),
        pid: process.pid
      }
    })

  } catch (error) {
    res.status(500).json({
      success: false,
      error: {
        code: 'DETAILED_HEALTH_CHECK_FAILED',
        message: 'Detailed health check failed',
        details: error.message
      }
    })
  }
})

// Liveness probe (for Kubernetes)
router.get('/live', (req, res) => {
  res.status(200).json({
    success: true,
    data: {
      status: 'alive',
      timestamp: new Date().toISOString(),
      uptime: process.uptime()
    }
  })
})

// Readiness probe (for Kubernetes)
router.get('/ready', async (req, res) => {
  try {
    // Quick checks for critical services
    await Promise.all([
      supabaseService.healthCheck(),
      redis.ping()
    ])

    res.status(200).json({
      success: true,
      data: {
        status: 'ready',
        timestamp: new Date().toISOString()
      }
    })

  } catch (error) {
    res.status(503).json({
      success: false,
      error: {
        code: 'SERVICE_NOT_READY',
        message: 'Service not ready',
        details: error.message
      }
    })
  }
})

export { router as healthRoutes }