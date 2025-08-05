import { Request, Response, NextFunction } from 'express'
import { ClerkExpressRequireAuth } from '@clerk/clerk-sdk-node'
import { AuthenticationError, AuthorizationError } from './errorHandler'

// Extend Express Request type to include Clerk auth
declare global {
  namespace Express {
    interface Request {
      auth?: {
        userId: string
        user?: any
        sessionId?: string
        sessionClaims?: any
      }
    }
  }
}

// Clerk authentication middleware
export const authenticateClerk = (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  // Use Clerk's built-in middleware
  const clerkAuth = ClerkExpressRequireAuth({
    onError: (error) => {
      // Transform Clerk errors to our error format
      if (error.message?.includes('Unauthenticated')) {
        return next(new AuthenticationError('Authentication required'))
      }
      if (error.message?.includes('Invalid token')) {
        return next(new AuthenticationError('Invalid authentication token'))
      }
      return next(new AuthenticationError(error.message || 'Authentication failed'))
    }
  })
  
  return clerkAuth(req, res, next)
}

// Optional authentication (doesn't fail if no token)
export const authenticateOptional = (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  const authHeader = req.headers.authorization
  
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    // No auth provided, continue without auth
    return next()
  }
  
  // Auth provided, validate it
  return authenticateClerk(req, res, next)
}

// Subscription tier authorization
export const requireSubscription = (
  requiredTier: 'free' | 'premium',
  allowHigherTiers: boolean = true
) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    try {
      if (!req.auth?.userId) {
        return next(new AuthenticationError('Authentication required'))
      }
      
      // In a full implementation, you would:
      // 1. Get user subscription from Supabase
      // 2. Check if their tier meets the requirement
      // For now, we'll implement a basic check
      
      // This would be replaced with actual subscription checking logic
      const userTier = 'free' // Placeholder - get from database
      
      const tierLevels = { free: 0, premium: 1 }
      const userLevel = tierLevels[userTier as keyof typeof tierLevels] || 0
      const requiredLevel = tierLevels[requiredTier]
      
      if (allowHigherTiers ? userLevel >= requiredLevel : userLevel === requiredLevel) {
        next()
      } else {
        next(new AuthorizationError(
          `This feature requires a ${requiredTier} subscription`
        ))
      }
      
    } catch (error) {
      next(error)
    }
  }
}

// Admin role authorization
export const requireAdmin = (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  if (!req.auth?.userId) {
    return next(new AuthenticationError('Authentication required'))
  }
  
  // Check if user has admin role
  // This would typically check user roles in the database
  // For now, we'll use a simple environment variable check
  const adminUserIds = process.env.ADMIN_USER_IDS?.split(',') || []
  
  if (!adminUserIds.includes(req.auth.userId)) {
    return next(new AuthorizationError('Admin access required'))
  }
  
  next()
}

// Rate limiting per user
export const rateLimitPerUser = (
  maxRequests: number,
  windowMs: number,
  message?: string
) => {
  const userRequests = new Map<string, { count: number; resetTime: number }>()
  
  return (req: Request, res: Response, next: NextFunction) => {
    const userId = req.auth?.userId || req.ip || 'anonymous'
    const now = Date.now()
    
    // Clean up expired entries
    for (const [id, data] of userRequests.entries()) {
      if (data.resetTime < now) {
        userRequests.delete(id)
      }
    }
    
    const current = userRequests.get(userId)
    
    if (!current || current.resetTime < now) {
      // New window
      userRequests.set(userId, { count: 1, resetTime: now + windowMs })
      next()
    } else if (current.count < maxRequests) {
      // Within limit
      current.count++
      next()
    } else {
      // Rate limit exceeded
      const resetTime = new Date(current.resetTime).toISOString()
      res.set({
        'X-RateLimit-Limit': maxRequests.toString(),
        'X-RateLimit-Remaining': '0',
        'X-RateLimit-Reset': resetTime,
      })
      
      next(new AuthorizationError(
        message || `Rate limit exceeded. Try again after ${resetTime}`
      ))
    }
  }
}

// Validate API key (for service-to-service communication)
export const validateApiKey = (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  const apiKey = req.headers['x-api-key'] as string
  const validApiKey = process.env.API_KEY
  
  if (!validApiKey) {
    return next(new Error('API key validation not configured'))
  }
  
  if (!apiKey) {
    return next(new AuthenticationError('API key required'))
  }
  
  if (apiKey !== validApiKey) {
    return next(new AuthenticationError('Invalid API key'))
  }
  
  next()
}

// CORS preflight helper
export const handleCors = (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  const origin = req.headers.origin
  const allowedOrigins = process.env.ALLOWED_ORIGINS?.split(',') || [
    'http://localhost:3000',
    'http://localhost:8081',
    'https://your-app.vercel.app'
  ]
  
  if (allowedOrigins.includes(origin || '')) {
    res.setHeader('Access-Control-Allow-Origin', origin || '*')
  }
  
  res.setHeader('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-API-Key')
  res.setHeader('Access-Control-Allow-Credentials', 'true')
  
  if (req.method === 'OPTIONS') {
    res.status(200).end()
    return
  }
  
  next()
}