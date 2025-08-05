import { Request, Response, NextFunction } from 'express'
import { logger } from '../utils/logger'

export interface AppError extends Error {
  statusCode?: number
  code?: string
  isOperational?: boolean
}

export class ValidationError extends Error {
  statusCode = 400
  code = 'VALIDATION_ERROR'
  isOperational = true
  
  constructor(message: string, public details?: any) {
    super(message)
    this.name = 'ValidationError'
  }
}

export class AuthenticationError extends Error {
  statusCode = 401
  code = 'AUTHENTICATION_ERROR'
  isOperational = true
  
  constructor(message: string = 'Authentication required') {
    super(message)
    this.name = 'AuthenticationError'
  }
}

export class AuthorizationError extends Error {
  statusCode = 403
  code = 'AUTHORIZATION_ERROR'
  isOperational = true
  
  constructor(message: string = 'Insufficient permissions') {
    super(message)
    this.name = 'AuthorizationError'
  }
}

export class NotFoundError extends Error {
  statusCode = 404
  code = 'NOT_FOUND'
  isOperational = true
  
  constructor(message: string = 'Resource not found') {
    super(message)
    this.name = 'NotFoundError'
  }
}

export class ConflictError extends Error {
  statusCode = 409
  code = 'CONFLICT_ERROR'
  isOperational = true
  
  constructor(message: string = 'Conflict with existing resource') {
    super(message)
    this.name = 'ConflictError'
  }
}

export class RateLimitError extends Error {
  statusCode = 429
  code = 'RATE_LIMIT_EXCEEDED'
  isOperational = true
  
  constructor(message: string = 'Rate limit exceeded') {
    super(message)
    this.name = 'RateLimitError'
  }
}

export const errorHandler = (
  error: AppError,
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  // Log error details
  const errorInfo = {
    message: error.message,
    stack: error.stack,
    statusCode: error.statusCode,
    code: error.code,
    url: req.url,
    method: req.method,
    userId: req.auth?.userId,
    userAgent: req.get('User-Agent'),
    ip: req.ip,
  }

  // Log based on severity
  if (error.statusCode && error.statusCode >= 500) {
    logger.error('Server error:', errorInfo)
  } else if (error.statusCode && error.statusCode >= 400) {
    logger.warn('Client error:', errorInfo)
  } else {
    logger.error('Unknown error:', errorInfo)
  }

  // Don't expose internal errors in production
  const isProduction = process.env.NODE_ENV === 'production'
  const isDevelopment = process.env.NODE_ENV === 'development'

  // Determine status code
  const statusCode = error.statusCode || 500

  // Prepare error response
  const errorResponse: any = {
    success: false,
    error: {
      code: error.code || 'INTERNAL_SERVER_ERROR',
      message: error.message || 'An unexpected error occurred',
    },
    timestamp: new Date().toISOString(),
    requestId: req.headers['x-request-id'] || 'unknown',
  }

  // Add additional error details in development
  if (isDevelopment) {
    errorResponse.error.stack = error.stack
    errorResponse.error.statusCode = statusCode
    errorResponse.debug = {
      url: req.url,
      method: req.method,
      headers: req.headers,
      body: req.body,
    }
  }

  // Handle specific error types
  switch (error.name) {
    case 'ValidationError':
      if (error instanceof ValidationError && error.details) {
        errorResponse.error.details = error.details
      }
      break
      
    case 'ClerkAPIResponseError':
      errorResponse.error.code = 'CLERK_API_ERROR'
      errorResponse.error.message = isProduction 
        ? 'Authentication service error' 
        : error.message
      break
      
    case 'DatabaseError':
    case 'PostgresError':
      errorResponse.error.code = 'DATABASE_ERROR'
      errorResponse.error.message = isProduction 
        ? 'Database operation failed' 
        : error.message
      break
      
    case 'SyntaxError':
      if (error.message.includes('JSON')) {
        errorResponse.error.code = 'INVALID_JSON'
        errorResponse.error.message = 'Invalid JSON in request body'
      }
      break
  }

  // Send error response
  res.status(statusCode).json(errorResponse)
}

// Async error wrapper
export const asyncHandler = (
  fn: (req: Request, res: Response, next: NextFunction) => Promise<any>
) => {
  return (req: Request, res: Response, next: NextFunction) => {
    Promise.resolve(fn(req, res, next)).catch(next)
  }
}

// 404 handler
export const notFoundHandler = (req: Request, res: Response) => {
  res.status(404).json({
    success: false,
    error: {
      code: 'NOT_FOUND',
      message: `Route ${req.method} ${req.url} not found`,
    },
    timestamp: new Date().toISOString(),
  })
}