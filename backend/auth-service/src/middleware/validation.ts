import { Request, Response, NextFunction } from 'express'
import { z, ZodError } from 'zod'
import { ValidationError } from './errorHandler'

// Middleware to validate request body, query params, or params using Zod schemas
export const validateRequest = (
  schema: z.ZodSchema,
  source: 'body' | 'query' | 'params' = 'body'
) => {
  return (req: Request, res: Response, next: NextFunction) => {
    try {
      const data = req[source]
      const validatedData = schema.parse(data)
      
      // Replace the original data with validated/transformed data
      req[source] = validatedData
      
      next()
    } catch (error) {
      if (error instanceof ZodError) {
        const validationError = new ValidationError(
          'Validation failed',
          {
            issues: error.issues.map(issue => ({
              field: issue.path.join('.'),
              message: issue.message,
              code: issue.code,
              received: issue.received,
            })),
            source,
          }
        )
        next(validationError)
      } else {
        next(error)
      }
    }
  }
}

// Validate request body
export const validateBody = (schema: z.ZodSchema) => {
  return validateRequest(schema, 'body')
}

// Validate query parameters
export const validateQuery = (schema: z.ZodSchema) => {
  return validateRequest(schema, 'query')
}

// Validate route parameters
export const validateParams = (schema: z.ZodSchema) => {
  return validateRequest(schema, 'params')
}

// Common validation schemas
export const commonSchemas = {
  // UUID parameter validation
  uuidParam: z.object({
    id: z.string().uuid('Invalid UUID format'),
  }),
  
  // Pagination query validation
  pagination: z.object({
    limit: z.coerce.number().min(1).max(100).default(20),
    offset: z.coerce.number().min(0).default(0),
    page: z.coerce.number().min(1).optional(),
  }).transform(data => {
    // Convert page to offset if provided
    if (data.page) {
      data.offset = (data.page - 1) * data.limit
    }
    return data
  }),
  
  // Email validation
  email: z.string().email('Invalid email format'),
  
  // Password validation
  password: z.string()
    .min(8, 'Password must be at least 8 characters')
    .max(100, 'Password must be less than 100 characters')
    .regex(/[a-z]/, 'Password must contain at least one lowercase letter')
    .regex(/[A-Z]/, 'Password must contain at least one uppercase letter')
    .regex(/[0-9]/, 'Password must contain at least one number'),
  
  // Name validation
  name: z.string().min(1).max(50).trim(),
  
  // URL validation
  url: z.string().url('Invalid URL format'),
  
  // Phone validation (UK format)
  phoneUK: z.string().regex(
    /^(?:(?:\+44)|(?:0))(?:[1-9]\d{8,9})$/,
    'Invalid UK phone number format'
  ),
}

// Sanitization helpers
export const sanitizers = {
  // Remove HTML tags and trim
  cleanString: (value: string): string => {
    return value.replace(/<[^>]*>/g, '').trim()
  },
  
  // Normalize email
  normalizeEmail: (email: string): string => {
    return email.toLowerCase().trim()
  },
  
  // Clean filename for storage
  cleanFilename: (filename: string): string => {
    return filename
      .replace(/[^a-zA-Z0-9.-]/g, '_')
      .replace(/_{2,}/g, '_')
      .replace(/^_+|_+$/g, '')
      .toLowerCase()
  },
}

// File upload validation
export const validateFileUpload = (
  allowedTypes: string[],
  maxSizeBytes: number = 10 * 1024 * 1024 // 10MB default
) => {
  return (req: Request, res: Response, next: NextFunction) => {
    const file = req.file
    
    if (!file) {
      return next(new ValidationError('No file provided'))
    }
    
    // Check file type
    if (!allowedTypes.includes(file.mimetype)) {
      return next(new ValidationError(
        `Invalid file type. Allowed types: ${allowedTypes.join(', ')}`,
        { received: file.mimetype, allowed: allowedTypes }
      ))
    }
    
    // Check file size
    if (file.size > maxSizeBytes) {
      return next(new ValidationError(
        `File too large. Maximum size: ${Math.round(maxSizeBytes / 1024 / 1024)}MB`,
        { received: `${Math.round(file.size / 1024 / 1024)}MB`, maxSize: `${Math.round(maxSizeBytes / 1024 / 1024)}MB` }
      ))
    }
    
    next()
  }
}

// Rate limiting validation
export const validateRateLimit = (
  windowMs: number,
  maxRequests: number,
  message?: string
) => {
  const requests = new Map<string, { count: number; resetTime: number }>()
  
  return (req: Request, res: Response, next: NextFunction) => {
    const key = req.ip || 'unknown'
    const now = Date.now()
    const windowStart = now - windowMs
    
    // Clean up old entries
    for (const [ip, data] of requests.entries()) {
      if (data.resetTime < now) {
        requests.delete(ip)
      }
    }
    
    const current = requests.get(key)
    
    if (!current || current.resetTime < now) {
      // New window
      requests.set(key, { count: 1, resetTime: now + windowMs })
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
      
      next(new ValidationError(
        message || `Rate limit exceeded. Try again after ${resetTime}`,
        { limit: maxRequests, resetTime }
      ))
    }
  }
}