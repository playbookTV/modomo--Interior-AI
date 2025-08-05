import jwt from 'jsonwebtoken'
import { logger } from './logger'

// Note: In the current architecture, Clerk handles JWT tokens
// This utility is mainly for service-to-service communication or custom tokens

interface TokenPayload {
  userId: string
  email?: string
  type: 'access' | 'refresh' | 'service' | 'verification'
  permissions?: string[]
  metadata?: Record<string, any>
}

interface JWTOptions {
  expiresIn?: string | number
  issuer?: string
  audience?: string
  subject?: string
}

class JWTService {
  private readonly secret: string
  private readonly refreshSecret: string
  private readonly defaultOptions: JWTOptions

  constructor() {
    this.secret = process.env.JWT_SECRET || 'your-secret-key'
    this.refreshSecret = process.env.JWT_REFRESH_SECRET || 'your-refresh-secret-key'
    
    if (!process.env.JWT_SECRET) {
      logger.warn('JWT_SECRET not set, using default (not secure for production)')
    }
    
    this.defaultOptions = {
      issuer: process.env.JWT_ISSUER || 'reroom-auth-service',
      audience: process.env.JWT_AUDIENCE || 'reroom-api',
    }
  }

  /**
   * Generate access token (short-lived)
   */
  generateAccessToken(payload: Omit<TokenPayload, 'type'>, options?: JWTOptions): string {
    try {
      const tokenPayload: TokenPayload = {
        ...payload,
        type: 'access',
      }

      const mergedOptions: JWTOptions = {
        ...this.defaultOptions,
        expiresIn: '15m', // 15 minutes
        ...options,
      }

      const token = jwt.sign(tokenPayload, this.secret, mergedOptions)
      
      logger.debug('Access token generated', {
        userId: payload.userId,
        expiresIn: mergedOptions.expiresIn,
      })
      
      return token
    } catch (error) {
      logger.error('Failed to generate access token:', error)
      throw new Error('Token generation failed')
    }
  }

  /**
   * Generate refresh token (long-lived)
   */
  generateRefreshToken(payload: Omit<TokenPayload, 'type'>, options?: JWTOptions): string {
    try {
      const tokenPayload: TokenPayload = {
        ...payload,
        type: 'refresh',
      }

      const mergedOptions: JWTOptions = {
        ...this.defaultOptions,
        expiresIn: '7d', // 7 days
        ...options,
      }

      const token = jwt.sign(tokenPayload, this.refreshSecret, mergedOptions)
      
      logger.debug('Refresh token generated', {
        userId: payload.userId,
        expiresIn: mergedOptions.expiresIn,
      })
      
      return token
    } catch (error) {
      logger.error('Failed to generate refresh token:', error)
      throw new Error('Refresh token generation failed')
    }
  }

  /**
   * Generate service token (for API-to-API communication)
   */
  generateServiceToken(service: string, permissions: string[], options?: JWTOptions): string {
    try {
      const tokenPayload: TokenPayload = {
        userId: `service:${service}`,
        type: 'service',
        permissions,
        metadata: { service },
      }

      const mergedOptions: JWTOptions = {
        ...this.defaultOptions,
        expiresIn: '1h', // 1 hour
        subject: service,
        ...options,
      }

      const token = jwt.sign(tokenPayload, this.secret, mergedOptions)
      
      logger.debug('Service token generated', {
        service,
        permissions,
        expiresIn: mergedOptions.expiresIn,
      })
      
      return token
    } catch (error) {
      logger.error('Failed to generate service token:', error)
      throw new Error('Service token generation failed')
    }
  }

  /**
   * Generate verification token (for email verification, password reset, etc.)
   */
  generateVerificationToken(userId: string, purpose: string, options?: JWTOptions): string {
    try {
      const tokenPayload: TokenPayload = {
        userId,
        type: 'verification',
        metadata: { purpose },
      }

      const mergedOptions: JWTOptions = {
        ...this.defaultOptions,
        expiresIn: '1h', // 1 hour
        ...options,
      }

      const token = jwt.sign(tokenPayload, this.secret, mergedOptions)
      
      logger.debug('Verification token generated', {
        userId,
        purpose,
        expiresIn: mergedOptions.expiresIn,
      })
      
      return token
    } catch (error) {
      logger.error('Failed to generate verification token:', error)
      throw new Error('Verification token generation failed')
    }
  }

  /**
   * Verify access token
   */
  verifyAccessToken(token: string): TokenPayload | null {
    try {
      const decoded = jwt.verify(token, this.secret, {
        issuer: this.defaultOptions.issuer,
        audience: this.defaultOptions.audience,
      }) as TokenPayload

      if (decoded.type !== 'access') {
        logger.warn('Invalid token type for access token verification', {
          type: decoded.type,
          userId: decoded.userId,
        })
        return null
      }

      return decoded
    } catch (error) {
      if (error instanceof jwt.JsonWebTokenError) {
        logger.warn('Access token verification failed:', error.message)
      } else {
        logger.error('Access token verification error:', error)
      }
      return null
    }
  }

  /**
   * Verify refresh token
   */
  verifyRefreshToken(token: string): TokenPayload | null {
    try {
      const decoded = jwt.verify(token, this.refreshSecret, {
        issuer: this.defaultOptions.issuer,
        audience: this.defaultOptions.audience,
      }) as TokenPayload

      if (decoded.type !== 'refresh') {
        logger.warn('Invalid token type for refresh token verification', {
          type: decoded.type,
          userId: decoded.userId,
        })
        return null
      }

      return decoded
    } catch (error) {
      if (error instanceof jwt.JsonWebTokenError) {
        logger.warn('Refresh token verification failed:', error.message)
      } else {
        logger.error('Refresh token verification error:', error)
      }
      return null
    }
  }

  /**
   * Verify service token
   */
  verifyServiceToken(token: string): TokenPayload | null {
    try {
      const decoded = jwt.verify(token, this.secret, {
        issuer: this.defaultOptions.issuer,
        audience: this.defaultOptions.audience,
      }) as TokenPayload

      if (decoded.type !== 'service') {
        logger.warn('Invalid token type for service token verification', {
          type: decoded.type,
          userId: decoded.userId,
        })
        return null
      }

      return decoded
    } catch (error) {
      if (error instanceof jwt.JsonWebTokenError) {
        logger.warn('Service token verification failed:', error.message)
      } else {
        logger.error('Service token verification error:', error)
      }
      return null
    }
  }

  /**
   * Verify verification token
   */
  verifyVerificationToken(token: string, expectedPurpose?: string): TokenPayload | null {
    try {
      const decoded = jwt.verify(token, this.secret, {
        issuer: this.defaultOptions.issuer,
        audience: this.defaultOptions.audience,
      }) as TokenPayload

      if (decoded.type !== 'verification') {
        logger.warn('Invalid token type for verification token verification', {
          type: decoded.type,
          userId: decoded.userId,
        })
        return null
      }

      if (expectedPurpose && decoded.metadata?.purpose !== expectedPurpose) {
        logger.warn('Token purpose mismatch', {
          expected: expectedPurpose,
          received: decoded.metadata?.purpose,
          userId: decoded.userId,
        })
        return null
      }

      return decoded
    } catch (error) {
      if (error instanceof jwt.JsonWebTokenError) {
        logger.warn('Verification token verification failed:', error.message)
      } else {
        logger.error('Verification token verification error:', error)
      }
      return null
    }
  }

  /**
   * Generate token pair (access + refresh)
   */
  generateTokens(userId: string, email?: string, metadata?: Record<string, any>) {
    const payload = { userId, email, metadata }
    
    return {
      accessToken: this.generateAccessToken(payload),
      refreshToken: this.generateRefreshToken(payload),
    }
  }

  /**
   * Decode token without verification (for debugging)
   */
  decodeToken(token: string): any {
    try {
      return jwt.decode(token, { complete: true })
    } catch (error) {
      logger.error('Token decode error:', error)
      return null
    }
  }

  /**
   * Get token expiration time
   */
  getTokenExpiration(token: string): Date | null {
    try {
      const decoded = jwt.decode(token) as any
      if (decoded && decoded.exp) {
        return new Date(decoded.exp * 1000)
      }
      return null
    } catch (error) {
      logger.error('Failed to get token expiration:', error)
      return null
    }
  }

  /**
   * Check if token is expired
   */
  isTokenExpired(token: string): boolean {
    const expiration = this.getTokenExpiration(token)
    return expiration ? expiration < new Date() : true
  }
}

// Singleton instance
export const jwtService = new JWTService()

// Convenience exports
export const generateTokens = (userId: string, email?: string, metadata?: Record<string, any>) => {
  return jwtService.generateTokens(userId, email, metadata)
}

export const verifyAccessToken = (token: string) => {
  return jwtService.verifyAccessToken(token)
}

export const verifyRefreshToken = (token: string) => {
  return jwtService.verifyRefreshToken(token)
}

export default jwtService