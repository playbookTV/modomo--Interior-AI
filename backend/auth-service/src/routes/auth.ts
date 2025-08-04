import { Router } from 'express'
import bcrypt from 'bcryptjs'
import jwt from 'jsonwebtoken'
import { z } from 'zod'

import { prisma } from '../utils/database'
import { redis } from '../utils/redis'
import { logger } from '../utils/logger'
import { validateRequest } from '../middleware/validation'
import { authenticateToken } from '../middleware/auth'
import { generateTokens, verifyRefreshToken } from '../utils/jwt'
import { sendWelcomeEmail, sendPasswordResetEmail } from '../utils/email'

const router = Router()

// Validation schemas
const registerSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8).max(100),
  firstName: z.string().min(1).max(50),
  lastName: z.string().min(1).max(50),
})

const loginSchema = z.object({
  email: z.string().email(),
  password: z.string().min(1),
})

const forgotPasswordSchema = z.object({
  email: z.string().email(),
})

const resetPasswordSchema = z.object({
  token: z.string(),
  password: z.string().min(8).max(100),
})

const refreshTokenSchema = z.object({
  refreshToken: z.string(),
})

// Register new user
router.post('/register', validateRequest(registerSchema), async (req, res) => {
  try {
    const { email, password, firstName, lastName } = req.body

    // Check if user already exists
    const existingUser = await prisma.user.findUnique({
      where: { email: email.toLowerCase() },
    })

    if (existingUser) {
      return res.status(400).json({
        success: false,
        error: {
          code: 'USER_EXISTS',
          message: 'User with this email already exists',
        },
      })
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(password, 12)

    // Create user
    const user = await prisma.user.create({
      data: {
        email: email.toLowerCase(),
        password: hashedPassword,
        firstName,
        lastName,
        preferences: {
          notifications: {
            priceAlerts: true,
            newFeatures: true,
            marketing: false,
          },
        },
      },
      select: {
        id: true,
        email: true,
        firstName: true,
        lastName: true,
        avatar: true,
        preferences: true,
        subscription: true,
        createdAt: true,
        updatedAt: true,
      },
    })

    // Generate tokens
    const { accessToken, refreshToken } = generateTokens(user.id)

    // Store refresh token
    await redis.setex(`refresh_token:${user.id}`, 7 * 24 * 60 * 60, refreshToken)

    // Send welcome email
    try {
      await sendWelcomeEmail(user.email, user.firstName)
    } catch (emailError) {
      logger.warn('Failed to send welcome email:', emailError)
    }

    logger.info(`User registered: ${user.email}`)

    res.status(201).json({
      success: true,
      data: {
        user,
        accessToken,
        refreshToken,
        expiresIn: 15 * 60, // 15 minutes
      },
    })
  } catch (error) {
    logger.error('Registration error:', error)
    res.status(500).json({
      success: false,
      error: {
        code: 'REGISTRATION_FAILED',
        message: 'Failed to register user',
      },
    })
  }
})

// Login user
router.post('/login', validateRequest(loginSchema), async (req, res) => {
  try {
    const { email, password } = req.body

    // Find user
    const user = await prisma.user.findUnique({
      where: { email: email.toLowerCase() },
      select: {
        id: true,
        email: true,
        password: true,
        firstName: true,
        lastName: true,
        avatar: true,
        preferences: true,
        subscription: true,
        createdAt: true,
        updatedAt: true,
      },
    })

    if (!user) {
      return res.status(401).json({
        success: false,
        error: {
          code: 'INVALID_CREDENTIALS',
          message: 'Invalid email or password',
        },
      })
    }

    // Verify password
    const isValidPassword = await bcrypt.compare(password, user.password)
    if (!isValidPassword) {
      return res.status(401).json({
        success: false,
        error: {
          code: 'INVALID_CREDENTIALS',
          message: 'Invalid email or password',
        },
      })
    }

    // Generate tokens
    const { accessToken, refreshToken } = generateTokens(user.id)

    // Store refresh token
    await redis.setex(`refresh_token:${user.id}`, 7 * 24 * 60 * 60, refreshToken)

    // Remove password from response
    const { password: _, ...userWithoutPassword } = user

    logger.info(`User logged in: ${user.email}`)

    res.json({
      success: true,
      data: {
        user: userWithoutPassword,
        accessToken,
        refreshToken,
        expiresIn: 15 * 60, // 15 minutes
      },
    })
  } catch (error) {
    logger.error('Login error:', error)
    res.status(500).json({
      success: false,
      error: {
        code: 'LOGIN_FAILED',
        message: 'Failed to login',
      },
    })
  }
})

// Refresh access token
router.post('/refresh', validateRequest(refreshTokenSchema), async (req, res) => {
  try {
    const { refreshToken } = req.body

    // Verify refresh token
    const decoded = verifyRefreshToken(refreshToken)
    if (!decoded) {
      return res.status(401).json({
        success: false,
        error: {
          code: 'INVALID_REFRESH_TOKEN',
          message: 'Invalid refresh token',
        },
      })
    }

    // Check if refresh token exists in Redis
    const storedToken = await redis.get(`refresh_token:${decoded.userId}`)
    if (storedToken !== refreshToken) {
      return res.status(401).json({
        success: false,
        error: {
          code: 'INVALID_REFRESH_TOKEN',
          message: 'Invalid refresh token',
        },
      })
    }

    // Generate new access token
    const { accessToken } = generateTokens(decoded.userId)

    res.json({
      success: true,
      data: {
        accessToken,
        expiresIn: 15 * 60, // 15 minutes
      },
    })
  } catch (error) {
    logger.error('Token refresh error:', error)
    res.status(500).json({
      success: false,
      error: {
        code: 'TOKEN_REFRESH_FAILED',
        message: 'Failed to refresh token',
      },
    })
  }
})

// Logout user
router.post('/logout', authenticateToken, async (req, res) => {
  try {
    const userId = req.user!.id

    // Remove refresh token from Redis
    await redis.del(`refresh_token:${userId}`)

    logger.info(`User logged out: ${userId}`)

    res.json({
      success: true,
      data: {
        message: 'Logged out successfully',
      },
    })
  } catch (error) {
    logger.error('Logout error:', error)
    res.status(500).json({
      success: false,
      error: {
        code: 'LOGOUT_FAILED',
        message: 'Failed to logout',
      },
    })
  }
})

// Get current user
router.get('/me', authenticateToken, async (req, res) => {
  try {
    const userId = req.user!.id

    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: {
        id: true,
        email: true,
        firstName: true,
        lastName: true,
        avatar: true,
        preferences: true,
        subscription: true,
        createdAt: true,
        updatedAt: true,
      },
    })

    if (!user) {
      return res.status(404).json({
        success: false,
        error: {
          code: 'USER_NOT_FOUND',
          message: 'User not found',
        },
      })
    }

    res.json({
      success: true,
      data: user,
    })
  } catch (error) {
    logger.error('Get current user error:', error)
    res.status(500).json({
      success: false,
      error: {
        code: 'GET_USER_FAILED',
        message: 'Failed to get user',
      },
    })
  }
})

// Forgot password
router.post('/forgot-password', validateRequest(forgotPasswordSchema), async (req, res) => {
  try {
    const { email } = req.body

    const user = await prisma.user.findUnique({
      where: { email: email.toLowerCase() },
    })

    // Always return success to prevent email enumeration
    if (!user) {
      return res.json({
        success: true,
        data: {
          message: 'If an account with this email exists, a password reset link has been sent.',
        },
      })
    }

    // Generate reset token
    const resetToken = jwt.sign(
      { userId: user.id, type: 'password_reset' },
      process.env.JWT_SECRET!,
      { expiresIn: '1h' }
    )

    // Store reset token
    await redis.setex(`reset_token:${user.id}`, 60 * 60, resetToken)

    // Send password reset email
    try {
      await sendPasswordResetEmail(user.email, user.firstName, resetToken)
    } catch (emailError) {
      logger.warn('Failed to send password reset email:', emailError)
    }

    logger.info(`Password reset requested: ${user.email}`)

    res.json({
      success: true,
      data: {
        message: 'If an account with this email exists, a password reset link has been sent.',
      },
    })
  } catch (error) {
    logger.error('Forgot password error:', error)
    res.status(500).json({
      success: false,
      error: {
        code: 'FORGOT_PASSWORD_FAILED',
        message: 'Failed to process password reset request',
      },
    })
  }
})

// Reset password
router.post('/reset-password', validateRequest(resetPasswordSchema), async (req, res) => {
  try {
    const { token, password } = req.body

    // Verify reset token
    const decoded = jwt.verify(token, process.env.JWT_SECRET!) as any
    if (decoded.type !== 'password_reset') {
      return res.status(400).json({
        success: false,
        error: {
          code: 'INVALID_RESET_TOKEN',
          message: 'Invalid reset token',
        },
      })
    }

    // Check if reset token exists in Redis
    const storedToken = await redis.get(`reset_token:${decoded.userId}`)
    if (storedToken !== token) {
      return res.status(400).json({
        success: false,
        error: {
          code: 'INVALID_RESET_TOKEN',
          message: 'Invalid or expired reset token',
        },
      })
    }

    // Hash new password
    const hashedPassword = await bcrypt.hash(password, 12)

    // Update password
    await prisma.user.update({
      where: { id: decoded.userId },
      data: { password: hashedPassword },
    })

    // Remove reset token
    await redis.del(`reset_token:${decoded.userId}`)

    // Remove all refresh tokens for this user
    await redis.del(`refresh_token:${decoded.userId}`)

    logger.info(`Password reset completed: ${decoded.userId}`)

    res.json({
      success: true,
      data: {
        message: 'Password reset successfully',
      },
    })
  } catch (error) {
    logger.error('Reset password error:', error)
    res.status(500).json({
      success: false,
      error: {
        code: 'RESET_PASSWORD_FAILED',
        message: 'Failed to reset password',
      },
    })
  }
})

export { router as authRoutes } 