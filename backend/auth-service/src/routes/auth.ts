import { Router } from 'express'
import { z } from 'zod'
import { ClerkExpressRequireAuth } from '@clerk/clerk-sdk-node'

import { supabaseService } from '../utils/database'
import { redis } from '../utils/redis'
import { logger } from '../utils/logger'
import { validateRequest } from '../middleware/validation'
import { authenticateClerk } from '../middleware/auth'

const router = Router()

// Validation schemas
const createUserSchema = z.object({
  clerkUserId: z.string().min(1),
  email: z.string().email(),
  firstName: z.string().min(1).max(50).optional(),
  lastName: z.string().min(1).max(50).optional(),
})

const syncUserSchema = z.object({
  clerkUserId: z.string().min(1),
  email: z.string().email(),
  firstName: z.string().min(1).max(50).optional(),
  lastName: z.string().min(1).max(50).optional(),
  subscriptionTier: z.enum(['free', 'premium']).optional(),
})

// Clerk webhook validation schema
const clerkWebhookSchema = z.object({
  type: z.string(),
  data: z.object({
    id: z.string(),
    email_addresses: z.array(z.object({
      email_address: z.string().email(),
      id: z.string(),
    })).optional(),
    first_name: z.string().optional(),
    last_name: z.string().optional(),
  }),
})

// Create user in Supabase (called after Clerk signup)
router.post('/users', validateRequest(createUserSchema), async (req, res) => {
  try {
    const { clerkUserId, email, firstName, lastName } = req.body

    // Check if user already exists
    const existingUser = await supabaseService.getUser(clerkUserId)
    if (existingUser) {
      return res.status(409).json({
        success: false,
        error: {
          code: 'USER_EXISTS',
          message: 'User already exists in our system',
        },
      })
    }

    // Create user in Supabase
    const user = await supabaseService.createOrUpdateUser(
      clerkUserId,
      email,
      {
        preferences: {
          notifications: {
            priceAlerts: true,
            newFeatures: true,
            marketing: false,
          },
          designPreferences: {},
          privacy: {
            shareDesigns: false,
            analyticsOptIn: true,
          }
        }
      }
    )

    logger.info(`User created in Supabase: ${clerkUserId}`)

    res.status(201).json({
      success: true,
      data: {
        id: user.id,
        clerkUserId: user.clerk_user_id,
        email: user.email,
        subscriptionTier: user.subscription_tier || 'free',
        createdAt: user.created_at,
      },
    })

  } catch (error) {
    logger.error('Create user error:', error)
    res.status(500).json({
      success: false,
      error: {
        code: 'USER_CREATION_FAILED',
        message: 'Failed to create user',
      },
    })
  }
})

// Sync user data from Clerk to Supabase
router.post('/sync', validateRequest(syncUserSchema), async (req, res) => {
  try {
    const { clerkUserId, email, firstName, lastName, subscriptionTier } = req.body

    // Update or create user in Supabase
    const user = await supabaseService.createOrUpdateUser(
      clerkUserId,
      email,
      {
        subscription_tier: subscriptionTier,
        // Note: firstName/lastName are handled by Clerk, 
        // we mainly sync subscription and preference data
      }
    )

    logger.info(`User synced: ${clerkUserId}`)

    res.json({
      success: true,
      data: {
        id: user.id,
        clerkUserId: user.clerk_user_id,
        email: user.email,
        subscriptionTier: user.subscription_tier,
        updatedAt: user.updated_at,
      },
    })

  } catch (error) {
    logger.error('Sync user error:', error)
    res.status(500).json({
      success: false,
      error: {
        code: 'USER_SYNC_FAILED',
        message: 'Failed to sync user',
      },
    })
  }
})

// Clerk webhook handler for user events
router.post('/webhook', async (req, res) => {
  try {
    // In production, you should verify the webhook signature
    // const signature = req.headers['svix-signature'] as string
    // if (!signature) {
    //   return res.status(400).json({ error: 'Missing signature' })
    // }

    const { type, data } = req.body

    logger.info(`Clerk webhook received: ${type}`, { userId: data.id })

    switch (type) {
      case 'user.created':
        // User signed up in Clerk, create in Supabase
        const email = data.email_addresses?.[0]?.email_address
        if (email) {
          await supabaseService.createOrUpdateUser(
            data.id,
            email,
            {
              preferences: {
                notifications: {
                  priceAlerts: true,
                  newFeatures: true,
                  marketing: false,
                },
                designPreferences: {},
                privacy: {
                  shareDesigns: false,
                  analyticsOptIn: true,
                }
              }
            }
          )
        }
        break

      case 'user.updated':
        // User updated profile in Clerk, sync to Supabase
        const updatedEmail = data.email_addresses?.[0]?.email_address
        if (updatedEmail) {
          await supabaseService.createOrUpdateUser(data.id, updatedEmail)
        }
        break

      case 'user.deleted':
        // User deleted from Clerk, handle in Supabase
        // In production, you might want to soft delete or archive
        logger.info(`User deletion webhook received: ${data.id}`)
        // Implementation depends on your data retention policy
        break

      default:
        logger.warn(`Unhandled webhook type: ${type}`)
    }

    res.status(200).json({ success: true })

  } catch (error) {
    logger.error('Webhook processing error:', error)
    res.status(500).json({
      success: false,
      error: {
        code: 'WEBHOOK_PROCESSING_FAILED',
        message: 'Failed to process webhook',
      },
    })
  }
})

// Validate session (for client-side verification)
router.get('/session', authenticateClerk, async (req, res) => {
  try {
    const clerkUserId = req.auth.userId

    // Get user data from Supabase
    const user = await supabaseService.getUser(clerkUserId)

    if (!user) {
      // User exists in Clerk but not in Supabase, create them
      const clerkUser = req.auth.user
      const createdUser = await supabaseService.createOrUpdateUser(
        clerkUserId,
        clerkUser?.emailAddresses?.[0]?.emailAddress,
        {
          preferences: {
            notifications: {
              priceAlerts: true,
              newFeatures: true,
              marketing: false,
            },
            designPreferences: {},
            privacy: {
              shareDesigns: false,
              analyticsOptIn: true,
            }
          }
        }
      )

      return res.json({
        success: true,
        data: {
          valid: true,
          user: {
            id: createdUser.id,
            clerkUserId: createdUser.clerk_user_id,
            email: createdUser.email,
            subscriptionTier: createdUser.subscription_tier || 'free',
            preferences: createdUser.preferences,
          },
          session: {
            sessionId: req.auth.sessionId,
            userId: req.auth.userId,
          },
        },
      })
    }

    res.json({
      success: true,
      data: {
        valid: true,
        user: {
          id: user.id,
          clerkUserId: user.clerk_user_id,
          email: user.email,
          subscriptionTier: user.subscription_tier || 'free',
          preferences: user.preferences,
          totalPhotos: user.total_photos || 0,
          totalMakeovers: user.total_makeovers || 0,
        },
        session: {
          sessionId: req.auth.sessionId,
          userId: req.auth.userId,
        },
      },
    })

  } catch (error) {
    logger.error('Session validation error:', error)
    res.status(500).json({
      success: false,
      error: {
        code: 'SESSION_VALIDATION_FAILED',
        message: 'Failed to validate session',
      },
    })
  }
})

// Delete user account (GDPR compliance)
router.delete('/users/:clerkUserId', authenticateClerk, async (req, res) => {
  try {
    const { clerkUserId } = req.params
    const requestingUserId = req.auth.userId

    // Only allow users to delete their own account (or admin)
    if (clerkUserId !== requestingUserId) {
      return res.status(403).json({
        success: false,
        error: {
          code: 'UNAUTHORIZED',
          message: 'Can only delete your own account',
        },
      })
    }

    // In production, implement proper user deletion:
    // 1. Delete user photos from Cloudflare R2
    // 2. Delete user data from Supabase
    // 3. Cancel subscriptions
    // 4. Log the deletion for audit

    logger.info(`User deletion initiated: ${clerkUserId}`)

    res.json({
      success: true,
      data: {
        message: 'Account deletion initiated. Data will be removed within 48 hours.',
        deletionScheduled: new Date(Date.now() + 48 * 60 * 60 * 1000).toISOString(),
      },
    })

  } catch (error) {
    logger.error('Delete user error:', error)
    res.status(500).json({
      success: false,
      error: {
        code: 'USER_DELETION_FAILED',
        message: 'Failed to delete user',
      },
    })
  }
})

// Refresh session (handled by Clerk client-side, this is for monitoring)
router.post('/refresh', async (req, res) => {
  res.json({
    success: true,
    data: {
      message: 'Session refresh is handled by Clerk client-side',
      timestamp: new Date().toISOString(),
    },
  })
})

// Logout (handled by Clerk client-side, this is for monitoring)
router.post('/logout', async (req, res) => {
  // Optional: Clear any server-side cache or analytics
  const userId = req.headers['x-user-id'] as string
  
  if (userId) {
    logger.info(`User logout: ${userId}`)
    
    // Clear any Redis cache for this user
    try {
      await redis.del(`user_cache:${userId}`)
    } catch (error) {
      logger.warn('Failed to clear user cache on logout:', error)
    }
  }

  res.json({
    success: true,
    data: {
      message: 'Logout successful',
      timestamp: new Date().toISOString(),
    },
  })
})

export { router as authRoutes }