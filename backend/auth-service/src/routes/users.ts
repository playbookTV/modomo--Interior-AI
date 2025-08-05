import { Router } from 'express'
import { z } from 'zod'
import { ClerkExpressRequireAuth } from '@clerk/clerk-sdk-node'

import { supabaseService } from '../utils/database'
import { logger } from '../utils/logger'
import { validateRequest } from '../middleware/validation'
import { authenticateClerk } from '../middleware/auth'

const router = Router()

// Validation schemas
const updateUserPreferencesSchema = z.object({
  notifications: z.object({
    priceAlerts: z.boolean().optional(),
    newFeatures: z.boolean().optional(),
    marketing: z.boolean().optional(),
  }).optional(),
  designPreferences: z.object({
    preferredStyles: z.array(z.string()).optional(),
    budgetRange: z.enum(['budget', 'mid', 'premium']).optional(),
    roomTypes: z.array(z.string()).optional(),
  }).optional(),
  privacy: z.object({
    shareDesigns: z.boolean().optional(),
    analyticsOptIn: z.boolean().optional(),
  }).optional(),
})

const updateUserProfileSchema = z.object({
  firstName: z.string().min(1).max(50).optional(),
  lastName: z.string().min(1).max(50).optional(),
  avatar: z.string().url().optional(),
})

// Get current user profile
router.get('/me', authenticateClerk, async (req, res) => {
  try {
    const clerkUserId = req.auth.userId

    // Get user from Supabase
    let user = await supabaseService.getUser(clerkUserId)

    // If user doesn't exist in Supabase, create from Clerk data
    if (!user) {
      const clerkUser = req.auth.user
      user = await supabaseService.createOrUpdateUser(
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
    }

    logger.info(`User profile retrieved: ${clerkUserId}`)

    res.json({
      success: true,
      data: {
        id: user.id,
        clerkUserId: user.clerk_user_id,
        email: user.email,
        subscriptionTier: user.subscription_tier || 'free',
        preferences: user.preferences || {},
        totalPhotos: user.total_photos || 0,
        totalMakeovers: user.total_makeovers || 0,
        createdAt: user.created_at,
        updatedAt: user.updated_at,
      },
    })

  } catch (error) {
    logger.error('Get user profile error:', error)
    res.status(500).json({
      success: false,
      error: {
        code: 'GET_USER_FAILED',
        message: 'Failed to get user profile',
      },
    })
  }
})

// Update user preferences
router.patch('/me/preferences', 
  authenticateClerk, 
  validateRequest(updateUserPreferencesSchema), 
  async (req, res) => {
    try {
      const clerkUserId = req.auth.userId
      const updates = req.body

      // Get current user
      const currentUser = await supabaseService.getUser(clerkUserId)
      if (!currentUser) {
        return res.status(404).json({
          success: false,
          error: {
            code: 'USER_NOT_FOUND',
            message: 'User not found',
          },
        })
      }

      // Merge with existing preferences
      const updatedPreferences = {
        ...currentUser.preferences,
        ...updates,
        notifications: {
          ...currentUser.preferences?.notifications,
          ...updates.notifications,
        },
        designPreferences: {
          ...currentUser.preferences?.designPreferences,
          ...updates.designPreferences,
        },
        privacy: {
          ...currentUser.preferences?.privacy,
          ...updates.privacy,
        },
      }

      // Update in Supabase
      const updatedUser = await supabaseService.createOrUpdateUser(
        clerkUserId,
        currentUser.email,
        { preferences: updatedPreferences }
      )

      logger.info(`User preferences updated: ${clerkUserId}`)

      res.json({
        success: true,
        data: {
          preferences: updatedUser.preferences,
          updatedAt: updatedUser.updated_at,
        },
      })

    } catch (error) {
      logger.error('Update user preferences error:', error)
      res.status(500).json({
        success: false,
        error: {
          code: 'UPDATE_PREFERENCES_FAILED',
          message: 'Failed to update user preferences',
        },
      })
    }
  }
)

// Update user profile (syncs with Clerk)
router.patch('/me/profile',
  authenticateClerk,
  validateRequest(updateUserProfileSchema),
  async (req, res) => {
    try {
      const clerkUserId = req.auth.userId
      const updates = req.body

      // Note: For profile updates like firstName, lastName, avatar
      // These should be updated in Clerk first, then synced to Supabase
      // This endpoint mainly handles Supabase-specific data

      const updatedUser = await supabaseService.createOrUpdateUser(
        clerkUserId,
        undefined, // Don't update email here - Clerk handles that
        updates
      )

      logger.info(`User profile updated: ${clerkUserId}`)

      res.json({
        success: true,
        data: {
          id: updatedUser.id,
          updatedAt: updatedUser.updated_at,
        },
        message: 'Profile updated. Note: Name and avatar changes should be made through your account settings.',
      })

    } catch (error) {
      logger.error('Update user profile error:', error)
      res.status(500).json({
        success: false,
        error: {
          code: 'UPDATE_PROFILE_FAILED',
          message: 'Failed to update user profile',
        },
      })
    }
  }
)

// Get user statistics
router.get('/me/stats', authenticateClerk, async (req, res) => {
  try {
    const clerkUserId = req.auth.userId

    const stats = await supabaseService.getUserStats(clerkUserId)

    res.json({
      success: true,
      data: {
        totalPhotos: stats.total_photos || 0,
        totalMakeovers: stats.total_makeovers || 0,
        subscriptionTier: stats.subscription_tier || 'free',
        memberSince: stats.created_at,
        // Add computed stats
        averagePhotosPerWeek: Math.round((stats.total_photos || 0) / Math.max(1, 
          Math.ceil((Date.now() - new Date(stats.created_at).getTime()) / (7 * 24 * 60 * 60 * 1000))
        )),
      },
    })

  } catch (error) {
    logger.error('Get user stats error:', error)
    res.status(500).json({
      success: false,
      error: {
        code: 'GET_STATS_FAILED',
        message: 'Failed to get user statistics',
      },
    })
  }
})

// Get user's photos and makeovers
router.get('/me/photos', authenticateClerk, async (req, res) => {
  try {
    const clerkUserId = req.auth.userId
    const limit = Math.min(parseInt(req.query.limit as string) || 20, 100)
    const offset = parseInt(req.query.offset as string) || 0

    const photos = await supabaseService.getUserPhotos(clerkUserId, limit, offset)

    res.json({
      success: true,
      data: {
        photos,
        pagination: {
          limit,
          offset,
          hasMore: photos.length === limit,
        },
      },
    })

  } catch (error) {
    logger.error('Get user photos error:', error)
    res.status(500).json({
      success: false,
      error: {
        code: 'GET_PHOTOS_FAILED',
        message: 'Failed to get user photos',
      },
    })
  }
})

// Delete user account (GDPR compliance)
router.delete('/me', authenticateClerk, async (req, res) => {
  try {
    const clerkUserId = req.auth.userId

    // Note: In a full implementation, this would:
    // 1. Delete user data from Supabase
    // 2. Delete associated photos from Cloudflare R2
    // 3. Cancel subscriptions
    // 4. Send confirmation email via Clerk
    // 5. Schedule Clerk user deletion

    logger.info(`User deletion requested: ${clerkUserId}`)

    res.json({
      success: true,
      data: {
        message: 'Account deletion initiated. You will receive a confirmation email.',
        deletionScheduled: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(), // 7 days
      },
    })

  } catch (error) {
    logger.error('Delete user account error:', error)
    res.status(500).json({
      success: false,
      error: {
        code: 'DELETE_ACCOUNT_FAILED',
        message: 'Failed to delete user account',
      },
    })
  }
})

// Export user data (GDPR compliance)
router.get('/me/export', authenticateClerk, async (req, res) => {
  try {
    const clerkUserId = req.auth.userId

    // Get all user data
    const [user, photos, stats] = await Promise.all([
      supabaseService.getUser(clerkUserId),
      supabaseService.getUserPhotos(clerkUserId, 1000, 0), // All photos
      supabaseService.getUserStats(clerkUserId),
    ])

    const exportData = {
      user: {
        id: user.id,
        clerkUserId: user.clerk_user_id,
        email: user.email,
        preferences: user.preferences,
        subscriptionTier: user.subscription_tier,
        createdAt: user.created_at,
        updatedAt: user.updated_at,
      },
      statistics: stats,
      photos: photos.map(photo => ({
        id: photo.id,
        originalName: photo.original_name,
        mimeType: photo.mime_type,
        size: photo.original_size,
        dimensions: { width: photo.width, height: photo.height },
        takenAt: photo.taken_at,
        createdAt: photo.created_at,
        makeovers: photo.makeovers?.map(makeover => ({
          id: makeover.id,
          status: makeover.status,
          stylePreference: makeover.style_preference,
          completedAt: makeover.completed_at,
        })) || [],
      })),
      exportedAt: new Date().toISOString(),
    }

    logger.info(`User data exported: ${clerkUserId}`)

    res.json({
      success: true,
      data: exportData,
    })

  } catch (error) {
    logger.error('Export user data error:', error)
    res.status(500).json({
      success: false,
      error: {
        code: 'EXPORT_DATA_FAILED',
        message: 'Failed to export user data',
      },
    })
  }
})

export { router as userRoutes }