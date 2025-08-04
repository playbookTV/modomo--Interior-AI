import express from 'express'
import multer from 'multer'
import { CloudflareR2Service } from '../services/cloudflareR2'
import { SupabaseService } from '../services/supabaseService'
import { RunPodService } from '../services/runpodService'
import { ClerkExpressRequireAuth } from '@clerk/clerk-sdk-node'

const router: any = express.Router()
const r2Service = new CloudflareR2Service()
const supabaseService = new SupabaseService()
const runpodService = new RunPodService()

// Configure multer for photo uploads
const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: { 
    fileSize: 50 * 1024 * 1024, // 50MB max file size
    files: 1 // Single file upload
  },
  fileFilter: (req, file, cb) => {
    // Accept only images
    if (file.mimetype.startsWith('image/')) {
      cb(null, true)
    } else {
      cb(new Error('Only image files are allowed') as any, false)
    }
  }
})

// =============================================================================
// PHOTO UPLOAD ENDPOINTS
// =============================================================================

/**
 * 📸 Upload photo to Cloudflare R2 + trigger AI makeover
 * POST /api/photos/upload
 */
router.post('/upload', 
  ClerkExpressRequireAuth(), // Clerk authentication middleware
  upload.single('photo'),
  async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ 
          success: false,
          error: 'No photo provided',
          code: 'NO_FILE'
        })
      }

      const { userId } = req.auth // From Clerk middleware
      const metadata = req.body.metadata ? JSON.parse(req.body.metadata) : {}

      console.log(`📸 Processing photo upload for user: ${userId}`)

      // Step 1: Ensure user exists in Supabase
      await supabaseService.createOrUpdateUser(userId, req.auth.emailAddress)

      // Step 2: Upload to Cloudflare R2
      const uploadResult = await r2Service.uploadPhoto(
        req.file.buffer,
        req.file.originalname,
        userId
      )

      // Step 3: Save metadata to Supabase
      const photoRecord = await supabaseService.createPhoto({
        clerk_user_id: userId,
        original_url: uploadResult.url,
        optimized_url: uploadResult.variants[2], // Medium size variant
        cloudflare_key: uploadResult.key,
        original_name: req.file.originalname,
        mime_type: req.file.mimetype,
        metadata: {
          ...metadata,
          originalSize: req.file.size,
          uploadedVia: 'mobile',
          processingVersion: '2.0'
        },
        original_size: req.file.size,
        taken_at: metadata.capturedAt || new Date().toISOString()
      })

      // Step 4: Trigger AI makeover if requested
      let makeoverRecord = null
      if (metadata.triggerAI !== false) {
        makeoverRecord = await supabaseService.createMakeover({
          photo_id: photoRecord.id,
          clerk_user_id: userId,
          style_preference: metadata.stylePreference || 'Modern',
          budget_range: metadata.budgetRange || 'mid-range',
          room_type: metadata.roomType || 'living-room'
        })

        // Submit to RunPod asynchronously
        runpodService.submitMakeoverJob({
          photo_url: uploadResult.url,
          photo_id: photoRecord.id,
          makeover_id: makeoverRecord.id,
          user_id: userId,
          style_preference: metadata.stylePreference,
          budget_range: metadata.budgetRange,
          room_type: metadata.roomType
        }).catch(error => {
          console.error('RunPod job submission failed:', error)
        })
      }

      // Step 5: Return success response
      res.json({
        success: true,
        data: {
          id: photoRecord.id,
          url: uploadResult.url,
          variants: uploadResult.variants,
          size: req.file.size,
          originalName: req.file.originalname,
          uploadedAt: photoRecord.created_at,
          makeover: makeoverRecord ? {
            id: makeoverRecord.id,
            status: makeoverRecord.status,
            style_preference: makeoverRecord.style_preference
          } : null
        },
        message: 'Photo uploaded successfully'
      })

    } catch (error) {
      console.error('❌ Photo upload failed:', error)
      res.status(500).json({ 
        success: false,
        error: 'Upload failed',
        message: error.message,
        code: 'UPLOAD_ERROR'
      })
    }
  }
)

/**
 * 📋 Get user photos with makeover status
 * GET /api/photos
 */
router.get('/', 
  ClerkExpressRequireAuth(),
  async (req, res) => {
    try {
      const { userId } = req.auth
      const limit = parseInt(req.query.limit as string) || 50
      const offset = parseInt(req.query.offset as string) || 0

      const photos = await supabaseService.getUserPhotos(userId, limit, offset)

      res.json({
        success: true,
        data: photos,
        count: photos.length,
        pagination: {
          limit,
          offset,
          hasMore: photos.length === limit
        }
      })

    } catch (error) {
      console.error('❌ Failed to fetch photos:', error)
      res.status(500).json({
        success: false,
        error: 'Failed to fetch photos',
        message: error.message,
        code: 'FETCH_ERROR'
      })
    }
  }
)

/**
 * 🔍 Get specific photo by ID
 * GET /api/photos/:photoId
 */
router.get('/:photoId', 
  ClerkExpressRequireAuth(),
  async (req, res) => {
    try {
      const { userId } = req.auth
      const { photoId } = req.params

      // Get photo with makeover details
      const { data: photos, error } = await supabaseService.supabase
        .from('photos')
        .select(`
          *,
          makeovers (
            id,
            status,
            progress,
            makeover_url,
            style_preference,
            completed_at,
            error_message,
            detected_objects,
            suggested_products
          )
        `)
        .eq('id', photoId)
        .eq('clerk_user_id', userId)

      if (error) throw error
      if (!photos || photos.length === 0) {
        return res.status(404).json({
          success: false,
          error: 'Photo not found',
          code: 'NOT_FOUND'
        })
      }

      res.json({
        success: true,
        data: photos[0]
      })

    } catch (error) {
      console.error('❌ Failed to get photo:', error)
      res.status(500).json({
        success: false,
        error: 'Failed to get photo',
        message: error.message,
        code: 'GET_ERROR'
      })
    }
  }
)

/**
 * 🗑️ Delete photo and associated data
 * DELETE /api/photos/:photoId
 */
router.delete('/:photoId', 
  ClerkExpressRequireAuth(),
  async (req, res) => {
    try {
      const { userId } = req.auth
      const { photoId } = req.params

      // Get photo to verify ownership and get Cloudflare key
      const { data: photos, error: fetchError } = await supabaseService.supabase
        .from('photos')
        .select('cloudflare_key, makeovers(runpod_job_id)')
        .eq('id', photoId)
        .eq('clerk_user_id', userId)

      if (fetchError) throw fetchError
      if (!photos || photos.length === 0) {
        return res.status(404).json({
          success: false,
          error: 'Photo not found',
          code: 'NOT_FOUND'
        })
      }

      const photo = photos[0]

      // Cancel any running RunPod jobs
      if (photo.makeovers?.length > 0) {
        for (const makeover of photo.makeovers) {
          if (makeover.runpod_job_id) {
            await runpodService.cancelJob(makeover.runpod_job_id)
          }
        }
      }

      // Delete from Cloudflare R2
      await r2Service.deletePhoto(photo.cloudflare_key)

      // Delete from Supabase (cascades to makeovers and product_suggestions)
      const { error: deleteError } = await supabaseService.supabase
        .from('photos')
        .delete()
        .eq('id', photoId)
        .eq('clerk_user_id', userId)

      if (deleteError) throw deleteError

      res.json({
        success: true,
        message: 'Photo deleted successfully'
      })

    } catch (error) {
      console.error('❌ Failed to delete photo:', error)
      res.status(500).json({
        success: false,
        error: 'Failed to delete photo',
        message: error.message,
        code: 'DELETE_ERROR'
      })
    }
  }
)

// =============================================================================
// DIRECT UPLOAD ENDPOINTS
// =============================================================================

/**
 * 🔗 Get signed upload URL for direct client uploads
 * POST /api/photos/signed-url
 */
router.post('/signed-url', 
  ClerkExpressRequireAuth(),
  async (req, res) => {
    try {
      const { userId } = req.auth
      const { originalName, contentType } = req.body

      if (!originalName) {
        return res.status(400).json({
          success: false,
          error: 'Original filename required',
          code: 'MISSING_FILENAME'
        })
      }

      const signedUrlData = await r2Service.getSignedUploadUrl(userId, originalName)

      res.json({
        success: true,
        data: signedUrlData
      })

    } catch (error) {
      console.error('❌ Failed to generate signed URL:', error)
      res.status(500).json({
        success: false,
        error: 'Failed to generate upload URL',
        message: error.message,
        code: 'SIGNED_URL_ERROR'
      })
    }
  }
)

// =============================================================================
// HEALTH CHECK
// =============================================================================

/**
 * 🏥 Health check for photo service
 * GET /api/photos/health
 */
router.get('/health', async (req, res) => {
  try {
    const r2Health = await r2Service.healthCheck()
    const supabaseHealth = await supabaseService.healthCheck()
    const runpodHealth = await runpodService.healthCheck()

    const overall = r2Health && supabaseHealth && runpodHealth

    res.json({
      success: true,
      status: overall ? 'healthy' : 'degraded',
      services: {
        cloudflare_r2: r2Health ? 'healthy' : 'unhealthy',
        supabase: supabaseHealth ? 'healthy' : 'unhealthy',
        runpod: runpodHealth ? 'healthy' : 'unhealthy'
      },
      timestamp: new Date().toISOString()
    })

  } catch (error) {
    console.error('❌ Health check failed:', error)
    res.status(500).json({
      success: false,
      status: 'unhealthy',
      error: error.message
    })
  }
})

export default router