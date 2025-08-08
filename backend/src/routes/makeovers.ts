import express, { Request, Response, Router } from 'express'
import crypto from 'crypto'
import { SupabaseService } from '../services/supabaseService'
import { RunPodService } from '../services/runpodService'
import { ClerkExpressRequireAuth } from '@clerk/clerk-sdk-node'

const router: Router = express.Router()
const supabaseService = new SupabaseService()
const runpodService = new RunPodService()

// Webhook signature verification utility
function verifyWebhookSignature(payload: string, signature: string, secret: string): boolean {
  if (!secret) {
    console.warn('⚠️ No webhook secret configured - skipping signature verification')
    return true // Allow if no secret is set (for development)
  }

  try {
    const expectedSignature = crypto
      .createHmac('sha256', secret)
      .update(payload)
      .digest('hex')
    
    const providedSignature = signature.replace('sha256=', '')
    
    return crypto.timingSafeEqual(
      Buffer.from(expectedSignature, 'hex'),
      Buffer.from(providedSignature, 'hex')
    )
  } catch (error) {
    console.error('❌ Webhook signature verification failed:', error)
    return false
  }
}

// =============================================================================
// MAKEOVER MANAGEMENT ENDPOINTS
// =============================================================================

/**
 * 🎨 Get makeover details by ID
 * GET /api/makeovers/:makeoverId
 */
router.get('/:makeoverId', 
  ClerkExpressRequireAuth(),
  async (req, res) => {
    try {
      const { userId } = req.auth as any
      const { makeoverId } = req.params

      const makeover = await supabaseService.getMakeover(makeoverId)

      // Verify ownership
      if (makeover.clerk_user_id !== userId) {
        return res.status(403).json({
          success: false,
          error: 'Access denied',
          code: 'FORBIDDEN'
        })
      }

      return res.json({
        success: true,
        data: makeover
      })

    } catch (error) {
      console.error('❌ Failed to get makeover:', error)
      return res.status(500).json({
        success: false,
        error: 'Failed to get makeover',
        message: error instanceof Error ? error.message : String(error),
        code: 'GET_ERROR'
      })
    }
  }
)

/**
 * 🔄 Retry failed makeover
 * POST /api/makeovers/:makeoverId/retry
 */
router.post('/:makeoverId/retry', 
  ClerkExpressRequireAuth(),
  async (req, res) => {
    try {
      const { userId } = req.auth as any
      const { makeoverId } = req.params

      // Get makeover details
      const makeover = await supabaseService.getMakeover(makeoverId)

      // Verify ownership
      if (makeover.clerk_user_id !== userId) {
        return res.status(403).json({
          success: false,
          error: 'Access denied',
          code: 'FORBIDDEN'
        })
      }

      // Check if makeover is in a retryable state
      if (makeover.status !== 'failed') {
        return res.status(400).json({
          success: false,
          error: 'Makeover is not in a failed state',
          code: 'INVALID_STATE'
        })
      }

      // Reset makeover status
      await supabaseService.updateMakeover(makeoverId, {
        status: 'queued',
        progress: 0,
        error_message: undefined,
        processing_started_at: undefined,
        completed_at: undefined
      })

      // Resubmit to RunPod
      const jobResponse = await runpodService.submitMakeoverJob({
        photo_url: makeover.photos.optimized_url,
        photo_id: makeover.photo_id,
        makeover_id: makeoverId,
        user_id: userId,
        style_preference: makeover.style_preference,
        budget_range: makeover.budget_range,
        room_type: makeover.room_type
      })

      return res.json({
        success: true,
        data: {
          makeover_id: makeoverId,
          runpod_job_id: jobResponse.id,
          status: 'queued'
        },
        message: 'Makeover retry initiated'
      })

    } catch (error) {
      console.error('❌ Failed to retry makeover:', error)
      return res.status(500).json({
        success: false,
        error: 'Failed to retry makeover',
        message: error instanceof Error ? error.message : String(error),
        code: 'RETRY_ERROR'
      })
    }
  }
)

/**
 * ❌ Cancel active makeover
 * POST /api/makeovers/:makeoverId/cancel
 */
router.post('/:makeoverId/cancel', 
  ClerkExpressRequireAuth(),
  async (req: Request, res: Response) => {
    try {
      const { userId } = req.auth as any
      const { makeoverId } = req.params

      // Get makeover details
      const { data: makeovers, error } = await supabaseService.supabase
        .from('makeovers')
        .select('status, runpod_job_id, clerk_user_id')
        .eq('id', makeoverId)

      if (error) throw error
      if (!makeovers || makeovers.length === 0) {
        return res.status(404).json({
          success: false,
          error: 'Makeover not found',
          code: 'NOT_FOUND'
        })
      }

      const makeover = makeovers[0]

      // Verify ownership
      if (makeover.clerk_user_id !== userId) {
        return res.status(403).json({
          success: false,
          error: 'Access denied',
          code: 'FORBIDDEN'
        })
      }

      // Check if makeover can be cancelled
      if (!['queued', 'processing'].includes(makeover.status)) {
        return res.status(400).json({
          success: false,
          error: 'Makeover cannot be cancelled in current state',
          code: 'INVALID_STATE'
        })
      }

      // Cancel RunPod job if exists
      if (makeover.runpod_job_id) {
        await runpodService.cancelJob(makeover.runpod_job_id)
      }

      // Update makeover status
      await supabaseService.updateMakeover(makeoverId, {
        status: 'failed',
        error_message: 'Cancelled by user',
        completed_at: new Date().toISOString()
      })

      return res.json({
        success: true,
        message: 'Makeover cancelled successfully'
      })

    } catch (error) {
      console.error('❌ Failed to cancel makeover:', error)
      return res.status(500).json({
        success: false,
        error: 'Failed to cancel makeover',
        message: error instanceof Error ? error.message : String(error),
        code: 'CANCEL_ERROR'
      })
    }
  }
)

/**
 * 📊 Get user's makeover history
 * GET /api/makeovers
 */
router.get('/', 
  ClerkExpressRequireAuth(),
  async (req: Request, res: Response) => {
    try {
      const { userId } = req.auth as any
      const limit = parseInt(req.query.limit as string) || 20
      const offset = parseInt(req.query.offset as string) || 0
      const status = req.query.status as string

      let query = supabaseService.supabase
        .from('makeovers')
        .select(`
          *,
          photos (
            id,
            original_url,
            optimized_url,
            cloudflare_key,
            original_name
          ),
          product_suggestions (*)
        `)
        .eq('clerk_user_id', userId)
        .order('created_at', { ascending: false })
        .range(offset, offset + limit - 1)

      // Filter by status if provided
      if (status && ['queued', 'processing', 'completed', 'failed'].includes(status)) {
        query = query.eq('status', status)
      }

      const { data: makeovers, error } = await query

      if (error) throw error

      return res.json({
        success: true,
        data: makeovers || [],
        count: makeovers?.length || 0,
        pagination: {
          limit,
          offset,
          hasMore: (makeovers?.length || 0) === limit
        }
      })

    } catch (error) {
      console.error('❌ Failed to get makeover history:', error)
      return res.status(500).json({
        success: false,
        error: 'Failed to get makeover history',
        message: error instanceof Error ? error.message : String(error),
        code: 'HISTORY_ERROR'
      })
    }
  }
)

// =============================================================================
// RUNPOD WEBHOOK HANDLERS
// =============================================================================

/**
 * 🔔 RunPod webhook callback handler
 * POST /api/makeovers/callback
 */
router.post('/callback', async (req: Request, res: Response) => {
  try {
    console.log('🔔 RunPod webhook received:', JSON.stringify(req.body, null, 2))

    // Verify webhook signature for security
    const signature = req.headers['x-signature'] as string
    const webhookSecret = process.env.RUNPOD_WEBHOOK_SECRET
    const rawPayload = JSON.stringify(req.body)

    if (!verifyWebhookSignature(rawPayload, signature || '', webhookSecret || '')) {
      console.error('❌ Invalid webhook signature')
      return res.status(401).json({
        success: false,
        error: 'Invalid webhook signature'
      })
    }

    const payload = req.body

    if (!payload.id) {
      return res.status(400).json({
        success: false,
        error: 'Invalid webhook payload: missing job ID'
      })
    }

    // Handle the webhook asynchronously
    runpodService.handleWebhookCallback(payload)
      .catch(error => {
        console.error('❌ Webhook handling failed:', error)
      })

    // Respond immediately to RunPod
    return res.json({
      success: true,
      message: 'Webhook received and processing'
    })

  } catch (error) {
    console.error('❌ Webhook callback failed:', error)
    return res.status(500).json({
      success: false,
      error: 'Webhook processing failed',
      message: error instanceof Error ? error.message : String(error)
    })
  }
})

/**
 * 🔍 Manual job status check
 * POST /api/makeovers/:makeoverId/check-status
 */
router.post('/:makeoverId/check-status', 
  ClerkExpressRequireAuth(),
  async (req: Request, res: Response) => {
    try {
      const { userId } = req.auth as any
      const { makeoverId } = req.params

      // Get makeover with RunPod job ID
      const { data: makeovers, error } = await supabaseService.supabase
        .from('makeovers')
        .select('runpod_job_id, clerk_user_id, status')
        .eq('id', makeoverId)

      if (error) throw error
      if (!makeovers || makeovers.length === 0) {
        return res.status(404).json({
          success: false,
          error: 'Makeover not found',
          code: 'NOT_FOUND'
        })
      }

      const makeover = makeovers[0]

      // Verify ownership
      if (makeover.clerk_user_id !== userId) {
        return res.status(403).json({
          success: false,
          error: 'Access denied',
          code: 'FORBIDDEN'
        })
      }

      if (!makeover.runpod_job_id) {
        return res.status(400).json({
          success: false,
          error: 'No RunPod job associated with this makeover',
          code: 'NO_JOB'
        })
      }

      // Check job status from RunPod
      const jobStatus = await runpodService.checkJobStatus(makeover.runpod_job_id)

      // Update makeover if status changed
      if (jobStatus.status === 'COMPLETED' && makeover.status !== 'completed') {
        await runpodService.handleWebhookCallback(jobStatus)
      }

      return res.json({
        success: true,
        data: {
          makeover_id: makeoverId,
          runpod_job_id: makeover.runpod_job_id,
          runpod_status: jobStatus.status,
          local_status: makeover.status
        }
      })

    } catch (error) {
      console.error('❌ Failed to check job status:', error)
      return res.status(500).json({
        success: false,
        error: 'Failed to check job status',
        message: error instanceof Error ? error.message : String(error),
        code: 'STATUS_CHECK_ERROR'
      })
    }
  }
)

// =============================================================================
// STATISTICS & ANALYTICS
// =============================================================================

/**
 * 📈 Get user makeover statistics
 * GET /api/makeovers/stats
 */
router.get('/stats', 
  ClerkExpressRequireAuth(),
  async (req: Request, res: Response) => {
    try {
      const { userId } = req.auth as any

      // Get makeover statistics
      const { data: stats, error } = await supabaseService.supabase
        .from('makeovers')
        .select('status')
        .eq('clerk_user_id', userId)

      if (error) throw error

      // Count by status
      const statusCounts: any = (stats || []).reduce((acc: any, makeover: any) => {
        acc[makeover.status] = (acc[makeover.status] || 0) + 1
        return acc
      }, {})

      // Get user profile stats
      const userStats = await supabaseService.getUserStats(userId)

      return res.json({
        success: true,
        data: {
          total_makeovers: userStats.total_makeovers,
          total_photos: userStats.total_photos,
          status_breakdown: {
            queued: statusCounts.queued || 0,
            processing: statusCounts.processing || 0,
            completed: statusCounts.completed || 0,
            failed: statusCounts.failed || 0
          },
          subscription_tier: userStats.subscription_tier,
          member_since: userStats.created_at
        }
      })

    } catch (error) {
      console.error('❌ Failed to get makeover stats:', error)
      return res.status(500).json({
        success: false,
        error: 'Failed to get makeover statistics',
        message: error instanceof Error ? error.message : String(error),
        code: 'STATS_ERROR'
      })
    }
  }
)

export default router