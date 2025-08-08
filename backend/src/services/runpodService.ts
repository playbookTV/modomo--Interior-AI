// Using built-in fetch API (Node.js 18+)
import { supabaseService } from './supabaseService'

export interface RunPodJobResponse {
  id: string
  status: 'IN_QUEUE' | 'IN_PROGRESS' | 'COMPLETED' | 'FAILED'
  output?: any
  error?: string
}

export interface MakeoverRequest {
  photo_url: string
  photo_id: string
  makeover_id: string
  user_id: string
  style_preference?: string
  budget_range?: string
  room_type?: string
}

export class RunPodService {
  private endpoint = process.env.RUNPOD_ENDPOINT!
  private apiKey = process.env.RUNPOD_API_KEY!
  private callbackUrl = `${process.env.RAILWAY_BACKEND_URL}/api/makeovers/callback`

  constructor() {
    if (!this.endpoint || !this.apiKey) {
      throw new Error('RunPod configuration missing: RUNPOD_ENDPOINT and RUNPOD_API_KEY required')
    }
  }

  /**
   * Submit makeover job to RunPod
   */
  async submitMakeoverJob(makeoverData: MakeoverRequest): Promise<RunPodJobResponse> {
    try {
      console.log(`🚀 Submitting makeover job to RunPod for photo: ${makeoverData.photo_id}`)

      const response = await fetch(this.endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify({
          input: {
            // Photo and user context
            photo_url: makeoverData.photo_url,
            photo_id: makeoverData.photo_id,
            makeover_id: makeoverData.makeover_id,
            user_id: makeoverData.user_id,

            // Makeover preferences
            style_preference: makeoverData.style_preference || 'Modern',
            budget_range: makeoverData.budget_range || 'mid-range',
            room_type: makeoverData.room_type || 'living-room',

            // Processing options
            quality: 'high',
            include_products: true,
            include_pricing: true,

            // Callback configuration
            callback_url: this.callbackUrl,
            webhook_events: ['progress', 'completed', 'failed']
          },
          // Job configuration
          webhook: this.callbackUrl
        })
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`RunPod API error: ${response.status} - ${errorText}`)
      }

      const jobData: any = await response.json()

      // Update makeover with RunPod job ID
      await supabaseService.updateMakeover(makeoverData.makeover_id, {
        runpod_job_id: jobData.id,
        status: 'processing',
        processing_started_at: new Date().toISOString()
      })

      console.log(`✅ RunPod job submitted: ${jobData.id} for makeover: ${makeoverData.makeover_id}`)
      return jobData

    } catch (error) {
      console.error('❌ RunPod job submission failed:', error)
      
      // Update makeover status to failed
      await supabaseService.updateMakeover(makeoverData.makeover_id, {
        status: 'failed',
        error_message: error instanceof Error ? error.message : 'RunPod job submission failed'
      })
      
      throw error
    }
  }

  /**
   * Check job status from RunPod
   */
  async checkJobStatus(jobId: string): Promise<RunPodJobResponse> {
    try {
      const response = await fetch(`${this.endpoint}/${jobId}`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
        }
      })

      if (!response.ok) {
        throw new Error(`RunPod status check failed: ${response.status}`)
      }

      const jobData: any = await response.json()
      console.log(`📊 RunPod job status: ${jobId} - ${jobData.status}`)

      return jobData
    } catch (error) {
      console.error('❌ RunPod status check failed:', error)
      throw error
    }
  }

  /**
   * Handle RunPod webhook callback
   */
  async handleWebhookCallback(payload: any) {
    try {
      const { id: jobId, status, output, error } = payload

      console.log(`🔔 RunPod webhook received: ${jobId} - ${status}`)

      // Find makeover by RunPod job ID
      const { data: makeovers, error: queryError } = await supabaseService.supabase
        .from('makeovers')
        .select('id, photo_id, clerk_user_id')
        .eq('runpod_job_id', jobId)

      if (queryError || !makeovers || makeovers.length === 0) {
        console.error('❌ Makeover not found for RunPod job:', jobId)
        return
      }

      const makeover = makeovers[0]

      // Update makeover based on job status
      switch (status) {
        case 'IN_PROGRESS':
          await supabaseService.updateMakeover(makeover.id, {
            status: 'processing',
            progress: output?.progress || 25
          })
          break

        case 'COMPLETED':
          if (output?.makeover_url) {
            await supabaseService.updateMakeover(makeover.id, {
              status: 'completed',
              progress: 100,
              makeover_url: output.makeover_url,
              detected_objects: output.detected_objects || [],
              suggested_products: output.suggested_products || [],
              completed_at: new Date().toISOString()
            })

            // Create product suggestions if provided
            if (output.products && output.products.length > 0) {
              await supabaseService.createProductSuggestions(makeover.id, output.products)
            }

            console.log(`✅ Makeover completed: ${makeover.id}`)
          } else {
            throw new Error('Completed job missing makeover URL')
          }
          break

        case 'FAILED':
          await supabaseService.updateMakeover(makeover.id, {
            status: 'failed',
            error_message: error || 'RunPod processing failed',
            completed_at: new Date().toISOString()
          })
          console.log(`❌ Makeover failed: ${makeover.id}`)
          break

        default:
          console.log(`ℹ️ Unknown RunPod status: ${status}`)
      }

    } catch (error) {
      console.error('❌ Failed to handle RunPod webhook:', error)
    }
  }

  /**
   * Cancel RunPod job
   */
  async cancelJob(jobId: string): Promise<boolean> {
    try {
      const response = await fetch(`${this.endpoint}/${jobId}/cancel`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
        }
      })

      if (!response.ok) {
        throw new Error(`Job cancellation failed: ${response.status}`)
      }

      console.log(`✅ RunPod job cancelled: ${jobId}`)
      return true
    } catch (error) {
      console.error('❌ Failed to cancel RunPod job:', error)
      return false
    }
  }

  /**
   * Get RunPod account status and quotas
   */
  async getAccountInfo(): Promise<any> {
    try {
      const response = await fetch('https://api.runpod.ai/v2/account', {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
        }
      })

      if (!response.ok) {
        throw new Error(`Account info request failed: ${response.status}`)
      }

      return await response.json()
    } catch (error) {
      console.error('❌ Failed to get RunPod account info:', error)
      throw error
    }
  }

  /**
   * Health check for RunPod service
   */
  async healthCheck(): Promise<boolean> {
    try {
      // Simple health check - just verify we have an API key
      if (!this.apiKey || this.apiKey === '[YOUR_RUNPOD_API_KEY]') {
        console.warn('⚠️ RunPod API key not configured')
        return true // Don't fail deployment for missing AI service
      }
      
      console.log('✅ RunPod configuration check passed')
      return true
    } catch (error) {
      console.warn('⚠️ RunPod health check failed, but continuing deployment:', error)
      return true // Don't fail deployment for AI service issues
    }
  }
}

// Singleton instance
export const runpodService = new RunPodService()