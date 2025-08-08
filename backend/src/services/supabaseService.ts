import { createClient, SupabaseClient } from '@supabase/supabase-js'

export interface UserData {
  clerk_user_id: string
  email?: string
  subscription_tier?: 'free' | 'pro' | 'premium'
  preferences?: any
}

export interface PhotoData {
  clerk_user_id: string
  original_url: string
  optimized_url?: string
  cloudflare_key: string
  original_name?: string
  mime_type?: string
  metadata?: any
  original_size?: number
  width?: number
  height?: number
  taken_at?: string
}

export interface MakeoverData {
  photo_id: string
  clerk_user_id: string
  style_preference?: string
  budget_range?: string
  room_type?: string
}

export class SupabaseService {
  public supabase: SupabaseClient

  constructor() {
    const supabaseUrl = process.env.SUPABASE_URL || process.env.EXPO_PUBLIC_SUPABASE_URL
    const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY
    
    if (!supabaseUrl) {
      throw new Error('Missing SUPABASE_URL environment variable')
    }
    if (!supabaseKey) {
      throw new Error('Missing SUPABASE_SERVICE_ROLE_KEY environment variable')
    }
    
    this.supabase = createClient(
      supabaseUrl,
      supabaseKey, // Service role for backend operations
      {
        auth: {
          autoRefreshToken: false,
          persistSession: false
        },
        db: {
          schema: 'public'
        }
      }
    )
  }

  // =============================================================================
  // USER MANAGEMENT
  // =============================================================================

  /**
   * Create or update user with Clerk integration
   */
  async createOrUpdateUser(clerkUserId: string, email?: string, additionalData?: Partial<UserData>) {
    try {
      const { data, error } = await this.supabase
        .from('users')
        .upsert({
          clerk_user_id: clerkUserId,
          email,
          ...additionalData,
          updated_at: new Date().toISOString()
        }, {
          onConflict: 'clerk_user_id'
        })
        .select()
        .single()

      if (error) throw error

      console.log(`✅ User upserted: ${clerkUserId}`)
      return data
    } catch (error) {
      console.error('❌ Failed to create/update user:', error)
      throw new Error(`User operation failed: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  /**
   * Get user by Clerk ID
   */
  async getUser(clerkUserId: string) {
    try {
      const { data, error } = await this.supabase
        .from('users')
        .select('*')
        .eq('clerk_user_id', clerkUserId)
        .single()

      if (error && error.code !== 'PGRST116') throw error // PGRST116 = not found

      return data
    } catch (error) {
      console.error('❌ Failed to get user:', error)
      throw error
    }
  }

  // =============================================================================
  // PHOTO MANAGEMENT
  // =============================================================================

  /**
   * Create photo record in database
   */
  async createPhoto(photoData: PhotoData) {
    try {
      const { data, error } = await this.supabase
        .from('photos')
        .insert({
          ...photoData,
          status: 'uploaded',
          created_at: new Date().toISOString()
        })
        .select()
        .single()

      if (error) throw error

      // Increment user photo count
      await this.incrementUserStats(photoData.clerk_user_id, { total_photos: 1 })

      console.log(`✅ Photo created: ${data.id}`)
      return data
    } catch (error) {
      console.error('❌ Failed to create photo:', error)
      throw new Error(`Photo creation failed: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  /**
   * Get user photos with makeover status
   */
  async getUserPhotos(clerkUserId: string, limit = 50, offset = 0) {
    try {
      const { data, error } = await this.supabase
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
            runpod_job_id
          )
        `)
        .eq('clerk_user_id', clerkUserId)
        .order('created_at', { ascending: false })
        .range(offset, offset + limit - 1)

      if (error) throw error

      console.log(`✅ Retrieved ${data.length} photos for user: ${clerkUserId}`)
      return data
    } catch (error) {
      console.error('❌ Failed to get user photos:', error)
      throw error
    }
  }

  /**
   * Update photo status
   */
  async updatePhoto(photoId: string, updates: Partial<PhotoData>) {
    try {
      const { data, error } = await this.supabase
        .from('photos')
        .update({
          ...updates,
          updated_at: new Date().toISOString()
        })
        .eq('id', photoId)
        .select()
        .single()

      if (error) throw error

      console.log(`✅ Photo updated: ${photoId}`)
      return data
    } catch (error) {
      console.error('❌ Failed to update photo:', error)
      throw error
    }
  }

  // =============================================================================
  // MAKEOVER MANAGEMENT
  // =============================================================================

  /**
   * Create makeover record
   */
  async createMakeover(makeoverData: MakeoverData) {
    try {
      const { data, error } = await this.supabase
        .from('makeovers')
        .insert({
          ...makeoverData,
          status: 'queued',
          progress: 0,
          created_at: new Date().toISOString()
        })
        .select()
        .single()

      if (error) throw error

      // Increment user makeover count
      await this.incrementUserStats(makeoverData.clerk_user_id, { total_makeovers: 1 })

      console.log(`✅ Makeover created: ${data.id}`)
      return data
    } catch (error) {
      console.error('❌ Failed to create makeover:', error)
      throw new Error(`Makeover creation failed: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  /**
   * Update makeover with RunPod job status
   */
  async updateMakeover(makeoverId: string, updates: {
    runpod_job_id?: string
    status?: 'queued' | 'processing' | 'completed' | 'failed'
    progress?: number
    makeover_url?: string
    error_message?: string
    processing_started_at?: string
    completed_at?: string
    detected_objects?: any[]
    suggested_products?: any[]
  }) {
    try {
      const { data, error } = await this.supabase
        .from('makeovers')
        .update({
          ...updates,
          updated_at: new Date().toISOString()
        })
        .eq('id', makeoverId)
        .select()
        .single()

      if (error) throw error

      console.log(`✅ Makeover updated: ${makeoverId} (${updates.status})`)
      return data
    } catch (error) {
      console.error('❌ Failed to update makeover:', error)
      throw error
    }
  }

  /**
   * Get makeover by ID
   */
  async getMakeover(makeoverId: string) {
    try {
      const { data, error } = await this.supabase
        .from('makeovers')
        .select(`
          *,
          photos (
            id,
            original_url,
            optimized_url,
            cloudflare_key
          ),
          product_suggestions (*)
        `)
        .eq('id', makeoverId)
        .single()

      if (error) throw error

      return data
    } catch (error) {
      console.error('❌ Failed to get makeover:', error)
      throw error
    }
  }

  // =============================================================================
  // PRODUCT SUGGESTIONS
  // =============================================================================

  /**
   * Create product suggestions for a makeover
   */
  async createProductSuggestions(makeoverId: string, products: Array<{
    product_name: string
    category?: string
    description?: string
    brand?: string
    amazon_price?: number
    amazon_url?: string
    ikea_price?: number
    ikea_url?: string
    wayfair_price?: number
    wayfair_url?: string
    image_url?: string
    confidence_score?: number
  }>) {
    try {
      const productsWithMakeoverId = products.map(product => ({
        ...product,
        makeover_id: makeoverId,
        created_at: new Date().toISOString()
      }))

      const { data, error } = await this.supabase
        .from('product_suggestions')
        .insert(productsWithMakeoverId)
        .select()

      if (error) throw error

      console.log(`✅ Created ${data.length} product suggestions for makeover: ${makeoverId}`)
      return data
    } catch (error) {
      console.error('❌ Failed to create product suggestions:', error)
      throw error
    }
  }

  // =============================================================================
  // STATISTICS & ANALYTICS
  // =============================================================================

  /**
   * Increment user statistics
   */
  async incrementUserStats(clerkUserId: string, increments: { 
    total_photos?: number
    total_makeovers?: number 
  }) {
    try {
      const { error } = await this.supabase.rpc('increment_user_stats', {
        user_clerk_id: clerkUserId,
        photo_increment: increments.total_photos || 0,
        makeover_increment: increments.total_makeovers || 0
      })

      if (error) throw error

      console.log(`✅ User stats updated for: ${clerkUserId}`)
    } catch (error) {
      console.error('❌ Failed to update user stats:', error)
      // Don't throw error for stats - it's not critical
    }
  }

  /**
   * Get user statistics
   */
  async getUserStats(clerkUserId: string) {
    try {
      const { data, error } = await this.supabase
        .from('users')
        .select('total_photos, total_makeovers, subscription_tier, created_at')
        .eq('clerk_user_id', clerkUserId)
        .single()

      if (error) throw error

      return data
    } catch (error) {
      console.error('❌ Failed to get user stats:', error)
      throw error
    }
  }

  // =============================================================================
  // HEALTH & UTILITIES
  // =============================================================================

  /**
   * Health check for Supabase connection
   */
  async healthCheck(): Promise<boolean> {
    try {
      const { error } = await this.supabase
        .from('users')
        .select('count', { count: 'exact', head: true })

      return !error
    } catch (error) {
      console.error('❌ Supabase health check failed:', error)
      return false
    }
  }
}

// Singleton instance
export const supabaseService = new SupabaseService()