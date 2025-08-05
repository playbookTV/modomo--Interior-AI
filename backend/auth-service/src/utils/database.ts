import { createClient, SupabaseClient } from '@supabase/supabase-js'
import { logger } from './logger'

// Re-export types and service from the main backend
// This allows the auth service to use the same Supabase integration
export interface UserData {
  clerk_user_id: string
  email?: string
  subscription_tier?: 'free' | 'pro' | 'premium'
  preferences?: any
  total_photos?: number
  total_makeovers?: number
}

export class SupabaseService {
  public supabase: SupabaseClient
  private isConnected: boolean = false

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
      supabaseKey,
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

    logger.info('Supabase client initialized', {
      url: supabaseUrl,
      hasServiceKey: !!process.env.SUPABASE_SERVICE_ROLE_KEY
    })
  }

  /**
   * Test database connection
   */
  async connect(): Promise<boolean> {
    try {
      const { error } = await this.supabase
        .from('users')
        .select('count', { count: 'exact', head: true })

      if (error) {
        logger.error('Database connection failed:', error)
        this.isConnected = false
        return false
      }

      this.isConnected = true
      logger.info('Database connection established')
      return true
    } catch (error) {
      logger.error('Database connection error:', error)
      this.isConnected = false
      return false
    }
  }

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
      logger.error('❌ Supabase health check failed:', error)
      return false
    }
  }

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

      logger.info(`✅ User upserted: ${clerkUserId}`)
      return data
    } catch (error) {
      logger.error('❌ Failed to create/update user:', error)
      throw new Error(`User operation failed: ${error.message}`)
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
      logger.error('❌ Failed to get user:', error)
      throw error
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
      logger.error('❌ Failed to get user stats:', error)
      throw error
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

      logger.info(`✅ Retrieved ${data.length} photos for user: ${clerkUserId}`)
      return data
    } catch (error) {
      logger.error('❌ Failed to get user photos:', error)
      throw error
    }
  }

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

      logger.info(`✅ User stats updated for: ${clerkUserId}`)
    } catch (error) {
      logger.error('❌ Failed to update user stats:', error)
      // Don't throw error for stats - it's not critical
    }
  }

  /**
   * Execute raw SQL query (use with caution)
   */
  async query(sql: string, params?: any[]) {
    try {
      const { data, error } = await this.supabase.rpc('execute_sql', {
        query: sql,
        params: params || []
      })

      if (error) throw error
      return data
    } catch (error) {
      logger.error('❌ Raw query failed:', error)
      throw error
    }
  }

  /**
   * Close database connection
   */
  async disconnect(): Promise<void> {
    // Supabase doesn't require explicit disconnection
    this.isConnected = false
    logger.info('Database connection closed')
  }

  /**
   * Get connection status
   */
  isConnectionActive(): boolean {
    return this.isConnected
  }
}

// Singleton instance
export const supabaseService = new SupabaseService()

// Connection helper
export const connectDatabase = async (): Promise<boolean> => {
  try {
    const connected = await supabaseService.connect()
    if (connected) {
      logger.info('✅ Database connected successfully')
    } else {
      logger.error('❌ Database connection failed')
    }
    return connected
  } catch (error) {
    logger.error('❌ Database connection error:', error)
    return false
  }
}

// Graceful shutdown helper
export const disconnectDatabase = async (): Promise<void> => {
  await supabaseService.disconnect()
}

export default supabaseService