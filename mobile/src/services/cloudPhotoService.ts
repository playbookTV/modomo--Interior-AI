import { useAuth } from '@clerk/clerk-expo'
import { createClient } from '@supabase/supabase-js'
import ImageResizer from '@bam.tech/react-native-image-resizer'
import { MMKV } from 'react-native-mmkv'
import { Logger } from '../utils/logger'

// Generate a unique encryption key based on device/user
function getOrCreateEncryptionKey(): string {
  const existingKey = MMKV.getString('reroom_encryption_key')
  if (existingKey) {
    return existingKey
  }
  
  // Generate a new key using device characteristics and timestamp
  const deviceId = require('react-native-device-info').getUniqueId?.() || 'unknown'
  const timestamp = Date.now().toString()
  const randomComponent = Math.random().toString(36)
  
  const newKey = `${deviceId}-${timestamp}-${randomComponent}`
  MMKV.set('reroom_encryption_key', newKey)
  return newKey
}

// Local storage for offline capability with dynamic encryption key
const photoStorage = new MMKV({
  id: 'cloud-photo-storage',
  encryptionKey: getOrCreateEncryptionKey()
})

export interface CloudPhotoResult {
  id: string
  url: string
  variants: string[]
  size: number
  originalName: string
  uploadedAt: string
  makeover?: {
    id: string
    status: string
    style_preference: string
  }
}

export interface PhotoWithMakeover {
  id: string
  original_url: string
  optimized_url: string
  cloudflare_key: string
  original_name: string
  created_at: string
  makeovers?: Array<{
    id: string
    status: 'queued' | 'processing' | 'completed' | 'failed'
    progress: number
    makeover_url?: string
    style_preference: string
    completed_at?: string
    error_message?: string
  }>
}

export interface LocalPhoto {
  id: string
  uri: string
  cloudUrl?: string
  variants?: string[]
  uploaded: boolean
  synced: boolean
  makeoverId?: string
  makeover?: any
  metadata?: any
  error?: string
  timestamp: number
}

export class CloudPhotoService {
  private supabase: any
  private auth: any
  private backendUrl = process.env.EXPO_PUBLIC_API_BASE_URL_CLOUD || 'https://reroom-production-dcb0.up.railway.app:6969'

  constructor(auth: ReturnType<typeof useAuth>) {
    this.auth = auth
    
    // Initialize Supabase with Clerk integration
    this.supabase = createClient(
      process.env.EXPO_PUBLIC_SUPABASE_URL!,
      process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY!,
      {
        global: {
          headers: (async () => {
            const token = await auth.getToken({ template: 'supabase' })
            return token ? { Authorization: `Bearer ${token}` } : {}
          }) as any,
        },
        auth: {
          autoRefreshToken: false,
          persistSession: false
        }
      }
    )
  }

  // =============================================================================
  // PHOTO CAPTURE & UPLOAD
  // =============================================================================

  /**
   * 📸 Capture photo, optimize, and upload to cloud
   */
  async captureAndUpload(
    photoUri: string, 
    options: {
      triggerAI?: boolean
      stylePreference?: string
      budgetRange?: string
      roomType?: string
      metadata?: any
    } = {}
  ): Promise<CloudPhotoResult> {
    try {
      Logger.info('Starting cloud photo capture and upload', { photoUri, options })

      // Step 1: Optimize photo locally for better upload performance
      const optimized = await this.optimizePhoto(photoUri)

      // Step 2: Prepare metadata
      const metadata = {
        triggerAI: options.triggerAI !== false,
        stylePreference: options.stylePreference || 'Modern',
        budgetRange: options.budgetRange || 'mid-range',
        roomType: options.roomType || 'living-room',
        optimized: true,
        capturedAt: new Date().toISOString(),
        version: '2.0',
        ...options.metadata
      }

      // Step 3: Upload to Railway backend → Cloudflare R2
      const result = await this.uploadToBackend(optimized.uri, metadata)

      // Step 4: Save locally for offline access and sync status
      await this.saveLocalPhoto({
        id: result.id,
        uri: optimized.uri,
        cloudUrl: result.url,
        variants: result.variants,
        uploaded: true,
        synced: true,
        makeoverId: result.makeover?.id,
        metadata,
        timestamp: Date.now()
      })

      // Step 5: Subscribe to real-time makeover updates if AI is triggered
      if (result.makeover) {
        this.subscribeMakeoverUpdates(result.makeover.id)
      }

      Logger.info('Photo uploaded successfully to cloud', { 
        photoId: result.id, 
        makeoverId: result.makeover?.id 
      })

      return result

    } catch (error) {
      Logger.error('Cloud photo upload failed', { error, photoUri })
      
      // Save locally as fallback for offline capability
      await this.saveLocalPhoto({
        id: `local_${Date.now()}`,
        uri: photoUri,
        uploaded: false,
        synced: false,
        error: error.message,
        timestamp: Date.now()
      })
      
      throw error
    }
  }

  /**
   * 🔧 Optimize photo for cloud upload
   */
  private async optimizePhoto(photoUri: string) {
    try {
      const optimized = await ImageResizer.createResizedImage(
        photoUri,
        1200, // Max width
        1200, // Max height
        'JPEG',
        80, // Quality
        0, // Rotation
        undefined, // Output path
        false, // Keep metadata
        {
          mode: 'contain',
          onlyScaleDown: true,
        }
      )

      Logger.info('Photo optimized for cloud upload', {
        originalUri: photoUri,
        optimizedUri: optimized.uri,
        size: optimized.size
      })

      return optimized
    } catch (error) {
      Logger.error('Photo optimization failed', { error, photoUri })
      throw new Error('Failed to optimize photo for upload')
    }
  }

  /**
   * 🚀 Upload to Railway backend
   */
  private async uploadToBackend(photoUri: string, metadata: any): Promise<CloudPhotoResult> {
    try {
      // Get Clerk authentication token
      const token = await this.auth.getToken()
      if (!token) {
        throw new Error('Authentication required')
      }

      // Prepare form data
      const formData = new FormData()
      formData.append('photo', {
        uri: photoUri,
        type: 'image/jpeg',
        name: `reroom_photo_${Date.now()}.jpg`,
      } as any)
      formData.append('metadata', JSON.stringify(metadata))

      // Upload with timeout and retry logic
      const response = await this.makeRequestWithRetry(
        `${this.backendUrl}/api/photos/upload`,
        {
          method: 'POST',
          body: formData,
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'multipart/form-data',
          },
        }
      )

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.message || `Upload failed: ${response.status}`)
      }

      const result = await response.json()
      
      if (!result.success) {
        throw new Error(result.message || 'Upload failed')
      }

      return result.data

    } catch (error) {
      Logger.error('Backend upload failed', { error })
      throw error
    }
  }

  /**
   * 🔄 Make HTTP request with retry logic
   */
  private async makeRequestWithRetry(
    url: string, 
    options: RequestInit, 
    maxRetries = 3
  ): Promise<Response> {
    let lastError: Error

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const timeoutPromise = new Promise<never>((_, reject) => {
          setTimeout(() => reject(new Error('Request timeout')), 60000) // 60 second timeout
        })

        const response = await Promise.race([
          fetch(url, options),
          timeoutPromise
        ])

        if (response.ok || response.status < 500) {
          return response // Success or client error (don't retry)
        }

        throw new Error(`Server error: ${response.status}`)
      } catch (error) {
        lastError = error
        
        if (attempt < maxRetries) {
          const delay = Math.pow(2, attempt) * 1000 // Exponential backoff
          Logger.warn(`Request failed, retrying in ${delay}ms`, { attempt, error })
          await new Promise(resolve => setTimeout(resolve, delay))
        }
      }
    }

    throw lastError
  }

  // =============================================================================
  // REAL-TIME UPDATES
  // =============================================================================

  /**
   * 📺 Subscribe to real-time makeover status updates
   */
  subscribeMakeoverUpdates(makeoverId: string) {
    try {
      return this.supabase
        .channel(`makeover-${makeoverId}`)
        .on('postgres_changes', {
          event: 'UPDATE',
          schema: 'public',
          table: 'makeovers',
          filter: `id=eq.${makeoverId}`
        }, (payload: any) => {
          Logger.info('Real-time makeover update received', { makeoverId, payload: payload.new })
          this.handleMakeoverUpdate(payload.new)
        })
        .subscribe((status: string) => {
          Logger.info('Makeover subscription status', { makeoverId, status })
        })
    } catch (error) {
      Logger.error('Failed to subscribe to makeover updates', { error, makeoverId })
    }
  }

  /**
   * 📱 Handle makeover update and update local storage
   */
  private async handleMakeoverUpdate(makeover: any) {
    try {
      // Update local storage with new makeover status
      const localPhotos = this.getLocalPhotos()
      const updatedPhotos = localPhotos.map(photo => 
        photo.makeoverId === makeover.id 
          ? { ...photo, makeover }
          : photo
      )
      
      photoStorage.set('photos', JSON.stringify(updatedPhotos))

      // Trigger UI update notification
      this.notifyMakeoverUpdate(makeover)

      Logger.info('Local storage updated with makeover progress', { 
        makeoverId: makeover.id, 
        status: makeover.status,
        progress: makeover.progress 
      })
    } catch (error) {
      Logger.error('Failed to handle makeover update', { error, makeover })
    }
  }

  // =============================================================================
  // DATA RETRIEVAL
  // =============================================================================

  /**
   * 🎨 Get all user photos with makeover status
   */
  async getUserPhotos(refresh = false): Promise<PhotoWithMakeover[]> {
    try {
      if (!this.auth.isSignedIn) {
        Logger.warn('User not signed in, returning local photos only')
        return this.getLocalPhotosAsCloudFormat()
      }

      if (refresh || !this.getCachedCloudPhotos()) {
        Logger.info('Fetching fresh photos from Supabase')

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
              error_message
            )
          `)
          .order('created_at', { ascending: false })

        if (error) {
          Logger.error('Failed to fetch photos from Supabase', { error })
          throw error
        }

        // Cache the results locally
        photoStorage.set('cloud_photos', JSON.stringify(data))
        Logger.info(`Fetched ${data?.length || 0} photos from cloud`)

        return data || []
      }

      // Return cached data
      const cached = this.getCachedCloudPhotos()
      Logger.info(`Returning ${cached.length} cached photos`)
      return cached

    } catch (error) {
      Logger.error('Failed to fetch user photos', { error })
      
      // Return local photos as fallback
      return this.getLocalPhotosAsCloudFormat()
    }
  }

  /**
   * 🔍 Get specific photo by ID
   */
  async getPhoto(photoId: string): Promise<PhotoWithMakeover | null> {
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
            detected_objects,
            suggested_products
          )
        `)
        .eq('id', photoId)
        .single()

      if (error) {
        Logger.error('Failed to fetch photo', { error, photoId })
        return null
      }

      return data
    } catch (error) {
      Logger.error('Failed to get photo', { error, photoId })
      return null
    }
  }

  // =============================================================================
  // LOCAL STORAGE MANAGEMENT
  // =============================================================================

  /**
   * 💾 Save photo to local storage
   */
  private async saveLocalPhoto(photoData: LocalPhoto) {
    try {
      const photos = this.getLocalPhotos()
      photos.unshift(photoData)
      
      // Keep only last 100 photos locally to avoid storage bloat
      const trimmedPhotos = photos.slice(0, 100)
      
      photoStorage.set('photos', JSON.stringify(trimmedPhotos))
      Logger.info('Photo saved to local storage', { photoId: photoData.id })
    } catch (error) {
      Logger.error('Failed to save photo locally', { error, photoData })
    }
  }

  /**
   * 📖 Get photos from local storage
   */
  private getLocalPhotos(): LocalPhoto[] {
    try {
      const stored = photoStorage.getString('photos')
      return stored ? JSON.parse(stored) : []
    } catch (error) {
      Logger.error('Failed to get local photos', { error })
      return []
    }
  }

  /**
   * 📋 Get cached cloud photos
   */
  private getCachedCloudPhotos(): PhotoWithMakeover[] {
    try {
      const cached = photoStorage.getString('cloud_photos')
      return cached ? JSON.parse(cached) : []
    } catch (error) {
      Logger.error('Failed to get cached cloud photos', { error })
      return []
    }
  }

  /**
   * 🔄 Convert local photos to cloud format for consistency
   */
  private getLocalPhotosAsCloudFormat(): PhotoWithMakeover[] {
    const localPhotos = this.getLocalPhotos()
    
    return localPhotos.map(local => ({
      id: local.id,
      original_url: local.cloudUrl || local.uri,
      optimized_url: local.cloudUrl || local.uri,
      cloudflare_key: local.id,
      original_name: `photo_${local.timestamp}.jpg`,
      created_at: new Date(local.timestamp).toISOString(),
      makeovers: local.makeover ? [local.makeover] : []
    }))
  }

  /**
   * 🔔 Notify UI of makeover updates
   */
  private notifyMakeoverUpdate(makeover: any) {
    // Implement your preferred state management notification here
    // Could be Zustand, Redux, EventEmitter, etc.
    Logger.info('Makeover update notification', { 
      makeoverId: makeover.id,
      status: makeover.status,
      progress: makeover.progress 
    })
  }

  // =============================================================================
  // UTILITY METHODS
  // =============================================================================

  /**
   * 🧹 Clear local cache
   */
  clearCache() {
    try {
      photoStorage.delete('cloud_photos')
      Logger.info('Cloud photo cache cleared')
    } catch (error) {
      Logger.error('Failed to clear cache', { error })
    }
  }

  /**
   * 📊 Get cache statistics
   */
  getCacheStats() {
    const localPhotos = this.getLocalPhotos()
    const cachedPhotos = this.getCachedCloudPhotos()
    
    return {
      localPhotos: localPhotos.length,
      cachedPhotos: cachedPhotos.length,
      syncedPhotos: localPhotos.filter(p => p.synced).length,
      pendingSync: localPhotos.filter(p => !p.synced).length
    }
  }
}

export default CloudPhotoService