import analytics from '@react-native-firebase/analytics'
import crashlytics from '@react-native-firebase/crashlytics'
import { Platform } from 'react-native'

import type { AnalyticsEvent, User } from '../types'

class AnalyticsService {
  private isInitialized = false
  private userId: string | null = null

  /**
   * Initialize analytics service
   */
  async initialize(): Promise<void> {
    try {
      // Enable analytics collection
      await analytics().setAnalyticsCollectionEnabled(true)
      
      // Enable crashlytics
      await crashlytics().setCrashlyticsCollectionEnabled(true)
      
      this.isInitialized = true
      console.log('✅ Analytics service initialized')
    } catch (error) {
      console.warn('⚠️ Failed to initialize analytics:', error)
      // Continue without analytics rather than crashing
    }
  }

  /**
   * Set user ID for analytics
   */
  async setUserId(userId: string): Promise<void> {
    if (!this.isInitialized) return

    try {
      this.userId = userId
      await analytics().setUserId(userId)
      await crashlytics().setUserId(userId)
    } catch (error) {
      console.warn('Failed to set analytics user ID:', error)
    }
  }

  /**
   * Set user properties
   */
  async setUserProperties(user: User): Promise<void> {
    if (!this.isInitialized) return

    try {
      await analytics().setUserProperties({
        user_type: user.subscription?.type || 'free',
        subscription_status: user.subscription?.status || 'none',
        preferred_style: user.preferences.stylePreferences?.[0] || 'unknown',
        country: user.preferences.location?.country || 'unknown',
      })

      await crashlytics().setAttributes({
        userId: user.id,
        email: user.email,
        subscriptionType: user.subscription?.type || 'free',
      })
    } catch (error) {
      console.warn('Failed to set user properties:', error)
    }
  }

  /**
   * Track custom events
   */
  async track(eventName: string, properties: Record<string, any> = {}): Promise<void> {
    if (!this.isInitialized) return

    try {
      // Add common properties
      const eventProperties = {
        ...properties,
        platform: Platform.OS,
        timestamp: new Date().toISOString(),
        user_id: this.userId,
      }

      await analytics().logEvent(eventName, eventProperties)
      
      // Log for debugging in development
      if (__DEV__) {
        console.log('📊 Analytics:', eventName, eventProperties)
      }
    } catch (error) {
      console.warn('Failed to track event:', error)
    }
  }

  /**
   * Track screen views
   */
  async trackScreen(screenName: string, properties: Record<string, any> = {}): Promise<void> {
    if (!this.isInitialized) return

    try {
      await analytics().logScreenView({
        screen_name: screenName,
        screen_class: screenName,
        ...properties,
      })
    } catch (error) {
      console.warn('Failed to track screen:', error)
    }
  }

  /**
   * Track user journey milestones
   */
  async trackPhotoCapture(metadata: {
    lighting_quality: number
    room_type: string
    photo_size: number
  }): Promise<void> {
    await this.track('photo_captured', {
      lighting_quality: metadata.lighting_quality,
      room_type: metadata.room_type,
      photo_size_mb: Math.round(metadata.photo_size / 1024 / 1024),
    })
  }

  async trackAIProcessing(data: {
    style: string
    processing_time: number
    success: boolean
    confidence?: number
  }): Promise<void> {
    await this.track('ai_processing_complete', {
      style: data.style,
      processing_time_seconds: Math.round(data.processing_time),
      success: data.success,
      confidence_score: data.confidence || 0,
    })
  }

  async trackProductInteraction(
    action: 'view' | 'click' | 'add_to_cart',
    product: {
      id: string
      price: number
      category: string
      retailer: string
    }
  ): Promise<void> {
    await this.track(`product_${action}`, {
      product_id: product.id,
      price: product.price,
      category: product.category,
      retailer: product.retailer,
      currency: 'GBP',
    })
  }

  async trackConversion(data: {
    total_value: number
    item_count: number
    style: string
    session_duration: number
  }): Promise<void> {
    await this.track('purchase', {
      value: data.total_value,
      currency: 'GBP',
      items: data.item_count,
      style: data.style,
      session_duration_minutes: Math.round(data.session_duration / 60),
    })
  }

  /**
   * Track performance metrics
   */
  async trackPerformance(
    metric: 'app_start' | 'ai_render' | 'image_load',
    duration: number
  ): Promise<void> {
    await this.track('performance_metric', {
      metric_type: metric,
      duration_ms: Math.round(duration),
    })
  }

  /**
   * Track errors
   */
  async trackError(error: Error, context: string): Promise<void> {
    if (!this.isInitialized) return

    try {
      // Record error in Crashlytics
      await crashlytics().recordError(error)
      
      // Log as analytics event
      await this.track('error_occurred', {
        error_type: error.name,
        context,
        error_message: error.message.substring(0, 100), // Truncate for privacy
      })
    } catch (trackingError) {
      console.warn('Failed to track error:', trackingError)
    }
  }

  /**
   * Track premium subscription events
   */
  async trackSubscription(action: 'viewed' | 'started' | 'completed' | 'cancelled'): Promise<void> {
    await this.track(`subscription_${action}`, {
      subscription_type: 'premium',
      price: 9.99,
      currency: 'GBP',
    })
  }

  /**
   * Track feature usage
   */
  async trackFeatureUsage(feature: string, properties: Record<string, any> = {}): Promise<void> {
    await this.track('feature_used', {
      feature_name: feature,
      ...properties,
    })
  }

  /**
   * Track app lifecycle events
   */
  async trackAppForeground(): Promise<void> {
    await this.track('app_foreground')
  }

  async trackAppBackground(): Promise<void> {
    await this.track('app_background')
  }

  /**
   * Set custom dimensions for detailed analysis
   */
  async setCustomDimensions(dimensions: Record<string, string>): Promise<void> {
    if (!this.isInitialized) return

    try {
      for (const [key, value] of Object.entries(dimensions)) {
        await analytics().setUserProperty(key, value)
      }
    } catch (error) {
      console.warn('Failed to set custom dimensions:', error)
    }
  }

  /**
   * Reset analytics data (e.g., on logout)
   */
  async reset(): Promise<void> {
    if (!this.isInitialized) return

    try {
      this.userId = null
      await analytics().resetAnalyticsData()
    } catch (error) {
      console.warn('Failed to reset analytics:', error)
    }
  }
}

export const analyticsService = new AnalyticsService() 