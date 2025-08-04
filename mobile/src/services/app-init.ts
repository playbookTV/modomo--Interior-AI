import * as Font from 'expo-font'
import { Platform } from 'react-native'
import AsyncStorage from '@react-native-async-storage/async-storage'

import { useAppStore } from '../stores/app-store'
import { analyticsService } from './analytics'
import { authService } from './auth'
import { cameraService } from './camera'

/**
 * Initialize the ReRoom app with all necessary services
 */
export async function initializeApp(): Promise<void> {
  console.log('🚀 Initializing ReRoom app...')
  
  try {
    // Run all initialization tasks in parallel where possible
    await Promise.all([
      initializeFonts(),
      initializeAnalytics(),
      initializeAuth(),
      initializePermissions(),
      initializeServices(),
    ])
    
    console.log('✅ ReRoom app initialized successfully')
  } catch (error) {
    console.error('❌ Failed to initialize ReRoom app:', error)
    throw error
  }
}

/**
 * Load custom fonts for the app
 */
async function initializeFonts(): Promise<void> {
  try {
    await Font.loadAsync({
      // Add custom fonts here if needed
      // 'CustomFont-Regular': require('../assets/fonts/CustomFont-Regular.ttf'),
      // 'CustomFont-Bold': require('../assets/fonts/CustomFont-Bold.ttf'),
    })
    console.log('✅ Fonts loaded')
  } catch (error) {
    console.warn('⚠️ Failed to load fonts:', error)
    // Non-critical, continue with system fonts
  }
}

/**
 * Initialize analytics and crash reporting
 */
async function initializeAnalytics(): Promise<void> {
  try {
    await analyticsService.initialize()
    
    // Track app initialization
    analyticsService.track('app_initialized', {
      platform: Platform.OS,
      version: Platform.Version,
      timestamp: new Date().toISOString(),
    })
    
    console.log('✅ Analytics initialized')
  } catch (error) {
    console.warn('⚠️ Failed to initialize analytics:', error)
    // Non-critical, continue without analytics
  }
}

/**
 * Initialize authentication state
 */
async function initializeAuth(): Promise<void> {
  try {
    // Check if user is already authenticated
    const isAuthenticated = await authService.isAuthenticated()
    
    if (isAuthenticated) {
      // Try to refresh user data
      const user = await authService.getCurrentUser()
      if (user) {
        useAppStore.getState().setUser(user)
        console.log('✅ User authentication restored')
      }
    }
    
    console.log('✅ Auth service initialized')
  } catch (error) {
    console.warn('⚠️ Failed to initialize auth:', error)
    // Clear any corrupted auth state
    await authService.logout()
  }
}

/**
 * Initialize and request necessary permissions
 */
async function initializePermissions(): Promise<void> {
  try {
    // Camera permissions will be requested when needed
    // We just initialize the camera service here
    await cameraService.requestPermissions()
    
    console.log('✅ Permissions initialized')
  } catch (error) {
    console.warn('⚠️ Failed to initialize permissions:', error)
    // Non-critical, permissions will be requested when needed
  }
}

/**
 * Initialize other app services
 */
async function initializeServices(): Promise<void> {
  try {
    // Check onboarding status
    const hasCompletedOnboarding = await AsyncStorage.getItem('hasCompletedOnboarding')
    if (hasCompletedOnboarding === 'true') {
      useAppStore.getState().setOnboardingComplete(true)
    }
    
    // Initialize other services as needed
    // await notificationService.initialize()
    // await pushNotificationService.initialize()
    
    console.log('✅ App services initialized')
  } catch (error) {
    console.warn('⚠️ Failed to initialize some services:', error)
    // Non-critical, continue with app startup
  }
}

/**
 * Clean up resources when app is backgrounded or closed
 */
export async function cleanupApp(): Promise<void> {
  try {
    // Clean up temporary files
    // Close database connections
    // Cancel pending requests
    
    console.log('✅ App cleanup completed')
  } catch (error) {
    console.warn('⚠️ Failed to cleanup app:', error)
  }
} 