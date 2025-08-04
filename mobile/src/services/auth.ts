import AsyncStorage from '@react-native-async-storage/async-storage'
import * as SecureStore from 'expo-secure-store'

import type { User, LoginRequest, RegisterRequest, AuthResponse } from '../types'
import { apiService } from './api'
import { analyticsService } from './analytics'

const ACCESS_TOKEN_KEY = 'access_token'
const REFRESH_TOKEN_KEY = 'refresh_token'
const USER_DATA_KEY = 'user_data'

class AuthService {
  private accessToken: string | null = null
  private refreshToken: string | null = null
  private user: User | null = null

  /**
   * Initialize auth service by loading stored tokens
   */
  async initialize(): Promise<void> {
    try {
      // Load tokens from secure storage
      this.accessToken = await SecureStore.getItemAsync(ACCESS_TOKEN_KEY)
      this.refreshToken = await SecureStore.getItemAsync(REFRESH_TOKEN_KEY)
      
      // Load user data from async storage
      const userData = await AsyncStorage.getItem(USER_DATA_KEY)
      if (userData) {
        this.user = JSON.parse(userData)
      }

      // Set up API authentication
      if (this.accessToken) {
        apiService.setAuthToken(this.accessToken)
      }
    } catch (error) {
      console.warn('Failed to initialize auth service:', error)
      await this.clearStoredAuth()
    }
  }

  /**
   * Check if user is currently authenticated
   */
  async isAuthenticated(): Promise<boolean> {
    return !!(this.accessToken && this.user)
  }

  /**
   * Get current user data
   */
  async getCurrentUser(): Promise<User | null> {
    if (this.user) {
      return this.user
    }

    // Try to fetch fresh user data if we have a token
    if (this.accessToken) {
      try {
        const response = await apiService.get<User>('/auth/me')
        this.user = response.data
        await this.storeUserData(this.user)
        return this.user
      } catch (error) {
        console.warn('Failed to fetch current user:', error)
        // Token might be expired, try to refresh
        await this.refreshAccessToken()
      }
    }

    return null
  }

  /**
   * Login with email and password
   */
  async login(credentials: LoginRequest): Promise<User> {
    try {
      const response = await apiService.post<AuthResponse>('/auth/login', credentials)
      
      await this.handleAuthResponse(response.data)
      
      // Track login event
      analyticsService.track('user_login', {
        method: 'email',
      })

      return this.user!
    } catch (error) {
      analyticsService.trackError(error as Error, 'login')
      throw error
    }
  }

  /**
   * Register new user account
   */
  async register(userData: RegisterRequest): Promise<User> {
    try {
      const response = await apiService.post<AuthResponse>('/auth/register', userData)
      
      await this.handleAuthResponse(response.data)
      
      // Track registration event
      analyticsService.track('user_register', {
        method: 'email',
      })

      return this.user!
    } catch (error) {
      analyticsService.trackError(error as Error, 'register')
      throw error
    }
  }

  /**
   * Logout user and clear all auth data
   */
  async logout(): Promise<void> {
    try {
      // Notify server about logout
      if (this.accessToken) {
        await apiService.post('/auth/logout', {
          refreshToken: this.refreshToken,
        })
      }
    } catch (error) {
      console.warn('Failed to notify server about logout:', error)
    } finally {
      // Clear local auth state
      await this.clearAuth()
      
      // Track logout event
      analyticsService.track('user_logout')
    }
  }

  /**
   * Refresh access token using refresh token
   */
  async refreshAccessToken(): Promise<string | null> {
    if (!this.refreshToken) {
      await this.clearAuth()
      return null
    }

    try {
      const response = await apiService.post<{ accessToken: string; expiresIn: number }>('/auth/refresh', {
        refreshToken: this.refreshToken,
      })

      this.accessToken = response.data.accessToken
      await SecureStore.setItemAsync(ACCESS_TOKEN_KEY, this.accessToken)
      apiService.setAuthToken(this.accessToken)

      return this.accessToken
    } catch (error) {
      console.warn('Failed to refresh access token:', error)
      await this.clearAuth()
      return null
    }
  }

  /**
   * Request password reset
   */
  async requestPasswordReset(email: string): Promise<void> {
    try {
      await apiService.post('/auth/forgot-password', { email })
      
      analyticsService.track('password_reset_requested')
    } catch (error) {
      analyticsService.trackError(error as Error, 'password_reset_request')
      throw error
    }
  }

  /**
   * Reset password with token
   */
  async resetPassword(token: string, newPassword: string): Promise<void> {
    try {
      await apiService.post('/auth/reset-password', {
        token,
        password: newPassword,
      })
      
      analyticsService.track('password_reset_completed')
    } catch (error) {
      analyticsService.trackError(error as Error, 'password_reset')
      throw error
    }
  }

  /**
   * Update user profile
   */
  async updateProfile(updates: Partial<User>): Promise<User> {
    try {
      const response = await apiService.put<User>('/auth/profile', updates)
      
      this.user = response.data
      await this.storeUserData(this.user)
      
      analyticsService.track('profile_updated')
      
      return this.user
    } catch (error) {
      analyticsService.trackError(error as Error, 'profile_update')
      throw error
    }
  }

  /**
   * Delete user account
   */
  async deleteAccount(): Promise<void> {
    try {
      await apiService.delete('/auth/account')
      
      await this.clearAuth()
      
      analyticsService.track('account_deleted')
    } catch (error) {
      analyticsService.trackError(error as Error, 'account_deletion')
      throw error
    }
  }

  /**
   * Get current access token (for API calls)
   */
  getAccessToken(): string | null {
    return this.accessToken
  }

  /**
   * Handle successful authentication response
   */
  private async handleAuthResponse(authData: AuthResponse): Promise<void> {
    this.accessToken = authData.accessToken
    this.refreshToken = authData.refreshToken
    this.user = authData.user

    // Store tokens securely
    await SecureStore.setItemAsync(ACCESS_TOKEN_KEY, this.accessToken)
    await SecureStore.setItemAsync(REFRESH_TOKEN_KEY, this.refreshToken)
    
    // Store user data
    await this.storeUserData(this.user)
    
    // Set up API authentication
    apiService.setAuthToken(this.accessToken)
    
    // Set up analytics
    await analyticsService.setUserId(this.user.id)
    await analyticsService.setUserProperties(this.user)
  }

  /**
   * Store user data in async storage
   */
  private async storeUserData(user: User): Promise<void> {
    try {
      await AsyncStorage.setItem(USER_DATA_KEY, JSON.stringify(user))
    } catch (error) {
      console.warn('Failed to store user data:', error)
    }
  }

  /**
   * Clear all authentication data
   */
  private async clearAuth(): Promise<void> {
    this.accessToken = null
    this.refreshToken = null
    this.user = null

    await this.clearStoredAuth()
    
    // Clear API authentication
    apiService.clearAuthToken()
    
    // Reset analytics
    await analyticsService.reset()
  }

  /**
   * Clear stored authentication data
   */
  private async clearStoredAuth(): Promise<void> {
    try {
      await Promise.all([
        SecureStore.deleteItemAsync(ACCESS_TOKEN_KEY),
        SecureStore.deleteItemAsync(REFRESH_TOKEN_KEY),
        AsyncStorage.removeItem(USER_DATA_KEY),
      ])
    } catch (error) {
      console.warn('Failed to clear stored auth data:', error)
    }
  }
}

export const authService = new AuthService() 