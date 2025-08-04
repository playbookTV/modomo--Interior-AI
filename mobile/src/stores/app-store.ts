import { create } from 'zustand'
import { createJSONStorage, persist } from 'zustand/middleware'
import { MMKV } from 'react-native-mmkv'

import type { User, Room, StyleType, CartItem, AIRender } from '../types'

const storage = new MMKV()

export interface AppState {
  // App initialization
  isAppReady: boolean
  hasCompletedOnboarding: boolean
  
  // User state
  user: User | null
  isAuthenticated: boolean
  
  // Photo and processing
  currentPhoto: string | null
  photoQuality: 'low' | 'medium' | 'high'
  
  // AI processing
  isProcessing: boolean
  processingProgress: number
  processingStage: string
  currentRender: AIRender | null
  renderQueue: string[]
  
  // Style and preferences
  selectedStyle: StyleType
  userPreferences: {
    budget?: { min: number; max: number }
    preferredRetailers?: string[]
    stylePreferences?: StyleType[]
  }
  
  // Shopping
  cartItems: CartItem[]
  savedRooms: Room[]
  
  // Premium features
  isPremiumUser: boolean
  remainingFreeRenders: number
  
  // Actions
  setAppReady: (ready: boolean) => void
  setOnboardingComplete: (complete: boolean) => void
  setUser: (user: User | null) => void
  setAuthenticated: (authenticated: boolean) => void
  
  // Photo actions
  setPhoto: (uri: string | null) => void
  setPhotoQuality: (quality: 'low' | 'medium' | 'high') => void
  
  // AI processing actions
  startAIProcessing: () => void
  updateProcessingProgress: (progress: number) => void
  setProcessingStage: (stage: string) => void
  setCurrentRender: (render: AIRender | null) => void
  completeProcessing: () => void
  
  // Style actions
  setSelectedStyle: (style: StyleType) => void
  updateUserPreferences: (preferences: Partial<AppState['userPreferences']>) => void
  
  // Shopping actions
  addToCart: (item: CartItem) => void
  removeFromCart: (itemId: string) => void
  clearCart: () => void
  
  // Room management
  saveRoom: (room: Room) => void
  removeRoom: (roomId: string) => void
  updateRoom: (roomId: string, updates: Partial<Room>) => void
  
  // Premium actions
  setPremiumUser: (isPremium: boolean) => void
  decrementFreeRenders: () => void
  
  // Utility actions
  reset: () => void
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      // Initial state
      isAppReady: false,
      hasCompletedOnboarding: false,
      
      user: null,
      isAuthenticated: false,
      
      currentPhoto: null,
      photoQuality: 'medium',
      
      isProcessing: false,
      processingProgress: 0,
      processingStage: '',
      currentRender: null,
      renderQueue: [],
      
      selectedStyle: 'modern',
      userPreferences: {},
      
      cartItems: [],
      savedRooms: [],
      
      isPremiumUser: false,
      remainingFreeRenders: 3,
      
      // Actions
      setAppReady: (ready) => set({ isAppReady: ready }),
      setOnboardingComplete: (complete) => set({ hasCompletedOnboarding: complete }),
      setUser: (user) => set({ user, isAuthenticated: !!user }),
      setAuthenticated: (authenticated) => set({ isAuthenticated: authenticated }),
      
      // Photo actions
      setPhoto: (uri) => set({ currentPhoto: uri }),
      setPhotoQuality: (quality) => set({ photoQuality: quality }),
      
      // AI processing actions
      startAIProcessing: () => set({ 
        isProcessing: true, 
        processingProgress: 0,
        processingStage: 'Analyzing your room...'
      }),
      updateProcessingProgress: (progress) => set({ processingProgress: progress }),
      setProcessingStage: (stage) => set({ processingStage: stage }),
      setCurrentRender: (render) => set({ currentRender: render }),
      completeProcessing: () => set({ 
        isProcessing: false, 
        processingProgress: 100,
        processingStage: 'Complete!'
      }),
      
      // Style actions
      setSelectedStyle: (style) => set({ selectedStyle: style }),
      updateUserPreferences: (preferences) => set(state => ({
        userPreferences: { ...state.userPreferences, ...preferences }
      })),
      
      // Shopping actions
      addToCart: (item) => set(state => ({
        cartItems: [...state.cartItems, item]
      })),
      removeFromCart: (itemId) => set(state => ({
        cartItems: state.cartItems.filter(item => item.id !== itemId)
      })),
      clearCart: () => set({ cartItems: [] }),
      
      // Room management
      saveRoom: (room) => set(state => ({
        savedRooms: [...state.savedRooms, room]
      })),
      removeRoom: (roomId) => set(state => ({
        savedRooms: state.savedRooms.filter(room => room.id !== roomId)
      })),
      updateRoom: (roomId, updates) => set(state => ({
        savedRooms: state.savedRooms.map(room => 
          room.id === roomId ? { ...room, ...updates } : room
        )
      })),
      
      // Premium actions
      setPremiumUser: (isPremium) => set({ isPremiumUser: isPremium }),
      decrementFreeRenders: () => set(state => ({
        remainingFreeRenders: Math.max(0, state.remainingFreeRenders - 1)
      })),
      
      // Utility actions
      reset: () => set({
        currentPhoto: null,
        isProcessing: false,
        processingProgress: 0,
        processingStage: '',
        currentRender: null,
        cartItems: [],
      }),
    }),
    {
      name: 'reroom-storage',
      storage: createJSONStorage(() => ({
        setItem: (name, value) => storage.set(name, value),
        getItem: (name) => storage.getString(name) ?? null,
        removeItem: (name) => storage.delete(name),
      })),
      partialize: (state) => ({
        hasCompletedOnboarding: state.hasCompletedOnboarding,
        user: state.user,
        isAuthenticated: state.isAuthenticated,
        userPreferences: state.userPreferences,
        savedRooms: state.savedRooms,
        isPremiumUser: state.isPremiumUser,
        remainingFreeRenders: state.remainingFreeRenders,
      }),
    }
  )
) 