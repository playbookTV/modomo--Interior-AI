// ReRoom App Store - Main state management with Zustand and MMKV persistence

import { create } from 'zustand'
import { createJSONStorage, persist } from 'zustand/middleware'
import { MMKV } from 'react-native-mmkv'
import { StyleType, CartItem, SavedRoom, AIRender, User, ProcessingStatus } from '../types'

const storage = new MMKV()

interface AppState {
  // Photo state
  currentPhoto: string | null
  photoQuality: 'low' | 'medium' | 'high'
  
  // AI processing
  isProcessing: boolean
  processingStatus: ProcessingStatus | null
  currentRender: AIRender | null
  renderQueue: string[]
  
  // Shopping
  selectedStyle: StyleType
  cartItems: CartItem[]
  savedRooms: SavedRoom[]
  
  // User
  user: User | null
  
  // UI state
  isOnboarded: boolean
  darkMode: boolean
  
  // Actions
  setPhoto: (uri: string) => void
  setPhotoQuality: (quality: 'low' | 'medium' | 'high') => void
  clearPhoto: () => void
  
  startAIProcessing: (photo: string, style: StyleType) => void
  updateProcessingProgress: (progress: number) => void
  setProcessingStage: (stage: string) => void
  setCurrentRender: (render: AIRender) => void
  clearProcessing: () => void
  
  setSelectedStyle: (style: StyleType) => void
  addToCart: (item: CartItem) => void
  removeFromCart: (itemId: string) => void
  clearCart: () => void
  
  saveRoom: (room: Omit<SavedRoom, 'id' | 'createdAt'>) => void
  removeRoom: (roomId: string) => void
  
  setUser: (user: User) => void
  updateUserPreferences: (preferences: Partial<User>) => void
  
  completeOnboarding: () => void
  toggleDarkMode: () => void
  
  // Computed
  getTotalSavings: () => number
  getCartTotal: () => number
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      // Initial state
      currentPhoto: null,
      photoQuality: 'medium',
      
      isProcessing: false,
      processingStatus: null,
      currentRender: null,
      renderQueue: [],
      
      selectedStyle: 'modern',
      cartItems: [],
      savedRooms: [],
      
      user: null,
      
      isOnboarded: false,
      darkMode: false,
      
      // Photo actions
      setPhoto: (uri) => set({ currentPhoto: uri }),
      setPhotoQuality: (quality) => set({ photoQuality: quality }),
      clearPhoto: () => set({ currentPhoto: null }),
      
      // AI processing actions
      startAIProcessing: (photo, style) => set({ 
        isProcessing: true,
        processingStatus: { progress: 0, stage: 'Analyzing your room...' },
        currentRender: null 
      }),
      
      updateProcessingProgress: (progress) => set((state) => ({
        processingStatus: state.processingStatus ? {
          ...state.processingStatus,
          progress
        } : null
      })),
      
      setProcessingStage: (stage) => set((state) => ({
        processingStatus: state.processingStatus ? {
          ...state.processingStatus,
          stage
        } : null
      })),
      
      setCurrentRender: (render) => set({ 
        currentRender: render,
        isProcessing: false,
        processingStatus: null 
      }),
      
      clearProcessing: () => set({
        isProcessing: false,
        processingStatus: null,
        currentRender: null
      }),
      
      // Style actions
      setSelectedStyle: (style) => set({ selectedStyle: style }),
      
      // Shopping actions
      addToCart: (item) => set((state) => ({
        cartItems: [...state.cartItems, item]
      })),
      
      removeFromCart: (itemId) => set((state) => ({
        cartItems: state.cartItems.filter(item => item.id !== itemId)
      })),
      
      clearCart: () => set({ cartItems: [] }),
      
      // Room management
      saveRoom: (roomData) => {
        const room: SavedRoom = {
          ...roomData,
          id: Date.now().toString(),
          createdAt: new Date()
        }
        set((state) => ({
          savedRooms: [...state.savedRooms, room]
        }))
      },
      
      removeRoom: (roomId) => set((state) => ({
        savedRooms: state.savedRooms.filter(room => room.id !== roomId)
      })),
      
      // User actions
      setUser: (user) => set({ user }),
      
      updateUserPreferences: (preferences) => set((state) => ({
        user: state.user ? { ...state.user, ...preferences } : null
      })),
      
      // App state
      completeOnboarding: () => set({ isOnboarded: true }),
      toggleDarkMode: () => set((state) => ({ darkMode: !state.darkMode })),
      
      // Computed values
      getTotalSavings: () => {
        const { savedRooms } = get()
        return savedRooms.reduce((total, room) => total + room.totalSavings, 0)
      },
      
      getCartTotal: () => {
        const { cartItems } = get()
        return cartItems.reduce((total, item) => total + item.price, 0)
      },
    }),
    {
      name: 'reroom-storage',
      storage: createJSONStorage(() => ({
        setItem: (name, value) => storage.set(name, value),
        getItem: (name) => storage.getString(name) ?? null,
        removeItem: (name) => storage.delete(name),
      })),
      // Only persist user data and saved content
      partialize: (state) => ({
        savedRooms: state.savedRooms,
        cartItems: state.cartItems,
        user: state.user,
        isOnboarded: state.isOnboarded,
        darkMode: state.darkMode,
        selectedStyle: state.selectedStyle,
      }),
    }
  )
)