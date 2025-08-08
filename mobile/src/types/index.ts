// Core types for ReRoom mobile app

export type StyleType = 'modern' | 'japandi' | 'boho' | 'scandinavian' | 'industrial' | 'minimalist'

export interface CartItem {
  id: string
  name: string
  price: number
  currency: string
  imageUrl: string
  retailer: string
  category: string
  savings?: number
  originalPrice?: number
}

export interface SavedRoom {
  id: string
  name: string
  style: StyleType
  imageUrl: string
  createdAt: Date
  totalCost: number
  totalSavings: number
  items: CartItem[]
}

export interface AIRender {
  id: string
  originalImageUrl: string
  styledImageUrl: string
  style: StyleType
  confidence: number
  processingTime: number
  products: CartItem[]
  createdAt: Date
}

export interface PhotoQualityCheck {
  isWellLit: boolean
  hasGoodFocus: boolean
  containsFurniture: boolean
  confidence: number
}

export interface User {
  id: string
  email?: string
  preferredStyle?: StyleType
  averageBudget?: number
  roomCount?: number
  totalSavings?: number
}

export interface ProcessingStatus {
  progress: number
  stage: string
  estimatedTimeRemaining?: number
}