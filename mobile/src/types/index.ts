// User types
export interface User {
  id: string
  email: string
  firstName: string
  lastName: string
  avatar?: string
  preferences: UserPreferences
  subscription?: Subscription
  createdAt: string
  updatedAt: string
}

export interface UserPreferences {
  budget?: { min: number; max: number }
  preferredRetailers?: string[]
  stylePreferences?: StyleType[]
  location?: {
    country: string
    city?: string
    postcode?: string
  }
  notifications: {
    priceAlerts: boolean
    newFeatures: boolean
    marketing: boolean
  }
}

export interface Subscription {
  id: string
  type: 'free' | 'premium'
  status: 'active' | 'cancelled' | 'expired'
  currentPeriodEnd: string
  cancelAtPeriodEnd: boolean
}

// Style types
export type StyleType = 
  | 'modern'
  | 'scandinavian'
  | 'boho'
  | 'industrial'
  | 'minimalist'
  | 'traditional'
  | 'eclectic'
  | 'farmhouse'
  | 'mid-century'
  | 'contemporary'
  | 'coastal'
  | 'rustic'

export interface Style {
  id: StyleType
  name: string
  description: string
  previewImage: string
  priceRange: { min: number; max: number }
  averageSavings: number
  isPremium: boolean
}

// Photo and AI types (updated to match PhotoService)
export interface PhotoMetadata {
  id: string;
  originalPath: string;
  optimizedPath: string;
  originalSize: number;
  optimizedSize: number;
  width: number;
  height: number;
  quality: number;
  format: string;
  optimized: boolean;
  timestamp: string;
  compressionRatio: number;
  // Backend sync fields
  backendId?: string;
  backendUrl?: string;
  uploadedAt?: string;
}

export interface AIRender {
  id: string
  originalPhoto: string
  styledImage: string
  style: StyleType
  confidence: number
  processingTime: number
  products: Product[]
  totalCost: number
  estimatedSavings: number
  createdAt: string
}

export interface ProcessingStatus {
  id: string
  stage: 'analyzing' | 'styling' | 'matching' | 'optimizing' | 'complete' | 'failed'
  progress: number
  message: string
  estimatedTimeRemaining?: number
}

// Product types
export interface Product {
  id: string
  title: string
  description: string
  price: number
  originalPrice?: number
  currency: string
  category: ProductCategory
  brand?: string
  retailer: Retailer
  imageUrls: string[]
  affiliateUrl: string
  availability: 'in_stock' | 'low_stock' | 'out_of_stock'
  rating?: number
  reviewCount?: number
  dimensions?: {
    width?: number
    height?: number
    depth?: number
    weight?: number
  }
  visualSimilarity: number
  lastUpdated: string
}

export type ProductCategory = 
  | 'sofa'
  | 'chair'
  | 'table'
  | 'lamp'
  | 'rug'
  | 'artwork'
  | 'decor'
  | 'storage'
  | 'bed'
  | 'dresser'
  | 'mirror'
  | 'plant'
  | 'cushion'
  | 'curtain'
  | 'other'

export interface Retailer {
  id: string
  name: string
  logo: string
  baseUrl: string
  shippingInfo: {
    freeShippingThreshold?: number
    standardDeliveryDays: number
    expressDeliveryDays?: number
  }
  returnPolicy: {
    days: number
    conditions: string[]
  }
  trustScore: number
}

export interface CartItem {
  id: string
  product: Product
  quantity: number
  addedAt: string
}

export interface PriceComparison {
  product: Product
  alternatives: Array<{
    retailer: Retailer
    price: number
    url: string
    availability: string
    deliveryTime: string
  }>
  bestPrice: {
    retailer: Retailer
    price: number
    savings: number
  }
}

// Room types
export interface Room {
  id: string
  name: string
  type: 'living_room' | 'bedroom' | 'kitchen' | 'bathroom' | 'dining_room' | 'office' | 'other'
  originalPhoto: string
  renders: AIRender[]
  favoriteRender?: string
  products: Product[]
  totalBudget?: number
  actualSpent?: number
  createdAt: string
  updatedAt: string
}

export interface SavedRoom {
  id: string
  room: Room
  notes?: string
  tags: string[]
  isPublic: boolean
  shareUrl?: string
}

// API types
export interface APIResponse<T = any> {
  success: boolean
  data?: T
  error?: {
    code: string
    message: string
    details?: any
  }
  pagination?: {
    page: number
    limit: number
    total: number
    pages: number
  }
}

export interface LoginRequest {
  email: string
  password: string
}

export interface RegisterRequest {
  email: string
  password: string
  firstName: string
  lastName: string
}

export interface AuthResponse {
  user: User
  accessToken: string
  refreshToken: string
  expiresIn: number
}

export interface PhotoUploadRequest {
  uri: string
  type: string
  name: string
}

export interface AIProcessingRequest {
  photoId: string
  style: StyleType
  preferences?: {
    budget?: { min: number; max: number }
    keepObjects?: string[]
    excludeCategories?: ProductCategory[]
  }
}

// Analytics types
export interface AnalyticsEvent {
  name: string
  properties: Record<string, any>
  timestamp: string
  userId?: string
  sessionId: string
}

// Navigation types
export type RootStackParamList = {
  '(tabs)': undefined
  onboarding: undefined
  camera: undefined
  processing: { style: StyleType; photoUri: string }
  result: { renderId: string }
  product: { productId: string }
  comparison: { productId: string }
  premium: undefined
}

export type TabParamList = {
  index: undefined
  camera: undefined
  saved: undefined
  profile: undefined
}

// Utility types
export interface Coordinates {
  latitude: number
  longitude: number
}

export interface Dimensions {
  width: number
  height: number
}

export interface ColorPalette {
  primary: string
  secondary: string
  accent: string
  background: string
  surface: string
  text: string
  textSecondary: string
  border: string
  success: string
  warning: string
  error: string
}

// Error types
export interface AppError extends Error {
  code: string
  statusCode?: number
  isRetryable?: boolean
  context?: Record<string, any>
}

// WebSocket types
export interface WebSocketMessage {
  type: 'processing_update' | 'price_update' | 'notification'
  data: any
  timestamp: string
}

export interface ProcessingUpdate {
  renderId: string
  stage: ProcessingStatus['stage']
  progress: number
  message: string
  estimatedTimeRemaining?: number
}

export interface PriceUpdate {
  productId: string
  oldPrice: number
  newPrice: number
  change: number
  retailer: string
} 