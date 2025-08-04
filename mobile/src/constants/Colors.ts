import type { ColorPalette } from '../types'

// ReRoom Design System Colors
export const Colors: ColorPalette = {
  // Primary colors (from UX documentation)
  primary: '#0066FF',        // Electric Blue - CTAs, interactive elements
  secondary: '#000000',      // Deep Black - Headers, primary text
  accent: '#10B981',         // Forest Green - Success, savings indicators
  
  // Background colors
  background: '#FFFFFF',     // Pure White - Main backgrounds
  surface: '#F3F4F6',        // Soft Grey - Cards, inactive elements
  
  // Text colors
  text: '#000000',           // Deep Black - Primary text
  textSecondary: '#6B7280',  // Medium Grey - Secondary text
  
  // UI colors
  border: '#E5E7EB',         // Light Grey - Borders, dividers
  success: '#10B981',        // Forest Green - Success states
  warning: '#F59E0B',        // Amber Gold - Warnings, alerts
  error: '#EF4444',          // Deep Red - Errors, urgent actions
}

// Dark mode colors (for future implementation)
export const DarkColors: ColorPalette = {
  primary: '#0066FF',
  secondary: '#FFFFFF',
  accent: '#10B981',
  
  background: '#000000',
  surface: '#1F1F1F',
  
  text: '#FFFFFF',
  textSecondary: '#A3A3A3',
  
  border: '#374151',
  success: '#10B981',
  warning: '#F59E0B',
  error: '#EF4444',
}

// Style-specific color palettes for AI rendering
export const StyleColors = {
  modern: {
    primary: '#2563EB',
    secondary: '#F8FAFC',
    accent: '#64748B',
  },
  scandinavian: {
    primary: '#F1F5F9',
    secondary: '#E2E8F0',
    accent: '#8B5CF6',
  },
  boho: {
    primary: '#D97706',
    secondary: '#FEF3C7',
    accent: '#DC2626',
  },
  industrial: {
    primary: '#374151',
    secondary: '#6B7280',
    accent: '#F59E0B',
  },
  minimalist: {
    primary: '#FFFFFF',
    secondary: '#F9FAFB',
    accent: '#111827',
  },
  traditional: {
    primary: '#7C2D12',
    secondary: '#FEF7FF',
    accent: '#059669',
  },
  eclectic: {
    primary: '#BE185D',
    secondary: '#FDF4FF',
    accent: '#7C3AED',
  },
  farmhouse: {
    primary: '#16A34A',
    secondary: '#F0FDF4',
    accent: '#DC2626',
  },
  'mid-century': {
    primary: '#EA580C',
    secondary: '#FFF7ED',
    accent: '#0891B2',
  },
  contemporary: {
    primary: '#1E40AF',
    secondary: '#EFF6FF',
    accent: '#059669',
  },
  coastal: {
    primary: '#0EA5E9',
    secondary: '#F0F9FF',
    accent: '#10B981',
  },
  rustic: {
    primary: '#92400E',
    secondary: '#FFFBEB',
    accent: '#059669',
  },
} as const

// Semantic colors for specific UI elements
export const SemanticColors = {
  // Savings and pricing
  savings: '#10B981',
  priceIncrease: '#EF4444',
  priceDecrease: '#10B981',
  priceStable: '#6B7280',
  
  // Processing states
  processing: '#F59E0B',
  completed: '#10B981',
  failed: '#EF4444',
  pending: '#6B7280',
  
  // Premium features
  premium: '#7C3AED',
  free: '#6B7280',
  
  // Social features
  like: '#EF4444',
  share: '#0066FF',
  bookmark: '#F59E0B',
  
  // Retailer indicators
  inStock: '#10B981',
  lowStock: '#F59E0B',
  outOfStock: '#EF4444',
  
  // Quality indicators
  highQuality: '#10B981',
  mediumQuality: '#F59E0B',
  lowQuality: '#EF4444',
} as const

// Opacity variants for overlays and states
export const OpacityColors = {
  overlay: 'rgba(0, 0, 0, 0.5)',
  modalOverlay: 'rgba(0, 0, 0, 0.6)',
  loadingOverlay: 'rgba(255, 255, 255, 0.9)',
  disabled: 'rgba(107, 114, 128, 0.5)',
  pressed: 'rgba(0, 102, 255, 0.1)',
  hover: 'rgba(0, 102, 255, 0.05)',
} as const

// Gradient definitions
export const Gradients = {
  primary: ['#0066FF', '#0052CC'],
  success: ['#10B981', '#059669'],
  warning: ['#F59E0B', '#D97706'],
  error: ['#EF4444', '#DC2626'],
  neutral: ['#F3F4F6', '#E5E7EB'],
  dark: ['#111827', '#1F2937'],
} as const 