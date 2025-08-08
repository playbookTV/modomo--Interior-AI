// ReRoom Color System - Based on UX Documentation

export const Colors = {
  // Primary Colors
  primary: {
    black: '#000000',        // Headers, primary text
    white: '#FFFFFF',        // Backgrounds, cards  
    blue: '#0066FF',         // CTAs, interactive elements
  },
  
  // Secondary Colors
  secondary: {
    green: '#10B981',        // Success, savings indicators
    amber: '#F59E0B',        // Warnings, alerts
    grey: '#F3F4F6',         // Inactive elements, borders
    red: '#EF4444',          // Errors, urgent actions
  },
  
  // Text Colors
  text: {
    primary: '#000000',
    secondary: '#666666',
    tertiary: '#999999',
    inverse: '#FFFFFF',
  },
  
  // Background Colors
  background: {
    primary: '#FFFFFF',
    secondary: '#F3F4F6',
    tertiary: '#F9FAFB',
    inverse: '#000000',
  },
  
  // Border Colors
  border: {
    primary: '#E5E7EB',
    secondary: '#D1D5DB',
    focus: '#0066FF',
  },
  
  // Semantic Colors
  semantic: {
    success: '#10B981',
    warning: '#F59E0B', 
    error: '#EF4444',
    info: '#0066FF',
  },
  
  // Dark mode support
  dark: {
    background: '#000000',
    surface: '#1C1C1E',
    text: '#FFFFFF',
    textSecondary: '#EBEBF5',
    border: '#38383A',
  },
} as const

export type ColorKeys = keyof typeof Colors