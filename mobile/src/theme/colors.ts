/**
 * BNA UI Colors Configuration
 * Integrated with ReRoom design system
 */

const lightColors = {
    // Primary brand colors from ReRoom design system
    primary: '#0066FF',         // Electric Blue
    primaryForeground: '#FFFFFF',
    
    // Secondary colors
    secondary: '#F3F4F6',       // Soft Grey
    secondaryForeground: '#000000',
    
    // Accent colors
    accent: '#10B981',          // Forest Green
    accentForeground: '#FFFFFF',
    
    // Background colors
    background: '#FFFFFF',      // Pure White
    backgroundSecondary: '#F8F9FA',
    foreground: '#000000',      // Deep Black
    
    // Surface colors
    surface: '#FFFFFF',
    surfaceSecondary: '#F3F4F6',
    
    // Text colors
    text: '#000000',            // Primary text
    textSecondary: '#6B7280',   // Secondary text
    textMuted: '#9CA3AF',       // Muted text
    
    // Border and dividers
    border: '#E5E7EB',          // Light Grey
    divider: '#F3F4F6',
    
    // Status colors
    success: '#10B981',         // Forest Green
    successForeground: '#FFFFFF',
    warning: '#F59E0B',         // Amber Gold
    warningForeground: '#FFFFFF',
    error: '#EF4444',           // Deep Red
    errorForeground: '#FFFFFF',
    info: '#0066FF',
    infoForeground: '#FFFFFF',
    
    // Interactive states
    interactive: '#0066FF',
    interactiveHover: '#0052CC',
    interactivePressed: '#003D99',
    interactiveDisabled: '#9CA3AF',
    
    // Overlay colors
    overlay: 'rgba(0, 0, 0, 0.5)',
    modalOverlay: 'rgba(0, 0, 0, 0.6)',
    
    // Camera specific colors
    cameraBackground: '#000000',
    cameraControls: 'rgba(255, 255, 255, 0.9)',
    cameraAccent: '#0066FF',
    
    // Gallery specific colors
    galleryBackground: '#F8F9FA',
    galleryCard: '#FFFFFF',
    gallerySelected: '#0066FF',
};

const darkColors = {
    // Primary brand colors adapted for dark mode
    primary: '#0066FF',
    primaryForeground: '#FFFFFF',
    
    // Secondary colors
    secondary: '#1F1F1F',
    secondaryForeground: '#FFFFFF',
    
    // Accent colors
    accent: '#10B981',
    accentForeground: '#FFFFFF',
    
    // Background colors
    background: '#000000',
    backgroundSecondary: '#111111',
    foreground: '#FFFFFF',
    
    // Surface colors
    surface: '#1F1F1F',
    surfaceSecondary: '#2A2A2A',
    
    // Text colors
    text: '#FFFFFF',
    textSecondary: '#A3A3A3',
    textMuted: '#6B7280',
    
    // Border and dividers
    border: '#374151',
    divider: '#2A2A2A',
    
    // Status colors
    success: '#10B981',
    successForeground: '#FFFFFF',
    warning: '#F59E0B',
    warningForeground: '#FFFFFF',
    error: '#EF4444',
    errorForeground: '#FFFFFF',
    info: '#0066FF',
    infoForeground: '#FFFFFF',
    
    // Interactive states
    interactive: '#0066FF',
    interactiveHover: '#1A7CFF',
    interactivePressed: '#0052CC',
    interactiveDisabled: '#6B7280',
    
    // Overlay colors
    overlay: 'rgba(0, 0, 0, 0.7)',
    modalOverlay: 'rgba(0, 0, 0, 0.8)',
    
    // Camera specific colors
    cameraBackground: '#000000',
    cameraControls: 'rgba(255, 255, 255, 0.9)',
    cameraAccent: '#0066FF',
    
    // Gallery specific colors
    galleryBackground: '#111111',
    galleryCard: '#1F1F1F',
    gallerySelected: '#0066FF',
};

export const colors = {
  light: lightColors,
  dark: darkColors,
} as const;

export type Colors = typeof lightColors;
export type ColorName = keyof Colors;