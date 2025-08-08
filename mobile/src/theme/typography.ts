// ReRoom Typography System - Based on UX Documentation

import { Platform } from 'react-native'

const fontFamily = {
  heading: Platform.select({
    ios: 'SF Pro Display',
    android: 'Roboto',
    default: 'System',
  }),
  body: Platform.select({
    ios: 'SF Pro Text', 
    android: 'Roboto',
    default: 'System',
  }),
} as const

export const Typography = {
  // Headings
  h1: {
    fontFamily: fontFamily.heading,
    fontSize: 28,
    fontWeight: '700' as const,
    lineHeight: 34,
  },
  h2: {
    fontFamily: fontFamily.heading,
    fontSize: 24,
    fontWeight: '600' as const,
    lineHeight: 29,
  },
  h3: {
    fontFamily: fontFamily.heading,
    fontSize: 20,
    fontWeight: '500' as const,
    lineHeight: 24,
  },
  
  // Body Text
  bodyLarge: {
    fontFamily: fontFamily.body,
    fontSize: 18,
    fontWeight: '400' as const,
    lineHeight: 26,
  },
  bodyMedium: {
    fontFamily: fontFamily.body,
    fontSize: 16,
    fontWeight: '400' as const,
    lineHeight: 22,
  },
  bodySmall: {
    fontFamily: fontFamily.body,
    fontSize: 14,
    fontWeight: '400' as const,
    lineHeight: 20,
  },
  caption: {
    fontFamily: fontFamily.body,
    fontSize: 12,
    fontWeight: '400' as const,
    lineHeight: 16,
  },
  
  // Interactive Elements
  button: {
    fontFamily: fontFamily.body,
    fontSize: 16,
    fontWeight: '500' as const,
    lineHeight: 22,
  },
  
  // Navigation
  tabLabel: {
    fontFamily: fontFamily.body,
    fontSize: 10,
    fontWeight: '500' as const,
    lineHeight: 12,
  },
  
  // Special
  price: {
    fontFamily: fontFamily.heading,
    fontSize: 20,
    fontWeight: '600' as const,
    lineHeight: 24,
  },
  savings: {
    fontFamily: fontFamily.heading,
    fontSize: 16,
    fontWeight: '600' as const,
    lineHeight: 20,
  },
} as const

export type TypographyKeys = keyof typeof Typography