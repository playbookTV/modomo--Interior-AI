/**
 * BNA UI Card Component
 * Themeable card container
 */

import React from 'react';
import { View, StyleSheet, ViewStyle, ViewProps } from 'react-native';
import { useTheme } from '@/theme/theme-provider';

interface CardProps extends ViewProps {
  variant?: 'default' | 'elevated' | 'outlined';
  padding?: keyof typeof import('@/theme/globals').globals.spacing;
  children: React.ReactNode;
}

export function Card({
  variant = 'default',
  padding = 4,
  children,
  style,
  ...props
}: CardProps) {
  const { theme } = useTheme();
  
  const getCardStyle = (): ViewStyle => {
    const baseStyle: ViewStyle = {
      backgroundColor: theme.colors.surface,
      borderRadius: theme.globals.borderRadius.xl,
      padding: theme.globals.spacing[padding],
    };

    switch (variant) {
      case 'elevated':
        return {
          ...baseStyle,
          ...theme.globals.shadow.md,
        };
      case 'outlined':
        return {
          ...baseStyle,
          borderWidth: 1,
          borderColor: theme.colors.border,
        };
      default:
        return baseStyle;
    }
  };

  return (
    <View style={[getCardStyle(), style]} {...props}>
      {children}
    </View>
  );
}