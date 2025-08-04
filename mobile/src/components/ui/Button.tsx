/**
 * BNA UI Button Component
 * Themeable button with multiple variants
 */

import React from 'react';
import {
  TouchableOpacity,
  Text,
  StyleSheet,
  ActivityIndicator,
  ViewStyle,
  TextStyle,
  TouchableOpacityProps,
} from 'react-native';
import { useTheme } from '@/theme/theme-provider';

interface ButtonProps extends TouchableOpacityProps {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: 'sm' | 'md' | 'lg' | 'xl';
  children: React.ReactNode;
  loading?: boolean;
  fullWidth?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

export function Button({
  variant = 'primary',
  size = 'md',
  children,
  loading = false,
  fullWidth = false,
  leftIcon,
  rightIcon,
  disabled,
  style,
  ...props
}: ButtonProps) {
  const { theme } = useTheme();
  
  const getButtonStyle = (): ViewStyle => {
    const baseStyle: ViewStyle = {
      height: theme.globals.components.button.height[size],
      paddingHorizontal: theme.globals.components.button.paddingHorizontal[size],
      borderRadius: theme.globals.borderRadius.lg,
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'center',
      gap: theme.globals.spacing[2],
    };

    if (fullWidth) {
      baseStyle.width = '100%';
    }

    switch (variant) {
      case 'primary':
        return {
          ...baseStyle,
          backgroundColor: disabled ? theme.colors.interactiveDisabled : theme.colors.primary,
          ...theme.globals.shadow.md,
        };
      case 'secondary':
        return {
          ...baseStyle,
          backgroundColor: disabled ? theme.colors.interactiveDisabled : theme.colors.secondary,
          borderWidth: 1,
          borderColor: theme.colors.border,
        };
      case 'outline':
        return {
          ...baseStyle,
          backgroundColor: 'transparent',
          borderWidth: 1,
          borderColor: disabled ? theme.colors.interactiveDisabled : theme.colors.primary,
        };
      case 'ghost':
        return {
          ...baseStyle,
          backgroundColor: 'transparent',
        };
      default:
        return baseStyle;
    }
  };

  const getTextStyle = (): TextStyle => {
    const baseStyle: TextStyle = {
      fontSize: size === 'sm' ? theme.globals.typography.fontSize.sm : 
                size === 'lg' ? theme.globals.typography.fontSize.lg :
                size === 'xl' ? theme.globals.typography.fontSize.xl :
                theme.globals.typography.fontSize.base,
      fontWeight: theme.globals.typography.fontWeight.semiBold,
    };

    switch (variant) {
      case 'primary':
        return {
          ...baseStyle,
          color: disabled ? theme.colors.textMuted : theme.colors.primaryForeground,
        };
      case 'secondary':
        return {
          ...baseStyle,
          color: disabled ? theme.colors.textMuted : theme.colors.secondaryForeground,
        };
      case 'outline':
        return {
          ...baseStyle,
          color: disabled ? theme.colors.interactiveDisabled : theme.colors.primary,
        };
      case 'ghost':
        return {
          ...baseStyle,
          color: disabled ? theme.colors.interactiveDisabled : theme.colors.text,
        };
      default:
        return baseStyle;
    }
  };

  return (
    <TouchableOpacity
      style={[getButtonStyle(), style]}
      disabled={disabled || loading}
      activeOpacity={0.7}
      {...props}
    >
      {loading ? (
        <ActivityIndicator
          size="small"
          color={variant === 'primary' ? theme.colors.primaryForeground : theme.colors.primary}
        />
      ) : (
        <>
          {leftIcon}
          <Text style={getTextStyle()}>{children}</Text>
          {rightIcon}
        </>
      )}
    </TouchableOpacity>
  );
}