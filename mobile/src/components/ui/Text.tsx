/**
 * BNA UI Text Component
 * Themeable text with typography variants
 */

import React from 'react';
import { Text as RNText, TextStyle, TextProps as RNTextProps } from 'react-native';
import { useTheme } from '@/theme/theme-provider';

interface TextProps extends RNTextProps {
  variant?: 'h1' | 'h2' | 'h3' | 'h4' | 'body' | 'caption' | 'label';
  color?: 'primary' | 'secondary' | 'muted' | 'success' | 'warning' | 'error';
  weight?: 'normal' | 'medium' | 'semiBold' | 'bold';
  align?: 'left' | 'center' | 'right';
  children: React.ReactNode;
}

export function Text({
  variant = 'body',
  color = 'primary',
  weight,
  align = 'left',
  children,
  style,
  ...props
}: TextProps) {
  const { theme } = useTheme();
  
  const getTextStyle = (): TextStyle => {
    const baseStyle: TextStyle = {
      textAlign: align,
    };

    // Font size and line height based on variant
    switch (variant) {
      case 'h1':
        baseStyle.fontSize = theme.globals.typography.fontSize['4xl'];
        baseStyle.fontWeight = theme.globals.typography.fontWeight.bold;
        baseStyle.lineHeight = theme.globals.typography.fontSize['4xl'] * 1.2;
        break;
      case 'h2':
        baseStyle.fontSize = theme.globals.typography.fontSize['3xl'];
        baseStyle.fontWeight = theme.globals.typography.fontWeight.bold;
        baseStyle.lineHeight = theme.globals.typography.fontSize['3xl'] * 1.2;
        break;
      case 'h3':
        baseStyle.fontSize = theme.globals.typography.fontSize['2xl'];
        baseStyle.fontWeight = theme.globals.typography.fontWeight.semiBold;
        baseStyle.lineHeight = theme.globals.typography.fontSize['2xl'] * 1.3;
        break;
      case 'h4':
        baseStyle.fontSize = theme.globals.typography.fontSize.xl;
        baseStyle.fontWeight = theme.globals.typography.fontWeight.semiBold;
        baseStyle.lineHeight = theme.globals.typography.fontSize.xl * 1.3;
        break;
      case 'body':
        baseStyle.fontSize = theme.globals.typography.fontSize.base;
        baseStyle.fontWeight = theme.globals.typography.fontWeight.normal;
        baseStyle.lineHeight = theme.globals.typography.fontSize.base * 1.4;
        break;
      case 'caption':
        baseStyle.fontSize = theme.globals.typography.fontSize.sm;
        baseStyle.fontWeight = theme.globals.typography.fontWeight.normal;
        baseStyle.lineHeight = theme.globals.typography.fontSize.sm * 1.4;
        break;
      case 'label':
        baseStyle.fontSize = theme.globals.typography.fontSize.sm;
        baseStyle.fontWeight = theme.globals.typography.fontWeight.medium;
        baseStyle.lineHeight = theme.globals.typography.fontSize.sm * 1.3;
        break;
    }

    // Override font weight if specified
    if (weight) {
      baseStyle.fontWeight = theme.globals.typography.fontWeight[weight];
    }

    // Color based on color prop
    switch (color) {
      case 'primary':
        baseStyle.color = theme.colors.text;
        break;
      case 'secondary':
        baseStyle.color = theme.colors.textSecondary;
        break;
      case 'muted':
        baseStyle.color = theme.colors.textMuted;
        break;
      case 'success':
        baseStyle.color = theme.colors.success;
        break;
      case 'warning':
        baseStyle.color = theme.colors.warning;
        break;
      case 'error':
        baseStyle.color = theme.colors.error;
        break;
    }

    return baseStyle;
  };

  return (
    <RNText style={[getTextStyle(), style]} {...props}>
      {children}
    </RNText>
  );
}