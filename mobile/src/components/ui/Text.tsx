// ReRoom Text Component - Based on BNA UI Design System

import React from 'react'
import { Text as RNText, TextStyle, TextProps as RNTextProps } from 'react-native'
import { Colors } from '../../theme/colors'
import { Typography, TypographyKeys } from '../../theme/typography'

interface TextProps extends RNTextProps {
  variant?: TypographyKeys
  color?: 'primary' | 'secondary' | 'tertiary' | 'inverse' | 'success' | 'warning' | 'error' | 'info'
  align?: 'left' | 'center' | 'right'
  weight?: 'normal' | 'medium' | 'semibold' | 'bold'
  children: React.ReactNode
  style?: TextStyle
}

export const Text: React.FC<TextProps> = ({
  variant = 'bodyMedium',
  color = 'primary',
  align = 'left',
  weight,
  children,
  style,
  ...props
}) => {
  const getColor = () => {
    switch (color) {
      case 'primary':
        return Colors.text.primary
      case 'secondary':
        return Colors.text.secondary
      case 'tertiary':
        return Colors.text.tertiary
      case 'inverse':
        return Colors.text.inverse
      case 'success':
        return Colors.semantic.success
      case 'warning':
        return Colors.semantic.warning
      case 'error':
        return Colors.semantic.error
      case 'info':
        return Colors.semantic.info
      default:
        return Colors.text.primary
    }
  }

  const getFontWeight = () => {
    if (weight) {
      switch (weight) {
        case 'normal':
          return '400'
        case 'medium':
          return '500'
        case 'semibold':
          return '600'
        case 'bold':
          return '700'
      }
    }
    return Typography[variant].fontWeight
  }

  const textStyles: TextStyle = {
    ...Typography[variant],
    color: getColor(),
    textAlign: align,
    fontWeight: getFontWeight() as any,
    ...style,
  }

  return (
    <RNText style={textStyles} {...props}>
      {children}
    </RNText>
  )
}