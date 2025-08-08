// ReRoom Card Component - Based on BNA UI Design System

import React from 'react'
import { View, ViewStyle, StyleSheet, TouchableOpacity, TouchableOpacityProps } from 'react-native'
import { Colors } from '../../theme/colors'

interface CardProps {
  children: React.ReactNode
  variant?: 'default' | 'elevated' | 'outlined'
  padding?: 'none' | 'small' | 'medium' | 'large'
  onPress?: () => void
  style?: ViewStyle
  testID?: string
  disabled?: boolean
}

export const Card: React.FC<CardProps> = ({
  children,
  variant = 'default',
  padding = 'medium',
  onPress,
  style,
  testID,
  disabled = false,
}) => {
  const cardStyles = [
    styles.base,
    styles[variant],
    styles[padding],
    disabled && styles.disabled,
    style,
  ]

  if (onPress) {
    return (
      <TouchableOpacity
        style={cardStyles}
        onPress={onPress}
        disabled={disabled}
        testID={testID}
        activeOpacity={0.7}
      >
        {children}
      </TouchableOpacity>
    )
  }

  return (
    <View style={cardStyles} testID={testID}>
      {children}
    </View>
  )
}

const styles = StyleSheet.create({
  base: {
    backgroundColor: Colors.background.primary,
    borderRadius: 12,
  },

  // Variants
  default: {
    backgroundColor: Colors.background.primary,
  },

  elevated: {
    backgroundColor: Colors.background.primary,
    shadowColor: Colors.primary.black,
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },

  outlined: {
    backgroundColor: Colors.background.primary,
    borderWidth: 1,
    borderColor: Colors.border.primary,
  },

  // Padding
  none: {
    padding: 0,
  },

  small: {
    padding: 12,
  },

  medium: {
    padding: 16,
  },

  large: {
    padding: 20,
  },

  disabled: {
    opacity: 0.6,
  },
})