// ReRoom Input Component - Based on BNA UI Design System

import React, { useState } from 'react'
import { 
  TextInput, 
  View, 
  StyleSheet, 
  TextInputProps,
  ViewStyle 
} from 'react-native'
import { Colors } from '../../theme/colors'
import { Typography } from '../../theme/typography'
import { Text } from './Text'

interface InputProps extends Omit<TextInputProps, 'style'> {
  label?: string
  error?: string
  helper?: string
  variant?: 'default' | 'outlined'
  size?: 'small' | 'medium' | 'large'
  fullWidth?: boolean
  disabled?: boolean
  containerStyle?: ViewStyle
}

export const Input: React.FC<InputProps> = ({
  label,
  error,
  helper,
  variant = 'outlined',
  size = 'medium',
  fullWidth = true,
  disabled = false,
  containerStyle,
  ...props
}) => {
  const [isFocused, setIsFocused] = useState(false)

  const containerStyles = [
    styles.container,
    fullWidth && styles.fullWidth,
    containerStyle,
  ]

  const inputStyles = [
    styles.base,
    styles[variant],
    styles[size],
    isFocused && styles.focused,
    error && styles.error,
    disabled && styles.disabled,
  ]

  return (
    <View style={containerStyles}>
      {label && (
        <Text variant="bodySmall" weight="medium" style={styles.label}>
          {label}
        </Text>
      )}
      
      <TextInput
        style={inputStyles}
        onFocus={() => setIsFocused(true)}
        onBlur={() => setIsFocused(false)}
        editable={!disabled}
        placeholderTextColor={Colors.text.tertiary}
        {...props}
      />
      
      {error && (
        <Text variant="caption" color="error" style={styles.message}>
          {error}
        </Text>
      )}
      
      {helper && !error && (
        <Text variant="caption" color="secondary" style={styles.message}>
          {helper}
        </Text>
      )}
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    marginBottom: 4,
  },

  fullWidth: {
    width: '100%',
  },

  label: {
    marginBottom: 6,
    color: Colors.text.primary,
  },

  message: {
    marginTop: 4,
  },

  base: {
    ...Typography.bodyMedium,
    color: Colors.text.primary,
    backgroundColor: Colors.background.primary,
    borderRadius: 8,
    borderWidth: 1,
  },

  // Variants
  default: {
    borderColor: 'transparent',
    backgroundColor: Colors.background.secondary,
  },

  outlined: {
    borderColor: Colors.border.primary,
    backgroundColor: Colors.background.primary,
  },

  // Sizes
  small: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    minHeight: 36,
    fontSize: 14,
  },

  medium: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    minHeight: 44,
    fontSize: 16,
  },

  large: {
    paddingHorizontal: 20,
    paddingVertical: 16,
    minHeight: 52,
    fontSize: 18,
  },

  // States
  focused: {
    borderColor: Colors.border.focus,
    borderWidth: 2,
  },

  error: {
    borderColor: Colors.semantic.error,
  },

  disabled: {
    opacity: 0.6,
    backgroundColor: Colors.background.secondary,
  },
})