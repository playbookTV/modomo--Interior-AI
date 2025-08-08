// ReRoom Loading Component - For AI processing states

import React from 'react'
import { View, ActivityIndicator, StyleSheet, ViewStyle } from 'react-native'
import { Colors } from '../../theme/colors'
import { Text } from './Text'

interface LoadingProps {
  message?: string
  progress?: number
  size?: 'small' | 'large'
  color?: string
  style?: ViewStyle
}

export const Loading: React.FC<LoadingProps> = ({
  message,
  progress,
  size = 'large',
  color = Colors.primary.blue,
  style,
}) => {
  return (
    <View style={[styles.container, style]}>
      <ActivityIndicator size={size} color={color} />
      
      {message && (
        <Text variant="bodyMedium" color="secondary" style={styles.message}>
          {message}
        </Text>
      )}
      
      {typeof progress === 'number' && (
        <View style={styles.progressContainer}>
          <View style={styles.progressBar}>
            <View 
              style={[
                styles.progressFill, 
                { width: `${Math.max(0, Math.min(100, progress))}%` }
              ]} 
            />
          </View>
          <Text variant="caption" color="secondary" style={styles.progressText}>
            {Math.round(progress)}%
          </Text>
        </View>
      )}
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },

  message: {
    marginTop: 12,
    textAlign: 'center',
  },

  progressContainer: {
    marginTop: 16,
    width: '100%',
    alignItems: 'center',
  },

  progressBar: {
    width: '100%',
    height: 4,
    backgroundColor: Colors.background.secondary,
    borderRadius: 2,
    overflow: 'hidden',
  },

  progressFill: {
    height: '100%',
    backgroundColor: Colors.primary.blue,
    borderRadius: 2,
  },

  progressText: {
    marginTop: 6,
  },
})