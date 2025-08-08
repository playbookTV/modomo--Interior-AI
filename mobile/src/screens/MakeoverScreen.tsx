// ReRoom Makeover Screen - AI processing and style selection

import React, { useState, useEffect } from 'react'
import { View, StyleSheet, ScrollView, Image } from 'react-native'
import { useRoute, RouteProp } from '@react-navigation/native'
import { Button, Text, Card, Loading } from '../components/ui'
import { Colors } from '../theme/colors'
import { useAppStore } from '../stores/app-store'
import { RootStackParamList } from '../navigation/AppNavigator'
import { StyleType } from '../types'

type MakeoverScreenRouteProp = RouteProp<RootStackParamList, 'Makeover'>

const STYLE_OPTIONS: { value: StyleType; label: string; description: string }[] = [
  { value: 'modern', label: 'Modern', description: 'Clean lines and minimalist aesthetic' },
  { value: 'japandi', label: 'Japandi', description: 'Japanese-Scandinavian fusion' },
  { value: 'boho', label: 'Boho', description: 'Eclectic and free-spirited' },
  { value: 'scandinavian', label: 'Scandinavian', description: 'Light, airy, and functional' },
  { value: 'industrial', label: 'Industrial', description: 'Raw materials and urban edge' },
  { value: 'minimalist', label: 'Minimalist', description: 'Less is more philosophy' },
]

export const MakeoverScreen = () => {
  const route = useRoute<MakeoverScreenRouteProp>()
  const { imageUri } = route.params
  
  const { 
    selectedStyle, 
    setSelectedStyle, 
    isProcessing, 
    processingStatus, 
    startAIProcessing,
    currentRender 
  } = useAppStore()
  
  const [hasStartedProcessing, setHasStartedProcessing] = useState(false)

  const handleStyleSelect = (style: StyleType) => {
    setSelectedStyle(style)
  }

  const handleStartProcessing = () => {
    setHasStartedProcessing(true)
    startAIProcessing(imageUri, selectedStyle)
    
    // Simulate AI processing progress
    let progress = 0
    const interval = setInterval(() => {
      progress += Math.random() * 15
      if (progress >= 100) {
        progress = 100
        clearInterval(interval)
      }
      // Update progress in store would happen here
    }, 1000)
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Card style={styles.imageCard}>
        <Image source={{ uri: imageUri }} style={styles.image} resizeMode="cover" />
        <Text variant="caption" color="secondary" align="center" style={styles.imageLabel}>
          Original Photo
        </Text>
      </Card>

      {!hasStartedProcessing && (
        <>
          <View style={styles.section}>
            <Text variant="h3" style={styles.sectionTitle}>
              Choose Your Style
            </Text>
            <Text variant="bodyMedium" color="secondary">
              Select the interior design style you'd like to apply
            </Text>
          </View>

          <View style={styles.styleGrid}>
            {STYLE_OPTIONS.map((option) => (
              <Card
                key={option.value}
                variant={selectedStyle === option.value ? 'elevated' : 'outlined'}
                onPress={() => handleStyleSelect(option.value)}
                style={{
                  ...styles.styleCard,
                  ...(selectedStyle === option.value && styles.selectedStyleCard),
                }}
              >
                <Text variant="bodyMedium" weight="medium">
                  {option.label}
                </Text>
                <Text variant="caption" color="secondary">
                  {option.description}
                </Text>
              </Card>
            ))}
          </View>

          <Button
            title="Generate AI Design"
            onPress={handleStartProcessing}
            size="large"
            fullWidth
            style={styles.generateButton}
          />
        </>
      )}

      {isProcessing && (
        <Card style={styles.processingCard}>
          <Loading
            message={processingStatus?.stage}
            progress={processingStatus?.progress}
          />
        </Card>
      )}

      {currentRender && !isProcessing && (
        <View style={styles.resultSection}>
          <Text variant="h3" style={styles.sectionTitle}>
            Your AI Design
          </Text>
          <Card style={styles.imageCard}>
            <Image 
              source={{ uri: currentRender.styledImageUrl }} 
              style={styles.image} 
              resizeMode="cover" 
            />
            <Text variant="caption" color="secondary" align="center" style={styles.imageLabel}>
              {selectedStyle.charAt(0).toUpperCase() + selectedStyle.slice(1)} Style
            </Text>
          </Card>
          
          <Button
            title="Save & Shop"
            onPress={() => {/* TODO: Implement save & shop functionality */}}
            size="large"
            fullWidth
            style={styles.shopButton}
          />
        </View>
      )}
    </ScrollView>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background.primary,
  },

  content: {
    padding: 20,
  },

  imageCard: {
    marginBottom: 24,
  },

  image: {
    width: '100%',
    height: 200,
    borderRadius: 8,
    marginBottom: 8,
  },

  imageLabel: {
    marginTop: 4,
  },

  section: {
    marginBottom: 20,
  },

  sectionTitle: {
    marginBottom: 8,
    color: Colors.text.primary,
  },

  styleGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginBottom: 24,
  },

  styleCard: {
    flex: 1,
    minWidth: '45%',
    padding: 16,
  },

  selectedStyleCard: {
    borderColor: Colors.primary.blue,
    borderWidth: 2,
  },

  generateButton: {
    marginTop: 12,
  },

  processingCard: {
    marginVertical: 40,
  },

  resultSection: {
    marginTop: 20,
  },

  shopButton: {
    marginTop: 20,
  },
})