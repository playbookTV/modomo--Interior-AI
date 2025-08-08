// ReRoom Home Screen - Main landing page

import React from 'react'
import { View, StyleSheet, ScrollView, Image } from 'react-native'
import { useNavigation } from '@react-navigation/native'
import { NativeStackNavigationProp } from '@react-navigation/native-stack'

import { Button, Text, Card } from '../components/ui'
import { Colors } from '../theme/colors'
import { useAppStore } from '../stores/app-store'
import { RootStackParamList } from '../navigation/AppNavigator'

type HomeScreenNavigationProp = NativeStackNavigationProp<RootStackParamList>

export const HomeScreen = () => {
  const navigation = useNavigation<HomeScreenNavigationProp>()
  const { savedRooms, getTotalSavings } = useAppStore()

  const totalSavings = getTotalSavings()

  const handleStartDesigning = () => {
    navigation.navigate('Camera')
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <View style={styles.header}>
        <View style={styles.branding}>
          <View style={styles.logoContainer}>
            <Text style={styles.logoText}>🏠</Text>
          </View>
          <Text variant="h1" style={styles.title}>
            ReRoom
          </Text>
          <Text variant="bodyLarge" color="secondary" style={styles.tagline}>
            Snap. Style. Save.
          </Text>
        </View>
        <Text variant="bodyLarge" color="secondary" style={styles.subtitle}>
          Transform your space with AI-powered interior design and discover amazing savings
        </Text>
      </View>

      {totalSavings > 0 && (
        <Card variant="elevated" style={styles.savingsCard}>
          <Text variant="h3" color="success">
            ${totalSavings.toFixed(2)} Saved
          </Text>
          <Text variant="bodyMedium" color="secondary">
            Total savings across all your room designs
          </Text>
        </Card>
      )}

      <Card style={styles.actionCard}>
        <View style={styles.cardHeader}>
          <Text style={styles.cardIcon}>📸</Text>
          <Text variant="h3" style={styles.cardTitle}>
            Ready to redesign?
          </Text>
        </View>
        <Text variant="bodyMedium" color="secondary" style={styles.cardDescription}>
          Take a photo of your room and let our AI show you amazing transformations with real product recommendations
        </Text>
        <Button
          title="🎨 Start Designing"
          onPress={handleStartDesigning}
          size="large"
          fullWidth
          style={styles.designButton}
        />
        <View style={styles.features}>
          <View style={styles.feature}>
            <Text style={styles.featureIcon}>⚡</Text>
            <Text variant="caption" color="secondary">AI-powered</Text>
          </View>
          <View style={styles.feature}>
            <Text style={styles.featureIcon}>💰</Text>
            <Text variant="caption" color="secondary">Save money</Text>
          </View>
          <View style={styles.feature}>
            <Text style={styles.featureIcon}>🎯</Text>
            <Text variant="caption" color="secondary">Instant results</Text>
          </View>
        </View>
      </Card>

      {savedRooms.length > 0 && (
        <View style={styles.recentSection}>
          <Text variant="h3" style={styles.sectionTitle}>
            Recent Designs
          </Text>
          {savedRooms.slice(0, 3).map((room) => (
            <Card key={room.id} variant="outlined" style={styles.roomCard}>
              <Text variant="bodyMedium" weight="medium">
                {room.name}
              </Text>
              <Text variant="caption" color="secondary">
                {room.style.charAt(0).toUpperCase() + room.style.slice(1)} style
              </Text>
              <Text variant="bodySmall" color="success">
                Saved ${room.totalSavings.toFixed(2)}
              </Text>
            </Card>
          ))}
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
    paddingTop: 60,
  },

  header: {
    marginBottom: 32,
    alignItems: 'center',
  },

  branding: {
    alignItems: 'center',
    marginBottom: 24,
  },

  logoContainer: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: Colors.primary.blue,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
    shadowColor: Colors.primary.black,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },

  logoText: {
    fontSize: 36,
  },

  title: {
    marginBottom: 4,
    color: Colors.text.primary,
    textAlign: 'center',
  },

  tagline: {
    fontWeight: '600',
    color: Colors.primary.blue,
    marginBottom: 8,
  },

  subtitle: {
    lineHeight: 24,
    textAlign: 'center',
  },

  savingsCard: {
    marginBottom: 24,
    backgroundColor: Colors.background.secondary,
  },

  actionCard: {
    marginBottom: 32,
    padding: 24,
    backgroundColor: Colors.background.primary,
    borderRadius: 16,
    shadowColor: Colors.primary.black,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },

  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },

  cardIcon: {
    fontSize: 24,
    marginRight: 12,
  },

  cardTitle: {
    color: Colors.text.primary,
    flex: 1,
  },

  cardDescription: {
    marginBottom: 20,
    lineHeight: 22,
  },

  designButton: {
    marginTop: 8,
    marginBottom: 16,
    backgroundColor: Colors.primary.blue,
  },

  features: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: Colors.border.primary,
  },

  feature: {
    alignItems: 'center',
  },

  featureIcon: {
    fontSize: 16,
    marginBottom: 4,
  },

  recentSection: {
    marginBottom: 32,
  },

  sectionTitle: {
    marginBottom: 16,
    color: Colors.text.primary,
  },

  roomCard: {
    marginBottom: 12,
    padding: 16,
  },
})