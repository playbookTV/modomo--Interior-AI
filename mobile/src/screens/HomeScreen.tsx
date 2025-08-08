// ReRoom Home Screen - Main landing page

import React from 'react'
import { View, StyleSheet, ScrollView } from 'react-native'
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
        <Text variant="h1" style={styles.title}>
          Welcome to ReRoom
        </Text>
        <Text variant="bodyLarge" color="secondary" style={styles.subtitle}>
          Transform your space with AI-powered interior design
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
        <Text variant="h3" style={styles.cardTitle}>
          Ready to redesign?
        </Text>
        <Text variant="bodyMedium" color="secondary" style={styles.cardDescription}>
          Take a photo of your room and let our AI show you amazing transformations
        </Text>
        <Button
          title="Start Designing"
          onPress={handleStartDesigning}
          size="large"
          fullWidth
          style={styles.designButton}
        />
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
  },

  title: {
    marginBottom: 8,
    color: Colors.text.primary,
  },

  subtitle: {
    lineHeight: 24,
  },

  savingsCard: {
    marginBottom: 24,
    backgroundColor: Colors.background.secondary,
  },

  actionCard: {
    marginBottom: 32,
    padding: 24,
  },

  cardTitle: {
    marginBottom: 8,
    color: Colors.text.primary,
  },

  cardDescription: {
    marginBottom: 20,
    lineHeight: 22,
  },

  designButton: {
    marginTop: 8,
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