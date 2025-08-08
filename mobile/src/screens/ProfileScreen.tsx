// ReRoom Profile Screen - User preferences and settings

import React from 'react'
import { View, StyleSheet, ScrollView } from 'react-native'
import { Button, Text, Card } from '../components/ui'
import { Colors } from '../theme/colors'
import { useAppStore } from '../stores/app-store'

export const ProfileScreen = () => {
  const { 
    user, 
    savedRooms, 
    getTotalSavings, 
    toggleDarkMode, 
    darkMode 
  } = useAppStore()

  const totalSavings = getTotalSavings()

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <View style={styles.header}>
        <Text variant="h1">Profile</Text>
      </View>

      <Card variant="elevated" style={styles.statsCard}>
        <Text variant="h3" color="primary" style={styles.statsTitle}>
          Your ReRoom Stats
        </Text>
        
        <View style={styles.statsRow}>
          <View style={styles.statItem}>
            <Text variant="h2" color="info">
              {savedRooms.length}
            </Text>
            <Text variant="bodySmall" color="secondary">
              Designs Created
            </Text>
          </View>
          
          <View style={styles.statItem}>
            <Text variant="h2" color="success">
              ${totalSavings.toFixed(0)}
            </Text>
            <Text variant="bodySmall" color="secondary">
              Total Saved
            </Text>
          </View>
        </View>
      </Card>

      <View style={styles.section}>
        <Text variant="h3" style={styles.sectionTitle}>
          Settings
        </Text>

        <Card variant="outlined" style={styles.settingCard}>
          <View style={styles.settingRow}>
            <View style={styles.settingInfo}>
              <Text variant="bodyMedium" weight="medium">
                Dark Mode
              </Text>
              <Text variant="bodySmall" color="secondary">
                Switch between light and dark themes
              </Text>
            </View>
            <Button
              title={darkMode ? 'On' : 'Off'}
              onPress={toggleDarkMode}
              variant={darkMode ? 'primary' : 'outline'}
              size="small"
            />
          </View>
        </Card>

        <Card variant="outlined" style={styles.settingCard}>
          <View style={styles.settingRow}>
            <View style={styles.settingInfo}>
              <Text variant="bodyMedium" weight="medium">
                Photo Quality
              </Text>
              <Text variant="bodySmall" color="secondary">
                Higher quality means better AI results
              </Text>
            </View>
            <Text variant="bodyMedium" color="info">
              Medium
            </Text>
          </View>
        </Card>
      </View>

      <View style={styles.section}>
        <Text variant="h3" style={styles.sectionTitle}>
          About
        </Text>
        
        <Text variant="bodyMedium" color="secondary" style={styles.aboutText}>
          ReRoom uses advanced AI to help you redesign your spaces and discover 
          furniture at the best prices across multiple retailers.
        </Text>
        
        <Text variant="caption" color="tertiary" align="center" style={styles.version}>
          Version 1.0.0
        </Text>
      </View>
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

  header: {
    paddingTop: 40,
    marginBottom: 24,
  },

  statsCard: {
    marginBottom: 32,
    padding: 20,
  },

  statsTitle: {
    marginBottom: 20,
    textAlign: 'center',
  },

  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },

  statItem: {
    alignItems: 'center',
  },

  section: {
    marginBottom: 32,
  },

  sectionTitle: {
    marginBottom: 16,
    color: Colors.text.primary,
  },

  settingCard: {
    marginBottom: 12,
    padding: 16,
  },

  settingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },

  settingInfo: {
    flex: 1,
    marginRight: 16,
  },

  aboutText: {
    lineHeight: 22,
    marginBottom: 20,
  },

  version: {
    marginTop: 20,
  },
})