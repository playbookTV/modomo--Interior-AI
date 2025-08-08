// ReRoom Gallery Screen - View saved room designs

import React from 'react'
import { View, StyleSheet, ScrollView, FlatList, Dimensions } from 'react-native'
import { Text, Card } from '../components/ui'
import { Colors } from '../theme/colors'
import { useAppStore } from '../stores/app-store'

const { width } = Dimensions.get('window')
const cardWidth = (width - 60) / 2

export const GalleryScreen = () => {
  const { savedRooms } = useAppStore()

  const renderRoomItem = ({ item }: { item: any }) => (
    <Card variant="elevated" style={styles.roomCard}>
      <View style={styles.imageContainer}>
        <Text variant="caption" color="secondary" align="center">
          Image Preview
        </Text>
      </View>
      <View style={styles.roomInfo}>
        <Text variant="bodyMedium" weight="medium" numberOfLines={1}>
          {item.name}
        </Text>
        <Text variant="caption" color="secondary">
          {item.style.charAt(0).toUpperCase() + item.style.slice(1)}
        </Text>
        <Text variant="bodySmall" color="success">
          Saved ${item.totalSavings.toFixed(2)}
        </Text>
      </View>
    </Card>
  )

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text variant="h1">Gallery</Text>
        <Text variant="bodyMedium" color="secondary">
          {savedRooms.length} saved design{savedRooms.length !== 1 ? 's' : ''}
        </Text>
      </View>

      {savedRooms.length === 0 ? (
        <View style={styles.emptyState}>
          <Text variant="h3" color="secondary" align="center">
            No designs yet
          </Text>
          <Text variant="bodyMedium" color="secondary" align="center" style={styles.emptyText}>
            Start by capturing a photo of your room and creating your first AI design
          </Text>
        </View>
      ) : (
        <FlatList
          data={savedRooms}
          renderItem={renderRoomItem}
          keyExtractor={(item) => item.id}
          numColumns={2}
          showsVerticalScrollIndicator={false}
          contentContainerStyle={styles.grid}
        />
      )}
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background.primary,
  },

  header: {
    padding: 20,
    paddingTop: 60,
  },

  grid: {
    padding: 20,
    paddingTop: 0,
  },

  roomCard: {
    width: cardWidth,
    marginRight: 20,
    marginBottom: 20,
  },

  imageContainer: {
    height: 120,
    backgroundColor: Colors.background.secondary,
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
  },

  roomInfo: {
    gap: 4,
  },

  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
  },

  emptyText: {
    marginTop: 12,
    textAlign: 'center',
  },
})