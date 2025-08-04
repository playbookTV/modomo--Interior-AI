import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Image,
  Alert,
  RefreshControl,
  Dimensions,
  ActivityIndicator,
} from 'react-native';
import { router, useFocusEffect } from 'expo-router';
import { useAuth } from '@clerk/clerk-expo';
import CloudPhotoService, { PhotoWithMakeover } from '../services/cloudPhotoService';
import { PhotoService } from '../services/photoService'; // For fallback operations
import { useTheme } from '@/theme/theme-provider';
import { Button, Text, Card } from '@/components/ui';

const { width } = Dimensions.get('window');
const PHOTO_SIZE = (width - 60) / 2; // 2 columns with padding

export default function GalleryScreen() {
  const { theme } = useTheme();
  const auth = useAuth();
  const [photos, setPhotos] = useState<PhotoWithMakeover[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [cacheStats, setCacheStats] = useState({
    localPhotos: 0,
    cachedPhotos: 0,
    syncedPhotos: 0,
    pendingSync: 0,
  });

  // Initialize cloud photo service
  const cloudPhotoService = auth.isSignedIn ? new CloudPhotoService(auth) : null;

  const loadPhotos = useCallback(async () => {
    if (!auth.isSignedIn || !cloudPhotoService) {
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      
      // Load photos from cloud (with local cache fallback)
      const cloudPhotos = await cloudPhotoService.getUserPhotos(refreshing);
      const stats = cloudPhotoService.getCacheStats();
      
      setPhotos(cloudPhotos);
      setCacheStats(stats);
    } catch (error) {
      console.error('Failed to load photos:', error);
      Alert.alert(
        'Error', 
        'Failed to load photos from cloud. Please check your connection and try again.'
      );
    } finally {
      setLoading(false);
    }
  }, [auth.isSignedIn, cloudPhotoService, refreshing]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadPhotos();
    setRefreshing(false);
  }, [loadPhotos]);

  useFocusEffect(
    useCallback(() => {
      loadPhotos();
    }, [loadPhotos])
  );

  const handleDeletePhoto = (photoId: string) => {
    Alert.alert(
      'Delete Photo',
      'Are you sure you want to delete this photo? This action cannot be undone.',
      [
        {
          text: 'Cancel',
          style: 'cancel',
        },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              const success = await PhotoService.deletePhoto(photoId);
              if (success) {
                await loadPhotos(); // Refresh the list
              } else {
                Alert.alert('Error', 'Failed to delete photo. Please try again.');
              }
            } catch (error) {
              console.error('Delete photo error:', error);
              Alert.alert('Error', 'Failed to delete photo. Please try again.');
            }
          },
        },
      ]
    );
  };

  const handleClearAll = () => {
    if (photos.length === 0) return;

    Alert.alert(
      'Clear All Photos',
      'Are you sure you want to delete all photos? This action cannot be undone.',
      [
        {
          text: 'Cancel',
          style: 'cancel',
        },
        {
          text: 'Clear All',
          style: 'destructive',
          onPress: async () => {
            try {
              await PhotoService.clearAllPhotos();
              await loadPhotos(); // Refresh the list
            } catch (error) {
              console.error('Clear all photos error:', error);
              Alert.alert('Error', 'Failed to clear photos. Please try again.');
            }
          },
        },
      ]
    );
  };

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleDateString();
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const renderPhotoItem = ({ item }: { item: PhotoWithMakeover }) => {
    const makeover = item.makeovers?.[0]; // Get the latest makeover

    const handleLongPress = () => {
      Alert.alert(
        'Photo Options',
        `Photo: ${item.original_name || 'Unknown'}\nUploaded: ${new Date(item.created_at).toLocaleString()}\nStorage: Cloudflare R2 (Global CDN)`,
        [
          { text: 'Cancel', style: 'cancel' },
          {
            text: 'View Details',
            onPress: () => router.push(`/photo/${item.id}`),
          },
          { 
            text: 'Delete', 
            style: 'destructive',
            onPress: () => handleDeletePhoto(item.id)
          },
        ]
      );
    };

    const handlePress = () => {
      if (makeover && makeover.status === 'completed') {
        // Navigate to completed makeover
        router.push(`/makeover/${makeover.id}`);
      } else {
        // Navigate to photo details
        router.push(`/photo/${item.id}`);
      }
    };

    const getCloudAIStatus = () => {
      if (!makeover) {
        return { icon: '📷', text: 'Ready for AI', color: '#666' };
      }
      
      switch (makeover.status) {
        case 'queued':
          return { icon: '⏳', text: 'Queued...', color: '#FF9800' };
        case 'processing':
          return { 
            icon: '🧠', 
            text: `Processing ${makeover.progress || 0}%`, 
            color: '#2196F3',
            progress: makeover.progress || 0
          };
        case 'completed':
          return { icon: '✨', text: 'Tap for Makeover', color: '#4CAF50' };
        case 'failed':
          return { icon: '❌', text: 'Processing Failed', color: '#f44336' };
        default:
          return { icon: '❓', text: 'Unknown Status', color: '#666' };
      }
    };

    const aiStatus = getCloudAIStatus();

    return (
      <TouchableOpacity 
        style={styles.photoItem} 
        onLongPress={handleLongPress}
        onPress={handlePress}
        activeOpacity={0.7}
      >
        <Image 
          source={{ uri: item.optimized_url || item.original_url }} 
          style={styles.photoImage}
          defaultSource={{ uri: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==' }}
        />
        
        <View style={styles.photoInfo}>
          <Text style={styles.photoId}>
            {item.original_name || `Photo ${item.id.slice(-8)}`}
          </Text>
          
          <View style={styles.statusRow}>
            <Text style={styles.statusText}>
              ☁️ Cloud Storage
            </Text>
            <Text style={styles.statusText}>
              🌍 Global CDN
            </Text>
          </View>

          {/* AI Status with Progress */}
          <View style={[styles.aiStatusRow, { backgroundColor: aiStatus.color + '20' }]}>
            <Text style={styles.aiStatusText}>
              {aiStatus.icon} {aiStatus.text}
            </Text>
            {aiStatus.progress && aiStatus.progress > 0 && (
              <View style={styles.progressContainer}>
                <View style={[styles.progressBar, { width: `${aiStatus.progress}%` }]} />
              </View>
            )}
          </View>

          {/* Makeover Results Preview */}
          {makeover?.status === 'completed' && (
            <View style={styles.makeoverPreview}>
              <Text style={styles.makeoverText}>
                Style: {makeover.style_preference}
              </Text>
              {makeover.makeover_url && (
                <Text style={styles.makeoverText}>
                  🎨 Tap to view makeover
                </Text>
              )}
            </View>
          )}

          {/* Error Display */}
          {makeover?.status === 'failed' && makeover.error_message && (
            <View style={styles.errorPreview}>
              <Text style={styles.errorText}>
                ❌ {makeover.error_message}
              </Text>
            </View>
          )}
          
          <Text style={styles.timestamp}>
            {new Date(item.created_at).toLocaleString()}
          </Text>
        </View>
      </TouchableOpacity>
    );
  };

  const handleClearCache = async () => {
    if (cloudPhotoService) {
      cloudPhotoService.clearCache();
      await loadPhotos();
    }
  };

  const renderHeader = () => (
    <View style={styles.header}>
      <Text variant="h2" style={styles.title}>☁️ Cloud Gallery</Text>
      
      {/* Authentication Status */}
      {!auth.isSignedIn ? (
        <View>
          <Text variant="body" color="secondary" style={styles.subtitle}>
            Sign in to access cloud storage and AI makeovers
          </Text>
          <Card style={{ marginTop: 16, padding: 16, backgroundColor: '#FFF3CD' }}>
            <Text variant="h4" style={{ marginBottom: 8 }}>🔐 Authentication Required</Text>
            <Text variant="body" color="secondary" style={{ marginBottom: 12 }}>
              Access cloud storage, AI makeovers, and real-time sync
            </Text>
            <Button 
              variant="primary" 
              size="md" 
              onPress={() => router.push('/auth')}
              fullWidth
            >
              Sign In to Continue
            </Button>
          </Card>
        </View>
      ) : (
        <View>
          <Text variant="body" color="secondary" style={styles.subtitle}>
            {photos.length} photo{photos.length !== 1 ? 's' : ''} stored in cloud with global CDN
          </Text>
          
          <View style={styles.actionButtons}>
            <Button 
              variant="primary" 
              size="md"
              onPress={() => router.push('/camera')}
              style={styles.primaryButton}
            >
              📸 Capture New Room
            </Button>
            
            <Button 
              variant="secondary" 
              size="md"
              onPress={handleClearCache}
              style={styles.secondaryButton}
            >
              🔄 Refresh
            </Button>
          </View>

          {/* Cloud Storage Stats */}
          {photos.length > 0 && (
            <Card style={{ marginTop: 16, padding: 16 }}>
              <Text variant="h4" style={{ marginBottom: 8 }}>📊 Statistics</Text>
              <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
                <View style={{ alignItems: 'center' }}>
                  <Text variant="h3" color="primary">{photos.length}</Text>
                  <Text variant="caption" color="secondary">Photos</Text>
                </View>
                <View style={{ alignItems: 'center' }}>
                  <Text variant="h3" color="primary">{photos.filter(p => p.makeovers?.length > 0).length}</Text>
                  <Text variant="caption" color="secondary">With AI</Text>
                </View>
                <View style={{ alignItems: 'center' }}>
                  <Text variant="h3" color="primary">{photos.filter(p => p.makeovers?.[0]?.status === 'completed').length}</Text>
                  <Text variant="caption" color="secondary">Ready</Text>
                </View>
              </View>
            </Card>
          )}
        </View>
      )}
    </View>
  );

  const renderEmptyState = () => (
    <View style={styles.emptyState}>
      <Text style={styles.emptyIcon}>☁️</Text>
      <Text variant="h3" align="center" style={styles.emptyTitle}>
        {auth.isSignedIn ? 'No Photos in Cloud' : 'Sign In Required'}
      </Text>
      <Text variant="body" color="secondary" align="center" style={styles.emptyMessage}>
        {auth.isSignedIn 
          ? 'Start by capturing your first room photo for AI-powered makeovers stored globally.'
          : 'Sign in to access cloud storage, AI makeovers, and sync across devices.'
        }
      </Text>
      <Button 
        variant="primary" 
        size="lg"
        onPress={() => auth.isSignedIn ? router.push('/camera') : router.push('/auth')}
        style={styles.emptyButton}
      >
        {auth.isSignedIn ? '📸 Capture Your First Room' : '🔐 Sign In to Continue'}
      </Button>
    </View>
  );

  return (
    <View style={[styles.container, { backgroundColor: theme.colors.galleryBackground }]}>
      {photos.length === 0 ? (
        <View style={styles.content}>
          {renderHeader()}
          {renderEmptyState()}
        </View>
      ) : (
        <FlatList
          data={photos}
          renderItem={renderPhotoItem}
          keyExtractor={(item) => item.id}
          numColumns={2}
          contentContainerStyle={styles.content}
          ListHeaderComponent={renderHeader}
          refreshControl={
            <RefreshControl 
              refreshing={refreshing} 
              onRefresh={onRefresh}
              tintColor={theme.colors.primary}
              colors={[theme.colors.primary]}
            />
          }
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    padding: 20,
  },
  header: {
    marginBottom: 20,
  },
  title: {
    marginBottom: 4,
  },
  subtitle: {
    marginBottom: 20,
  },
  actionButtons: {
    flexDirection: 'row',
    gap: 12,
    flexWrap: 'wrap',
  },
  primaryButton: {
    flex: 1,
    minWidth: 160,
  },
  secondaryButton: {
    minWidth: 100,
  },
  photoItem: {
    marginRight: 20,
    marginBottom: 20,
  },
  photoImage: {
    width: '100%',
    height: PHOTO_SIZE * 0.75,
    backgroundColor: '#f0f0f0',
    borderTopLeftRadius: 12,
    borderTopRightRadius: 12,
  },
  photoInfo: {
    padding: 12,
    gap: 2,
  },
  photoId: {
    fontSize: 12,
    color: '#666',
  },
  statusRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 4,
  },
  statusText: {
    fontSize: 12,
    color: '#666',
  },
  aiStatusRow: {
    marginTop: 8,
    padding: 8,
    borderRadius: 6,
    alignItems: 'center',
  },
  aiStatusText: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#333',
  },
  makeoverPreview: {
    marginTop: 4,
    padding: 6,
    backgroundColor: '#28a74520',
    borderRadius: 4,
  },
  makeoverText: {
    fontSize: 11,
    color: '#28a745',
    fontWeight: 'bold',
  },
  timestamp: {
    fontSize: 10,
    color: '#999',
    marginTop: 4,
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyIcon: {
    fontSize: 64,
    marginBottom: 16,
  },
  emptyTitle: {
    marginBottom: 8,
  },
  emptyMessage: {
    marginBottom: 32,
    maxWidth: 280,
  },
  emptyButton: {
    width: 280,
  },
  progressContainer: {
    marginTop: 4,
    backgroundColor: 'rgba(255,255,255,0.3)',
    borderRadius: 3,
    height: 4,
    overflow: 'hidden',
  },
  progressBar: {
    backgroundColor: '#4CAF50',
    borderRadius: 3,
    height: 4,
  },
  errorPreview: {
    marginTop: 4,
    padding: 6,
    backgroundColor: '#f4433620',
    borderRadius: 4,
  },
  errorText: {
    fontSize: 11,
    color: '#f44336',
    fontWeight: 'bold',
  },
}); 