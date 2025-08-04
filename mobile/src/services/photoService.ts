import ImageResizer from 'react-native-image-resizer';
import { MMKV } from 'react-native-mmkv';
import BackendService from './backendService';
import { Logger } from '../utils/logger';

// Initialize MMKV storage for photos
const photoStorage = new MMKV({
  id: 'photo-storage',
  encryptionKey: 'reroom-photos-key',
});

export interface PhotoMetadata {
  id: string;
  originalPath: string;
  optimizedPath: string;
  takenAt: string;
  originalSize: number;
  optimizedSize: number;
  optimized: boolean;
  backendId?: string;
  backendUrl?: string;
  uploadedAt?: string;
  // AI Makeover data
  makeoverId?: string;
  makeover?: any; // Full makeover response
  aiProcessing?: boolean;
  aiProcessedAt?: string;
  // Additional properties for compatibility
  timestamp?: number;
  size?: number;
}

export interface PhotoOptimizationOptions {
  maxWidth?: number;
  maxHeight?: number;
  quality?: number;
  format?: 'JPEG' | 'PNG';
  compressThreshold?: number; // MB
  compressImageMaxWidth?: number;
  compressImageMaxHeight?: number;
  onlyScaleDown?: boolean;
}

export class PhotoService {
  private static readonly DEFAULT_OPTIONS: Required<PhotoOptimizationOptions> = {
    maxWidth: 1920,
    maxHeight: 1080,
    quality: 80,
    format: 'JPEG',
    compressThreshold: 2, // 2MB
    compressImageMaxWidth: 1920,
    compressImageMaxHeight: 1080,
    onlyScaleDown: true,
  };

  private static storage = new MMKV({
    id: 'reroom-photos',
    encryptionKey: 'reroom-photos-key-2024',
  });

  /**
   * Optimize photo and trigger AI makeover
   */
  static async optimizePhoto(
    originalPath: string, 
    options: PhotoOptimizationOptions = {},
    triggerAI: boolean = true
  ): Promise<PhotoMetadata> {
    try {
      Logger.info('Starting photo optimization', { originalPath, triggerAI });

      const {
        maxWidth = 1920,
        maxHeight = 1080,
        quality = 80,
        format = 'JPEG',
        compressImageMaxWidth = maxWidth,
        compressImageMaxHeight = maxHeight,
        onlyScaleDown = true,
      } = options;

      // Get original image info
      const originalStats = await this.getImageStats(originalPath);
      const shouldOptimize = originalStats.size > 2 * 1024 * 1024; // 2MB threshold

      let optimizedPath = originalPath;
      let finalQuality = quality;

      if (shouldOptimize) {
        const resizeResult = await ImageResizer.createResizedImage(
          originalPath,
          compressImageMaxWidth,
          compressImageMaxHeight,
          format,
          finalQuality,
          0, // rotation
          undefined, // output path
          false, // keep meta
          {
            mode: 'contain',
            onlyScaleDown,
          }
        );

        optimizedPath = resizeResult.uri;
      }

      // Get final stats
      const finalStats = await this.getImageStats(optimizedPath);
      
      const photoId = this.generatePhotoId();
      const metadata: PhotoMetadata = {
        id: photoId,
        originalPath,
        optimizedPath,
        takenAt: new Date().toISOString(),
        originalSize: originalStats.size,
        optimizedSize: finalStats.size,
        optimized: true,
        aiProcessing: triggerAI, // Mark as AI processing if enabled
      };

      // Save metadata locally first
      await this.savePhotoMetadata(metadata);

      // Upload to backend
      try {
        const uploadResult = await BackendService.uploadPhoto(optimizedPath, {
          originalSize: metadata.originalSize,
          optimizedSize: metadata.optimizedSize,
          timestamp: metadata.takenAt,
        });

        // Update metadata with backend info
        if (uploadResult.data) {
          metadata.backendId = uploadResult.data.id;
          metadata.backendUrl = uploadResult.data.url;
          metadata.uploadedAt = uploadResult.data.uploadedAt;
        }

        // Trigger AI makeover if enabled and backend upload successful
        if (triggerAI && metadata.backendUrl) {
          this.triggerAIMakeover(metadata);
        }

        await this.savePhotoMetadata(metadata);
        Logger.info('Photo uploaded to backend successfully', { 
          photoId, 
          backendId: uploadResult.data?.id,
          triggerAI 
        });

      } catch (backendError) {
        Logger.warn('Backend upload failed, photo saved locally only', { 
          photoId, 
          error: backendError 
        });
        // Continue without backend - offline-first approach
      }

      return metadata;

    } catch (error) {
      Logger.error('Photo optimization failed', { originalPath, error });
      throw error;
    }
  }

  /**
   * Save photo metadata to local storage
   */
  static async savePhotoMetadata(metadata: PhotoMetadata): Promise<void> {
    try {
      photoStorage.set(`photo:${metadata.id}`, JSON.stringify(metadata));
      
      // Update photos list
      const photosList = this.getAllPhotoIds();
      if (!photosList.includes(metadata.id)) {
        photosList.unshift(metadata.id); // Add to beginning (most recent first)
        photoStorage.set('photos:list', JSON.stringify(photosList));
      }
    } catch (error) {
      console.error('Failed to save photo metadata:', error);
      throw new Error('Failed to save photo data');
    }
  }

  /**
   * Get all saved photos
   */
  static getAllPhotos(): PhotoMetadata[] {
    try {
      const photoIds = this.getAllPhotoIds();
      const photos: PhotoMetadata[] = [];
      
      for (const id of photoIds) {
        const photoData = photoStorage.getString(`photo:${id}`);
        if (photoData) {
          photos.push(JSON.parse(photoData));
        }
      }
      
      return photos.sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0)); // Most recent first
    } catch (error) {
      console.error('Failed to get photos:', error);
      return [];
    }
  }

  /**
   * Get photo by ID
   */
  static getPhoto(id: string): PhotoMetadata | null {
    try {
      const photoData = photoStorage.getString(`photo:${id}`);
      return photoData ? JSON.parse(photoData) : null;
    } catch (error) {
      console.error('Failed to get photo:', error);
      return null;
    }
  }

  /**
   * Delete photo
   */
  static async deletePhoto(id: string): Promise<boolean> {
    try {
      // Remove from metadata storage
      photoStorage.delete(`photo:${id}`);
      
      // Update photos list
      const photosList = this.getAllPhotoIds().filter(photoId => photoId !== id);
      photoStorage.set('photos:list', JSON.stringify(photosList));
      
      return true;
    } catch (error) {
      console.error('Failed to delete photo:', error);
      return false;
    }
  }

  /**
   * Clear all photos
   */
  static async clearAllPhotos(): Promise<void> {
    try {
      const photoIds = this.getAllPhotoIds();
      
      // Delete all photo metadata
      for (const id of photoIds) {
        photoStorage.delete(`photo:${id}`);
      }
      
      // Clear photos list
      photoStorage.delete('photos:list');
    } catch (error) {
      console.error('Failed to clear photos:', error);
      throw new Error('Failed to clear photo storage');
    }
  }

  /**
   * Get storage usage statistics
   */
  static getStorageStats(): {
    totalPhotos: number;
    totalSize: number;
    averageSize: number;
  } {
    try {
      const photos = this.getAllPhotos();
      const totalSize = photos.reduce((sum, photo) => sum + (photo.size || photo.optimizedSize), 0);
      
      return {
        totalPhotos: photos.length,
        totalSize,
        averageSize: photos.length > 0 ? totalSize / photos.length : 0,
      };
    } catch (error) {
      console.error('Failed to get storage stats:', error);
      return { totalPhotos: 0, totalSize: 0, averageSize: 0 };
    }
  }

  // Private helper methods
  private static getAllPhotoIds(): string[] {
    try {
      const photosListData = photoStorage.getString('photos:list');
      return photosListData ? JSON.parse(photosListData) : [];
    } catch (error) {
      console.error('Failed to get photo IDs:', error);
      return [];
    }
  }

  /**
   * Generate unique photo ID
   */
  private static generatePhotoId(): string {
    return `photo_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private static async getImageStats(imagePath: string): Promise<{
    width: number;
    height: number;
    size: number;
  }> {
    // This is a simplified version - in a real app you'd use a proper image info library
    // For now, we'll return default values and let ImageResizer handle the details
    return {
      width: 1920,
      height: 1080,
      size: 2 * 1024 * 1024, // 2MB default
    };
  }

  /**
   * Validate photo before processing
   */
  static validatePhoto(photoPath: string): {
    isValid: boolean;
    error?: string;
  } {
    if (!photoPath) {
      return { isValid: false, error: 'Photo path is required' };
    }

    // Check file extension
    const validExtensions = ['.jpg', '.jpeg', '.png', '.heic'];
    const extension = photoPath.toLowerCase().split('.').pop();
    
    if (!extension || !validExtensions.includes(`.${extension}`)) {
      return { 
        isValid: false, 
        error: 'Invalid file format. Please use JPG, PNG, or HEIC files.' 
      };
    }

    return { isValid: true };
  }

  /**
   * Trigger AI room makeover analysis
   */
  private static async triggerAIMakeover(metadata: PhotoMetadata): Promise<void> {
    if (!metadata.backendUrl || !metadata.id) {
      Logger.warn('Cannot trigger AI makeover - missing backend URL or ID');
      return;
    }

    try {
      Logger.info('Triggering AI room makeover', { photoId: metadata.id });

      // Start AI processing
      metadata.aiProcessing = true;
      await this.savePhotoMetadata(metadata);

      // Call AI service
      const makeoverResult = await BackendService.createRoomMakeover(
        metadata.id,
        metadata.backendUrl,
        'Modern', // Default style, could be user preference
        'medium'  // Default budget
      );

      // Update metadata with AI results
      metadata.makeoverId = makeoverResult.makeover_id;
      metadata.makeover = makeoverResult;
      metadata.aiProcessing = false;
      metadata.aiProcessedAt = new Date().toISOString();

      await this.savePhotoMetadata(metadata);

      Logger.info('AI makeover completed', { 
        photoId: metadata.id, 
        makeoverId: makeoverResult.makeover_id,
        productsCount: makeoverResult.transformation?.suggested_products?.length || 0
      });

    } catch (error) {
      Logger.error('AI makeover failed', { photoId: metadata.id, error });
      
      // Mark AI processing as failed
      metadata.aiProcessing = false;
      await this.savePhotoMetadata(metadata);
    }
  }

  /**
   * Get makeover data for a photo
   */
  static async getPhotoMakeover(photoId: string): Promise<any | null> {
    try {
      const metadata = await this.getPhoto(photoId);
      if (!metadata?.makeoverId) {
        return null;
      }

      // Get fresh makeover data from AI service
      return await BackendService.getRoomMakeover(metadata.makeoverId);
    } catch (error) {
      Logger.error('Failed to get photo makeover', { photoId, error });
      return null;
    }
  }

  /**
   * Refresh AI makeover for a photo
   */
  static async refreshMakeover(photoId: string, stylePreference?: string): Promise<void> {
    try {
      const metadata = await this.getPhoto(photoId);
      if (!metadata?.backendUrl) {
        throw new Error('Photo not uploaded to backend');
      }

      await this.triggerAIMakeover({
        ...metadata,
        aiProcessing: true
      });
    } catch (error) {
      Logger.error('Failed to refresh makeover', { photoId, error });
      throw error;
    }
  }
} 