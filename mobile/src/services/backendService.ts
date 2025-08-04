import { Logger } from '../utils/logger';

const API_BASE_URL = process.env.EXPO_PUBLIC_API_URL || 'http://localhost:3002';

export interface UploadResponse {
  success: boolean;
  data?: {
    id: string;
    url: string;
    size: number;
    originalName: string;
    mimeType: string;
    uploadedAt: string;
    metadata?: any;
  };
  error?: string;
  code?: string;
  message?: string;
}

export interface PhotoMetadata {
  width?: number;
  height?: number;
  quality?: number;
  optimized?: boolean;
  originalSize?: number;
  optimizedSize?: number;
  timestamp?: string;
}

export class BackendService {
  private static readonly TIMEOUT = 30000; // 30 seconds

  /**
   * Upload a photo to the backend photo service
   */
  static async uploadPhoto(
    photoUri: string,
    metadata?: PhotoMetadata,
    userId?: string
  ): Promise<UploadResponse> {
    try {
      // Create FormData for multipart upload
      const formData = new FormData();
      
      // Add the photo file
      formData.append('photo', {
        uri: photoUri,
        type: 'image/jpeg',
        name: `photo_${Date.now()}.jpg`,
      } as any);

      // Add metadata if provided
      if (metadata) {
        formData.append('metadata', JSON.stringify(metadata));
      }

      // Add userId if provided
      if (userId) {
        formData.append('userId', userId);
      }

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.TIMEOUT);

      const response = await fetch(`${API_BASE_URL}/api/photos/upload`, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      const result: UploadResponse = await response.json();

      if (!response.ok) {
        console.error('Upload failed:', result);
        return {
          success: false,
          error: result.error || 'Upload failed',
          code: result.code || 'UPLOAD_ERROR',
          message: result.message || 'Failed to upload photo',
        };
      }

      console.log('Photo uploaded successfully:', result.data?.id);
      return result;
    } catch (error) {
      console.error('Upload error:', error);
      return {
        success: false,
        error: 'Network error',
        code: 'NETWORK_ERROR',
        message: error instanceof Error ? error.message : 'Network request failed',
      };
    }
  }

  /**
   * Get photo information by ID
   */
  static async getPhoto(photoId: string): Promise<UploadResponse> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.TIMEOUT);

      const response = await fetch(`${API_BASE_URL}/api/photos/${photoId}`, {
        method: 'GET',
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      const result: UploadResponse = await response.json();

      if (!response.ok) {
        return {
          success: false,
          error: result.error || 'Failed to get photo',
          code: result.code || 'GET_ERROR',
        };
      }

      return result;
    } catch (error) {
      console.error('Get photo error:', error);
      return {
        success: false,
        error: 'Network error',
        code: 'NETWORK_ERROR',
        message: error instanceof Error ? error.message : 'Network request failed',
      };
    }
  }

  /**
   * Delete a photo by ID
   */
  static async deletePhoto(photoId: string): Promise<UploadResponse> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.TIMEOUT);

      const response = await fetch(`${API_BASE_URL}/api/photos/${photoId}`, {
        method: 'DELETE',
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      const result: UploadResponse = await response.json();

      if (!response.ok) {
        return {
          success: false,
          error: result.error || 'Failed to delete photo',
          code: result.code || 'DELETE_ERROR',
        };
      }

      return result;
    } catch (error) {
      console.error('Delete photo error:', error);
      return {
        success: false,
        error: 'Network error',
        code: 'NETWORK_ERROR',
        message: error instanceof Error ? error.message : 'Network request failed',
      };
    }
  }

  /**
   * List all photos
   */
  static async listPhotos(): Promise<UploadResponse> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.TIMEOUT);

      const response = await fetch(`${API_BASE_URL}/api/photos`, {
        method: 'GET',
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      const result: UploadResponse = await response.json();

      if (!response.ok) {
        return {
          success: false,
          error: result.error || 'Failed to list photos',
          code: result.code || 'LIST_ERROR',
        };
      }

      return result;
    } catch (error) {
      console.error('List photos error:', error);
      return {
        success: false,
        error: 'Network error',
        code: 'NETWORK_ERROR',
        message: error instanceof Error ? error.message : 'Network request failed',
      };
    }
  }

  /**
   * Trigger AI room makeover analysis
   */
  static async createRoomMakeover(
    photoId: string, 
    photoUrl: string, 
    stylePreference: string = 'Modern',
    budgetRange?: string
  ): Promise<any> {
    try {
      const response = await fetch(`${AI_SERVICE_URL}/makeover`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          photo_id: photoId,
          photo_url: photoUrl,
          style_preference: stylePreference,
          budget_range: budgetRange,
        }),
      });

      if (!response.ok) {
        throw new Error(`AI makeover failed: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      Logger.error('Room makeover failed:', error);
      throw error;
    }
  }

  /**
   * Get room makeover results by ID
   */
  static async getRoomMakeover(makeoverId: string): Promise<any> {
    try {
      const response = await fetch(`${AI_SERVICE_URL}/makeover/${makeoverId}`);
      
      if (!response.ok) {
        throw new Error(`Failed to get makeover: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      Logger.error('Failed to get makeover:', error);
      throw error;
    }
  }

  /**
   * Get product prices from multiple retailers
   */
  static async getProductPrices(productId: string): Promise<any> {
    try {
      const response = await fetch(`${AI_SERVICE_URL}/products/${productId}/prices`);
      
      if (!response.ok) {
        throw new Error(`Failed to get product prices: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      Logger.error('Failed to get product prices:', error);
      throw error;
    }
  }

  /**
   * Check AI service health
   */
  static async checkAIHealth(): Promise<any> {
    try {
      const response = await fetch(`${AI_SERVICE_URL}/health`);
      return await response.json();
    } catch (error) {
      Logger.error('AI health check failed:', error);
      return { status: 'unhealthy', error: error.message };
    }
  }

  /**
   * Check backend service health
   */
  static async healthCheck(): Promise<boolean> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);

      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        const health = await response.json();
        return health.status === 'healthy';
      }

      return false;
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }
}

export default BackendService; 

// AI Service URL
const AI_SERVICE_URL = __DEV__ 
  ? 'http://localhost:8000'  // Development
  : 'https://api.reroom.app/ai';  // Production 