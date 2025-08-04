import { Client as MinioClient } from 'minio';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../utils/logger';

export interface UploadResult {
  key: string;
  url: string;
  bucket: string;
  size: number;
  etag: string;
}

export interface PhotoMetadata {
  originalName: string;
  mimeType: string;
  size: number;
  uploadedAt: Date;
  userId?: string;
}

export class StorageService {
  private minioClient: MinioClient;
  private bucketName: string;

  constructor() {
    // Validate required environment variables
    const requiredEnvVars = {
      AWS_ACCESS_KEY_ID: process.env.AWS_ACCESS_KEY_ID,
      AWS_SECRET_ACCESS_KEY: process.env.AWS_SECRET_ACCESS_KEY,
      S3_ENDPOINT: process.env.S3_ENDPOINT,
    };

    for (const [key, value] of Object.entries(requiredEnvVars)) {
      if (!value) {
        throw new Error(`Missing required environment variable: ${key}`);
      }
    }

    this.bucketName = process.env.S3_BUCKET || 'reroom-photos';
    
    this.minioClient = new MinioClient({
      endPoint: process.env.S3_ENDPOINT!.replace('http://', '').replace('https://', ''),
      port: parseInt(process.env.S3_PORT || '9000'),
      useSSL: process.env.S3_USE_SSL === 'true',
      accessKey: process.env.AWS_ACCESS_KEY_ID!,
      secretKey: process.env.AWS_SECRET_ACCESS_KEY!,
    });

    this.initializeBucket();
  }

  private async initializeBucket(): Promise<void> {
    try {
      const exists = await this.minioClient.bucketExists(this.bucketName);
      if (!exists) {
        await this.minioClient.makeBucket(this.bucketName, 'us-east-1');
        logger.info(`Created bucket: ${this.bucketName}`);
      }
    } catch (error) {
      logger.error('Failed to initialize bucket:', error);
      throw new Error('Storage initialization failed');
    }
  }

  async uploadPhoto(
    buffer: Buffer,
    metadata: PhotoMetadata
  ): Promise<UploadResult> {
    try {
      const fileExtension = this.getFileExtension(metadata.originalName);
      const key = `photos/${uuidv4()}${fileExtension}`;

      // Set object metadata
      const objectMetadata = {
        'Content-Type': metadata.mimeType,
        'X-Upload-Date': metadata.uploadedAt.toISOString(),
        'X-Original-Name': metadata.originalName,
        'X-User-Id': metadata.userId || 'anonymous',
      };

      // Upload to MinIO
      const uploadInfo = await this.minioClient.putObject(
        this.bucketName,
        key,
        buffer,
        buffer.length,
        objectMetadata
      );

      // Generate public URL
      const url = await this.getPhotoUrl(key);

      const result: UploadResult = {
        key,
        url,
        bucket: this.bucketName,
        size: buffer.length,
        etag: uploadInfo.etag,
      };

      logger.info('Photo uploaded successfully', {
        key,
        size: buffer.length,
        originalName: metadata.originalName,
      });

      return result;
    } catch (error) {
      logger.error('Photo upload failed:', error);
      throw new Error('Failed to upload photo');
    }
  }

  async getPhotoUrl(key: string): Promise<string> {
    try {
      // Generate a presigned URL valid for 24 hours
      const url = await this.minioClient.presignedGetObject(
        this.bucketName,
        key,
        24 * 60 * 60 // 24 hours
      );
      return url;
    } catch (error) {
      logger.error('Failed to generate photo URL:', error);
      throw new Error('Failed to generate photo URL');
    }
  }

  async deletePhoto(key: string): Promise<boolean> {
    try {
      await this.minioClient.removeObject(this.bucketName, key);
      logger.info('Photo deleted successfully', { key });
      return true;
    } catch (error) {
      logger.error('Failed to delete photo:', error);
      return false;
    }
  }

  async getPhotoMetadata(key: string): Promise<any> {
    try {
      const stat = await this.minioClient.statObject(this.bucketName, key);
      return {
        size: stat.size,
        lastModified: stat.lastModified,
        etag: stat.etag,
        metadata: stat.metaData,
      };
    } catch (error) {
      logger.error('Failed to get photo metadata:', error);
      throw new Error('Photo not found');
    }
  }

  async listPhotos(prefix?: string): Promise<string[]> {
    try {
      const photos: string[] = [];
      const stream = this.minioClient.listObjects(
        this.bucketName,
        prefix || 'photos/',
        true
      );

      return new Promise((resolve, reject) => {
        stream.on('data', (obj) => {
          if (obj.name) {
            photos.push(obj.name);
          }
        });

        stream.on('end', () => {
          resolve(photos);
        });

        stream.on('error', (error) => {
          logger.error('Failed to list photos:', error);
          reject(new Error('Failed to list photos'));
        });
      });
    } catch (error) {
      logger.error('Failed to list photos:', error);
      throw new Error('Failed to list photos');
    }
  }

  private getFileExtension(filename: string): string {
    const ext = filename.split('.').pop();
    return ext ? `.${ext.toLowerCase()}` : '.jpg';
  }

  async healthCheck(): Promise<boolean> {
    try {
      await this.minioClient.bucketExists(this.bucketName);
      return true;
    } catch (error) {
      logger.error('Storage health check failed:', error);
      return false;
    }
  }
}

// Singleton instance
export const storageService = new StorageService(); 