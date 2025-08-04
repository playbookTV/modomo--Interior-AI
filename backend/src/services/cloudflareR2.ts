import { S3Client, PutObjectCommand, GetObjectCommand, DeleteObjectCommand } from '@aws-sdk/client-s3'
import { getSignedUrl } from '@aws-sdk/s3-request-presigner'
import { v4 as uuidv4 } from 'uuid'

export interface UploadResult {
  key: string
  url: string
  variants: string[]
  size: number
}

export class CloudflareR2Service {
  private client: S3Client
  private bucketUrl = process.env.CLOUDFLARE_R2_ENDPOINT!
  private bucketName = process.env.CLOUDFLARE_R2_BUCKET!
  private publicUrl = process.env.CLOUDFLARE_R2_PUBLIC_URL!

  constructor() {
    this.client = new S3Client({
      region: 'auto',
      endpoint: this.bucketUrl,
      credentials: {
        accessKeyId: process.env.CLOUDFLARE_R2_ACCESS_KEY_ID!,
        secretAccessKey: process.env.CLOUDFLARE_R2_SECRET_ACCESS_KEY!,
      },
      forcePathStyle: true, // Required for R2 compatibility
    })
  }

  /**
   * Upload photo to Cloudflare R2 with automatic optimization
   */
  async uploadPhoto(buffer: Buffer, originalName: string, userId: string): Promise<UploadResult> {
    try {
      // Generate unique key with user prefix for organization
      const fileExt = this.getFileExtension(originalName)
      const timestamp = Date.now()
      const key = `photos/${userId}/${timestamp}-${uuidv4()}${fileExt}`

      // Upload original to Cloudflare R2
      await this.client.send(new PutObjectCommand({
        Bucket: this.bucketName,
        Key: key,
        Body: buffer,
        ContentType: 'image/jpeg',
        Metadata: {
          'original-name': originalName,
          'uploaded-by': userId,
          'upload-date': new Date().toISOString(),
          'service': 'reroom-backend'
        },
        // Cache control for better performance
        CacheControl: 'public, max-age=31536000', // 1 year
      }))

      // Generate CDN URLs for different sizes using Cloudflare Image Resizing
      const baseUrl = `${this.publicUrl}/${key}`
      const variants = this.generateImageVariants(baseUrl)

      console.log(`✅ Photo uploaded to Cloudflare R2: ${key}`)

      return {
        key,
        url: baseUrl,
        variants,
        size: buffer.length
      }
    } catch (error) {
      console.error('❌ Cloudflare R2 upload failed:', error)
      throw new Error(`Failed to upload to Cloudflare R2: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  /**
   * Delete photo from Cloudflare R2
   */
  async deletePhoto(key: string): Promise<boolean> {
    try {
      await this.client.send(new DeleteObjectCommand({
        Bucket: this.bucketName,
        Key: key,
      }))

      console.log(`✅ Photo deleted from Cloudflare R2: ${key}`)
      return true
    } catch (error) {
      console.error('❌ Cloudflare R2 delete failed:', error)
      return false
    }
  }

  /**
   * Get signed upload URL for direct mobile uploads
   */
  async getSignedUploadUrl(userId: string, originalName: string): Promise<{
    uploadUrl: string
    key: string
    publicUrl: string
  }> {
    try {
      const fileExt = this.getFileExtension(originalName)
      const key = `photos/${userId}/${Date.now()}-${uuidv4()}${fileExt}`

      const command = new PutObjectCommand({
        Bucket: this.bucketName,
        Key: key,
        ContentType: 'image/jpeg',
        Metadata: {
          'uploaded-by': userId,
          'original-name': originalName,
        }
      })

      const uploadUrl = await getSignedUrl(this.client, command, { 
        expiresIn: 3600 // 1 hour
      })

      return {
        uploadUrl,
        key,
        publicUrl: `${this.publicUrl}/${key}`
      }
    } catch (error) {
      console.error('❌ Failed to generate signed upload URL:', error)
      throw new Error(`Failed to generate upload URL: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  /**
   * Generate image variants using Cloudflare Image Resizing
   */
  private generateImageVariants(baseUrl: string): string[] {
    return [
      `${baseUrl}/thumbnail`,  // Cloudflare will auto-resize to thumbnail
      `${baseUrl}/small`,      // Small version  
      `${baseUrl}/medium`,     // Medium version
      `${baseUrl}/large`,      // Large version
      baseUrl                  // Original
    ]
  }

  /**
   * Extract file extension from filename
   */
  private getFileExtension(filename: string): string {
    const ext = filename.split('.').pop()
    return ext ? `.${ext.toLowerCase()}` : '.jpg'
  }

  /**
   * Health check for Cloudflare R2 connection
   */
  async healthCheck(): Promise<boolean> {
    try {
      // Try to list objects to verify connection
      await this.client.send(new PutObjectCommand({
        Bucket: this.bucketName,
        Key: 'health-check.txt',
        Body: Buffer.from('ReRoom health check'),
        ContentType: 'text/plain'
      }))

      // Clean up health check file
      await this.client.send(new DeleteObjectCommand({
        Bucket: this.bucketName,
        Key: 'health-check.txt'
      }))

      console.log('✅ Cloudflare R2 health check passed')
      return true
    } catch (error) {
      console.error('❌ Cloudflare R2 health check failed:', error)
      return false
    }
  }

  /**
   * Get file metadata from R2
   */
  async getFileMetadata(key: string): Promise<any> {
    try {
      const command = new GetObjectCommand({
        Bucket: this.bucketName,
        Key: key
      })

      // Get metadata without downloading the file
      const response = await this.client.send(command)
      
      return {
        size: response.ContentLength,
        lastModified: response.LastModified,
        contentType: response.ContentType,
        metadata: response.Metadata
      }
    } catch (error) {
      console.error('❌ Failed to get file metadata:', error)
      throw new Error(`File not found: ${key}`)
    }
  }
}

// Singleton instance for use across the application
export const cloudflareR2Service = new CloudflareR2Service()