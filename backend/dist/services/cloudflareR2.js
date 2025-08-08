"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.cloudflareR2Service = exports.CloudflareR2Service = void 0;
const client_s3_1 = require("@aws-sdk/client-s3");
const s3_request_presigner_1 = require("@aws-sdk/s3-request-presigner");
const uuid_1 = require("uuid");
class CloudflareR2Service {
    client;
    bucketUrl;
    bucketName;
    publicUrl;
    constructor() {
        const requiredEnvVars = {
            CLOUDFLARE_R2_ENDPOINT: process.env.CLOUDFLARE_R2_ENDPOINT,
            CLOUDFLARE_R2_BUCKET: process.env.CLOUDFLARE_R2_BUCKET,
            CLOUDFLARE_R2_ACCESS_KEY_ID: process.env.CLOUDFLARE_R2_ACCESS_KEY_ID,
            CLOUDFLARE_R2_SECRET_ACCESS_KEY: process.env.CLOUDFLARE_R2_SECRET_ACCESS_KEY,
        };
        const missingVars = Object.entries(requiredEnvVars)
            .filter(([_, value]) => !value)
            .map(([key, _]) => key);
        if (missingVars.length > 0) {
            throw new Error(`Missing required Cloudflare R2 environment variables: ${missingVars.join(', ')}`);
        }
        this.bucketUrl = requiredEnvVars.CLOUDFLARE_R2_ENDPOINT;
        this.bucketName = requiredEnvVars.CLOUDFLARE_R2_BUCKET;
        this.publicUrl = process.env.CLOUDFLARE_R2_PUBLIC_URL || `${this.bucketUrl}/${this.bucketName}`;
        this.client = new client_s3_1.S3Client({
            region: 'auto',
            endpoint: this.bucketUrl,
            credentials: {
                accessKeyId: requiredEnvVars.CLOUDFLARE_R2_ACCESS_KEY_ID,
                secretAccessKey: requiredEnvVars.CLOUDFLARE_R2_SECRET_ACCESS_KEY,
            },
            forcePathStyle: true,
        });
    }
    async uploadPhoto(buffer, originalName, userId) {
        try {
            const fileExt = this.getFileExtension(originalName);
            const timestamp = Date.now();
            const key = `photos/${userId}/${timestamp}-${(0, uuid_1.v4)()}${fileExt}`;
            await this.client.send(new client_s3_1.PutObjectCommand({
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
                CacheControl: 'public, max-age=31536000',
            }));
            const baseUrl = `${this.publicUrl}/${key}`;
            const variants = this.generateImageVariants(baseUrl);
            console.log(`✅ Photo uploaded to Cloudflare R2: ${key}`);
            return {
                key,
                url: baseUrl,
                variants,
                size: buffer.length
            };
        }
        catch (error) {
            console.error('❌ Cloudflare R2 upload failed:', error);
            throw new Error(`Failed to upload to Cloudflare R2: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    }
    async deletePhoto(key) {
        try {
            await this.client.send(new client_s3_1.DeleteObjectCommand({
                Bucket: this.bucketName,
                Key: key,
            }));
            console.log(`✅ Photo deleted from Cloudflare R2: ${key}`);
            return true;
        }
        catch (error) {
            console.error('❌ Cloudflare R2 delete failed:', error);
            return false;
        }
    }
    async getSignedUploadUrl(userId, originalName) {
        try {
            const fileExt = this.getFileExtension(originalName);
            const key = `photos/${userId}/${Date.now()}-${(0, uuid_1.v4)()}${fileExt}`;
            const command = new client_s3_1.PutObjectCommand({
                Bucket: this.bucketName,
                Key: key,
                ContentType: 'image/jpeg',
                Metadata: {
                    'uploaded-by': userId,
                    'original-name': originalName,
                }
            });
            const uploadUrl = await (0, s3_request_presigner_1.getSignedUrl)(this.client, command, {
                expiresIn: 3600
            });
            return {
                uploadUrl,
                key,
                publicUrl: `${this.publicUrl}/${key}`
            };
        }
        catch (error) {
            console.error('❌ Failed to generate signed upload URL:', error);
            throw new Error(`Failed to generate upload URL: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    }
    generateImageVariants(baseUrl) {
        return [
            `${baseUrl}/thumbnail`,
            `${baseUrl}/small`,
            `${baseUrl}/medium`,
            `${baseUrl}/large`,
            baseUrl
        ];
    }
    getFileExtension(filename) {
        const ext = filename.split('.').pop();
        return ext ? `.${ext.toLowerCase()}` : '.jpg';
    }
    async healthCheck() {
        try {
            await this.client.send(new client_s3_1.PutObjectCommand({
                Bucket: this.bucketName,
                Key: 'health-check.txt',
                Body: Buffer.from('ReRoom health check'),
                ContentType: 'text/plain'
            }));
            await this.client.send(new client_s3_1.DeleteObjectCommand({
                Bucket: this.bucketName,
                Key: 'health-check.txt'
            }));
            console.log('✅ Cloudflare R2 health check passed');
            return true;
        }
        catch (error) {
            console.error('❌ Cloudflare R2 health check failed:', error);
            return false;
        }
    }
    async getFileMetadata(key) {
        try {
            const command = new client_s3_1.GetObjectCommand({
                Bucket: this.bucketName,
                Key: key
            });
            const response = await this.client.send(command);
            return {
                size: response.ContentLength,
                lastModified: response.LastModified,
                contentType: response.ContentType,
                metadata: response.Metadata
            };
        }
        catch (error) {
            console.error('❌ Failed to get file metadata:', error);
            throw new Error(`File not found: ${key}`);
        }
    }
}
exports.CloudflareR2Service = CloudflareR2Service;
exports.cloudflareR2Service = new CloudflareR2Service();
//# sourceMappingURL=cloudflareR2.js.map