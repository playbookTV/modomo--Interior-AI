export interface UploadResult {
    key: string;
    url: string;
    variants: string[];
    size: number;
}
export declare class CloudflareR2Service {
    private client;
    private bucketUrl;
    private bucketName;
    private publicUrl;
    constructor();
    uploadPhoto(buffer: Buffer, originalName: string, userId: string): Promise<UploadResult>;
    deletePhoto(key: string): Promise<boolean>;
    getSignedUploadUrl(userId: string, originalName: string): Promise<{
        uploadUrl: string;
        key: string;
        publicUrl: string;
    }>;
    private generateImageVariants;
    private getFileExtension;
    healthCheck(): Promise<boolean>;
    getFileMetadata(key: string): Promise<any>;
}
export declare const cloudflareR2Service: CloudflareR2Service;
//# sourceMappingURL=cloudflareR2.d.ts.map