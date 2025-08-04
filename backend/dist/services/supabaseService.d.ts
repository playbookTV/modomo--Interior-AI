import { SupabaseClient } from '@supabase/supabase-js';
export interface UserData {
    clerk_user_id: string;
    email?: string;
    subscription_tier?: 'free' | 'pro' | 'premium';
    preferences?: any;
}
export interface PhotoData {
    clerk_user_id: string;
    original_url: string;
    optimized_url?: string;
    cloudflare_key: string;
    original_name?: string;
    mime_type?: string;
    metadata?: any;
    original_size?: number;
    width?: number;
    height?: number;
    taken_at?: string;
}
export interface MakeoverData {
    photo_id: string;
    clerk_user_id: string;
    style_preference?: string;
    budget_range?: string;
    room_type?: string;
}
export declare class SupabaseService {
    supabase: SupabaseClient;
    constructor();
    createOrUpdateUser(clerkUserId: string, email?: string, additionalData?: Partial<UserData>): Promise<any>;
    getUser(clerkUserId: string): Promise<any>;
    createPhoto(photoData: PhotoData): Promise<any>;
    getUserPhotos(clerkUserId: string, limit?: number, offset?: number): Promise<any[]>;
    updatePhoto(photoId: string, updates: Partial<PhotoData>): Promise<any>;
    createMakeover(makeoverData: MakeoverData): Promise<any>;
    updateMakeover(makeoverId: string, updates: {
        runpod_job_id?: string;
        status?: 'queued' | 'processing' | 'completed' | 'failed';
        progress?: number;
        makeover_url?: string;
        error_message?: string;
        processing_started_at?: string;
        completed_at?: string;
        detected_objects?: any[];
        suggested_products?: any[];
    }): Promise<any>;
    getMakeover(makeoverId: string): Promise<any>;
    createProductSuggestions(makeoverId: string, products: Array<{
        product_name: string;
        category?: string;
        description?: string;
        brand?: string;
        amazon_price?: number;
        amazon_url?: string;
        ikea_price?: number;
        ikea_url?: string;
        wayfair_price?: number;
        wayfair_url?: string;
        image_url?: string;
        confidence_score?: number;
    }>): Promise<any[]>;
    incrementUserStats(clerkUserId: string, increments: {
        total_photos?: number;
        total_makeovers?: number;
    }): Promise<void>;
    getUserStats(clerkUserId: string): Promise<{
        total_photos: any;
        total_makeovers: any;
        subscription_tier: any;
        created_at: any;
    }>;
    healthCheck(): Promise<boolean>;
}
export declare const supabaseService: SupabaseService;
//# sourceMappingURL=supabaseService.d.ts.map