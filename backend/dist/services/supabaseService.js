"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.supabaseService = exports.SupabaseService = void 0;
const supabase_js_1 = require("@supabase/supabase-js");
class SupabaseService {
    supabase;
    constructor() {
        const supabaseUrl = process.env.SUPABASE_URL || process.env.EXPO_PUBLIC_SUPABASE_URL;
        const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY;
        if (!supabaseUrl) {
            throw new Error('Missing SUPABASE_URL environment variable');
        }
        if (!supabaseKey) {
            throw new Error('Missing SUPABASE_SERVICE_ROLE_KEY environment variable');
        }
        this.supabase = (0, supabase_js_1.createClient)(supabaseUrl, supabaseKey, {
            auth: {
                autoRefreshToken: false,
                persistSession: false
            },
            db: {
                schema: 'public'
            }
        });
    }
    async createOrUpdateUser(clerkUserId, email, additionalData) {
        try {
            const { data, error } = await this.supabase
                .from('users')
                .upsert({
                clerk_user_id: clerkUserId,
                email,
                ...additionalData,
                updated_at: new Date().toISOString()
            }, {
                onConflict: 'clerk_user_id'
            })
                .select()
                .single();
            if (error)
                throw error;
            console.log(`✅ User upserted: ${clerkUserId}`);
            return data;
        }
        catch (error) {
            console.error('❌ Failed to create/update user:', error);
            throw new Error(`User operation failed: ${error.message}`);
        }
    }
    async getUser(clerkUserId) {
        try {
            const { data, error } = await this.supabase
                .from('users')
                .select('*')
                .eq('clerk_user_id', clerkUserId)
                .single();
            if (error && error.code !== 'PGRST116')
                throw error;
            return data;
        }
        catch (error) {
            console.error('❌ Failed to get user:', error);
            throw error;
        }
    }
    async createPhoto(photoData) {
        try {
            const { data, error } = await this.supabase
                .from('photos')
                .insert({
                ...photoData,
                status: 'uploaded',
                created_at: new Date().toISOString()
            })
                .select()
                .single();
            if (error)
                throw error;
            await this.incrementUserStats(photoData.clerk_user_id, { total_photos: 1 });
            console.log(`✅ Photo created: ${data.id}`);
            return data;
        }
        catch (error) {
            console.error('❌ Failed to create photo:', error);
            throw new Error(`Photo creation failed: ${error.message}`);
        }
    }
    async getUserPhotos(clerkUserId, limit = 50, offset = 0) {
        try {
            const { data, error } = await this.supabase
                .from('photos')
                .select(`
          *,
          makeovers (
            id,
            status,
            progress,
            makeover_url,
            style_preference,
            completed_at,
            error_message,
            runpod_job_id
          )
        `)
                .eq('clerk_user_id', clerkUserId)
                .order('created_at', { ascending: false })
                .range(offset, offset + limit - 1);
            if (error)
                throw error;
            console.log(`✅ Retrieved ${data.length} photos for user: ${clerkUserId}`);
            return data;
        }
        catch (error) {
            console.error('❌ Failed to get user photos:', error);
            throw error;
        }
    }
    async updatePhoto(photoId, updates) {
        try {
            const { data, error } = await this.supabase
                .from('photos')
                .update({
                ...updates,
                updated_at: new Date().toISOString()
            })
                .eq('id', photoId)
                .select()
                .single();
            if (error)
                throw error;
            console.log(`✅ Photo updated: ${photoId}`);
            return data;
        }
        catch (error) {
            console.error('❌ Failed to update photo:', error);
            throw error;
        }
    }
    async createMakeover(makeoverData) {
        try {
            const { data, error } = await this.supabase
                .from('makeovers')
                .insert({
                ...makeoverData,
                status: 'queued',
                progress: 0,
                created_at: new Date().toISOString()
            })
                .select()
                .single();
            if (error)
                throw error;
            await this.incrementUserStats(makeoverData.clerk_user_id, { total_makeovers: 1 });
            console.log(`✅ Makeover created: ${data.id}`);
            return data;
        }
        catch (error) {
            console.error('❌ Failed to create makeover:', error);
            throw new Error(`Makeover creation failed: ${error.message}`);
        }
    }
    async updateMakeover(makeoverId, updates) {
        try {
            const { data, error } = await this.supabase
                .from('makeovers')
                .update({
                ...updates,
                updated_at: new Date().toISOString()
            })
                .eq('id', makeoverId)
                .select()
                .single();
            if (error)
                throw error;
            console.log(`✅ Makeover updated: ${makeoverId} (${updates.status})`);
            return data;
        }
        catch (error) {
            console.error('❌ Failed to update makeover:', error);
            throw error;
        }
    }
    async getMakeover(makeoverId) {
        try {
            const { data, error } = await this.supabase
                .from('makeovers')
                .select(`
          *,
          photos (
            id,
            original_url,
            optimized_url,
            cloudflare_key
          ),
          product_suggestions (*)
        `)
                .eq('id', makeoverId)
                .single();
            if (error)
                throw error;
            return data;
        }
        catch (error) {
            console.error('❌ Failed to get makeover:', error);
            throw error;
        }
    }
    async createProductSuggestions(makeoverId, products) {
        try {
            const productsWithMakeoverId = products.map(product => ({
                ...product,
                makeover_id: makeoverId,
                created_at: new Date().toISOString()
            }));
            const { data, error } = await this.supabase
                .from('product_suggestions')
                .insert(productsWithMakeoverId)
                .select();
            if (error)
                throw error;
            console.log(`✅ Created ${data.length} product suggestions for makeover: ${makeoverId}`);
            return data;
        }
        catch (error) {
            console.error('❌ Failed to create product suggestions:', error);
            throw error;
        }
    }
    async incrementUserStats(clerkUserId, increments) {
        try {
            const { error } = await this.supabase.rpc('increment_user_stats', {
                user_clerk_id: clerkUserId,
                photo_increment: increments.total_photos || 0,
                makeover_increment: increments.total_makeovers || 0
            });
            if (error)
                throw error;
            console.log(`✅ User stats updated for: ${clerkUserId}`);
        }
        catch (error) {
            console.error('❌ Failed to update user stats:', error);
        }
    }
    async getUserStats(clerkUserId) {
        try {
            const { data, error } = await this.supabase
                .from('users')
                .select('total_photos, total_makeovers, subscription_tier, created_at')
                .eq('clerk_user_id', clerkUserId)
                .single();
            if (error)
                throw error;
            return data;
        }
        catch (error) {
            console.error('❌ Failed to get user stats:', error);
            throw error;
        }
    }
    async healthCheck() {
        try {
            const { error } = await this.supabase
                .from('users')
                .select('count', { count: 'exact', head: true });
            return !error;
        }
        catch (error) {
            console.error('❌ Supabase health check failed:', error);
            return false;
        }
    }
}
exports.SupabaseService = SupabaseService;
exports.supabaseService = new SupabaseService();
//# sourceMappingURL=supabaseService.js.map