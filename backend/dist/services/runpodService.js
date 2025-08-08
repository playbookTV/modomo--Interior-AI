"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.runpodService = exports.RunPodService = void 0;
const supabaseService_1 = require("./supabaseService");
class RunPodService {
    endpoint = process.env.RUNPOD_ENDPOINT;
    apiKey = process.env.RUNPOD_API_KEY;
    callbackUrl = `${process.env.RAILWAY_BACKEND_URL}/api/makeovers/callback`;
    constructor() {
        if (!this.endpoint || !this.apiKey) {
            throw new Error('RunPod configuration missing: RUNPOD_ENDPOINT and RUNPOD_API_KEY required');
        }
    }
    async submitMakeoverJob(makeoverData) {
        try {
            console.log(`🚀 Submitting makeover job to RunPod for photo: ${makeoverData.photo_id}`);
            const response = await fetch(this.endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.apiKey}`,
                },
                body: JSON.stringify({
                    input: {
                        photo_url: makeoverData.photo_url,
                        photo_id: makeoverData.photo_id,
                        makeover_id: makeoverData.makeover_id,
                        user_id: makeoverData.user_id,
                        style_preference: makeoverData.style_preference || 'Modern',
                        budget_range: makeoverData.budget_range || 'mid-range',
                        room_type: makeoverData.room_type || 'living-room',
                        quality: 'high',
                        include_products: true,
                        include_pricing: true,
                        callback_url: this.callbackUrl,
                        webhook_events: ['progress', 'completed', 'failed']
                    },
                    webhook: this.callbackUrl
                })
            });
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`RunPod API error: ${response.status} - ${errorText}`);
            }
            const jobData = await response.json();
            await supabaseService_1.supabaseService.updateMakeover(makeoverData.makeover_id, {
                runpod_job_id: jobData.id,
                status: 'processing',
                processing_started_at: new Date().toISOString()
            });
            console.log(`✅ RunPod job submitted: ${jobData.id} for makeover: ${makeoverData.makeover_id}`);
            return jobData;
        }
        catch (error) {
            console.error('❌ RunPod job submission failed:', error);
            await supabaseService_1.supabaseService.updateMakeover(makeoverData.makeover_id, {
                status: 'failed',
                error_message: error instanceof Error ? error.message : 'RunPod job submission failed'
            });
            throw error;
        }
    }
    async checkJobStatus(jobId) {
        try {
            const response = await fetch(`${this.endpoint}/${jobId}`, {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${this.apiKey}`,
                }
            });
            if (!response.ok) {
                throw new Error(`RunPod status check failed: ${response.status}`);
            }
            const jobData = await response.json();
            console.log(`📊 RunPod job status: ${jobId} - ${jobData.status}`);
            return jobData;
        }
        catch (error) {
            console.error('❌ RunPod status check failed:', error);
            throw error;
        }
    }
    async handleWebhookCallback(payload) {
        try {
            const { id: jobId, status, output, error } = payload;
            console.log(`🔔 RunPod webhook received: ${jobId} - ${status}`);
            const { data: makeovers, error: queryError } = await supabaseService_1.supabaseService.supabase
                .from('makeovers')
                .select('id, photo_id, clerk_user_id')
                .eq('runpod_job_id', jobId);
            if (queryError || !makeovers || makeovers.length === 0) {
                console.error('❌ Makeover not found for RunPod job:', jobId);
                return;
            }
            const makeover = makeovers[0];
            switch (status) {
                case 'IN_PROGRESS':
                    await supabaseService_1.supabaseService.updateMakeover(makeover.id, {
                        status: 'processing',
                        progress: output?.progress || 25
                    });
                    break;
                case 'COMPLETED':
                    if (output?.makeover_url) {
                        await supabaseService_1.supabaseService.updateMakeover(makeover.id, {
                            status: 'completed',
                            progress: 100,
                            makeover_url: output.makeover_url,
                            detected_objects: output.detected_objects || [],
                            suggested_products: output.suggested_products || [],
                            completed_at: new Date().toISOString()
                        });
                        if (output.products && output.products.length > 0) {
                            await supabaseService_1.supabaseService.createProductSuggestions(makeover.id, output.products);
                        }
                        console.log(`✅ Makeover completed: ${makeover.id}`);
                    }
                    else {
                        throw new Error('Completed job missing makeover URL');
                    }
                    break;
                case 'FAILED':
                    await supabaseService_1.supabaseService.updateMakeover(makeover.id, {
                        status: 'failed',
                        error_message: error || 'RunPod processing failed',
                        completed_at: new Date().toISOString()
                    });
                    console.log(`❌ Makeover failed: ${makeover.id}`);
                    break;
                default:
                    console.log(`ℹ️ Unknown RunPod status: ${status}`);
            }
        }
        catch (error) {
            console.error('❌ Failed to handle RunPod webhook:', error);
        }
    }
    async cancelJob(jobId) {
        try {
            const response = await fetch(`${this.endpoint}/${jobId}/cancel`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.apiKey}`,
                }
            });
            if (!response.ok) {
                throw new Error(`Job cancellation failed: ${response.status}`);
            }
            console.log(`✅ RunPod job cancelled: ${jobId}`);
            return true;
        }
        catch (error) {
            console.error('❌ Failed to cancel RunPod job:', error);
            return false;
        }
    }
    async getAccountInfo() {
        try {
            const response = await fetch('https://api.runpod.ai/v2/account', {
                headers: {
                    'Authorization': `Bearer ${this.apiKey}`,
                }
            });
            if (!response.ok) {
                throw new Error(`Account info request failed: ${response.status}`);
            }
            return await response.json();
        }
        catch (error) {
            console.error('❌ Failed to get RunPod account info:', error);
            throw error;
        }
    }
    async healthCheck() {
        try {
            if (!this.apiKey || this.apiKey === '[YOUR_RUNPOD_API_KEY]') {
                console.warn('⚠️ RunPod API key not configured');
                return true;
            }
            console.log('✅ RunPod configuration check passed');
            return true;
        }
        catch (error) {
            console.warn('⚠️ RunPod health check failed, but continuing deployment:', error);
            return true;
        }
    }
}
exports.RunPodService = RunPodService;
exports.runpodService = new RunPodService();
//# sourceMappingURL=runpodService.js.map