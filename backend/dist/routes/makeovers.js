"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const supabaseService_1 = require("../services/supabaseService");
const runpodService_1 = require("../services/runpodService");
const clerk_sdk_node_1 = require("@clerk/clerk-sdk-node");
const router = express_1.default.Router();
const supabaseService = new supabaseService_1.SupabaseService();
const runpodService = new runpodService_1.RunPodService();
router.get('/:makeoverId', (0, clerk_sdk_node_1.ClerkExpressRequireAuth)(), async (req, res) => {
    try {
        const { userId } = req.auth;
        const { makeoverId } = req.params;
        const makeover = await supabaseService.getMakeover(makeoverId);
        if (makeover.clerk_user_id !== userId) {
            return res.status(403).json({
                success: false,
                error: 'Access denied',
                code: 'FORBIDDEN'
            });
        }
        res.json({
            success: true,
            data: makeover
        });
    }
    catch (error) {
        console.error('❌ Failed to get makeover:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to get makeover',
            message: error.message,
            code: 'GET_ERROR'
        });
    }
});
router.post('/:makeoverId/retry', (0, clerk_sdk_node_1.ClerkExpressRequireAuth)(), async (req, res) => {
    try {
        const { userId } = req.auth;
        const { makeoverId } = req.params;
        const makeover = await supabaseService.getMakeover(makeoverId);
        if (makeover.clerk_user_id !== userId) {
            return res.status(403).json({
                success: false,
                error: 'Access denied',
                code: 'FORBIDDEN'
            });
        }
        if (makeover.status !== 'failed') {
            return res.status(400).json({
                success: false,
                error: 'Makeover is not in a failed state',
                code: 'INVALID_STATE'
            });
        }
        await supabaseService.updateMakeover(makeoverId, {
            status: 'queued',
            progress: 0,
            error_message: undefined,
            processing_started_at: undefined,
            completed_at: undefined
        });
        const jobResponse = await runpodService.submitMakeoverJob({
            photo_url: makeover.photos.optimized_url,
            photo_id: makeover.photo_id,
            makeover_id: makeoverId,
            user_id: userId,
            style_preference: makeover.style_preference,
            budget_range: makeover.budget_range,
            room_type: makeover.room_type
        });
        res.json({
            success: true,
            data: {
                makeover_id: makeoverId,
                runpod_job_id: jobResponse.id,
                status: 'queued'
            },
            message: 'Makeover retry initiated'
        });
    }
    catch (error) {
        console.error('❌ Failed to retry makeover:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to retry makeover',
            message: error.message,
            code: 'RETRY_ERROR'
        });
    }
});
router.post('/:makeoverId/cancel', (0, clerk_sdk_node_1.ClerkExpressRequireAuth)(), async (req, res) => {
    try {
        const { userId } = req.auth;
        const { makeoverId } = req.params;
        const { data: makeovers, error } = await supabaseService.supabase
            .from('makeovers')
            .select('status, runpod_job_id, clerk_user_id')
            .eq('id', makeoverId);
        if (error)
            throw error;
        if (!makeovers || makeovers.length === 0) {
            return res.status(404).json({
                success: false,
                error: 'Makeover not found',
                code: 'NOT_FOUND'
            });
        }
        const makeover = makeovers[0];
        if (makeover.clerk_user_id !== userId) {
            return res.status(403).json({
                success: false,
                error: 'Access denied',
                code: 'FORBIDDEN'
            });
        }
        if (!['queued', 'processing'].includes(makeover.status)) {
            return res.status(400).json({
                success: false,
                error: 'Makeover cannot be cancelled in current state',
                code: 'INVALID_STATE'
            });
        }
        if (makeover.runpod_job_id) {
            await runpodService.cancelJob(makeover.runpod_job_id);
        }
        await supabaseService.updateMakeover(makeoverId, {
            status: 'failed',
            error_message: 'Cancelled by user',
            completed_at: new Date().toISOString()
        });
        res.json({
            success: true,
            message: 'Makeover cancelled successfully'
        });
    }
    catch (error) {
        console.error('❌ Failed to cancel makeover:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to cancel makeover',
            message: error.message,
            code: 'CANCEL_ERROR'
        });
    }
});
router.get('/', (0, clerk_sdk_node_1.ClerkExpressRequireAuth)(), async (req, res) => {
    try {
        const { userId } = req.auth;
        const limit = parseInt(req.query.limit) || 20;
        const offset = parseInt(req.query.offset) || 0;
        const status = req.query.status;
        let query = supabaseService.supabase
            .from('makeovers')
            .select(`
          *,
          photos (
            id,
            original_url,
            optimized_url,
            cloudflare_key,
            original_name
          ),
          product_suggestions (*)
        `)
            .eq('clerk_user_id', userId)
            .order('created_at', { ascending: false })
            .range(offset, offset + limit - 1);
        if (status && ['queued', 'processing', 'completed', 'failed'].includes(status)) {
            query = query.eq('status', status);
        }
        const { data: makeovers, error } = await query;
        if (error)
            throw error;
        res.json({
            success: true,
            data: makeovers || [],
            count: makeovers?.length || 0,
            pagination: {
                limit,
                offset,
                hasMore: (makeovers?.length || 0) === limit
            }
        });
    }
    catch (error) {
        console.error('❌ Failed to get makeover history:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to get makeover history',
            message: error.message,
            code: 'HISTORY_ERROR'
        });
    }
});
router.post('/callback', async (req, res) => {
    try {
        console.log('🔔 RunPod webhook received:', JSON.stringify(req.body, null, 2));
        const payload = req.body;
        if (!payload.id) {
            return res.status(400).json({
                success: false,
                error: 'Invalid webhook payload: missing job ID'
            });
        }
        runpodService.handleWebhookCallback(payload)
            .catch(error => {
            console.error('❌ Webhook handling failed:', error);
        });
        res.json({
            success: true,
            message: 'Webhook received and processing'
        });
    }
    catch (error) {
        console.error('❌ Webhook callback failed:', error);
        res.status(500).json({
            success: false,
            error: 'Webhook processing failed',
            message: error.message
        });
    }
});
router.post('/:makeoverId/check-status', (0, clerk_sdk_node_1.ClerkExpressRequireAuth)(), async (req, res) => {
    try {
        const { userId } = req.auth;
        const { makeoverId } = req.params;
        const { data: makeovers, error } = await supabaseService.supabase
            .from('makeovers')
            .select('runpod_job_id, clerk_user_id, status')
            .eq('id', makeoverId);
        if (error)
            throw error;
        if (!makeovers || makeovers.length === 0) {
            return res.status(404).json({
                success: false,
                error: 'Makeover not found',
                code: 'NOT_FOUND'
            });
        }
        const makeover = makeovers[0];
        if (makeover.clerk_user_id !== userId) {
            return res.status(403).json({
                success: false,
                error: 'Access denied',
                code: 'FORBIDDEN'
            });
        }
        if (!makeover.runpod_job_id) {
            return res.status(400).json({
                success: false,
                error: 'No RunPod job associated with this makeover',
                code: 'NO_JOB'
            });
        }
        const jobStatus = await runpodService.checkJobStatus(makeover.runpod_job_id);
        if (jobStatus.status === 'COMPLETED' && makeover.status !== 'completed') {
            await runpodService.handleWebhookCallback(jobStatus);
        }
        res.json({
            success: true,
            data: {
                makeover_id: makeoverId,
                runpod_job_id: makeover.runpod_job_id,
                runpod_status: jobStatus.status,
                local_status: makeover.status
            }
        });
    }
    catch (error) {
        console.error('❌ Failed to check job status:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to check job status',
            message: error.message,
            code: 'STATUS_CHECK_ERROR'
        });
    }
});
router.get('/stats', (0, clerk_sdk_node_1.ClerkExpressRequireAuth)(), async (req, res) => {
    try {
        const { userId } = req.auth;
        const { data: stats, error } = await supabaseService.supabase
            .from('makeovers')
            .select('status')
            .eq('clerk_user_id', userId);
        if (error)
            throw error;
        const statusCounts = (stats || []).reduce((acc, makeover) => {
            acc[makeover.status] = (acc[makeover.status] || 0) + 1;
            return acc;
        }, {});
        const userStats = await supabaseService.getUserStats(userId);
        res.json({
            success: true,
            data: {
                total_makeovers: userStats.total_makeovers,
                total_photos: userStats.total_photos,
                status_breakdown: {
                    queued: statusCounts.queued || 0,
                    processing: statusCounts.processing || 0,
                    completed: statusCounts.completed || 0,
                    failed: statusCounts.failed || 0
                },
                subscription_tier: userStats.subscription_tier,
                member_since: userStats.created_at
            }
        });
    }
    catch (error) {
        console.error('❌ Failed to get makeover stats:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to get makeover statistics',
            message: error.message,
            code: 'STATS_ERROR'
        });
    }
});
exports.default = router;
//# sourceMappingURL=makeovers.js.map