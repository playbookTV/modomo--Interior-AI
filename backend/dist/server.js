"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
require("dotenv/config");
const express_1 = __importDefault(require("express"));
const cors_1 = __importDefault(require("cors"));
const helmet_1 = __importDefault(require("helmet"));
const compression_1 = __importDefault(require("compression"));
const express_rate_limit_1 = __importDefault(require("express-rate-limit"));
const clerk_sdk_node_1 = require("@clerk/clerk-sdk-node");
const cloudPhotos_1 = __importDefault(require("./routes/cloudPhotos"));
const makeovers_1 = __importDefault(require("./routes/makeovers"));
const cloudflareR2_1 = require("./services/cloudflareR2");
const supabaseService_1 = require("./services/supabaseService");
const runpodService_1 = require("./services/runpodService");
const app = (0, express_1.default)();
const PORT = process.env.PORT || 6969;
app.use((0, helmet_1.default)({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            styleSrc: ["'self'", "'unsafe-inline'"],
            scriptSrc: ["'self'"],
            imgSrc: ["'self'", "data:", "https:"],
            connectSrc: ["'self'", "https:"],
        },
    },
    crossOriginEmbedderPolicy: false
}));
app.use((0, cors_1.default)({
    origin: [
        'http://localhost:8081',
        'https://reroom.app',
        'https://app.reroom.app',
        /\.reroom\.app$/
    ],
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'x-clerk-auth-token']
}));
app.use(express_1.default.json({ limit: '10mb' }));
app.use(express_1.default.urlencoded({ extended: true, limit: '10mb' }));
app.use((0, compression_1.default)());
const limiter = (0, express_rate_limit_1.default)({
    windowMs: 15 * 60 * 1000,
    max: 1000,
    message: {
        success: false,
        error: 'Too many requests, please try again later',
        code: 'RATE_LIMIT_EXCEEDED'
    },
    standardHeaders: true,
    legacyHeaders: false,
});
app.use(limiter);
const uploadLimiter = (0, express_rate_limit_1.default)({
    windowMs: 15 * 60 * 1000,
    max: 50,
    message: {
        success: false,
        error: 'Upload rate limit exceeded, please try again later',
        code: 'UPLOAD_RATE_LIMIT'
    }
});
app.use((0, clerk_sdk_node_1.ClerkExpressWithAuth)());
app.use((req, res, next) => {
    const start = Date.now();
    res.on('finish', () => {
        const duration = Date.now() - start;
        console.log(`${req.method} ${req.originalUrl} - ${res.statusCode} (${duration}ms)`);
    });
    next();
});
app.get('/health', async (req, res) => {
    try {
        const startTime = Date.now();
        const [r2Health, supabaseHealth, runpodHealth] = await Promise.allSettled([
            cloudflareR2_1.cloudflareR2Service.healthCheck(),
            supabaseService_1.supabaseService.healthCheck(),
            runpodService_1.runpodService.healthCheck()
        ]);
        const healthData = {
            status: 'healthy',
            timestamp: new Date().toISOString(),
            uptime: process.uptime(),
            environment: process.env.NODE_ENV || 'development',
            version: '2.0.0',
            response_time: Date.now() - startTime,
            services: {
                cloudflare_r2: {
                    status: r2Health.status === 'fulfilled' && r2Health.value ? 'healthy' : 'unhealthy',
                    error: r2Health.status === 'rejected' ? r2Health.reason?.message : null
                },
                supabase: {
                    status: supabaseHealth.status === 'fulfilled' && supabaseHealth.value ? 'healthy' : 'unhealthy',
                    error: supabaseHealth.status === 'rejected' ? supabaseHealth.reason?.message : null
                },
                runpod: {
                    status: runpodHealth.status === 'fulfilled' && runpodHealth.value ? 'healthy' : 'unhealthy',
                    error: runpodHealth.status === 'rejected' ? runpodHealth.reason?.message : null
                }
            }
        };
        const coreServicesHealthy = healthData.services.supabase.status === 'healthy' &&
            healthData.services.cloudflare_r2.status === 'healthy';
        if (!coreServicesHealthy) {
            healthData.status = 'unhealthy';
        }
        else if (healthData.services.runpod.status !== 'healthy') {
            healthData.status = 'degraded';
        }
        const statusCode = healthData.status === 'unhealthy' ? 503 : 200;
        res.status(statusCode).json(healthData);
    }
    catch (error) {
        console.error('❌ Health check failed:', error);
        res.status(503).json({
            status: 'unhealthy',
            timestamp: new Date().toISOString(),
            error: error.message
        });
    }
});
app.get('/status', (req, res) => {
    res.json({
        status: 'ok',
        timestamp: new Date().toISOString()
    });
});
app.use('/api/photos', uploadLimiter, cloudPhotos_1.default);
app.use('/api/makeovers', makeovers_1.default);
app.get('/', (req, res) => {
    res.json({
        service: 'ReRoom Cloud Backend',
        version: '2.0.0',
        environment: process.env.NODE_ENV || 'development',
        timestamp: new Date().toISOString(),
        endpoints: {
            health: '/health',
            photos: '/api/photos',
            makeovers: '/api/makeovers'
        },
        documentation: 'https://docs.reroom.app'
    });
});
app.use('*', (req, res) => {
    res.status(404).json({
        success: false,
        error: 'Endpoint not found',
        code: 'NOT_FOUND',
        path: req.originalUrl,
        method: req.method
    });
});
app.use((error, req, res, next) => {
    console.error('❌ Unhandled error:', error);
    if (error.name === 'ClerkAPIError') {
        return res.status(401).json({
            success: false,
            error: 'Authentication failed',
            code: 'AUTH_ERROR',
            message: error.message
        });
    }
    if (error.code === 'LIMIT_FILE_SIZE') {
        return res.status(413).json({
            success: false,
            error: 'File too large',
            code: 'FILE_TOO_LARGE',
            message: 'Maximum file size is 50MB'
        });
    }
    if (error.code === 'LIMIT_UNEXPECTED_FILE') {
        return res.status(400).json({
            success: false,
            error: 'Unexpected file field',
            code: 'INVALID_FILE_FIELD'
        });
    }
    res.status(500).json({
        success: false,
        error: 'Internal server error',
        code: 'INTERNAL_ERROR',
        message: process.env.NODE_ENV === 'development' ? error.message : 'Something went wrong'
    });
});
app.listen(PORT, () => {
    console.log(`🚀 ReRoom Cloud Backend running on port ${PORT}`);
    console.log(`🌍 Environment: ${process.env.NODE_ENV || 'development'}`);
    console.log(`🏥 Health check: http://localhost:${PORT}/health`);
    console.log(`📸 Photos API: http://localhost:${PORT}/api/photos`);
    console.log(`🎨 Makeovers API: http://localhost:${PORT}/api/makeovers`);
    console.log('\n🔧 Service Configuration:');
    console.log(`  Supabase: ${process.env.SUPABASE_URL ? '✅ Configured' : '❌ Missing'}`);
    console.log(`  Cloudflare R2: ${process.env.CLOUDFLARE_R2_ENDPOINT ? '✅ Configured' : '❌ Missing'}`);
    console.log(`  RunPod: ${process.env.RUNPOD_ENDPOINT ? '✅ Configured' : '❌ Missing'}`);
    console.log(`  Clerk: ${process.env.CLERK_SECRET_KEY ? '✅ Configured' : '❌ Missing'}`);
});
process.on('SIGTERM', () => {
    console.log('🛑 SIGTERM received, shutting down gracefully');
    process.exit(0);
});
process.on('SIGINT', () => {
    console.log('🛑 SIGINT received, shutting down gracefully');
    process.exit(0);
});
exports.default = app;
//# sourceMappingURL=server.js.map