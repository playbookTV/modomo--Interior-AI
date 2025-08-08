import axios from 'axios';
import type {
  RoomMakeoverRequest,
  RoomMakeoverResponse,
  EnhancedMakeoverRequest,
  HealthStatus,
  StyleComparisonRequest,
  StyleComparisonResult,
} from '@/types/api';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  timeout: 60000, // 60 seconds for AI processing
  headers: {
    'Content-Type': 'application/json',
  },
});

export const aiService = {
  // Health check
  async getHealth(): Promise<HealthStatus> {
    const response = await api.get('/health/detailed');
    return response.data;
  },

  // Basic room makeover
  async createMakeover(request: RoomMakeoverRequest): Promise<RoomMakeoverResponse> {
    const response = await api.post('/makeover', request);
    return response.data;
  },

  // Enhanced room makeover with custom parameters
  async createEnhancedMakeover(request: EnhancedMakeoverRequest): Promise<RoomMakeoverResponse> {
    const response = await api.post('/makeover/enhanced', request);
    return response.data;
  },

  // Get makeover by ID
  async getMakeover(makeoverId: string): Promise<RoomMakeoverResponse> {
    const response = await api.get(`/makeover/${makeoverId}`);
    return response.data;
  },

  // Style comparison
  async createStyleComparison(request: StyleComparisonRequest): Promise<StyleComparisonResult> {
    const response = await api.post('/analyze/style-comparison', request);
    return response.data;
  },

  // Upload image and convert to URL (mock for now)
  async uploadImage(file: File): Promise<string> {
    // In a real implementation, this would upload to storage
    // For demo purposes, we'll use a data URL
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => {
        const dataUrl = reader.result as string;
        // For the existing backend, we need to provide a URL
        // In production, this would be uploaded to S3/R2 and return the URL
        resolve(`https://example.com/uploads/${Date.now()}_${file.name}`);
      };
      reader.readAsDataURL(file);
    });
  },

  // Convert file to base64 for direct processing
  async fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        // Remove data:image/jpeg;base64, prefix
        const base64 = result.split(',')[1];
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  },

  // Get performance metrics
  async getPerformanceMetrics() {
    const response = await api.get('/metrics/performance');
    return response.data;
  },

  // Get monitoring status
  async getMonitoringStatus() {
    const response = await api.get('/monitoring/status');
    return response.data;
  },

  // Get usage analytics
  async getUsageAnalytics() {
    const response = await api.get('/analytics/usage');
    return response.data;
  },

  // List available models
  async getModels() {
    const response = await api.get('/models');
    return response.data;
  },

  // Get cache statistics
  async getCacheStats() {
    const response = await api.get('/cache/redis');
    return response.data;
  },

  // Reset cache (admin)
  async clearCache() {
    const response = await api.post('/cache/redis/clear');
    return response.data;
  },
};

export default api;