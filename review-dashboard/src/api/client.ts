import axios from 'axios'
import { Scene, DetectedObject, Product, ProductSimilarity, ScrapingJob, DatasetExportJob, DatasetStats, CategoryStats, ReviewUpdate } from '../types'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://ovalay-recruitment-production.up.railway.app' || '/api'

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000, // Reduced from 30s to 15s for better UX
  headers: {
    'Content-Type': 'application/json',
  },
})

// Add request interceptor for debugging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`)
    return config
  },
  (error) => {
    console.error('API Request Error:', error)
    return Promise.reject(error)
  }
)

// Add response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Response Error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

// Health check
export const healthCheck = async (): Promise<{ status: string; timestamp: string }> => {
  const response = await apiClient.get('/health')
  return response.data
}

// Scene scraping
export const startSceneScraping = async (params: {
  limit: number
  room_types?: string[]
}): Promise<{ job_id: string; status: string }> => {
  const response = await apiClient.post('/scrape/scenes', params)
  return response.data
}

export const getScrapingStatus = async (jobId: string): Promise<ScrapingJob> => {
  const response = await apiClient.get(`/scrape/scenes/${jobId}/status`)
  return response.data
}

// Object detection
export const startDetection = async (sceneIds: string[]): Promise<{ job_id: string; status: string }> => {
  const response = await apiClient.post('/detect/process', sceneIds)
  return response.data
}

// Review queue
export const getReviewQueue = async (params: {
  limit?: number
  room_type?: string
  category?: string
}): Promise<Scene[]> => {
  const response = await apiClient.get('/review/queue', { params })
  return response.data.scenes || response.data
}

export const updateReview = async (updates: ReviewUpdate[]): Promise<{ status: string; count: number }> => {
  try {
    const response = await apiClient.post('/review/update', updates, {
      timeout: 30000, // Increase timeout for review operations
    })
    return response.data
  } catch (error: any) {
    if (error.code === 'ECONNABORTED') {
      throw new Error('Review update is taking longer than expected. The operation may still be processing in the background.')
    }
    throw error
  }
}

export const approveScene = async (sceneId: string): Promise<{ status: string; scene_id: string }> => {
  try {
    const response = await apiClient.post(`/review/approve/${sceneId}`, {}, {
      timeout: 30000, // Increase timeout for scene approval
    })
    return response.data
  } catch (error: any) {
    if (error.code === 'ECONNABORTED') {
      throw new Error('Scene approval is taking longer than expected. The operation may still be processing in the background.')
    }
    throw error
  }
}

export const rejectScene = async (sceneId: string): Promise<{ status: string; scene_id: string }> => {
  try {
    const response = await apiClient.post(`/review/reject/${sceneId}`, {}, {
      timeout: 30000, // Increase timeout for scene rejection
    })
    return response.data
  } catch (error: any) {
    if (error.code === 'ECONNABORTED') {
      throw new Error('Scene rejection is taking longer than expected. The operation may still be processing in the background.')
    }
    throw error
  }
}

// Dataset export
export const startDatasetExport = async (params: {
  train_ratio: number
  val_ratio: number
  test_ratio: number
}): Promise<{ export_id: string; status: string }> => {
  const response = await apiClient.post('/export/dataset', params)
  return response.data
}

export const getDatasetStats = async (): Promise<DatasetStats> => {
  const response = await apiClient.get('/stats/dataset')
  return response.data
}

export const getCategoryStats = async (): Promise<CategoryStats[]> => {
  const response = await apiClient.get('/stats/categories')
  return response.data
}

// Product search
export const searchProducts = async (params: {
  query?: string
  category?: string
  limit?: number
}): Promise<Product[]> => {
  const response = await apiClient.get('/products/search', { params })
  return response.data
}

export const getProductSimilarities = async (
  objectId: string,
  limit = 5
): Promise<ProductSimilarity[]> => {
  const response = await apiClient.get(`/products/similar/${objectId}`, {
    params: { limit }
  })
  return response.data
}

// Jobs monitoring  
export const getActiveJobs = async (): Promise<ScrapingJob[]> => {
  const response = await apiClient.get('/jobs/active', {
    timeout: 5000 // Shorter timeout for jobs endpoint
  })
  return response.data
}

export const getJobStatus = async (jobId: string): Promise<ScrapingJob> => {
  const response = await apiClient.get(`/jobs/${jobId}/status`)
  return response.data
}

export const getRecentErrors = async (): Promise<{errors: ScrapingJob[], total_error_jobs: number}> => {
  const response = await apiClient.get('/jobs/errors/recent')
  return response.data
}

export const getAllExports = async (): Promise<DatasetExportJob[]> => {
  const response = await apiClient.get('/exports')
  return response.data
}

// Batch operations
export const batchApproveObjects = async (objectIds: string[]): Promise<{ count: number }> => {
  const updates = objectIds.map(id => ({ object_id: id, approved: true }))
  const response = await updateReview(updates)
  return { count: response.count }
}

export const batchRejectObjects = async (objectIds: string[]): Promise<{ count: number }> => {
  const updates = objectIds.map(id => ({ object_id: id, approved: false }))
  const response = await updateReview(updates)
  return { count: response.count }
}

// Classification endpoints
export const testImageClassification = async (params: {
  image_url: string
  caption?: string
}): Promise<{
  image_url: string
  caption?: string
  classification: {
    image_type: string
    is_primary_object: boolean
    primary_category?: string
    confidence: number
    reason: string
    metadata: any
  }
  status: string
}> => {
  const response = await apiClient.get('/classify/test', { params })
  return response.data
}

export const startSceneReclassification = async (params: {
  limit: number
  force_redetection: boolean
}): Promise<{
  job_id: string
  status: string
  message: string
  features: string[]
}> => {
  const response = await apiClient.post('/classify/reclassify-scenes', null, { params })
  return response.data
}

// Enhanced scene fetching with classification data
export const getScenesWithClassification = async (params: {
  limit?: number
  offset?: number
  status?: string
  image_type?: string
  room_type?: string
}): Promise<{
  scenes: (Scene & {
    image_type?: string
    is_primary_object?: boolean
    primary_category?: string
    metadata?: {
      classification_confidence?: number
      classification_reason?: string
      detected_room_type?: string
      detected_styles?: string[]
      scores?: {
        object: number
        scene: number
        hybrid: number
        style: number
      }
    }
  })[]
  total: number
}> => {
  const response = await apiClient.get('/scenes', { params })
  return response.data
}