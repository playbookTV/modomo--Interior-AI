import axios from 'axios'
import { Scene, DetectedObject, Product, ProductSimilarity, ScrapingJob, DatasetExportJob, DatasetStats, CategoryStats, ReviewUpdate } from '../types'

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api'

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
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
  return response.data
}

export const updateReview = async (updates: ReviewUpdate[]): Promise<{ status: string; count: number }> => {
  const response = await apiClient.post('/review/update', updates)
  return response.data
}

export const approveScene = async (sceneId: string): Promise<{ status: string; scene_id: string }> => {
  const response = await apiClient.post(`/review/approve/${sceneId}`)
  return response.data
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
  const response = await apiClient.get('/jobs/active')
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