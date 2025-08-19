import { apiClient } from './client'

export interface ComponentStatus {
  status: 'healthy' | 'error' | 'unavailable' | 'disconnected'
  message: string
  [key: string]: any
}

export interface SyncStatus {
  timestamp: string
  components: {
    frontend: ComponentStatus
    redis: ComponentStatus
    celery: ComponentStatus
    railway: ComponentStatus
  }
  pipeline_health: 'healthy' | 'degraded' | 'partial' | 'unknown'
  active_imports: number
  active_detections: number
  pending_handoffs: number
}

export interface PipelineJob {
  job_id: string
  status: string
  progress: number
  message?: string
  stage: string
  processed_items?: number
  total_items?: number
  updated_at?: string
  completed_at?: string
  processing_mode?: string
  waiting_for?: string
}

export interface PipelineJobs {
  import_stage: PipelineJob[]
  handoff_stage: PipelineJob[]
  detection_stage: PipelineJob[]
  storage_stage: PipelineJob[]
  completed: PipelineJob[]
}

export interface DetectionActivity {
  active_detections: Array<{
    job_id: string
    progress: number
    processed_items: number
    total_items: number
    duration_minutes: number
    original_job: string
    processing_mode: string
  }>
  recent_triggers: Array<{
    job_id: string
    status: string
    started_at: string
    original_job: string
    scenes_processed: number
    duration_minutes?: number
  }>
  detection_queue_depth: number
  avg_detection_time: number
  success_rate: number
}

export interface StorageStatus {
  recent_maps_stored: number
  recent_masks_stored: number
  storage_health: 'active' | 'idle' | 'error'
  db_operations: number
  r2_operations: number
}

/**
 * Get comprehensive sync status for FE/Celery/Redis/Railway
 */
export const getSyncStatus = async (): Promise<SyncStatus> => {
  const response = await apiClient.get('/sync/status')
  return response.data
}

/**
 * Get all jobs currently in the pipeline with their stage information
 */
export const getActivePipelineJobs = async (): Promise<PipelineJobs> => {
  const response = await apiClient.get('/sync/pipeline/active')
  return response.data
}

/**
 * Monitor when AI detection is being/has been triggered
 */
export const getAIDetectionTriggers = async (): Promise<DetectionActivity> => {
  const response = await apiClient.get('/sync/detection/triggers')
  return response.data
}

/**
 * Manually trigger AI detection for completed import jobs that are waiting for handoff
 */
export const triggerPendingHandoffs = async (): Promise<{
  status: string
  message: string
  triggered_jobs: number
  pending_handoffs: number
}> => {
  const response = await apiClient.post('/sync/handoff/trigger')
  return response.data
}

/**
 * Monitor storage operations for maps and masks
 */
export const getStorageOperationsStatus = async (): Promise<StorageStatus> => {
  const response = await apiClient.get('/sync/storage/operations')
  return response.data
}

/**
 * Get real-time sync data in a single call (for efficiency)
 */
export const getAllSyncData = async (): Promise<{
  sync_status: SyncStatus
  pipeline_jobs: PipelineJobs
  detection_activity: DetectionActivity
  storage_status: StorageStatus
}> => {
  const [syncRes, pipelineRes, detectionRes, storageRes] = await Promise.all([
    getSyncStatus(),
    getActivePipelineJobs(),
    getAIDetectionTriggers(),
    getStorageOperationsStatus()
  ])

  return {
    sync_status: syncRes,
    pipeline_jobs: pipelineRes,
    detection_activity: detectionRes,
    storage_status: storageRes
  }
}