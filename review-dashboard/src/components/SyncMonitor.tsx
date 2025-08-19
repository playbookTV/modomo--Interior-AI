import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Progress } from '@/components/ui/progress'
import { 
  RefreshCw, 
  CheckCircle, 
  AlertTriangle, 
  XCircle, 
  Activity, 
  Database,
  Server,
  Brain,
  HardDrive,
  Clock,
  Zap
} from 'lucide-react'

interface ComponentStatus {
  status: 'healthy' | 'error' | 'unavailable' | 'disconnected'
  message: string
  [key: string]: any
}

interface SyncStatus {
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

interface PipelineJobs {
  import_stage: Array<{
    job_id: string
    status: string
    progress: number
    message: string
    stage: string
  }>
  handoff_stage: Array<{
    job_id: string
    completed_at: string
    processed_items: number
    waiting_for: string
  }>
  detection_stage: Array<{
    job_id: string
    status: string
    progress: number
    processed_items: number
    total_items: number
    stage: string
    updated_at: string
  }>
  storage_stage: Array<any>
  completed: Array<{
    job_id: string
    status: string
    progress: number
    completed_at: string
    stage: string
    processing_mode: string
  }>
}

interface DetectionActivity {
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

const StatusIcon = ({ status }: { status: ComponentStatus['status'] }) => {
  switch (status) {
    case 'healthy':
      return <CheckCircle className="h-4 w-4 text-green-500" />
    case 'error':
      return <XCircle className="h-4 w-4 text-red-500" />
    case 'unavailable':
    case 'disconnected':
      return <AlertTriangle className="h-4 w-4 text-yellow-500" />
    default:
      return <RefreshCw className="h-4 w-4 text-gray-400" />
  }
}

const StatusBadge = ({ status }: { status: ComponentStatus['status'] }) => {
  const variants = {
    healthy: 'default',
    error: 'destructive',
    unavailable: 'secondary',
    disconnected: 'secondary'
  } as const

  return (
    <Badge variant={variants[status] || 'secondary'}>
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </Badge>
  )
}

export default function SyncMonitor() {
  const [syncStatus, setSyncStatus] = useState<SyncStatus | null>(null)
  const [pipelineJobs, setPipelineJobs] = useState<PipelineJobs | null>(null)
  const [detectionActivity, setDetectionActivity] = useState<DetectionActivity | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(true)

  const fetchSyncData = async () => {
    try {
      setLoading(true)
      
      const [syncRes, pipelineRes, detectionRes] = await Promise.all([
        fetch('/api/sync/status'),
        fetch('/api/sync/pipeline/active'),
        fetch('/api/sync/detection/triggers')
      ])

      if (!syncRes.ok || !pipelineRes.ok || !detectionRes.ok) {
        throw new Error('Failed to fetch sync data')
      }

      const [syncData, pipelineData, detectionData] = await Promise.all([
        syncRes.json(),
        pipelineRes.json(),
        detectionRes.json()
      ])

      setSyncStatus(syncData)
      setPipelineJobs(pipelineData)
      setDetectionActivity(detectionData)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const triggerPendingHandoffs = async () => {
    try {
      const response = await fetch('/api/sync/handoff/trigger', { method: 'POST' })
      if (!response.ok) throw new Error('Failed to trigger handoffs')
      
      const result = await response.json()
      // Show success message or update UI
      await fetchSyncData() // Refresh data
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to trigger handoffs')
    }
  }

  useEffect(() => {
    fetchSyncData()
    
    if (autoRefresh) {
      const interval = setInterval(fetchSyncData, 10000) // Refresh every 10 seconds
      return () => clearInterval(interval)
    }
  }, [autoRefresh])

  if (loading && !syncStatus) {
    return (
      <div className="flex items-center justify-center p-8">
        <RefreshCw className="h-6 w-6 animate-spin mr-2" />
        <span>Loading sync status...</span>
      </div>
    )
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>
          Error loading sync data: {error}
          <Button
            variant="outline"
            size="sm"
            onClick={fetchSyncData}
            className="ml-2"
          >
            Retry
          </Button>
        </AlertDescription>
      </Alert>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header with controls */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">System Synchronization Monitor</h2>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            <Activity className={`h-4 w-4 mr-2 ${autoRefresh ? 'text-green-500' : 'text-gray-400'}`} />
            Auto-refresh {autoRefresh ? 'On' : 'Off'}
          </Button>
          <Button variant="outline" size="sm" onClick={fetchSyncData} disabled={loading}>
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* System Health Overview */}
      {syncStatus && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Server className="h-5 w-5" />
              System Health Overview
              <StatusBadge status={syncStatus.pipeline_health as ComponentStatus['status']} />
            </CardTitle>
            <CardDescription>
              Real-time status of all system components
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {/* Frontend */}
              <div className="flex items-center gap-3 p-3 rounded-lg border">
                <div className="flex items-center gap-2">
                  <StatusIcon status={syncStatus.components.frontend.status} />
                  <span className="font-medium">Frontend</span>
                </div>
                <div className="text-sm text-gray-600">
                  {syncStatus.components.frontend.message}
                </div>
              </div>

              {/* Redis */}
              <div className="flex items-center gap-3 p-3 rounded-lg border">
                <div className="flex items-center gap-2">
                  <StatusIcon status={syncStatus.components.redis.status} />
                  <Database className="h-4 w-4" />
                  <span className="font-medium">Redis</span>
                </div>
                <div className="text-sm text-gray-600">
                  {syncStatus.components.redis.active_jobs || 0} active jobs
                </div>
              </div>

              {/* Celery */}
              <div className="flex items-center gap-3 p-3 rounded-lg border">
                <div className="flex items-center gap-2">
                  <StatusIcon status={syncStatus.components.celery.status} />
                  <Activity className="h-4 w-4" />
                  <span className="font-medium">Celery</span>
                </div>
                <div className="text-sm text-gray-600">
                  {syncStatus.components.celery.active_tasks || 0} active tasks
                </div>
              </div>

              {/* Railway */}
              <div className="flex items-center gap-3 p-3 rounded-lg border">
                <div className="flex items-center gap-2">
                  <StatusIcon status={syncStatus.components.railway.status} />
                  <Brain className="h-4 w-4" />
                  <span className="font-medium">Railway</span>
                </div>
                <div className="text-sm text-gray-600">
                  {syncStatus.pending_handoffs || 0} pending handoffs
                </div>
              </div>
            </div>

            {/* Pipeline Activity Summary */}
            <Separator className="my-4" />
            <div className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-4">
                <span className="flex items-center gap-1">
                  <RefreshCw className="h-4 w-4" />
                  {syncStatus.active_imports} Importing
                </span>
                <span className="flex items-center gap-1">
                  <Brain className="h-4 w-4" />
                  {syncStatus.active_detections} AI Processing
                </span>
                <span className="flex items-center gap-1">
                  <Clock className="h-4 w-4" />
                  {syncStatus.pending_handoffs} Pending Handoffs
                </span>
              </div>
              {syncStatus.pending_handoffs > 0 && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={triggerPendingHandoffs}
                >
                  <Zap className="h-4 w-4 mr-2" />
                  Trigger Handoffs
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Pipeline Flow Visualization */}
      {pipelineJobs && (
        <Card>
          <CardHeader>
            <CardTitle>Pipeline Flow Status</CardTitle>
            <CardDescription>
              Import → AI Detection → Storage Operations
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {/* Import Stage */}
              <div className="space-y-2">
                <h4 className="font-medium text-sm text-blue-600">1. Import Stage (Celery)</h4>
                <div className="border rounded-lg p-3 min-h-[120px]">
                  {pipelineJobs.import_stage.length > 0 ? (
                    pipelineJobs.import_stage.map((job) => (
                      <div key={job.job_id} className="text-xs p-2 bg-blue-50 rounded mb-2">
                        <div className="font-mono">{job.job_id.slice(0, 8)}...</div>
                        <Progress value={job.progress} className="h-1 mt-1" />
                        <div className="text-gray-600 mt-1">{job.message}</div>
                      </div>
                    ))
                  ) : (
                    <div className="text-gray-500 text-xs">No active imports</div>
                  )}
                </div>
              </div>

              {/* Handoff Stage */}
              <div className="space-y-2">
                <h4 className="font-medium text-sm text-yellow-600">2. Handoff Stage</h4>
                <div className="border rounded-lg p-3 min-h-[120px]">
                  {pipelineJobs.handoff_stage.length > 0 ? (
                    pipelineJobs.handoff_stage.map((job) => (
                      <div key={job.job_id} className="text-xs p-2 bg-yellow-50 rounded mb-2">
                        <div className="font-mono">{job.job_id.slice(0, 8)}...</div>
                        <div className="text-gray-600">
                          {job.processed_items} items → Waiting for {job.waiting_for}
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="text-gray-500 text-xs">No pending handoffs</div>
                  )}
                </div>
              </div>

              {/* AI Detection Stage */}
              <div className="space-y-2">
                <h4 className="font-medium text-sm text-purple-600">3. AI Detection (Railway)</h4>
                <div className="border rounded-lg p-3 min-h-[120px]">
                  {pipelineJobs.detection_stage.length > 0 ? (
                    pipelineJobs.detection_stage.map((job) => (
                      <div key={job.job_id} className="text-xs p-2 bg-purple-50 rounded mb-2">
                        <div className="font-mono">{job.job_id.slice(0, 8)}...</div>
                        <Progress value={job.progress} className="h-1 mt-1" />
                        <div className="text-gray-600 mt-1">
                          {job.processed_items}/{job.total_items} scenes
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="text-gray-500 text-xs">No active AI detection</div>
                  )}
                </div>
              </div>

              {/* Completed Stage */}
              <div className="space-y-2">
                <h4 className="font-medium text-sm text-green-600">4. Recent Completed</h4>
                <div className="border rounded-lg p-3 min-h-[120px]">
                  {pipelineJobs.completed.length > 0 ? (
                    pipelineJobs.completed.slice(0, 3).map((job) => (
                      <div key={job.job_id} className="text-xs p-2 bg-green-50 rounded mb-2">
                        <div className="font-mono">{job.job_id.slice(0, 8)}...</div>
                        <div className="flex items-center gap-1">
                          <CheckCircle className="h-3 w-3 text-green-500" />
                          <span className="text-green-600">{job.processing_mode}</span>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="text-gray-500 text-xs">No recent completions</div>
                  )}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* AI Detection Activity */}
      {detectionActivity && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
              AI Detection Activity
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Metrics */}
              <div className="space-y-4">
                <h4 className="font-medium">Detection Metrics</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Queue Depth:</span>
                    <span className="font-mono">{detectionActivity.detection_queue_depth}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Avg Time:</span>
                    <span className="font-mono">{detectionActivity.avg_detection_time}m</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Success Rate:</span>
                    <span className="font-mono">{detectionActivity.success_rate}%</span>
                  </div>
                </div>
              </div>

              {/* Active Detections */}
              <div className="space-y-4">
                <h4 className="font-medium">Currently Active</h4>
                <div className="space-y-2 max-h-32 overflow-y-auto">
                  {detectionActivity.active_detections.length > 0 ? (
                    detectionActivity.active_detections.map((detection) => (
                      <div key={detection.job_id} className="text-xs p-2 bg-blue-50 rounded">
                        <div className="font-mono">{detection.job_id.slice(0, 12)}...</div>
                        <Progress value={detection.progress} className="h-1 mt-1" />
                        <div className="text-gray-600 mt-1">
                          {detection.processed_items}/{detection.total_items} • {detection.duration_minutes}m
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="text-gray-500 text-xs">No active detections</div>
                  )}
                </div>
              </div>

              {/* Recent Triggers */}
              <div className="space-y-4">
                <h4 className="font-medium">Recent Triggers</h4>
                <div className="space-y-2 max-h-32 overflow-y-auto">
                  {detectionActivity.recent_triggers.slice(0, 5).map((trigger) => (
                    <div key={trigger.job_id} className="text-xs p-2 bg-gray-50 rounded">
                      <div className="flex items-center justify-between">
                        <span className="font-mono">{trigger.job_id.slice(0, 12)}...</span>
                        <StatusBadge status={trigger.status as ComponentStatus['status']} />
                      </div>
                      <div className="text-gray-600 mt-1">
                        {trigger.scenes_processed} scenes
                        {trigger.duration_minutes && ` • ${trigger.duration_minutes}m`}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Storage Operations Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <HardDrive className="h-5 w-5" />
            Storage Operations
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm text-gray-600">
            Maps and masks storage monitoring will be displayed here.
            This shows database operations and R2 storage activity.
          </div>
        </CardContent>
      </Card>
    </div>
  )
}