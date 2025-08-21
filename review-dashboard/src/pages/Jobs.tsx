import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { 
  Play, Loader2, CheckCircle, XCircle, AlertTriangle, Clock, 
  Pause, RefreshCw, Filter, Search, Calendar, Activity,
  Database, Sparkles, Eye, Download, Settings, Zap, Palette,
  Mountain, Layers, RotateCcw, Trash2
} from 'lucide-react'
import { getActiveJobs, getRecentErrors, getJobHistory, retryJob, cancelJob, retryPendingJobs } from '../api/client'

interface Job {
  job_id: string
  status: 'pending' | 'running' | 'processing' | 'completed' | 'failed' | 'error'
  job_type?: string
  message: string
  progress: number
  total: number
  processed: number
  created_at?: string
  updated_at?: string
  error_message?: string
  dataset?: string
  features?: string[]
}

// API functions
async function fetchActiveJobs(): Promise<Job[]> {
  return await getActiveJobs()
}

// Map generation API functions
const API_BASE = 'https://ovalay-recruitment-production.up.railway.app'

async function generateMapsBatch(limit: number = 10, mapTypes: string[] = ['depth', 'edge'], forceRegenerate: boolean = false) {
  const mapTypesQuery = mapTypes.map(type => `map_types=${type}`).join('&')
  const response = await fetch(`${API_BASE}/jobs/generate-maps/batch?limit=${limit}&${mapTypesQuery}&force_regenerate=${forceRegenerate}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    }
  })
  
  if (!response.ok) {
    const errorText = await response.text()
    let errorMessage = `Map generation failed: ${response.status}`
    
    // Handle specific backend errors
    if (response.status === 500 && errorText.includes("cannot import name 'DepthAnythingV2'")) {
      errorMessage = "Backend model import error - this feature needs a backend deployment to fix the import issues."
    } else if (response.status === 500 && errorText.includes("'SyncSelectRequestBuilder' object has no attribute 'or_'")) {
      errorMessage = "Backend database query error - this feature needs a backend update to fix the Supabase query syntax."
    } else if (errorText) {
      try {
        const errorData = JSON.parse(errorText)
        errorMessage = errorData.detail || errorMessage
      } catch {
        errorMessage = errorText
      }
    }
    
    throw new Error(errorMessage)
  }
  
  return await response.json()
}

async function fetchPerformanceStatus() {
  const response = await fetch(`${API_BASE}/performance/status`)
  if (!response.ok) {
    throw new Error(`Performance status fetch failed: ${response.status}`)
  }
  return await response.json()
}

async function fetchRecentErrors(): Promise<{errors: Job[], total_error_jobs: number}> {
  return await getRecentErrors()
}

async function fetchAllJobs(): Promise<Job[]> {
  try {
    // Fetch active jobs, recent errors, and historical jobs in parallel
    const [activeJobs, errorData, historyData] = await Promise.all([
      fetchActiveJobs().catch(() => []),
      fetchRecentErrors().catch(() => ({errors: [], total_error_jobs: 0})),
      getJobHistory({ limit: 20, status: 'all' }).catch(() => ({jobs: [], total: 0, limit: 20, offset: 0, has_more: false}))
    ])
    
    // Combine all job sources and deduplicate by job_id
    const allJobs = [...activeJobs, ...errorData.errors, ...historyData.jobs]
    const uniqueJobs = allJobs.filter((job, index, array) => 
      array.findIndex(j => j.job_id === job.job_id) === index
    )
    
    // Sort by updated_at or created_at, most recent first
    return uniqueJobs.sort((a, b) => {
      const aTime = a.updated_at || a.created_at || ''
      const bTime = b.updated_at || b.created_at || ''
      return bTime.localeCompare(aTime)
    })
  } catch (error) {
    console.error('Failed to fetch jobs:', error)
    // Return empty array as fallback
    return []
  }
}

export function Jobs() {
  const [filterStatus, setFilterStatus] = useState<string>('all')
  const [searchTerm, setSearchTerm] = useState('')
  const [refreshInterval, setRefreshInterval] = useState(5000) // 5 seconds default
  const [isAutoRefresh, setIsAutoRefresh] = useState(true)
  
  // Map generation state
  const [isGeneratingMaps, setIsGeneratingMaps] = useState(false)
  const [mapGenerationResult, setMapGenerationResult] = useState<any>(null)
  const [mapGenerationError, setMapGenerationError] = useState<string | null>(null)
  const [mapBatchLimit, setMapBatchLimit] = useState(10)
  const [selectedMapTypes, setSelectedMapTypes] = useState(['depth', 'edge'])
  const [forceRegenerate, setForceRegenerate] = useState(false)

  // Job retry state
  const [retryingJobs, setRetryingJobs] = useState<Set<string>>(new Set())
  const [bulkRetryInProgress, setBulkRetryInProgress] = useState(false)
  const [retryResult, setRetryResult] = useState<string | null>(null)

  // Fetch all jobs with real-time updates
  const { data: jobs = [], isLoading, error, refetch } = useQuery({
    queryKey: ['all-jobs'],
    queryFn: fetchAllJobs,
    refetchInterval: isAutoRefresh ? refreshInterval : false,
    refetchIntervalInBackground: true
  })

  // Fetch performance status
  const { data: performanceStatus, isLoading: perfLoading } = useQuery({
    queryKey: ['performance-status'],
    queryFn: fetchPerformanceStatus,
    refetchInterval: isAutoRefresh ? 10000 : false, // Update every 10 seconds
    retry: 1, // Only retry once for performance data
    onError: (error) => {
      console.warn('Performance status fetch failed:', error)
    }
  })

  // Filter jobs based on status and search
  const filteredJobs = jobs.filter(job => {
    const matchesStatus = filterStatus === 'all' || job.status === filterStatus
    const matchesSearch = !searchTerm || 
      job.job_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      job.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
      job.dataset?.toLowerCase().includes(searchTerm.toLowerCase())
    
    return matchesStatus && matchesSearch
  })

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending': return <Clock className="h-4 w-4" />
      case 'running':
      case 'processing': return <Loader2 className="h-4 w-4 animate-spin" />
      case 'completed': return <CheckCircle className="h-4 w-4" />
      case 'failed':
      case 'error': return <XCircle className="h-4 w-4" />
      default: return <AlertTriangle className="h-4 w-4" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return 'text-yellow-600 bg-yellow-100 border-yellow-200'
      case 'running':
      case 'processing': return 'text-blue-600 bg-blue-100 border-blue-200'
      case 'completed': return 'text-green-600 bg-green-100 border-green-200'
      case 'failed':
      case 'error': return 'text-red-600 bg-red-100 border-red-200'
      default: return 'text-gray-600 bg-gray-100 border-gray-200'
    }
  }

  // Map generation handler
  const handleMapGeneration = async () => {
    try {
      setIsGeneratingMaps(true)
      setMapGenerationError(null)
      setMapGenerationResult(null)
      
      console.log('Starting batch map generation:', { 
        limit: mapBatchLimit, 
        mapTypes: selectedMapTypes, 
        forceRegenerate 
      })
      
      const result = await generateMapsBatch(mapBatchLimit, selectedMapTypes, forceRegenerate)
      setMapGenerationResult(result)
      
      // Refresh jobs list to show the new generation job
      refetch()
      
    } catch (error) {
      console.error('Map generation failed:', error)
      setMapGenerationError(error instanceof Error ? error.message : 'Unknown error occurred')
    } finally {
      setIsGeneratingMaps(false)
    }
  }

  // Job retry handlers
  const handleRetryJob = async (jobId: string) => {
    try {
      setRetryingJobs(prev => new Set(prev).add(jobId))
      setRetryResult(null)
      
      const result = await retryJob(jobId)
      setRetryResult(`Job ${jobId.slice(-8)} retry: ${result.message}`)
      
      // Refresh jobs list
      refetch()
      
    } catch (error) {
      console.error(`Failed to retry job ${jobId}:`, error)
      setRetryResult(`Failed to retry job ${jobId.slice(-8)}: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setRetryingJobs(prev => {
        const newSet = new Set(prev)
        newSet.delete(jobId)
        return newSet
      })
    }
  }

  const handleBulkRetryPending = async () => {
    try {
      setBulkRetryInProgress(true)
      setRetryResult(null)
      
      // Retry jobs that are older than 1 hour and still pending
      const result = await retryPendingJobs({
        older_than_hours: 1,
        limit: 50 // Retry up to 50 jobs at once
      })
      
      setRetryResult(`Bulk retry complete: ${result.retried_jobs} jobs restarted, ${result.skipped_jobs} skipped. ${result.message}`)
      
      // Refresh jobs list
      refetch()
      
    } catch (error) {
      console.error('Bulk retry failed:', error)
      setRetryResult(`Bulk retry failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setBulkRetryInProgress(false)
    }
  }

  const getJobTypeIcon = (message: string, jobType?: string) => {
    const msgLower = (message || '').toLowerCase()
    const typeStr = (jobType || '').toLowerCase()
    
    if (typeStr === 'import' || msgLower.includes('dataset') || msgLower.includes('import')) return <Database className="h-5 w-5" />
    if (typeStr === 'scenes' || msgLower.includes('scraping') || msgLower.includes('scrape')) return <Eye className="h-5 w-5" />
    if (typeStr === 'detection' || msgLower.includes('detection') || msgLower.includes('ai')) return <Sparkles className="h-5 w-5" />
    if (typeStr === 'maps' || msgLower.includes('map') || msgLower.includes('depth') || msgLower.includes('edge')) return <Mountain className="h-5 w-5" />
    if (msgLower.includes('color')) return <Palette className="h-5 w-5" />
    if (msgLower.includes('export')) return <Download className="h-5 w-5" />
    return <Activity className="h-5 w-5" />
  }

  const formatDuration = (seconds: number | undefined): string => {
    if (!seconds) return ''
    
    if (seconds < 60) return `${Math.round(seconds)}s`
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`
    return `${Math.round(seconds / 3600)}h`
  }

  // Summary statistics
  const stats = {
    total: jobs.length,
    running: jobs.filter(j => ['running', 'processing', 'pending'].includes(j.status)).length,
    completed: jobs.filter(j => j.status === 'completed').length,
    failed: jobs.filter(j => ['failed', 'error'].includes(j.status)).length
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Job Monitor</h1>
          <p className="text-gray-600">Real-time tracking of all background processing jobs</p>
        </div>
        
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setIsAutoRefresh(!isAutoRefresh)}
            className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm ${
              isAutoRefresh 
                ? 'bg-green-100 text-green-800 border border-green-200' 
                : 'bg-gray-100 text-gray-600 border border-gray-200'
            }`}
          >
            <Zap className="h-4 w-4" />
            <span>{isAutoRefresh ? 'Auto-refresh ON' : 'Auto-refresh OFF'}</span>
          </button>
          
          <select
            value={refreshInterval}
            onChange={(e) => setRefreshInterval(Number(e.target.value))}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm"
            disabled={!isAutoRefresh}
          >
            <option value={2000}>2s</option>
            <option value={5000}>5s</option>
            <option value={10000}>10s</option>
            <option value={30000}>30s</option>
          </select>
          
          <button
            onClick={() => refetch()}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            <RefreshCw className="h-4 w-4" />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Jobs</p>
              <p className="text-2xl font-bold text-gray-900">{stats.total}</p>
            </div>
            <Activity className="h-8 w-8 text-gray-400" />
          </div>
        </div>
        
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Running</p>
              <p className="text-2xl font-bold text-blue-600">{stats.running}</p>
            </div>
            <Loader2 className="h-8 w-8 text-blue-400" />
          </div>
        </div>
        
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Completed</p>
              <p className="text-2xl font-bold text-green-600">{stats.completed}</p>
            </div>
            <CheckCircle className="h-8 w-8 text-green-400" />
          </div>
        </div>
        
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Failed</p>
              <p className="text-2xl font-bold text-red-600">{stats.failed}</p>
            </div>
            <XCircle className="h-8 w-8 text-red-400" />
          </div>
        </div>
      </div>

      {/* Job Management Section */}
      {stats.running > 0 && (
        <div className="bg-gradient-to-r from-orange-50 to-red-50 rounded-xl p-6 border border-orange-200">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-orange-100 rounded-lg">
              <RotateCcw className="h-6 w-6 text-orange-600" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-900">Job Management</h2>
              <p className="text-gray-600">Retry stuck jobs or manage pending operations</p>
            </div>
          </div>
          
          <div className="flex items-center gap-4 mb-4">
            <button
              onClick={handleBulkRetryPending}
              disabled={bulkRetryInProgress || stats.running === 0}
              className="flex items-center gap-2 px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {bulkRetryInProgress ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Retrying...
                </>
              ) : (
                <>
                  <RotateCcw className="h-4 w-4" />
                  Retry Stuck Jobs
                </>
              )}
            </button>
            
            <div className="text-sm text-gray-600">
              This will retry jobs that have been pending for more than 1 hour
            </div>
          </div>
          
          {retryResult && (
            <div className={`mt-4 p-4 rounded-lg border ${
              retryResult.includes('Failed') ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'
            }`}>
              <div className="flex items-start">
                {retryResult.includes('Failed') ? (
                  <XCircle className="h-5 w-5 text-red-600 mr-3 mt-0.5 flex-shrink-0" />
                ) : (
                  <CheckCircle className="h-5 w-5 text-green-600 mr-3 mt-0.5 flex-shrink-0" />
                )}
                <div className="flex-1">
                  <h4 className={`text-sm font-medium mb-1 ${
                    retryResult.includes('Failed') ? 'text-red-800' : 'text-green-800'
                  }`}>
                    {retryResult.includes('Failed') ? 'Retry Failed' : 'Retry Complete'}
                  </h4>
                  <p className={`text-sm ${
                    retryResult.includes('Failed') ? 'text-red-700' : 'text-green-700'
                  }`}>
                    {retryResult}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Map Generation Section */}
      <div className="bg-gradient-to-r from-purple-50 to-orange-50 rounded-xl p-6 border border-purple-200">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 bg-purple-100 rounded-lg">
            <Mountain className="h-6 w-6 text-purple-600" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-gray-900">Batch Map Generation</h2>
            <p className="text-gray-600">Generate depth and edge maps for multiple scenes</p>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          {/* Batch Limit */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Scenes to Process
            </label>
            <input
              type="number"
              min="1"
              max="100"
              value={mapBatchLimit}
              onChange={(e) => setMapBatchLimit(Number(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
              disabled={isGeneratingMaps}
            />
          </div>
          
          {/* Map Types */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Map Types
            </label>
            <div className="space-y-2">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={selectedMapTypes.includes('depth')}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setSelectedMapTypes([...selectedMapTypes, 'depth'])
                    } else {
                      setSelectedMapTypes(selectedMapTypes.filter(t => t !== 'depth'))
                    }
                  }}
                  className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                  disabled={isGeneratingMaps}
                />
                <span className="ml-2 text-sm text-gray-700">Depth Maps</span>
              </label>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={selectedMapTypes.includes('edge')}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setSelectedMapTypes([...selectedMapTypes, 'edge'])
                    } else {
                      setSelectedMapTypes(selectedMapTypes.filter(t => t !== 'edge'))
                    }
                  }}
                  className="rounded border-gray-300 text-orange-600 focus:ring-orange-500"
                  disabled={isGeneratingMaps}
                />
                <span className="ml-2 text-sm text-gray-700">Edge Maps</span>
              </label>
            </div>
          </div>
          
          {/* Force Regenerate */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Options
            </label>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={forceRegenerate}
                onChange={(e) => setForceRegenerate(e.target.checked)}
                className="rounded border-gray-300 text-red-600 focus:ring-red-500"
                disabled={isGeneratingMaps}
              />
              <span className="ml-2 text-sm text-gray-700">Force Regenerate</span>
            </label>
            <p className="text-xs text-gray-500 mt-1">Regenerate existing maps</p>
          </div>
          
          {/* Generate Button */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Action
            </label>
            <button
              onClick={handleMapGeneration}
              disabled={isGeneratingMaps || selectedMapTypes.length === 0}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isGeneratingMaps ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Layers className="h-4 w-4" />
                  Generate Maps
                </>
              )}
            </button>
          </div>
        </div>
        
        {/* Results */}
        {mapGenerationResult && (
          <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
            <div className="flex items-start">
              <CheckCircle className="h-5 w-5 text-green-600 mr-3 mt-0.5 flex-shrink-0" />
              <div className="flex-1">
                <h4 className="text-sm font-medium text-green-800 mb-1">Generation Complete</h4>
                <p className="text-sm text-green-700">{mapGenerationResult.message}</p>
                <div className="mt-2 text-xs text-green-600">
                  Processed: {mapGenerationResult.processed} | 
                  Successful: {mapGenerationResult.successful} | 
                  Failed: {mapGenerationResult.failed}
                </div>
              </div>
            </div>
          </div>
        )}
        
        {mapGenerationError && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-start">
              <XCircle className="h-5 w-5 text-red-600 mr-3 mt-0.5 flex-shrink-0" />
              <div className="flex-1">
                <h4 className="text-sm font-medium text-red-800 mb-1">Generation Failed</h4>
                <p className="text-sm text-red-700">{mapGenerationError}</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* System Performance Status */}
      {performanceStatus && (
        <div className="bg-gradient-to-r from-blue-50 to-green-50 rounded-xl p-6 border border-blue-200">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-blue-100 rounded-lg">
              <Activity className="h-6 w-6 text-blue-600" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-900">System Performance</h2>
              <p className="text-gray-600">Real-time AI processing capabilities and recommendations</p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
            {/* System Specs */}
            <div className="bg-white p-4 rounded-lg border border-gray-200">
              <h3 className="text-sm font-medium text-gray-700 mb-2">System Specs</h3>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">CPU Cores:</span>
                  <span className="font-medium">{performanceStatus.system_specs?.cpu_count || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Memory:</span>
                  <span className="font-medium">
                    {performanceStatus.system_specs?.memory_available_gb?.toFixed(1) || 'N/A'}GB / 
                    {performanceStatus.system_specs?.memory_total_gb?.toFixed(1) || 'N/A'}GB
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">GPU:</span>
                  <span className={`font-medium ${performanceStatus.system_specs?.gpu_available ? 'text-green-600' : 'text-orange-600'}`}>
                    {performanceStatus.system_specs?.gpu_available ? '✅ Available' : '❌ CPU Only'}
                  </span>
                </div>
                {performanceStatus.system_specs?.gpu_name && (
                  <div className="text-xs text-gray-500 mt-1">
                    {performanceStatus.system_specs.gpu_name}
                  </div>
                )}
              </div>
            </div>
            
            {/* Performance Status */}
            <div className="bg-white p-4 rounded-lg border border-gray-200">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Performance Status</h3>
              <div className={`text-lg font-bold mb-2 ${
                performanceStatus.performance_status === 'optimal' ? 'text-green-600' :
                performanceStatus.performance_status === 'good' ? 'text-blue-600' :
                performanceStatus.performance_status === 'limited' ? 'text-orange-600' :
                'text-red-600'
              }`}>
                {performanceStatus.performance_status === 'optimal' ? '🚀 Optimal' :
                 performanceStatus.performance_status === 'good' ? '✅ Good' :
                 performanceStatus.performance_status === 'limited' ? '⚠️ Limited' :
                 '❌ Poor'}
              </div>
              {performanceStatus.warnings && Array.isArray(performanceStatus.warnings) && performanceStatus.warnings.length > 0 && (
                <div className="text-xs text-orange-600">
                  {performanceStatus.warnings.slice(0, 2).map((warning, index) => (
                    <div key={index}>• {warning}</div>
                  ))}
                </div>
              )}
            </div>
            
            {/* Estimated Times */}
            <div className="bg-white p-4 rounded-lg border border-gray-200">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Estimated Times</h3>
              <div className="space-y-1 text-sm">
                {performanceStatus.estimated_times?.depth_per_image && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Depth Map:</span>
                    <span className="font-medium">{performanceStatus.estimated_times.depth_per_image}s</span>
                  </div>
                )}
                {performanceStatus.estimated_times?.edge_per_image && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Edge Map:</span>
                    <span className="font-medium">{performanceStatus.estimated_times.edge_per_image}s</span>
                  </div>
                )}
                {performanceStatus.estimated_times?.detection_per_image && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Detection:</span>
                    <span className="font-medium">{performanceStatus.estimated_times.detection_per_image}s</span>
                  </div>
                )}
                {performanceStatus.estimated_times?.segmentation_per_object && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Segmentation:</span>
                    <span className="font-medium">{performanceStatus.estimated_times.segmentation_per_object}s</span>
                  </div>
                )}
              </div>
            </div>
            
            {/* Latest Operation */}
            <div className="bg-white p-4 rounded-lg border border-gray-200">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Latest Operation</h3>
              {performanceStatus.latest_operation ? (
                <div className="space-y-1 text-sm">
                  <div className="font-medium text-gray-900">{performanceStatus.latest_operation.name}</div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Duration:</span>
                    <span className="font-medium">{performanceStatus.latest_operation.duration}s</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">CPU Usage:</span>
                    <span className="font-medium">{performanceStatus.latest_operation.cpu_usage}%</span>
                  </div>
                </div>
              ) : (
                <div className="text-sm text-gray-500">No recent operations</div>
              )}
            </div>
          </div>
          
          {/* Performance Suggestions */}
          {performanceStatus.suggestions && performanceStatus.suggestions.length > 0 && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h4 className="text-sm font-medium text-blue-800 mb-2">💡 Performance Suggestions</h4>
              <div className="space-y-1">
                {performanceStatus.suggestions.slice(0, 3).map((suggestion, index) => (
                  <div key={index} className="text-sm text-blue-700">• {suggestion}</div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {performanceStatus === undefined && !perfLoading && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-center">
            <AlertTriangle className="h-5 w-5 text-yellow-600 mr-2" />
            <span className="text-sm text-yellow-800">
              Performance monitoring unavailable - AI service may be offline
            </span>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-col sm:flex-row space-y-3 sm:space-y-0 sm:space-x-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search jobs by ID, message, or dataset..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        
        <div className="flex items-center space-x-2">
          <Filter className="h-5 w-5 text-gray-400" />
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="all">All Status</option>
            <option value="pending">Pending</option>
            <option value="running">Running</option>
            <option value="processing">Processing</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="error">Error</option>
          </select>
        </div>
      </div>

      {/* Jobs List */}
      <div className="space-y-4">
        {filteredJobs.length === 0 ? (
          <div className="text-center py-12 bg-white rounded-lg border border-gray-200">
            <Activity className="h-12 w-12 mx-auto text-gray-400 mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No jobs found</h3>
            <p className="text-gray-600">
              {filterStatus !== 'all' || searchTerm 
                ? 'Try adjusting your filters or search terms'
                : 'No background jobs are currently running or have been executed recently'
              }
            </p>
          </div>
        ) : (
          filteredJobs.map((job) => (
            <div key={job.job_id} className="bg-white rounded-lg border border-gray-200 p-6 hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-start space-x-3">
                  <div className="flex-shrink-0 mt-1">
                    {getJobTypeIcon(job.message, job.job_type)}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <h3 className="font-semibold text-gray-900">
                        Job #{job.job_id.slice(-8)}
                      </h3>
                      <span className={`inline-flex items-center space-x-1 px-3 py-1 rounded-full text-xs font-medium border ${getStatusColor(job.status)}`}>
                        {getStatusIcon(job.status)}
                        <span className="capitalize">{job.status}</span>
                      </span>
                    </div>
                    
                    <p className="text-gray-700 mb-2">{job.message}</p>
                    
                    {job.dataset && (
                      <div className="text-sm text-gray-600 mb-2">
                        <span className="font-medium">Dataset:</span> {job.dataset}
                      </div>
                    )}
                    
                    <div className="flex items-center space-x-4 text-sm text-gray-600">
                      {job.created_at && (
                        <div className="flex items-center space-x-1">
                          <Calendar className="h-4 w-4" />
                          <span>Started: {new Date(job.created_at).toLocaleString()}</span>
                        </div>
                      )}
                      {job.updated_at && job.updated_at !== job.created_at && (
                        <div className="flex items-center space-x-1">
                          <Clock className="h-4 w-4" />
                          <span>Updated: {new Date(job.updated_at).toLocaleString()}</span>
                        </div>
                      )}
                      {job.duration_seconds && (
                        <div className="flex items-center space-x-1">
                          <Activity className="h-4 w-4" />
                          <span>Duration: {formatDuration(job.duration_seconds)}</span>
                        </div>
                      )}
                      {job.job_type && (
                        <div className="flex items-center space-x-1">
                          <span className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full">
                            {job.job_type}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
                
                {/* Job Actions */}
                <div className="flex items-center space-x-2 ml-4">
                  {(job.status === 'pending' || job.status === 'failed' || job.status === 'error') && (
                    <button
                      onClick={() => handleRetryJob(job.job_id)}
                      disabled={retryingJobs.has(job.job_id)}
                      className="flex items-center space-x-1 px-3 py-1.5 bg-orange-100 text-orange-700 hover:bg-orange-200 text-sm rounded-lg border border-orange-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                      title="Retry this job"
                    >
                      {retryingJobs.has(job.job_id) ? (
                        <>
                          <Loader2 className="h-3 w-3 animate-spin" />
                          <span>Retrying...</span>
                        </>
                      ) : (
                        <>
                          <RotateCcw className="h-3 w-3" />
                          <span>Retry</span>
                        </>
                      )}
                    </button>
                  )}
                  
                  {(job.status === 'running' || job.status === 'processing') && (
                    <button
                      className="flex items-center space-x-1 px-3 py-1.5 bg-red-100 text-red-700 hover:bg-red-200 text-sm rounded-lg border border-red-200 transition-colors"
                      title="Cancel this job"
                    >
                      <XCircle className="h-3 w-3" />
                      <span>Cancel</span>
                    </button>
                  )}
                </div>
              </div>
              
              {/* Progress Bar */}
              <div className="mb-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">Progress</span>
                  <span className="text-sm text-gray-600">
                    {job.processed} / {job.total} items ({job.progress}%)
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div 
                    className={`h-2.5 rounded-full transition-all duration-300 ${
                      job.status === 'completed' ? 'bg-green-500' :
                      job.status === 'failed' || job.status === 'error' ? 'bg-red-500' :
                      'bg-blue-500'
                    }`}
                    style={{ width: `${Math.min(100, Math.max(0, job.progress))}%` }}
                  />
                </div>
              </div>
              
              {/* Error Message */}
              {job.error_message && (job.status === 'failed' || job.status === 'error') && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                  <div className="flex items-start">
                    <AlertTriangle className="h-5 w-5 text-red-600 mr-3 mt-0.5 flex-shrink-0" />
                    <div className="flex-1">
                      <h4 className="text-sm font-medium text-red-800 mb-1">Error Details</h4>
                      <p className="text-sm text-red-700">{job.error_message}</p>
                    </div>
                  </div>
                </div>
              )}
              
              {/* Features */}
              {job.features && job.features.length > 0 && (
                <div className="mt-4">
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Features:</h4>
                  <div className="flex flex-wrap gap-2">
                    {job.features.map((feature, index) => (
                      <span key={index} className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full">
                        {feature}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))
        )}
      </div>
      
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center">
            <XCircle className="h-5 w-5 text-red-600 mr-2" />
            <span className="text-sm text-red-800">
              Failed to fetch jobs: {error.message}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}