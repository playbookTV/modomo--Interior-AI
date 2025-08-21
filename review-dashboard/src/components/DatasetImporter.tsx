import React, { useState, useEffect, useCallback } from 'react'
import { useMutation, useQueryClient, useQuery } from '@tanstack/react-query'
import { 
  Play, Loader2, Database, Sparkles, AlertTriangle, Eye, CheckCircle, Clock, 
  Grid, List, Download, Target, Palette, Filter, BarChart3, ImageIcon, ChevronLeft, 
  ChevronRight, ArrowLeft, X, Info, AlertCircle
} from 'lucide-react'

interface ImportJob {
  job_id: string
  status: string
  message: string
  dataset: string
  features: string[]
}

interface Toast {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message: string
  duration?: number
}

interface JobProgress {
  status: string
  progress: number
  processed: number
  total: number
  message?: string
  error_message?: string
}

interface ProcessedImagesResponse {
  scenes: any[]
  total: number
  page: number
  totalPages: number
}

const PRESET_DATASETS = [
  {
    name: 'Houzz Interior Design',
    description: '1,600 high-quality interior design images with style labels',
    url: 'sk2003/houzzdata',
    recommended: true
  },
  {
    name: 'Custom HuggingFace Dataset',
    description: 'Enter any HuggingFace dataset ID',
    url: 'custom',
    recommended: false
  }
]

async function importDataset(params: {
  dataset: string
  offset: number
  limit: number
  include_detection: boolean
}): Promise<ImportJob> {
  const response = await fetch(
    `https://ovalay-recruitment-production.up.railway.app/import/huggingface-dataset?dataset=${encodeURIComponent(params.dataset)}&offset=${params.offset}&limit=${params.limit}&include_detection=${params.include_detection}`,
    { method: 'POST' }
  )
  if (!response.ok) throw new Error('Import failed')
  return response.json()
}

async function fetchJobStatus(jobId: string): Promise<JobProgress> {
  const response = await fetch(
    `https://ovalay-recruitment-production.up.railway.app/jobs/${jobId}/status`
  )
  if (!response.ok) {
    if (response.status === 404) {
      // Job not found - might be completed or expired
      return {
        status: 'not_found',
        progress: 0,
        processed: 0,
        total: 0,
        message: 'Job not found - may have completed or expired'
      }
    }
    throw new Error('Failed to fetch job status')
  }
  return response.json()
}

async function fetchScenes(limit = 10): Promise<{scenes: any[], total: number}> {
  const response = await fetch(
    `https://ovalay-recruitment-production.up.railway.app/review/queue?limit=${limit}&offset=0`
  )
  if (!response.ok) throw new Error('Failed to fetch scenes')
  const data = await response.json()
  return {
    scenes: data.scenes || [],
    total: data.pagination?.total || 0
  }
}

async function fetchDatasetStats(): Promise<any> {
  const response = await fetch(
    'https://ovalay-recruitment-production.up.railway.app/stats/dataset'
  )
  if (!response.ok) throw new Error('Failed to fetch stats')
  return response.json()
}

async function fetchCategoryStats(): Promise<any[]> {
  const response = await fetch(
    'https://ovalay-recruitment-production.up.railway.app/stats/categories'
  )
  if (!response.ok) throw new Error('Failed to fetch category stats')
  return response.json()
}

async function fetchRecentErrors(): Promise<any> {
  const response = await fetch(
    'https://ovalay-recruitment-production.up.railway.app/jobs/errors/recent'
  )
  if (!response.ok) throw new Error('Failed to fetch recent errors')
  return response.json()
}

async function fetchObjects(limit = 10, category?: string): Promise<{objects: any[], total: number}> {
  const params = new URLSearchParams({ limit: limit.toString(), offset: '0' })
  if (category) params.set('category', category)
  
  const response = await fetch(
    `https://ovalay-recruitment-production.up.railway.app/objects?${params}`
  )
  if (!response.ok) throw new Error('Failed to fetch objects')
  return response.json()
}

async function exportTrainingDataset(): Promise<any> {
  const response = await fetch(
    'https://ovalay-recruitment-production.up.railway.app/export/training-dataset'
  )
  if (!response.ok) throw new Error('Failed to export training dataset')
  return response.json()
}

async function fetchAllProcessedImages(page = 1, limit = 20): Promise<ProcessedImagesResponse> {
  const offset = (page - 1) * limit
  const response = await fetch(
    `https://ovalay-recruitment-production.up.railway.app/scenes?limit=${limit}&offset=${offset}&include_objects=true`
  )
  if (!response.ok) throw new Error('Failed to fetch processed images')
  
  const data = await response.json()
  const totalPages = Math.ceil(data.total / limit)
  
  return {
    scenes: data.scenes || [],
    total: data.total || 0,
    page: page,
    totalPages: totalPages
  }
}

export function DatasetImporter() {
  const [selectedDataset, setSelectedDataset] = useState('sk2003/houzzdata')
  const [customDataset, setCustomDataset] = useState('')
  const [offset, setOffset] = useState(0)
  const [limit, setLimit] = useState(50)
  const [includeDetection, setIncludeDetection] = useState(true)
  const [lastJob, setLastJob] = useState<ImportJob | null>(null)
  const [isMonitoring, setIsMonitoring] = useState(false)
  const [viewMode, setViewMode] = useState<'scenes' | 'objects'>('scenes')
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [jobProgress, setJobProgress] = useState<JobProgress | null>(null)
  const [showGallery, setShowGallery] = useState(false)
  const [galleryPage, setGalleryPage] = useState(1)
  const [toasts, setToasts] = useState<Toast[]>([])

  // Toast management functions
  const removeToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(toast => toast.id !== id))
  }, [])

  const addToast = useCallback((toast: Omit<Toast, 'id'>) => {
    const id = Date.now().toString()
    const newToast = { ...toast, id }
    setToasts(prev => [...prev, newToast])
    
    // Auto-dismiss after duration (default 5 seconds)
    const duration = toast.duration || 5000
    if (duration > 0) {
      setTimeout(() => removeToast(id), duration)
    }
  }, [removeToast])

  const queryClient = useQueryClient()

  // Real-time monitoring of scenes and stats
  const { data: scenes, refetch: refetchScenes } = useQuery({
    queryKey: ['scenes'],
    queryFn: () => fetchScenes(10),
    enabled: isMonitoring,
    refetchInterval: isMonitoring ? 3000 : false
  })

  const { data: liveStats, refetch: refetchStats } = useQuery({
    queryKey: ['live-stats'],
    queryFn: fetchDatasetStats,
    enabled: isMonitoring,
    refetchInterval: isMonitoring ? 3000 : false
  })

  const { data: categoryStats } = useQuery({
    queryKey: ['category-stats'],
    queryFn: fetchCategoryStats,
    enabled: isMonitoring,
    refetchInterval: isMonitoring ? 5000 : false
  })

  const { data: objects, refetch: refetchObjects } = useQuery({
    queryKey: ['objects', selectedCategory],
    queryFn: () => fetchObjects(10, selectedCategory || undefined),
    enabled: isMonitoring && viewMode === 'objects',
    refetchInterval: isMonitoring && viewMode === 'objects' ? 3000 : false
  })

  // Fetch recent errors for display
  const { data: recentErrors, refetch: refetchErrors } = useQuery({
    queryKey: ['recent-errors'],
    queryFn: fetchRecentErrors,
    enabled: true,
    refetchInterval: 10000 // Check for errors every 10 seconds
  })

  // Job progress tracking
  const { data: currentJobProgress } = useQuery({
    queryKey: ['job-status', lastJob?.job_id],
    queryFn: () => fetchJobStatus(lastJob!.job_id),
    enabled: !!lastJob?.job_id && (lastJob?.status === 'running' || jobProgress?.status === 'processing'),
    refetchInterval: !!lastJob?.job_id && (lastJob?.status === 'running' || jobProgress?.status === 'processing') ? 2000 : false,
    
  })

  // Handle job progress updates
  useEffect(() => {
    if (currentJobProgress) {
      setJobProgress(currentJobProgress)
      // Stop monitoring when job is complete
      if (currentJobProgress.status === 'completed') {
        setIsMonitoring(true) // Keep monitoring dashboard stats for a bit
        addToast({
          type: 'success',
          title: 'Dataset Import Complete!',
          message: `Successfully processed ${currentJobProgress.processed || 0} images from ${lastJob?.dataset}`,
          duration: 8000
        })
        // Auto-refresh dashboard stats
        refetchStats()
        refetchScenes()
        queryClient.invalidateQueries({ queryKey: ['category-stats'] })
      } else if (currentJobProgress.status === 'failed' || currentJobProgress.status === 'error') {
        addToast({
          type: 'error',
          title: 'Dataset Import Failed',
          message: currentJobProgress.error_message || currentJobProgress.message || 'Import failed. Check the error details below.',
          duration: 15000 // Show error longer
        })
        setIsMonitoring(false)
        // Refresh error list when a job fails
        refetchErrors()
      }
    }
  }, [currentJobProgress, lastJob?.dataset, addToast, refetchStats, refetchScenes, queryClient, refetchErrors])

  // Gallery data for processed images
  const { data: galleryData, isLoading: galleryLoading } = useQuery({
    queryKey: ['processed-images', galleryPage],
    queryFn: () => fetchAllProcessedImages(galleryPage, 20),
    enabled: showGallery,
    placeholderData: (previousData) => previousData
  })

  const exportMutation = useMutation({
    mutationFn: exportTrainingDataset,
    onSuccess: (data) => {
      // Create download link
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `reroom-training-dataset-${new Date().toISOString().split('T')[0]}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      
      addToast({
        type: 'success',
        title: 'Dataset Export Complete!',
        message: `Training dataset downloaded with ${data.total_scenes || 0} scenes and ${data.total_objects || 0} objects`,
        duration: 6000
      })
    },
    onError: (error: any) => {
      addToast({
        type: 'error',
        title: 'Export Failed',
        message: error.message || 'Failed to export training dataset',
        duration: 8000
      })
    }
  })

  // Auto-start monitoring when import starts
  useEffect(() => {
    if (lastJob && lastJob.status === 'running') {
      setIsMonitoring(true)
      
      // Stop monitoring after 2 minutes
      const timeout = setTimeout(() => {
        setIsMonitoring(false)
      }, 120000)
      
      return () => clearTimeout(timeout)
    }
  }, [lastJob])

  const importMutation = useMutation({
    mutationFn: importDataset,
    onSuccess: (data) => {
      setLastJob(data)
      addToast({
        type: 'info',
        title: 'Dataset Import Started',
        message: `Started importing ${limit} images from ${data.dataset}. Job ID: ${data.job_id}`,
        duration: 6000
      })
      // Refresh dashboard stats
      queryClient.invalidateQueries({ queryKey: ['dataset-stats'] })
      queryClient.invalidateQueries({ queryKey: ['category-stats'] })
    },
    onError: (error: any) => {
      addToast({
        type: 'error',
        title: 'Import Failed to Start',
        message: error.message || 'Failed to start dataset import. Check the dataset URL and try again.',
        duration: 8000
      })
      console.error('Import failed:', error)
    }
  })

  const handleImport = () => {
    // Validation
    if (!datasetUrl || (selectedDataset === 'custom' && !customDataset.trim())) {
      addToast({
        type: 'error',
        title: 'Invalid Dataset',
        message: 'Please select a dataset or enter a custom HuggingFace dataset ID',
        duration: 5000
      })
      return
    }

    if (limit < 1 || limit > 100) {
      addToast({
        type: 'error',
        title: 'Invalid Limit',
        message: 'Number of images must be between 1 and 100',
        duration: 5000
      })
      return
    }

    importMutation.mutate({
      dataset: datasetUrl,
      offset,
      limit,
      include_detection: includeDetection
    })
  }

  const datasetUrl = selectedDataset === 'custom' ? customDataset : selectedDataset

  return (
    <div className="space-y-6">
      {/* Dataset Selection */}
      <div className="space-y-4">
        <h3 className="font-semibold text-slate-800 flex items-center">
          <Database className="h-5 w-5 mr-2" />
          Select Dataset
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {PRESET_DATASETS.map((dataset) => (
            <div key={dataset.name} className="relative">
              <label className={`block p-4 border-2 rounded-lg cursor-pointer transition-all ${
                selectedDataset === dataset.url
                  ? 'border-blue-500 bg-blue-50' 
                  : 'border-slate-200 hover:border-slate-300'
              }`}>
                <input
                  type="radio"
                  name="dataset"
                  value={dataset.url}
                  checked={selectedDataset === dataset.url}
                  onChange={(e) => setSelectedDataset(e.target.value)}
                  className="sr-only"
                />
                <div className="flex items-start justify-between">
                  <div>
                    <h4 className="font-medium text-slate-800">{dataset.name}</h4>
                    <p className="text-sm text-slate-600 mt-1">{dataset.description}</p>
                  </div>
                  {dataset.recommended && (
                    <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                      <Sparkles className="h-3 w-3 mr-1" />
                      Recommended
                    </span>
                  )}
                </div>
              </label>
            </div>
          ))}
        </div>

        {selectedDataset === 'custom' && (
          <div className="ml-4">
            <label className="block text-sm font-medium text-slate-700 mb-2">
              HuggingFace Dataset ID
            </label>
            <input
              type="text"
              value={customDataset}
              onChange={(e) => setCustomDataset(e.target.value)}
              placeholder="e.g. username/dataset-name"
              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
        )}
      </div>

      {/* Import Parameters */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Starting Offset
          </label>
          <input
            type="number"
            value={offset}
            onChange={(e) => setOffset(parseInt(e.target.value) || 0)}
            min="0"
            className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Number of Images
          </label>
          <input
            type="number"
            value={limit}
            onChange={(e) => setLimit(parseInt(e.target.value) || 10)}
            min="1"
            max="100"
            className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        
        <div className="flex items-center">
          <label className="flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={includeDetection}
              onChange={(e) => setIncludeDetection(e.target.checked)}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-slate-300 rounded"
            />
            <span className="ml-3 text-sm font-medium text-slate-700">
              Run AI Detection
            </span>
          </label>
        </div>
      </div>

      {/* Import Button */}
      <div className="flex items-center justify-between">
        <div className="text-sm text-slate-600">
          {includeDetection ? (
            <span className="flex items-center">
              <Sparkles className="h-4 w-4 mr-1" />
              Will import images and run AI object detection
            </span>
          ) : (
            <span className="flex items-center">
              <AlertTriangle className="h-4 w-4 mr-1" />
              Import only - no AI processing
            </span>
          )}
        </div>
        
        <button
          onClick={handleImport}
          disabled={importMutation.isPending || !datasetUrl}
          className="inline-flex items-center px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {importMutation.isPending ? (
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
          ) : (
            <Play className="h-4 w-4 mr-2" />
          )}
          {importMutation.isPending ? 'Importing...' : 'Import Dataset'}
        </button>
      </div>

      {/* Last Job Status */}
      {lastJob && (
        <div className="space-y-4">
          <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <h4 className="font-medium text-blue-900 mb-2">Latest Import Job</h4>
            <div className="text-sm space-y-1">
              <p><span className="font-medium">Job ID:</span> {lastJob.job_id}</p>
              <p><span className="font-medium">Status:</span> {lastJob.status}</p>
              <p><span className="font-medium">Dataset:</span> {lastJob.dataset}</p>
              <p><span className="font-medium">Message:</span> {lastJob.message}</p>
            </div>
          </div>

          {/* Job Progress */}
          {jobProgress && (
            <div className="p-4 bg-white border border-slate-200 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-medium text-slate-900">Import Progress</h4>
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  jobProgress.status === 'completed' ? 'bg-green-100 text-green-800' :
                  (jobProgress.status === 'failed' || jobProgress.status === 'error') ? 'bg-red-100 text-red-800' :
                  jobProgress.status === 'processing' ? 'bg-blue-100 text-blue-800' :
                  'bg-gray-100 text-gray-800'
                }`}>
                  {jobProgress.status}
                </span>
              </div>
              
              {/* Progress bar */}
              <div className="mb-2">
                <div className="flex items-center justify-between text-sm text-slate-600 mb-1">
                  <span>{jobProgress.processed || 0} of {jobProgress.total || 0} images processed</span>
                  <span>{jobProgress.progress || 0}%</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full transition-all duration-300 ${
                      jobProgress.status === 'completed' ? 'bg-green-500' :
                      (jobProgress.status === 'failed' || jobProgress.status === 'error') ? 'bg-red-500' :
                      'bg-blue-500'
                    }`}
                    style={{ width: `${Math.min(100, Math.max(0, jobProgress.progress || 0))}%` }}
                  />
                </div>
              </div>
              
              {/* Status message */}
              {jobProgress.message && (
                <p className="text-sm text-slate-600">{jobProgress.message}</p>
              )}
              
              {/* Completion notification */}
              {jobProgress.status === 'completed' && (
                <div className="mt-3 p-3 bg-green-50 border border-green-200 rounded-lg">
                  <div className="flex items-center">
                    <CheckCircle className="h-5 w-5 text-green-600 mr-2" />
                    <span className="text-sm font-medium text-green-800">
                      Import completed successfully! 
                    </span>
                  </div>
                </div>
              )}
              
              {(jobProgress.status === 'failed' || jobProgress.status === 'error') && (
                <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg">
                  <div className="flex items-start">
                    <AlertTriangle className="h-5 w-5 text-red-600 mr-2 mt-0.5 flex-shrink-0" />
                    <div className="flex-1">
                      <span className="text-sm font-medium text-red-800 block mb-1">
                        Import failed {jobProgress.processed ? `after processing ${jobProgress.processed}/${jobProgress.total} images` : ''}
                      </span>
                      {(jobProgress.error_message || jobProgress.message) && (
                        <div className="text-sm text-red-700 bg-red-100 p-2 rounded border">
                          <strong>Error:</strong> {jobProgress.error_message || jobProgress.message}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Recent Errors Display */}
      {recentErrors && recentErrors.errors && recentErrors.errors.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="font-medium text-red-900 flex items-center">
              <AlertCircle className="h-5 w-5 mr-2" />
              Recent Import Errors ({recentErrors.total_error_jobs})
            </h4>
            <button
              onClick={() => refetchErrors()}
              className="text-sm text-red-600 hover:text-red-800"
            >
              Refresh
            </button>
          </div>
          
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {recentErrors.errors.slice(0, 5).map((error: any, index: number) => (
              <div key={error.job_id} className="p-3 bg-red-50 border border-red-200 rounded-lg">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="text-sm font-medium text-red-800">
                        Job #{error.job_id.slice(-8)}
                      </span>
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                        error.status === 'failed' ? 'bg-red-100 text-red-800' : 'bg-orange-100 text-orange-800'
                      }`}>
                        {error.status}
                      </span>
                    </div>
                    
                    {error.error_message && (
                      <div className="text-sm text-red-700 bg-red-100 p-2 rounded mb-2">
                        <strong>Error:</strong> {error.error_message}
                      </div>
                    )}
                    
                    <div className="text-xs text-red-600 space-y-1">
                      <div>Progress: {error.processed}/{error.total} images processed</div>
                      {error.updated_at && (
                        <div>Failed: {new Date(error.updated_at).toLocaleString()}</div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Import Error Display */}
      {importMutation.error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
          <h4 className="font-medium text-red-900 mb-2 flex items-center">
            <AlertTriangle className="h-5 w-5 mr-2" />
            Failed to Start Import
          </h4>
          <p className="text-sm text-red-700">
            {importMutation.error.message}
          </p>
        </div>
      )}

      {/* Live Monitoring */}
      {isMonitoring && (
        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-center">
              <Loader2 className="h-5 w-5 text-blue-600 animate-spin mr-2" />
              <span className="font-medium text-blue-900">Monitoring Import Progress...</span>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => exportMutation.mutate()}
                disabled={exportMutation.isPending}
                className="inline-flex items-center px-3 py-1 text-sm bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
              >
                {exportMutation.isPending ? (
                  <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                ) : (
                  <Download className="h-3 w-3 mr-1" />
                )}
                Export Dataset
              </button>
              <button
                onClick={() => setShowGallery(true)}
                className="inline-flex items-center px-3 py-1 text-sm bg-purple-600 text-white rounded hover:bg-purple-700"
              >
                <ImageIcon className="h-3 w-3 mr-1" />
                View Gallery
              </button>
              <button
                onClick={() => setIsMonitoring(false)}
                className="text-sm text-blue-600 hover:text-blue-800"
              >
                Stop Monitoring
              </button>
            </div>
          </div>
          
          {/* View Mode Toggle */}
          <div className="flex items-center gap-4 p-3 bg-white border border-slate-200 rounded-lg">
            <span className="text-sm font-medium text-slate-700">View:</span>
            <div className="flex bg-slate-100 rounded-lg p-1">
              <button
                onClick={() => setViewMode('scenes')}
                className={`px-3 py-1 text-sm rounded ${
                  viewMode === 'scenes'
                    ? 'bg-white text-blue-600 shadow-sm'
                    : 'text-slate-600 hover:text-slate-800'
                }`}
              >
                <Grid className="h-4 w-4 mr-1 inline" />
                Scenes
              </button>
              <button
                onClick={() => setViewMode('objects')}
                className={`px-3 py-1 text-sm rounded ${
                  viewMode === 'objects'
                    ? 'bg-white text-purple-600 shadow-sm'
                    : 'text-slate-600 hover:text-slate-800'
                }`}
              >
                <Target className="h-4 w-4 mr-1 inline" />
                Objects
              </button>
            </div>
          </div>

          {/* Live Stats */}
          {liveStats && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-4 bg-white border border-slate-200 rounded-lg">
                <div className="flex items-center">
                  <Database className="h-5 w-5 text-green-600 mr-2" />
                  <div>
                    <p className="text-sm text-slate-600">Total Scenes</p>
                    <p className="text-lg font-bold text-slate-800">{liveStats.total_scenes || 0}</p>
                  </div>
                </div>
              </div>
              <div className="p-4 bg-white border border-slate-200 rounded-lg">
                <div className="flex items-center">
                  <Target className="h-5 w-5 text-purple-600 mr-2" />
                  <div>
                    <p className="text-sm text-slate-600">Objects Detected</p>
                    <p className="text-lg font-bold text-slate-800">{liveStats.total_objects || 0}</p>
                  </div>
                </div>
              </div>
              <div className="p-4 bg-white border border-slate-200 rounded-lg">
                <div className="flex items-center">
                  <CheckCircle className="h-5 w-5 text-blue-600 mr-2" />
                  <div>
                    <p className="text-sm text-slate-600">Avg Confidence</p>
                    <p className="text-lg font-bold text-slate-800">
                      {liveStats.avg_confidence ? (liveStats.avg_confidence * 100).toFixed(1) + '%' : '0%'}
                    </p>
                  </div>
                </div>
              </div>
              <div className="p-4 bg-white border border-slate-200 rounded-lg">
                <div className="flex items-center">
                  <Sparkles className="h-5 w-5 text-orange-600 mr-2" />
                  <div>
                    <p className="text-sm text-slate-600">Categories Found</p>
                    <p className="text-lg font-bold text-slate-800">{categoryStats?.length || 0}</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Object Category Breakdown */}
          {categoryStats && categoryStats.length > 0 && (
            <div className="bg-white border border-slate-200 rounded-lg p-4">
              <div className="flex items-center justify-between mb-4">
                <h4 className="font-medium text-slate-800 flex items-center">
                  <BarChart3 className="h-5 w-5 mr-2" />
                  Object Categories
                </h4>
                <button
                  onClick={() => setSelectedCategory(null)}
                  className={`text-xs px-2 py-1 rounded ${
                    !selectedCategory ? 'bg-blue-100 text-blue-800' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                  }`}
                >
                  All
                </button>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {categoryStats.slice(0, 8).map((category: any) => (
                  <button
                    key={category.category}
                    onClick={() => setSelectedCategory(category.category === selectedCategory ? null : category.category)}
                    className={`p-2 text-left rounded-lg border transition-all ${
                      selectedCategory === category.category
                        ? 'border-purple-500 bg-purple-50 text-purple-800'
                        : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50'
                    }`}
                  >
                    <div className="text-xs font-medium truncate">{category.category}</div>
                    <div className="text-lg font-bold">{category.total_objects}</div>
                    <div className="text-xs text-slate-500">
                      {(category.avg_confidence * 100).toFixed(0)}% conf
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Recent Images or Objects */}
          {viewMode === 'scenes' && scenes && scenes.scenes && scenes.scenes.length > 0 && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h4 className="font-medium text-slate-800 flex items-center">
                  <Eye className="h-5 w-5 mr-2" />
                  Recent Images ({scenes.total} total)
                </h4>
                <button
                  onClick={() => refetchScenes()}
                  className="text-sm text-blue-600 hover:text-blue-800"
                >
                  Refresh
                </button>
              </div>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {scenes.scenes.slice(0, 8).map((scene: any) => (
                  <div key={scene.scene_id} className="bg-white border border-slate-200 rounded-lg p-3">
                    <div className="relative">
                      <img
                        src={scene.image_url}
                        alt={scene.houzz_id}
                        className="w-full h-20 object-cover rounded mb-2"
                        onError={(e) => {
                          (e.target as HTMLImageElement).src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiBmaWxsPSIjZjMzNDU2Ii8+Cjx0ZXh0IHg9IjEyIiB5PSIxMiIgZmlsbD0iI2ZmZmZmZiIgZm9udC1zaXplPSI4IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+4p2MPC90ZXh0Pgo8L3N2Zz4K'
                        }}
                      />
                      {/* Image Type Badge */}
                      {scene.image_type && (
                        <span className={`absolute top-1 right-1 px-1 py-0.5 text-xs font-medium rounded ${
                          scene.image_type === 'scene' ? 'bg-blue-100 text-blue-800' :
                          scene.image_type === 'object' ? 'bg-purple-100 text-purple-800' :
                          scene.image_type === 'hybrid' ? 'bg-orange-100 text-orange-800' :
                          scene.image_type === 'product' ? 'bg-green-100 text-green-800' :
                          'bg-gray-100 text-gray-800'
                        }`}>
                          {scene.image_type}
                        </span>
                      )}
                    </div>
                    
                    <div className="space-y-1">
                      <p className="text-xs text-slate-600 truncate">{scene.houzz_id}</p>
                      
                      {/* Room Type & Object Count */}
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-500">
                          {scene.metadata?.detected_room_type || scene.room_type}
                        </span>
                        <span className="flex items-center text-green-600">
                          <CheckCircle className="h-3 w-3 mr-1" />
                          {scene.object_count || 0}
                        </span>
                      </div>
                      
                      {/* Primary Category for Object/Hybrid types */}
                      {scene.primary_category && (
                        <p className="text-xs text-purple-600 truncate">
                          {scene.primary_category.replace(/_/g, ' ')}
                        </p>
                      )}
                      
                      {/* Detected Styles */}
                      {scene.metadata?.detected_styles && scene.metadata.detected_styles.length > 0 && (
                        <p className="text-xs text-pink-600 truncate">
                          {scene.metadata.detected_styles[0]}
                        </p>
                      )}
                      
                      {/* Style Tags (fallback) */}
                      {(!scene.metadata?.detected_styles || scene.metadata.detected_styles.length === 0) && 
                       scene.style_tags && scene.style_tags.length > 0 && (
                        <p className="text-xs text-blue-600 truncate">
                          {scene.style_tags[0]}
                        </p>
                      )}
                      
                      {/* Classification Confidence */}
                      {scene.metadata?.classification_confidence && (
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-slate-400">Confidence</span>
                          <span className={`font-medium ${
                            scene.metadata.classification_confidence >= 0.8 ? 'text-green-600' :
                            scene.metadata.classification_confidence >= 0.6 ? 'text-yellow-600' :
                            'text-red-600'
                          }`}>
                            {(scene.metadata.classification_confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Objects Gallery */}
          {viewMode === 'objects' && objects && objects.objects && objects.objects.length > 0 && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h4 className="font-medium text-slate-800 flex items-center">
                  <Target className="h-5 w-5 mr-2" />
                  Detected Objects ({objects.total} total)
                  {selectedCategory && (
                    <span className="ml-2 px-2 py-1 text-xs bg-purple-100 text-purple-800 rounded">
                      {selectedCategory}
                    </span>
                  )}
                </h4>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => refetchObjects()}
                    className="text-sm text-blue-600 hover:text-blue-800"
                  >
                    Refresh
                  </button>
                  <button
                    onClick={() => setSelectedCategory(null)}
                    className="text-sm text-purple-600 hover:text-purple-800"
                    disabled={!selectedCategory}
                  >
                    Clear Filter
                  </button>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {objects.objects.slice(0, 12).map((obj: any) => (
                  <div key={obj.object_id} className="bg-white border border-slate-200 rounded-lg p-3">
                    <div className="relative mb-3">
                      {obj.scene_info && (
                        <img
                          src={obj.scene_info.image_url}
                          alt={`Object in ${obj.scene_info.houzz_id}`}
                          className="w-full h-32 object-cover rounded"
                          onError={(e) => {
                            (e.target as HTMLImageElement).src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiBmaWxsPSIjZjMzNDU2Ii8+Cjx0ZXh0IHg9IjEyIiB5PSIxMiIgZmlsbD0iI2ZmZmZmZiIgZm9udC1zaXplPSI4IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+4p2MPC90ZXh0Pgo8L3N2Zz4K'
                          }}
                        />
                      )}
                      {/* Bounding Box Overlay Simulation */}
                      {obj.bbox && (
                        <div className="absolute inset-0 pointer-events-none">
                          <div 
                            className="absolute border-2 border-purple-500 bg-purple-500 bg-opacity-10 rounded"
                            style={{
                              left: `${(obj.bbox[0] / (obj.scene_info?.width || 800)) * 100}%`,
                              top: `${(obj.bbox[1] / (obj.scene_info?.height || 600)) * 100}%`,
                              width: `${(obj.bbox[2] / (obj.scene_info?.width || 800)) * 100}%`,
                              height: `${(obj.bbox[3] / (obj.scene_info?.height || 600)) * 100}%`,
                            }}
                          />
                        </div>
                      )}
                    </div>
                    
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-purple-800 bg-purple-100 px-2 py-1 rounded">
                          {obj.category}
                        </span>
                        <span className="text-xs text-green-600 font-medium">
                          {(obj.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      
                      {obj.scene_info && (
                        <div className="text-xs text-slate-600">
                          <p className="truncate">Scene: {obj.scene_info.houzz_id}</p>
                          <p>Room: {obj.scene_info.room_type}</p>
                        </div>
                      )}
                      
                      {obj.tags && obj.tags.length > 0 && (
                        <div className="flex flex-wrap gap-1">
                          {obj.tags.slice(0, 3).map((tag: string, idx: number) => (
                            <span key={idx} className="text-xs bg-slate-100 text-slate-600 px-2 py-1 rounded">
                              {tag}
                            </span>
                          ))}
                        </div>
                      )}

                      {obj.metadata?.colors && (
                        <div className="flex items-center gap-1">
                          <Palette className="h-3 w-3 text-slate-400" />
                          {obj.metadata.colors.colors?.slice(0, 3).map((color: any, idx: number) => (
                            <div
                              key={idx}
                              className="w-3 h-3 rounded-full border border-slate-300"
                              style={{ backgroundColor: color.hex }}
                              title={color.name}
                            />
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Processed Images Gallery */}
      {showGallery && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg max-w-6xl max-h-[90vh] w-full mx-4 overflow-hidden">
            {/* Gallery Header */}
            <div className="flex items-center justify-between p-6 border-b border-slate-200">
              <h2 className="text-xl font-semibold text-slate-900 flex items-center">
                <ImageIcon className="h-6 w-6 mr-2" />
                Processed Images Gallery
                {galleryData && (
                  <span className="ml-2 text-sm font-normal text-slate-600">
                    ({galleryData.total} total images)
                  </span>
                )}
              </h2>
              <button
                onClick={() => {
                  setShowGallery(false)
                  setGalleryPage(1)
                }}
                className="text-slate-400 hover:text-slate-600"
              >
                <ArrowLeft className="h-6 w-6" />
              </button>
            </div>

            {/* Gallery Content */}
            <div className="p-6 overflow-y-auto max-h-[calc(90vh-8rem)]">
              {galleryLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
                  <span className="ml-2 text-slate-600">Loading gallery...</span>
                </div>
              ) : !galleryData || galleryData.scenes.length === 0 ? (
                <div className="text-center py-12">
                  <ImageIcon className="h-16 w-16 text-slate-300 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-slate-900 mb-2">No processed images</h3>
                  <p className="text-slate-600">Import some datasets to see processed images here.</p>
                </div>
              ) : (
                <div className="space-y-6">
                  {/* Gallery Grid */}
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {galleryData?.scenes.map((scene: any) => (
                      <div key={scene.scene_id} className="bg-white border border-slate-200 rounded-lg overflow-hidden shadow-sm hover:shadow-md transition-shadow">
                        {/* Image */}
                        <div className="relative">
                          <img
                            src={scene.image_url}
                            alt={scene.houzz_id}
                            className="w-full h-48 object-cover"
                            onError={(e) => {
                              (e.target as HTMLImageElement).src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiBmaWxsPSIjZjMzNDU2Ii8+Cjx0ZXh0IHg9IjEyIiB5PSIxMiIgZmlsbD0iI2ZmZmZmZiIgZm9udC1zaXplPSI4IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+4p2MPC90ZXh0Pgo8L3N2Zz4K'
                            }}
                          />
                          {/* Status badge */}
                          <div className="absolute top-2 right-2">
                            <span className={`px-2 py-1 rounded text-xs font-medium ${
                              scene.status === 'scraped' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                            }`}>
                              {scene.status}
                            </span>
                          </div>
                          {/* Objects count overlay */}
                          {scene.object_count > 0 && (
                            <div className="absolute bottom-2 right-2 bg-purple-600 text-white px-2 py-1 rounded text-xs font-medium flex items-center">
                              <Target className="h-3 w-3 mr-1" />
                              {scene.object_count} objects
                            </div>
                          )}
                        </div>

                        {/* Image Details */}
                        <div className="p-4">
                          <h4 className="font-medium text-slate-900 truncate mb-2">{scene.houzz_id}</h4>
                          
                          <div className="space-y-2 text-sm text-slate-600">
                            <div className="flex items-center justify-between">
                              <span>Room Type:</span>
                              <span className="font-medium">{scene.room_type}</span>
                            </div>
                            
                            {scene.style_tags && scene.style_tags.length > 0 && (
                              <div>
                                <span className="block mb-1">Style Tags:</span>
                                <div className="flex flex-wrap gap-1">
                                  {scene.style_tags.slice(0, 2).map((tag: string, idx: number) => (
                                    <span key={idx} className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
                                      {tag}
                                    </span>
                                  ))}
                                  {scene.style_tags.length > 2 && (
                                    <span className="text-xs text-slate-500">
                                      +{scene.style_tags.length - 2} more
                                    </span>
                                  )}
                                </div>
                              </div>
                            )}

                            {scene.image_r2_key && (
                              <div className="flex items-center text-green-600">
                                <CheckCircle className="h-4 w-4 mr-1" />
                                <span className="text-xs">Stored in R2</span>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Pagination */}
                  {galleryData && galleryData.totalPages > 1 && (
                    <div className="flex items-center justify-center space-x-4 pt-6 border-t border-slate-200">
                      <button
                        onClick={() => setGalleryPage(Math.max(1, galleryPage - 1))}
                        disabled={galleryPage === 1}
                        className="flex items-center px-3 py-2 text-sm border border-slate-300 rounded-md hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        <ChevronLeft className="h-4 w-4 mr-1" />
                        Previous
                      </button>
                      
                      <span className="text-sm text-slate-600">
                        Page {galleryData.page} of {galleryData.totalPages}
                      </span>
                      
                      <button
                        onClick={() => setGalleryPage(Math.min(galleryData.totalPages, galleryPage + 1))}
                        disabled={galleryPage === galleryData.totalPages}
                        className="flex items-center px-3 py-2 text-sm border border-slate-300 rounded-md hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        Next
                        <ChevronRight className="h-4 w-4 ml-1" />
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Toast Notifications */}
      {toasts.length > 0 && (
        <div className="fixed top-4 right-4 z-50 space-y-2">
          {toasts.map((toast) => (
            <div
              key={toast.id}
              className={`max-w-sm w-full bg-white shadow-lg rounded-lg pointer-events-auto ring-1 ring-black ring-opacity-5 overflow-hidden transform transition-all duration-300 ease-in-out ${
                toast.type === 'success' ? 'border-l-4 border-green-500' :
                toast.type === 'error' ? 'border-l-4 border-red-500' :
                toast.type === 'warning' ? 'border-l-4 border-yellow-500' :
                'border-l-4 border-blue-500'
              }`}
            >
              <div className="p-4">
                <div className="flex items-start">
                  <div className="flex-shrink-0">
                    {toast.type === 'success' && <CheckCircle className="h-5 w-5 text-green-400" />}
                    {toast.type === 'error' && <AlertCircle className="h-5 w-5 text-red-400" />}
                    {toast.type === 'warning' && <AlertTriangle className="h-5 w-5 text-yellow-400" />}
                    {toast.type === 'info' && <Info className="h-5 w-5 text-blue-400" />}
                  </div>
                  <div className="ml-3 w-0 flex-1">
                    <p className="text-sm font-medium text-gray-900">{toast.title}</p>
                    <p className="mt-1 text-sm text-gray-500">{toast.message}</p>
                  </div>
                  <div className="ml-4 flex-shrink-0 flex">
                    <button
                      className="bg-white rounded-md inline-flex text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                      onClick={() => removeToast(toast.id)}
                    >
                      <span className="sr-only">Close</span>
                      <X className="h-5 w-5" />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}