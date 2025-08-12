import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { 
  Play, Loader2, CheckCircle, XCircle, AlertTriangle, Clock, 
  Pause, RefreshCw, Filter, Search, Calendar, Activity,
  Database, Sparkles, Eye, Download, Settings, Zap, Palette
} from 'lucide-react'

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
  const response = await fetch('https://ovalay-recruitment-production.up.railway.app/jobs/active')
  if (!response.ok) throw new Error('Failed to fetch active jobs')
  return response.json()
}

async function fetchJobStatus(jobId: string): Promise<Job> {
  const response = await fetch(`https://ovalay-recruitment-production.up.railway.app/jobs/${jobId}/status`)
  if (!response.ok) throw new Error('Failed to fetch job status')
  return response.json()
}

async function fetchRecentErrors(): Promise<{errors: Job[], total_error_jobs: number}> {
  const response = await fetch('https://ovalay-recruitment-production.up.railway.app/jobs/errors/recent')
  if (!response.ok) throw new Error('Failed to fetch recent errors')
  return response.json()
}

async function fetchAllJobs(): Promise<Job[]> {
  try {
    // Fetch both active jobs and recent errors, then combine them
    const [activeJobs, errorData] = await Promise.all([
      fetchActiveJobs().catch(() => []),
      fetchRecentErrors().catch(() => ({errors: [], total_error_jobs: 0}))
    ])
    
    // Combine and deduplicate by job_id
    const allJobs = [...activeJobs, ...errorData.errors]
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
    return []
  }
}

export function Jobs() {
  const [filterStatus, setFilterStatus] = useState<string>('all')
  const [searchTerm, setSearchTerm] = useState('')
  const [refreshInterval, setRefreshInterval] = useState(5000) // 5 seconds default
  const [isAutoRefresh, setIsAutoRefresh] = useState(true)

  // Fetch all jobs with real-time updates
  const { data: jobs = [], isLoading, error, refetch } = useQuery({
    queryKey: ['all-jobs'],
    queryFn: fetchAllJobs,
    refetchInterval: isAutoRefresh ? refreshInterval : false,
    refetchIntervalInBackground: true
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

  const getJobTypeIcon = (message: string) => {
    const msgLower = message.toLowerCase()
    if (msgLower.includes('dataset') || msgLower.includes('import')) return <Database className="h-5 w-5" />
    if (msgLower.includes('detection') || msgLower.includes('ai')) return <Sparkles className="h-5 w-5" />
    if (msgLower.includes('scraping') || msgLower.includes('scrape')) return <Eye className="h-5 w-5" />
    if (msgLower.includes('color')) return <Palette className="h-5 w-5" />
    if (msgLower.includes('export')) return <Download className="h-5 w-5" />
    return <Activity className="h-5 w-5" />
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
                    {getJobTypeIcon(job.message)}
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
                    </div>
                  </div>
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