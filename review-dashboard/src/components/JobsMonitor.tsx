import React from 'react'
import { ScrapingJob } from '../types'

interface JobsMonitorProps {
  jobs: ScrapingJob[]
}

export function JobsMonitor({ jobs }: JobsMonitorProps) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'text-blue-600 bg-blue-100'
      case 'completed': return 'text-green-600 bg-green-100'
      case 'failed': return 'text-red-600 bg-red-100'
      case 'pending': return 'text-yellow-600 bg-yellow-100'
      case 'processing': return 'text-blue-600 bg-blue-100'
      case 'cancelled': return 'text-gray-600 bg-gray-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  const getProcessingModeDisplay = (job: ScrapingJob) => {
    // Check if this is a detection job (has _detection suffix)
    if (job.job_id.includes('_detection')) {
      return { mode: 'AI Detection', icon: '🤖', color: 'text-purple-600' }
    }
    
    // Check job type and parameters for processing mode hints
    const jobType = job.job_type || 'processing'
    if (jobType === 'import') {
      return { mode: 'Import', icon: '📥', color: 'text-blue-600' }
    } else if (jobType === 'scenes') {
      return { mode: 'Scraping', icon: '🕷️', color: 'text-green-600' }
    } else if (jobType === 'detection') {
      return { mode: 'AI Detection', icon: '🤖', color: 'text-purple-600' }
    }
    
    return { mode: jobType, icon: '⚙️', color: 'text-gray-600' }
  }

  return (
    <div className="space-y-3">
      {Array.isArray(jobs) ? jobs.map((job) => {
        const processingInfo = getProcessingModeDisplay(job)
        
        return (
          <div key={job.job_id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
            <div className="flex-1">
              <div className="flex items-center space-x-3">
                <span className={`text-lg ${processingInfo.color}`}>{processingInfo.icon}</span>
                <div>
                  <div className="flex items-center space-x-2">
                    <span className="font-medium">{processingInfo.mode}</span>
                    <span className={`px-2 py-1 text-xs rounded-full ${getStatusColor(job.status)}`}>
                      {job.status}
                    </span>
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    Job #{job.job_id.slice(-8)}
                  </div>
                </div>
              </div>
              <div className="text-sm text-gray-600 mt-2">
                {job.processed_items || 0}/{job.total_items || 0} items • {job.progress || 0}% complete
              </div>
              
              {/* Show dataset info if available */}
              {(job as any).dataset && (
                <div className="text-xs text-gray-500 mt-1">
                  Dataset: {(job as any).dataset}
                </div>
              )}
            </div>
            
            <div className="flex-1 mx-4">
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all duration-300 ${
                    job.status === 'completed' ? 'bg-green-500' :
                    job.status === 'failed' ? 'bg-red-500' :
                    processingInfo.mode === 'AI Detection' ? 'bg-purple-500' :
                    'bg-blue-500'
                  }`}
                  style={{ width: `${Math.min(100, Math.max(0, job.progress || 0))}%` }}
                ></div>
              </div>
              <div className="text-xs text-gray-500 mt-1 text-center">
                {job.progress || 0}%
              </div>
            </div>
            
            <div className="text-right">
              <div className="text-xs text-gray-500">
                {job.created_at ? new Date(job.created_at).toLocaleTimeString() : 'Unknown'}
              </div>
              {job.error_message && (
                <div className="text-xs text-red-600 mt-1 max-w-xs truncate" title={job.error_message}>
                  Error: {job.error_message}
                </div>
              )}
              
              {/* Show processing stage hint */}
              {job.status === 'running' && (
                <div className="text-xs text-blue-600 mt-1">
                  {processingInfo.mode === 'Import' ? 'Downloading...' :
                   processingInfo.mode === 'Scraping' ? 'Crawling...' :
                   processingInfo.mode === 'AI Detection' ? 'Analyzing...' :
                   'Processing...'}
                </div>
              )}
            </div>
          </div>
        )
      }) : (
        <div className="text-center text-gray-500 py-4">
          No active jobs
        </div>
      )}
    </div>
  )
}