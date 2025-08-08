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
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  return (
    <div className="space-y-3">
      {jobs.map((job) => (
        <div key={job.job_id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
          <div className="flex-1">
            <div className="flex items-center space-x-3">
              <span className="font-medium capitalize">{job.job_type} Job</span>
              <span className={`px-2 py-1 text-xs rounded-full ${getStatusColor(job.status)}`}>
                {job.status}
              </span>
            </div>
            <div className="text-sm text-gray-600 mt-1">
              {job.processed_items}/{job.total_items} items processed
            </div>
          </div>
          
          <div className="flex-1 mx-4">
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${job.progress}%` }}
              ></div>
            </div>
            <div className="text-xs text-gray-500 mt-1 text-center">
              {job.progress}%
            </div>
          </div>
          
          <div className="text-right">
            <div className="text-xs text-gray-500">
              Started: {new Date(job.created_at).toLocaleTimeString()}
            </div>
            {job.error_message && (
              <div className="text-xs text-red-600 mt-1">
                Error: {job.error_message}
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}