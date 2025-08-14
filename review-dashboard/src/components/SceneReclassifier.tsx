import React, { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { 
  RefreshCw, 
  Loader2, 
  CheckCircle, 
  AlertTriangle, 
  BarChart3,
  Eye,
  Target,
  Info,
  Play,
  Settings
} from 'lucide-react'
import { startSceneReclassification, getJobStatus, getDatasetStats } from '../api/client'

export function SceneReclassifier() {
  const [limit, setLimit] = useState(100)
  const [forceRedetection, setForceRedetection] = useState(false)
  const [lastJobId, setLastJobId] = useState<string | null>(null)
  const [isMonitoring, setIsMonitoring] = useState(false)

  const queryClient = useQueryClient()

  // Monitor current stats
  const { data: currentStats, refetch: refetchStats } = useQuery({
    queryKey: ['dataset-stats-reclassify'],
    queryFn: getDatasetStats,
    refetchInterval: isMonitoring ? 5000 : false
  })

  // Monitor job progress
  const { data: jobProgress } = useQuery({
    queryKey: ['reclassify-job-status', lastJobId],
    queryFn: () => getJobStatus(lastJobId!),
    enabled: !!lastJobId && isMonitoring,
    refetchInterval: !!lastJobId && isMonitoring ? 3000 : false,
    onSuccess: (data) => {
      if (data.status === 'completed') {
        setIsMonitoring(false)
        // Refresh stats after completion
        refetchStats()
        queryClient.invalidateQueries({ queryKey: ['scenes'] })
      } else if (data.status === 'failed' || data.status === 'error') {
        setIsMonitoring(false)
      }
    }
  })

  const reclassifyMutation = useMutation({
    mutationFn: startSceneReclassification,
    onSuccess: (data) => {
      setLastJobId(data.job_id)
      setIsMonitoring(true)
    }
  })

  const handleReclassify = () => {
    reclassifyMutation.mutate({
      limit,
      force_redetection: forceRedetection
    })
  }

  const getJobStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-800 bg-green-100'
      case 'failed':
      case 'error': return 'text-red-800 bg-red-100'
      case 'running':
      case 'processing': return 'text-blue-800 bg-blue-100'
      default: return 'text-gray-800 bg-gray-100'
    }
  }

  return (
    <div className="bg-white rounded-xl p-8 shadow-sm border border-slate-200/60">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center">
          <RefreshCw className="h-6 w-6 text-blue-600 mr-3" />
          <div>
            <h2 className="text-2xl font-bold text-slate-800">Scene Reclassification</h2>
            <p className="text-slate-600 text-sm mt-1">
              Improve dataset quality using enhanced scene vs object detection
            </p>
          </div>
        </div>
        {currentStats && (
          <div className="text-right">
            <div className="text-2xl font-bold text-slate-800">{currentStats.total_scenes}</div>
            <div className="text-sm text-slate-600">Total Scenes</div>
          </div>
        )}
      </div>

      {/* Current Stats Overview */}
      {currentStats?.scenes_by_type && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6 p-4 bg-slate-50 rounded-lg">
          <div className="text-center">
            <div className="text-lg font-bold text-blue-600">{currentStats.scenes_by_type.scene}</div>
            <div className="text-xs text-slate-600">Scenes</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-purple-600">{currentStats.scenes_by_type.object}</div>
            <div className="text-xs text-slate-600">Objects</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-orange-600">{currentStats.scenes_by_type.hybrid}</div>
            <div className="text-xs text-slate-600">Hybrid</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-green-600">{currentStats.scenes_by_type.product}</div>
            <div className="text-xs text-slate-600">Products</div>
          </div>
        </div>
      )}

      {/* Configuration */}
      <div className="space-y-4 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Number of Scenes to Reclassify
            </label>
            <input
              type="number"
              value={limit}
              onChange={(e) => setLimit(parseInt(e.target.value) || 100)}
              min="1"
              max="1000"
              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
            <p className="text-xs text-slate-500 mt-1">
              Start with a smaller number to test improvements
            </p>
          </div>

          <div className="flex items-center">
            <label className="flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={forceRedetection}
                onChange={(e) => setForceRedetection(e.target.checked)}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-slate-300 rounded"
              />
              <div className="ml-3">
                <div className="text-sm font-medium text-slate-700">Force Re-detection</div>
                <div className="text-xs text-slate-500">
                  Re-run AI object detection for better classification
                </div>
              </div>
            </label>
          </div>
        </div>

        <div className="flex items-start p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <Info className="h-5 w-5 text-blue-600 mr-2 mt-0.5 flex-shrink-0" />
          <div className="text-sm text-blue-800">
            <p className="font-medium mb-1">How Reclassification Works:</p>
            <ul className="list-disc list-inside space-y-1 text-blue-700">
              <li><strong>Text Analysis:</strong> Enhanced keyword matching with 280+ furniture terms</li>
              <li><strong>AI Detection:</strong> Object count and distribution analysis</li>
              <li><strong>Confidence Scoring:</strong> Multi-heuristic classification with reliability metrics</li>
              <li><strong>Style Detection:</strong> Automatic interior design style identification</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Action Button */}
      <div className="flex items-center justify-between mb-6">
        <div className="text-sm text-slate-600">
          {forceRedetection ? (
            <span className="flex items-center">
              <Target className="h-4 w-4 mr-1 text-purple-600" />
              Will reclassify and re-run object detection
            </span>
          ) : (
            <span className="flex items-center">
              <Eye className="h-4 w-4 mr-1 text-blue-600" />
              Classification only - no new object detection
            </span>
          )}
        </div>
        
        <button
          onClick={handleReclassify}
          disabled={reclassifyMutation.isPending || isMonitoring}
          className="inline-flex items-center px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {reclassifyMutation.isPending ? (
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
          ) : (
            <Play className="h-4 w-4 mr-2" />
          )}
          {reclassifyMutation.isPending ? 'Starting...' : 'Start Reclassification'}
        </button>
      </div>

      {/* Job Progress */}
      {jobProgress && (
        <div className="space-y-4">
          <div className="p-4 bg-white border border-slate-200 rounded-lg">
            <div className="flex items-center justify-between mb-3">
              <h4 className="font-medium text-slate-900 flex items-center">
                <Settings className="h-5 w-5 mr-2" />
                Reclassification Progress
              </h4>
              <span className={`px-2 py-1 rounded text-xs font-medium ${getJobStatusColor(jobProgress.status)}`}>
                {jobProgress.status}
              </span>
            </div>
            
            {/* Progress Bar */}
            <div className="mb-3">
              <div className="flex items-center justify-between text-sm text-slate-600 mb-1">
                <span>
                  {jobProgress.processed || 0} of {jobProgress.total || 0} scenes processed
                </span>
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
            
            {/* Status Message */}
            {jobProgress.message && (
              <p className="text-sm text-slate-600 mb-3">{jobProgress.message}</p>
            )}
            
            {/* Completion Status */}
            {jobProgress.status === 'completed' && (
              <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
                <div className="flex items-center">
                  <CheckCircle className="h-5 w-5 text-green-600 mr-2" />
                  <span className="text-sm font-medium text-green-800">
                    Reclassification completed successfully!
                  </span>
                </div>
                <p className="text-sm text-green-700 mt-1">
                  Scene classifications have been updated with improved accuracy.
                </p>
              </div>
            )}
            
            {(jobProgress.status === 'failed' || jobProgress.status === 'error') && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                <div className="flex items-start">
                  <AlertTriangle className="h-5 w-5 text-red-600 mr-2 mt-0.5 flex-shrink-0" />
                  <div className="flex-1">
                    <span className="text-sm font-medium text-red-800 block mb-1">
                      Reclassification failed
                    </span>
                    {jobProgress.error_message && (
                      <div className="text-sm text-red-700 bg-red-100 p-2 rounded border">
                        <strong>Error:</strong> {jobProgress.error_message}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Error Display */}
      {reclassifyMutation.error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
          <h4 className="font-medium text-red-900 mb-2 flex items-center">
            <AlertTriangle className="h-5 w-5 mr-2" />
            Failed to Start Reclassification
          </h4>
          <p className="text-sm text-red-700">
            {reclassifyMutation.error.message}
          </p>
        </div>
      )}

      {/* Benefits Info */}
      <div className="mt-6 p-4 bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-lg">
        <h4 className="font-medium text-slate-800 mb-2 flex items-center">
          <BarChart3 className="h-5 w-5 mr-2 text-blue-600" />
          Expected Improvements
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <h5 className="font-medium text-slate-700 mb-1">Better Scene Detection</h5>
            <p className="text-slate-600">
              Improved room type and style recognition with 150+ contextual keywords
            </p>
          </div>
          <div>
            <h5 className="font-medium text-slate-700 mb-1">Enhanced Object Classification</h5>
            <p className="text-slate-600">
              Single furniture pieces identified with 280+ product-specific terms
            </p>
          </div>
          <div>
            <h5 className="font-medium text-slate-700 mb-1">Hybrid Image Handling</h5>
            <p className="text-slate-600">
              Scenes with dominant focal objects properly categorized
            </p>
          </div>
          <div>
            <h5 className="font-medium text-slate-700 mb-1">Confidence Scoring</h5>
            <p className="text-slate-600">
              Reliability metrics to identify high-quality training data
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}