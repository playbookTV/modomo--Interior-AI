import React, { useState, useEffect } from 'react'
import { useMutation, useQueryClient, useQuery } from '@tanstack/react-query'
import { Play, Loader2, Database, Sparkles, AlertTriangle, Eye, CheckCircle, Clock } from 'lucide-react'

interface ImportJob {
  job_id: string
  status: string
  message: string
  dataset: string
  features: string[]
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

async function fetchScenes(limit = 10): Promise<{scenes: any[], total: number}> {
  const response = await fetch(
    `https://ovalay-recruitment-production.up.railway.app/scenes?limit=${limit}&offset=0`
  )
  if (!response.ok) throw new Error('Failed to fetch scenes')
  return response.json()
}

async function fetchDatasetStats(): Promise<any> {
  const response = await fetch(
    'https://ovalay-recruitment-production.up.railway.app/stats/dataset'
  )
  if (!response.ok) throw new Error('Failed to fetch stats')
  return response.json()
}

export function DatasetImporter() {
  const [selectedDataset, setSelectedDataset] = useState('sk2003/houzzdata')
  const [customDataset, setCustomDataset] = useState('')
  const [offset, setOffset] = useState(0)
  const [limit, setLimit] = useState(50)
  const [includeDetection, setIncludeDetection] = useState(true)
  const [lastJob, setLastJob] = useState<ImportJob | null>(null)
  const [isMonitoring, setIsMonitoring] = useState(false)

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
      // Refresh dashboard stats
      queryClient.invalidateQueries({ queryKey: ['dataset-stats'] })
      queryClient.invalidateQueries({ queryKey: ['category-stats'] })
    },
    onError: (error) => {
      console.error('Import failed:', error)
    }
  })

  const handleImport = () => {
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
        <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <h4 className="font-medium text-blue-900 mb-2">Latest Import Job</h4>
          <div className="text-sm space-y-1">
            <p><span className="font-medium">Job ID:</span> {lastJob.job_id}</p>
            <p><span className="font-medium">Status:</span> {lastJob.status}</p>
            <p><span className="font-medium">Dataset:</span> {lastJob.dataset}</p>
            <p><span className="font-medium">Message:</span> {lastJob.message}</p>
          </div>
        </div>
      )}

      {/* Error Display */}
      {importMutation.error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
          <h4 className="font-medium text-red-900 mb-2">Import Failed</h4>
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
            <button
              onClick={() => setIsMonitoring(false)}
              className="text-sm text-blue-600 hover:text-blue-800"
            >
              Stop Monitoring
            </button>
          </div>

          {/* Live Stats */}
          {liveStats && (
            <div className="grid grid-cols-3 gap-4">
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
                  <Sparkles className="h-5 w-5 text-purple-600 mr-2" />
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
            </div>
          )}

          {/* Recent Images */}
          {scenes && scenes.scenes && scenes.scenes.length > 0 && (
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
                    <img
                      src={scene.image_url}
                      alt={scene.houzz_id}
                      className="w-full h-20 object-cover rounded mb-2"
                      onError={(e) => {
                        (e.target as HTMLImageElement).src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiBmaWxsPSIjZjMzNDU2Ii8+Cjx0ZXh0IHg9IjEyIiB5PSIxMiIgZmlsbD0iI2ZmZmZmZiIgZm9udC1zaXplPSI4IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+4p2MPC90ZXh0Pgo8L3N2Zz4K'
                      }}
                    />
                    <div className="space-y-1">
                      <p className="text-xs text-slate-600 truncate">{scene.houzz_id}</p>
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-500">{scene.room_type}</span>
                        <span className="flex items-center text-green-600">
                          <CheckCircle className="h-3 w-3 mr-1" />
                          {scene.object_count || 0}
                        </span>
                      </div>
                      {scene.style_tags && scene.style_tags.length > 0 && (
                        <p className="text-xs text-blue-600 truncate">
                          {scene.style_tags[0]}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}