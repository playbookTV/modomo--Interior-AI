import React, { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { 
  TestTube, 
  Loader2, 
  CheckCircle, 
  AlertTriangle, 
  Eye,
  Target,
  Palette,
  Home,
  Tag,
  BarChart3,
  Info
} from 'lucide-react'
import { testImageClassification } from '../api/client'
import { ClassificationTestResult, ImageClassification } from '../types'

export function ClassificationTester() {
  const [imageUrl, setImageUrl] = useState('')
  const [caption, setCaption] = useState('')
  const [result, setResult] = useState<ClassificationTestResult | null>(null)

  const testMutation = useMutation({
    mutationFn: testImageClassification,
    onSuccess: (data) => {
      setResult(data)
    },
    onError: (error: any) => {
      setResult({
        image_url: imageUrl,
        caption: caption || undefined,
        classification: {
          image_type: 'scene',
          is_primary_object: false,
          confidence: 0,
          reason: 'error',
          metadata: {
            scores: { object: 0, scene: 0, hybrid: 0, style: 0 },
            detected_styles: [],
            keyword_matches: { object_matches: [], scene_matches: [] }
          }
        },
        status: 'failed',
        error: error.message
      })
    }
  })

  const handleTest = () => {
    if (!imageUrl.trim()) return
    
    testMutation.mutate({
      image_url: imageUrl.trim(),
      caption: caption.trim() || undefined
    })
  }

  const getImageTypeColor = (imageType: string) => {
    switch (imageType) {
      case 'scene': return 'text-blue-800 bg-blue-100 border-blue-200'
      case 'object': return 'text-purple-800 bg-purple-100 border-purple-200'
      case 'hybrid': return 'text-orange-800 bg-orange-100 border-orange-200'
      case 'product': return 'text-green-800 bg-green-100 border-green-200'
      default: return 'text-gray-800 bg-gray-100 border-gray-200'
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-800 bg-green-100'
    if (confidence >= 0.6) return 'text-yellow-800 bg-yellow-100'
    return 'text-red-800 bg-red-100'
  }

  return (
    <div className="bg-white rounded-xl p-8 shadow-sm border border-slate-200/60">
      <div className="flex items-center mb-6">
        <TestTube className="h-6 w-6 text-purple-600 mr-3" />
        <h2 className="text-2xl font-bold text-slate-800">Image Classification Tester</h2>
      </div>
      
      <div className="space-y-6">
        {/* Input Form */}
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Image URL <span className="text-red-500">*</span>
            </label>
            <input
              type="url"
              value={imageUrl}
              onChange={(e) => setImageUrl(e.target.value)}
              placeholder="https://example.com/image.jpg"
              className="w-full px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Caption/Description (Optional)
            </label>
            <input
              type="text"
              value={caption}
              onChange={(e) => setCaption(e.target.value)}
              placeholder="e.g., Modern living room with sectional sofa"
              className="w-full px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
            />
            <p className="text-sm text-slate-500 mt-1">
              Providing a description helps improve classification accuracy
            </p>
          </div>
          
          <button
            onClick={handleTest}
            disabled={testMutation.isPending || !imageUrl.trim()}
            className="inline-flex items-center px-6 py-3 bg-purple-600 text-white font-medium rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {testMutation.isPending ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <TestTube className="h-4 w-4 mr-2" />
            )}
            {testMutation.isPending ? 'Testing...' : 'Test Classification'}
          </button>
        </div>

        {/* Results */}
        {result && (
          <div className="border-t border-slate-200 pt-6">
            <h3 className="text-lg font-semibold text-slate-800 mb-4">Classification Results</h3>
            
            {result.status === 'failed' && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-lg mb-4">
                <div className="flex items-center">
                  <AlertTriangle className="h-5 w-5 text-red-600 mr-2" />
                  <span className="font-medium text-red-800">Classification Failed</span>
                </div>
                {result.error && (
                  <p className="text-sm text-red-700 mt-1">{result.error}</p>
                )}
              </div>
            )}

            {result.status === 'success' && (
              <div className="space-y-6">
                {/* Image Preview */}
                <div className="flex flex-col lg:flex-row gap-6">
                  <div className="lg:w-1/3">
                    <img
                      src={result.image_url}
                      alt="Test image"
                      className="w-full h-48 lg:h-64 object-cover rounded-lg border border-slate-200"
                      onError={(e) => {
                        (e.target as HTMLImageElement).src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiBmaWxsPSIjZjMzNDU2Ii8+Cjx0ZXh0IHg9IjEyIiB5PSIxMiIgZmlsbD0iI2ZmZmZmZiIgZm9udC1zaXplPSI4IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+4p2MPC90ZXh0Pgo8L3N2Zz4K'
                      }}
                    />
                  </div>
                  
                  <div className="lg:w-2/3 space-y-4">
                    {/* Primary Classification */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="p-4 border border-slate-200 rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-slate-600">Image Type</span>
                          <Eye className="h-4 w-4 text-slate-400" />
                        </div>
                        <div className="flex items-center gap-2">
                          <span className={`px-3 py-1 rounded-full text-sm font-medium border ${getImageTypeColor(result.classification.image_type)}`}>
                            {result.classification.image_type}
                          </span>
                          {result.classification.is_primary_object && (
                            <span className="px-2 py-1 text-xs bg-orange-100 text-orange-800 rounded border border-orange-200">
                              Primary Object
                            </span>
                          )}
                        </div>
                      </div>
                      
                      <div className="p-4 border border-slate-200 rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-slate-600">Confidence</span>
                          <BarChart3 className="h-4 w-4 text-slate-400" />
                        </div>
                        <span className={`px-3 py-1 rounded-full text-sm font-bold ${getConfidenceColor(result.classification.confidence)}`}>
                          {(result.classification.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>

                    {/* Primary Category */}
                    {result.classification.primary_category && (
                      <div className="p-4 border border-slate-200 rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-slate-600">Primary Category</span>
                          <Target className="h-4 w-4 text-slate-400" />
                        </div>
                        <span className="inline-flex items-center px-3 py-1 text-sm font-medium text-indigo-800 bg-indigo-100 rounded-full border border-indigo-200">
                          {result.classification.primary_category.replace(/_/g, ' ')}
                        </span>
                      </div>
                    )}

                    {/* Room Type */}
                    {result.classification.metadata.detected_room_type && (
                      <div className="p-4 border border-slate-200 rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-slate-600">Detected Room Type</span>
                          <Home className="h-4 w-4 text-slate-400" />
                        </div>
                        <span className="inline-flex items-center px-3 py-1 text-sm font-medium text-teal-800 bg-teal-100 rounded-full border border-teal-200">
                          {result.classification.metadata.detected_room_type.replace(/_/g, ' ')}
                        </span>
                      </div>
                    )}

                    {/* Detected Styles */}
                    {result.classification.metadata.detected_styles.length > 0 && (
                      <div className="p-4 border border-slate-200 rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-slate-600">Detected Styles</span>
                          <Palette className="h-4 w-4 text-slate-400" />
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {result.classification.metadata.detected_styles.map((style, idx) => (
                            <span key={idx} className="px-2 py-1 text-xs font-medium text-pink-800 bg-pink-100 rounded border border-pink-200">
                              {style}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Detailed Scores */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {Object.entries(result.classification.metadata.scores).map(([scoreType, score]) => (
                    <div key={scoreType} className="p-3 bg-slate-50 rounded-lg border">
                      <div className="text-xs font-medium text-slate-600 mb-1 capitalize">
                        {scoreType} Score
                      </div>
                      <div className="text-lg font-bold text-slate-800">
                        {score.toFixed(1)}
                      </div>
                      <div className="w-full bg-slate-200 rounded-full h-1.5 mt-2">
                        <div 
                          className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                          style={{ width: `${Math.min(100, (score / 10) * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                {/* Keyword Matches */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
                    <div className="flex items-center mb-2">
                      <Tag className="h-4 w-4 text-purple-600 mr-2" />
                      <span className="text-sm font-medium text-purple-800">Object Keywords</span>
                    </div>
                    {result.classification.metadata.keyword_matches.object_matches.length > 0 ? (
                      <div className="flex flex-wrap gap-1">
                        {result.classification.metadata.keyword_matches.object_matches.slice(0, 6).map((keyword, idx) => (
                          <span key={idx} className="px-2 py-1 text-xs bg-purple-100 text-purple-700 rounded border border-purple-300">
                            {keyword}
                          </span>
                        ))}
                        {result.classification.metadata.keyword_matches.object_matches.length > 6 && (
                          <span className="text-xs text-purple-600">
                            +{result.classification.metadata.keyword_matches.object_matches.length - 6} more
                          </span>
                        )}
                      </div>
                    ) : (
                      <span className="text-xs text-purple-600">No object keywords found</span>
                    )}
                  </div>

                  <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                    <div className="flex items-center mb-2">
                      <Tag className="h-4 w-4 text-blue-600 mr-2" />
                      <span className="text-sm font-medium text-blue-800">Scene Keywords</span>
                    </div>
                    {result.classification.metadata.keyword_matches.scene_matches.length > 0 ? (
                      <div className="flex flex-wrap gap-1">
                        {result.classification.metadata.keyword_matches.scene_matches.slice(0, 6).map((keyword, idx) => (
                          <span key={idx} className="px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded border border-blue-300">
                            {keyword}
                          </span>
                        ))}
                        {result.classification.metadata.keyword_matches.scene_matches.length > 6 && (
                          <span className="text-xs text-blue-600">
                            +{result.classification.metadata.keyword_matches.scene_matches.length - 6} more
                          </span>
                        )}
                      </div>
                    ) : (
                      <span className="text-xs text-blue-600">No scene keywords found</span>
                    )}
                  </div>
                </div>

                {/* Classification Reasoning */}
                <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                  <div className="flex items-center mb-2">
                    <Info className="h-4 w-4 text-gray-600 mr-2" />
                    <span className="text-sm font-medium text-gray-800">Classification Reasoning</span>
                  </div>
                  <p className="text-sm text-gray-700">
                    <span className="font-medium">Reason:</span> {result.classification.reason}
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}