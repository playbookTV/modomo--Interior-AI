import React from 'react'
import { TestTube, RefreshCw } from 'lucide-react'
import { ClassificationTester } from '../components/ClassificationTester'
import { SceneReclassifier } from '../components/SceneReclassifier'

export function Classification() {
  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="text-center py-8">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-slate-800 to-slate-600 bg-clip-text text-transparent mb-4">
          Image Classification Tools
        </h1>
        <p className="text-slate-600 text-lg max-w-3xl mx-auto">
          Test and improve scene vs object classification using advanced AI analysis with comprehensive keyword matching and confidence scoring
        </p>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl p-6 border border-purple-200">
          <div className="flex items-center mb-4">
            <TestTube className="h-8 w-8 text-purple-600 mr-3" />
            <div>
              <h3 className="text-xl font-bold text-purple-900">Classification Testing</h3>
              <p className="text-purple-700 text-sm">Test single images with real-time analysis</p>
            </div>
          </div>
          <div className="space-y-2 text-sm text-purple-800">
            <div className="flex items-center justify-between">
              <span>Keywords Analyzed</span>
              <span className="font-bold">280+ furniture terms</span>
            </div>
            <div className="flex items-center justify-between">
              <span>Scene Contexts</span>
              <span className="font-bold">150+ room indicators</span>
            </div>
            <div className="flex items-center justify-between">
              <span>Style Recognition</span>
              <span className="font-bold">80+ design styles</span>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-6 border border-blue-200">
          <div className="flex items-center mb-4">
            <RefreshCw className="h-8 w-8 text-blue-600 mr-3" />
            <div>
              <h3 className="text-xl font-bold text-blue-900">Batch Reclassification</h3>
              <p className="text-blue-700 text-sm">Improve existing dataset quality</p>
            </div>
          </div>
          <div className="space-y-2 text-sm text-blue-800">
            <div className="flex items-center justify-between">
              <span>Processing Speed</span>
              <span className="font-bold">~50 images/minute</span>
            </div>
            <div className="flex items-center justify-between">
              <span>Classification Types</span>
              <span className="font-bold">Scene, Object, Hybrid</span>
            </div>
            <div className="flex items-center justify-between">
              <span>Confidence Scoring</span>
              <span className="font-bold">Multi-heuristic analysis</span>
            </div>
          </div>
        </div>
      </div>

      {/* Classification Tester */}
      <ClassificationTester />

      {/* Scene Reclassifier */}  
      <SceneReclassifier />

      {/* Algorithm Details */}
      <div className="bg-white rounded-xl p-8 shadow-sm border border-slate-200/60">
        <h2 className="text-2xl font-bold text-slate-800 mb-6">Enhanced Classification Algorithm</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
            <h3 className="font-semibold text-blue-900 mb-2">Text Analysis</h3>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>• Comprehensive keyword matching</li>
              <li>• Phrase-aware scoring (2-3 words)</li>
              <li>• Fuzzy matching for variations</li>
              <li>• Multi-language support ready</li>
            </ul>
          </div>

          <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
            <h3 className="font-semibold text-purple-900 mb-2">AI Detection</h3>
            <ul className="text-sm text-purple-800 space-y-1">
              <li>• Object count analysis</li>
              <li>• Category distribution</li>
              <li>• Confidence thresholds</li>
              <li>• Dominant object detection</li>
            </ul>
          </div>

          <div className="p-4 bg-orange-50 rounded-lg border border-orange-200">
            <h3 className="font-semibold text-orange-900 mb-2">Style Recognition</h3>
            <ul className="text-sm text-orange-800 space-y-1">
              <li>• Modern, Traditional, Eclectic</li>
              <li>• Scandinavian, Industrial, Boho</li>
              <li>• Farmhouse, Coastal, Art Deco</li>
              <li>• Emerging trend detection</li>
            </ul>
          </div>

          <div className="p-4 bg-green-50 rounded-lg border border-green-200">
            <h3 className="font-semibold text-green-900 mb-2">Quality Metrics</h3>
            <ul className="text-sm text-green-800 space-y-1">
              <li>• Multi-heuristic confidence</li>
              <li>• Classification reasoning</li>
              <li>• Keyword match tracking</li>
              <li>• Performance analytics</li>
            </ul>
          </div>
        </div>

        <div className="mt-6 p-4 bg-gradient-to-r from-slate-50 to-slate-100 rounded-lg border border-slate-200">
          <h3 className="font-semibold text-slate-800 mb-2">Classification Categories</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="inline-block w-3 h-3 bg-blue-500 rounded-full mr-2"></span>
              <strong>Scene:</strong> Full room contexts with 3+ diverse objects
            </div>
            <div>
              <span className="inline-block w-3 h-3 bg-purple-500 rounded-full mr-2"></span>
              <strong>Object:</strong> 1-2 furniture pieces, product-focused
            </div>
            <div>
              <span className="inline-block w-3 h-3 bg-orange-500 rounded-full mr-2"></span>
              <strong>Hybrid:</strong> Scenes with dominant focal objects
            </div>
            <div>
              <span className="inline-block w-3 h-3 bg-green-500 rounded-full mr-2"></span>
              <strong>Product:</strong> Catalog images, studio photography
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}