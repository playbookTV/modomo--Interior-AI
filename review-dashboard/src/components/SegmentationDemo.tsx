import React, { useState } from 'react'
import { Target, Eye, EyeOff } from 'lucide-react'

const DEMO_OBJECTS = [
  {
    object_id: 'demo-1',
    scene_id: 'demo-scene',
    category: 'sofa',
    confidence: 0.92,
    bbox: [150, 200, 300, 180] as [number, number, number, number],
    mask_url: '/api/demo/mask-sofa.png', // This would be a real SAM2 mask
    tags: ['comfortable', 'gray', 'modern'],
    metadata: {
      colors: {
        dominant_color: { hex: '#8B8680', name: 'gray', rgb: [139, 134, 128] as [number, number, number] },
        colors: [
          { hex: '#8B8680', name: 'gray', percentage: 65, rgb: [139, 134, 128] as [number, number, number] },
          { hex: '#2C2C2C', name: 'charcoal', percentage: 25, rgb: [44, 44, 44] as [number, number, number] }
        ]
      }
    }
  },
  {
    object_id: 'demo-2',
    scene_id: 'demo-scene', 
    category: 'coffee_table',
    confidence: 0.87,
    bbox: [250, 450, 200, 100] as [number, number, number, number],
    // No mask_url - this demonstrates bounding box fallback
    tags: ['wood', 'rectangular'],
    metadata: {
      colors: {
        dominant_color: { hex: '#8B4513', name: 'brown', rgb: [139, 69, 19] as [number, number, number] }
      }
    }
  }
]

export function SegmentationDemo() {
  const [showMasks, setShowMasks] = useState(true)
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(false)
  const [currentObjectIndex, setCurrentObjectIndex] = useState(0)

  const currentObject = DEMO_OBJECTS[currentObjectIndex]

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          Enhanced Object Identification with SAM2
        </h2>
        <p className="text-gray-600">
          Experience the difference between traditional bounding boxes and precise neural segmentation
        </p>
      </div>

      {/* Visualization Controls */}
      <div className="flex items-center gap-4 mb-4 p-3 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border border-blue-200">
        <div className="text-sm font-semibold text-gray-800">Visualization:</div>
        
        <button
          onClick={() => setShowMasks(!showMasks)}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
            showMasks 
              ? 'bg-green-500 text-white shadow-md' 
              : 'bg-white text-gray-600 border border-gray-300 hover:border-green-400'
          }`}
        >
          {showMasks ? <Eye size={14} /> : <EyeOff size={14} />}
          SAM2 Masks
        </button>
        
        <button
          onClick={() => setShowBoundingBoxes(!showBoundingBoxes)}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
            showBoundingBoxes 
              ? 'bg-blue-500 text-white shadow-md' 
              : 'bg-white text-gray-600 border border-gray-300 hover:border-blue-400'
          }`}
        >
          <div className={`w-3 h-3 border-2 rounded-sm ${showBoundingBoxes ? 'border-white' : 'border-gray-400'}`} />
          Bounding Boxes
        </button>
      </div>

      {/* Demo Image Area */}
      <div className="relative bg-gray-100 rounded-lg h-96 mb-4 border-2 border-dashed border-gray-300 flex items-center justify-center">
        <div className="text-center text-gray-500">
          <div className="text-6xl mb-4">🏠</div>
          <div className="text-lg font-medium">Demo Interior Scene</div>
          <div className="text-sm">Hover to see object identification overlays</div>
        </div>
        
        {/* Simulated overlays would go here */}
        <div className="absolute inset-4 border-2 border-green-400 rounded opacity-50 shadow-lg">
          <div className="absolute -top-8 left-0 bg-green-500 text-white text-xs px-2 py-1 rounded-full font-semibold">
            Sofa (92% confidence)
          </div>
        </div>
      </div>

      {/* Object Selection */}
      <div className="flex gap-2 mb-4">
        {DEMO_OBJECTS.map((obj, index) => (
          <button
            key={obj.object_id}
            onClick={() => setCurrentObjectIndex(index)}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg border font-medium text-sm transition-all ${
              index === currentObjectIndex
                ? 'bg-blue-500 text-white border-blue-500 shadow-md'
                : 'bg-white text-gray-700 border-gray-300 hover:border-blue-300'
            }`}
          >
            {obj.mask_url ? <Target size={14} /> : <div className="w-3 h-3 border border-current rounded-sm" />}
            {obj.category}
          </button>
        ))}
      </div>

      {/* Object Details */}
      <div className="bg-white rounded-lg p-4 border shadow-sm">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold capitalize">{currentObject.category}</h3>
            <p className="text-sm text-gray-600">
              Confidence: {(currentObject.confidence * 100).toFixed(1)}%
            </p>
          </div>
          
          <div className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${
            currentObject.mask_url 
              ? 'bg-green-100 text-green-800'
              : 'bg-orange-100 text-orange-800'
          }`}>
            {currentObject.mask_url ? (
              <>
                <Target size={12} />
                Neural Segmentation
              </>
            ) : (
              <>
                <div className="w-3 h-3 border border-orange-600 rounded-sm" />
                Bounding Box
              </>
            )}
          </div>
        </div>

        {/* Key Improvements */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-blue-50 rounded-lg p-3 border border-blue-200">
            <h4 className="font-semibold text-blue-800 mb-2">🎯 Precision Benefits</h4>
            <ul className="text-sm text-blue-700 space-y-1">
              <li>• Exact object boundaries</li>
              <li>• Handles complex shapes</li>
              <li>• Separates overlapping objects</li>
              <li>• Better training data quality</li>
            </ul>
          </div>
          
          <div className="bg-green-50 rounded-lg p-3 border border-green-200">
            <h4 className="font-semibold text-green-800 mb-2">🔬 SAM2 Features</h4>
            <ul className="text-sm text-green-700 space-y-1">
              <li>• Neural network segmentation</li>
              <li>• State-of-the-art accuracy</li>
              <li>• Real-time processing</li>
              <li>• Fine-grained object masks</li>
            </ul>
          </div>
        </div>

        {/* Statistics */}
        <div className="mt-4 pt-4 border-t">
          <div className="grid grid-cols-3 gap-4 text-center">
            <div className="bg-green-50 rounded-lg p-2 border border-green-200">
              <div className="text-lg font-bold text-green-800">
                {DEMO_OBJECTS.filter(obj => obj.mask_url).length}
              </div>
              <div className="text-xs text-green-600">SAM2 Segmented</div>
            </div>
            <div className="bg-orange-50 rounded-lg p-2 border border-orange-200">
              <div className="text-lg font-bold text-orange-800">
                {DEMO_OBJECTS.filter(obj => !obj.mask_url).length}
              </div>
              <div className="text-xs text-orange-600">Bounding Box Only</div>
            </div>
            <div className="bg-blue-50 rounded-lg p-2 border border-blue-200">
              <div className="text-lg font-bold text-blue-800">
                {((DEMO_OBJECTS.filter(obj => obj.mask_url).length / DEMO_OBJECTS.length) * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-blue-600">Segmentation Rate</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}