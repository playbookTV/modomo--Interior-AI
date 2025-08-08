import React, { useState, useCallback } from 'react'
import { useHotkeys } from 'react-hotkeys-hook'
import { Check, X, Edit3, Package, Zap } from 'lucide-react'
import { DetectedObject, Scene, Product } from '../types'
import { ObjectOverlay } from './ObjectOverlay'
import { TagEditor } from './TagEditor'
import { ProductMatcher } from './ProductMatcher'

interface ReviewInterfaceProps {
  scene: Scene
  currentObjectIndex: number
  onObjectUpdate: (objectId: string, updates: Partial<DetectedObject>) => void
  onNext: () => void
  onPrevious: () => void
  onApproveScene: () => void
  onRejectScene: () => void
  isLastObject: boolean
}

export function ReviewInterface({
  scene,
  currentObjectIndex,
  onObjectUpdate,
  onNext,
  onPrevious,
  onApproveScene,
  onRejectScene,
  isLastObject
}: ReviewInterfaceProps) {
  const [showTagEditor, setShowTagEditor] = useState(false)
  const [showProductMatcher, setShowProductMatcher] = useState(false)
  
  const currentObject = scene.objects[currentObjectIndex]
  
  // Keyboard shortcuts
  useHotkeys('a', () => handleApprove(), [currentObject])
  useHotkeys('r', () => handleReject(), [currentObject])
  useHotkeys('n', onNext, [onNext])
  useHotkeys('p', onPrevious, [onPrevious])
  useHotkeys('t', () => setShowTagEditor(true), [])
  useHotkeys('m', () => setShowProductMatcher(true), [])
  useHotkeys('escape', () => {
    setShowTagEditor(false)
    setShowProductMatcher(false)
  }, [])
  
  const handleApprove = useCallback(() => {
    onObjectUpdate(currentObject.object_id, { approved: true })
    if (isLastObject) {
      onApproveScene()
    } else {
      onNext()
    }
  }, [currentObject, onObjectUpdate, isLastObject, onApproveScene, onNext])
  
  const handleReject = useCallback(() => {
    onObjectUpdate(currentObject.object_id, { approved: false })
    if (isLastObject) {
      onApproveScene() // Still complete the scene
    } else {
      onNext()
    }
  }, [currentObject, onObjectUpdate, isLastObject, onApproveScene, onNext])
  
  const handleCategoryChange = useCallback((category: string) => {
    onObjectUpdate(currentObject.object_id, { category })
  }, [currentObject, onObjectUpdate])
  
  const handleTagsChange = useCallback((tags: string[]) => {
    onObjectUpdate(currentObject.object_id, { tags })
    setShowTagEditor(false)
  }, [currentObject, onObjectUpdate])
  
  const handleProductMatch = useCallback((productId: string) => {
    onObjectUpdate(currentObject.object_id, { matched_product_id: productId })
    setShowProductMatcher(false)
  }, [currentObject, onObjectUpdate])

  return (
    <div className="flex flex-col h-full">
      {/* Image and Object Overlay */}
      <div className="flex-1 relative bg-black rounded-lg overflow-hidden">
        <img
          src={scene.image_url}
          alt={`Room scene ${scene.scene_id}`}
          className="w-full h-full object-contain"
        />
        
        <ObjectOverlay
          objects={scene.objects}
          currentObjectIndex={currentObjectIndex}
          imageWidth={1920} // Would need to get actual dimensions
          imageHeight={1080}
        />
      </div>
      
      {/* Object Info Panel */}
      <div className="bg-white rounded-lg p-4 mt-4 border">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold">
              Object {currentObjectIndex + 1} of {scene.objects.length}
            </h3>
            <p className="text-sm text-gray-600">
              Category: <span className="font-medium">{currentObject.category}</span>
              {' • '}
              Confidence: <span className="font-medium">{(currentObject.confidence * 100).toFixed(1)}%</span>
            </p>
          </div>
          
          <div className="flex gap-2">
            <button
              onClick={() => setShowTagEditor(true)}
              className="flex items-center gap-2 px-3 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
              title="Edit Tags (T)"
            >
              <Edit3 size={16} />
              Tags
            </button>
            
            <button
              onClick={() => setShowProductMatcher(true)}
              className="flex items-center gap-2 px-3 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors"
              title="Match Product (M)"
            >
              <Package size={16} />
              Products
            </button>
          </div>
        </div>
        
        {/* Tags Display */}
        {currentObject.tags && currentObject.tags.length > 0 && (
          <div className="mb-4">
            <p className="text-sm font-medium text-gray-700 mb-2">Tags:</p>
            <div className="flex flex-wrap gap-2">
              {currentObject.tags.map((tag, index) => (
                <span
                  key={index}
                  className="px-2 py-1 bg-gray-100 text-gray-800 text-xs rounded-full"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )}
        
        {/* Product Match Display */}
        {currentObject.matched_product_id && (
          <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-lg">
            <p className="text-sm font-medium text-green-800 mb-1">Matched Product:</p>
            <p className="text-sm text-green-700">ID: {currentObject.matched_product_id}</p>
          </div>
        )}
        
        {/* Action Buttons */}
        <div className="flex justify-between items-center">
          <div className="flex gap-2">
            <button
              onClick={onPrevious}
              disabled={currentObjectIndex === 0}
              className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Previous (P)
            </button>
            
            <button
              onClick={onNext}
              disabled={isLastObject}
              className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Next (N)
            </button>
          </div>
          
          <div className="flex gap-2">
            <button
              onClick={handleReject}
              className="flex items-center gap-2 px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors"
              title="Reject (R)"
            >
              <X size={16} />
              Reject
            </button>
            
            <button
              onClick={handleApprove}
              className="flex items-center gap-2 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
              title="Approve (A)"
            >
              <Check size={16} />
              {isLastObject ? 'Complete Scene' : 'Approve'}
            </button>
          </div>
        </div>
        
        {/* Keyboard Shortcuts Help */}
        <div className="mt-4 pt-4 border-t text-xs text-gray-500">
          <p>Shortcuts: A=Approve, R=Reject, N=Next, P=Previous, T=Tags, M=Products</p>
        </div>
      </div>
      
      {/* Modals */}
      {showTagEditor && (
        <TagEditor
          currentTags={currentObject.tags || []}
          category={currentObject.category}
          onSave={handleTagsChange}
          onClose={() => setShowTagEditor(false)}
        />
      )}
      
      {showProductMatcher && (
        <ProductMatcher
          object={currentObject}
          onMatch={handleProductMatch}
          onClose={() => setShowProductMatcher(false)}
        />
      )}
    </div>
  )
}