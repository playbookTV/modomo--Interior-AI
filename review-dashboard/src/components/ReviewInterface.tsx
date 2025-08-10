import React, { useState, useCallback } from 'react'
import { useHotkeys } from 'react-hotkeys-hook'
import { Check, X, Edit3, Package } from 'lucide-react'
import { DetectedObject, Scene, Product } from '../types'
import { searchProducts } from '../api/client'

// Inline UI components to avoid missing module imports

type ObjectOverlayProps = {
  objects: DetectedObject[]
  currentObjectIndex: number
  imageWidth: number
  imageHeight: number
}

function ObjectOverlay({ objects, currentObjectIndex, imageWidth, imageHeight }: ObjectOverlayProps) {
  return (
    <div className="absolute inset-0 pointer-events-none">
      {objects.map((obj, index) => {
        const [x, y, w, h] = obj.bbox
        // Using given imageWidth/Height as source dimensions; boxes scale with container via percentages
        const leftPct = (x / imageWidth) * 100
        const topPct = (y / imageHeight) * 100
        const widthPct = (w / imageWidth) * 100
        const heightPct = (h / imageHeight) * 100
        const isActive = index === currentObjectIndex
        return (
          <div
            key={obj.object_id}
            className={`absolute border-2 rounded ${isActive ? 'border-green-400' : 'border-blue-400'} shadow-[0_0_0_1px_rgba(0,0,0,0.2)]`}
            style={{ left: `${leftPct}%`, top: `${topPct}%`, width: `${widthPct}%`, height: `${heightPct}%` }}
            aria-label={`bbox-${obj.category}`}
          />
        )
      })}
    </div>
  )
}

type TagEditorProps = {
  currentTags: string[]
  category: string
  onSave: (tags: string[]) => void
  onClose: () => void
}

function TagEditor({ currentTags, category, onSave, onClose }: TagEditorProps) {
  const [value, setValue] = useState(currentTags.join(', '))

  const handleSave = () => {
    const tags = value
      .split(',')
      .map(t => t.trim())
      .filter(Boolean)
    onSave(tags)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-white rounded-lg p-6 w-full max-w-lg shadow-xl border">
        <h3 className="text-lg font-semibold mb-2">Edit Tags</h3>
        <p className="text-sm text-gray-600 mb-4">Category: {category}</p>
        <input
          type="text"
          className="w-full px-3 py-2 border rounded mb-4 focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="comma, separated, tags"
          value={value}
          onChange={(e) => setValue(e.target.value)}
        />
        <div className="flex justify-end gap-2">
          <button onClick={onClose} className="px-4 py-2 rounded border">Cancel</button>
          <button onClick={handleSave} className="px-4 py-2 rounded bg-blue-600 text-white hover:bg-blue-700">Save</button>
        </div>
      </div>
    </div>
  )
}

type ProductMatcherProps = {
  object: DetectedObject
  onMatch: (productId: string) => void
  onClose: () => void
}

function ProductMatcher({ object, onMatch, onClose }: ProductMatcherProps) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<Product[]>([])
  const [loading, setLoading] = useState(false)

  const handleSearch = async () => {
    try {
      setLoading(true)
      const products = await searchProducts({ query, limit: 5 })
      setResults(products)
    } catch (err) {
      console.error('Product search failed', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-white rounded-lg p-6 w-full max-w-2xl shadow-xl border">
        <h3 className="text-lg font-semibold mb-4">Match Product</h3>
        <div className="flex gap-2 mb-4">
          <input
            type="text"
            className="flex-1 px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-purple-500"
            placeholder={`Search for products for ${object.category}`}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button onClick={handleSearch} className="px-4 py-2 rounded bg-purple-600 text-white hover:bg-purple-700" disabled={loading}>
            {loading ? 'Searching...' : 'Search'}
          </button>
          <button onClick={onClose} className="px-4 py-2 rounded border">Close</button>
        </div>
        <div className="max-h-80 overflow-auto space-y-2">
          {results.map((p) => (
            <div key={p.product_id} className="flex items-center justify-between p-3 border rounded">
              <div className="flex items-center gap-3">
                <img src={p.image_url} alt={p.name} className="w-14 h-14 object-cover rounded" />
                <div>
                  <p className="font-medium text-sm">{p.name}</p>
                  <p className="text-xs text-gray-500">{p.brand} • {p.category}</p>
                </div>
              </div>
              <button
                onClick={() => onMatch(p.product_id)}
                className="px-3 py-1.5 rounded bg-emerald-600 text-white hover:bg-emerald-700"
              >
                Match
              </button>
            </div>
          ))}
          {!loading && results.length === 0 && (
            <p className="text-sm text-gray-500">No results yet. Try searching.</p>
          )}
        </div>
      </div>
    </div>
  )
}

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
        
        {/* Color Display */}
        {currentObject.metadata?.colors && (
          <div className="mb-4">
            <p className="text-sm font-medium text-gray-700 mb-2">Colors:</p>
            
            {/* Dominant Color */}
            {currentObject.metadata.colors.dominant_color && (
              <div className="mb-3">
                <p className="text-xs text-gray-600 mb-1">Dominant:</p>
                <div className="flex items-center gap-2">
                  <div 
                    className="w-8 h-8 rounded border border-gray-300 shadow-sm"
                    style={{ backgroundColor: currentObject.metadata.colors.dominant_color.hex }}
                    title={`${currentObject.metadata.colors.dominant_color.name} (${currentObject.metadata.colors.dominant_color.hex})`}
                  />
                  <div>
                    <p className="text-sm font-medium capitalize">{currentObject.metadata.colors.dominant_color.name}</p>
                    <p className="text-xs text-gray-500">{currentObject.metadata.colors.dominant_color.hex}</p>
                  </div>
                </div>
              </div>
            )}
            
            {/* Color Palette */}
            {currentObject.metadata.colors.colors && currentObject.metadata.colors.colors.length > 0 && (
              <div>
                <p className="text-xs text-gray-600 mb-2">Color Palette:</p>
                <div className="flex flex-wrap gap-2">
                  {currentObject.metadata.colors.colors.slice(0, 5).map((color, index) => (
                    <div 
                      key={index}
                      className="flex items-center gap-1 px-2 py-1 bg-gray-50 border border-gray-200 rounded text-xs"
                      title={`${color.name} (${color.hex}) - ${color.percentage}%`}
                    >
                      <div 
                        className="w-4 h-4 rounded border border-gray-300"
                        style={{ backgroundColor: color.hex }}
                      />
                      <span className="capitalize">{color.name}</span>
                      <span className="text-gray-500">({color.percentage}%)</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {/* Color Properties */}
            {currentObject.metadata.colors.properties && (
              <div className="mt-2 text-xs text-gray-600">
                <span>Brightness: {(currentObject.metadata.colors.properties.brightness * 100).toFixed(0)}%</span>
                {currentObject.metadata.colors.properties.color_temperature && (
                  <span className="ml-3 capitalize">
                    Tone: {currentObject.metadata.colors.properties.color_temperature}
                  </span>
                )}
                {currentObject.metadata.colors.properties.is_neutral && (
                  <span className="ml-3 px-1 bg-gray-200 rounded">Neutral</span>
                )}
              </div>
            )}
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