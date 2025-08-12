import React, { useState, useCallback } from 'react'
import { useHotkeys } from 'react-hotkeys-hook'
import { Check, X, Edit3, Package, Loader2, Eye, EyeOff, Target } from 'lucide-react'
import { DetectedObject, Scene, Product } from '../types'
import { searchProducts } from '../api/client'

// Inline UI components to avoid missing module imports

function SegmentationQualityBadge({ object }: { object: DetectedObject }) {
  if (object.mask_url) {
    return (
      <div className="flex items-center gap-1 px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs font-medium">
        <Target size={12} />
        Neural Segmentation
      </div>
    )
  } else {
    return (
      <div className="flex items-center gap-1 px-2 py-1 bg-orange-100 text-orange-800 rounded-full text-xs font-medium">
        <div className="w-3 h-3 border border-orange-600 rounded-sm" />
        Bounding Box
      </div>
    )
  }
}

type ObjectOverlayProps = {
  objects: DetectedObject[]
  currentObjectIndex: number
  imageWidth: number
  imageHeight: number
  showMasks?: boolean
  showBoundingBoxes?: boolean
}

function MaskOverlay({ 
  maskUrl, 
  isActive, 
  opacity = 0.6 
}: { 
  maskUrl: string
  isActive: boolean
  opacity?: number 
}) {
  return (
    <div className="absolute inset-0 pointer-events-none">
      {/* Colored overlay */}
      <div
        className="absolute inset-0"
        style={{
          backgroundColor: isActive ? '#10b981' : '#3b82f6', // Green for active, blue for inactive
          opacity: isActive ? 0.3 : 0.2,
          maskImage: `url(${maskUrl})`,
          WebkitMaskImage: `url(${maskUrl})`,
          maskRepeat: 'no-repeat',
          maskSize: 'contain',
          maskPosition: 'center',
          WebkitMaskRepeat: 'no-repeat',
          WebkitMaskSize: 'contain', 
          WebkitMaskPosition: 'center'
        }}
      />
      
      {/* Mask outline */}
      <img
        src={maskUrl}
        alt="Object mask"
        className="w-full h-full object-contain"
        style={{
          opacity: isActive ? 0.8 : 0.5,
          filter: isActive 
            ? 'brightness(0) invert(1) sepia(1) saturate(5) hue-rotate(120deg)' // Green outline for active
            : 'brightness(0) invert(1) sepia(1) saturate(5) hue-rotate(240deg)', // Blue outline for inactive
          mixBlendMode: 'overlay'
        }}
        onError={(e) => {
          // Hide broken masks gracefully
          e.currentTarget.style.display = 'none'
          console.warn('Failed to load mask:', maskUrl)
        }}
      />
    </div>
  )
}

function ObjectOverlay({ 
  objects, 
  currentObjectIndex, 
  imageWidth, 
  imageHeight,
  showMasks = true,
  showBoundingBoxes = false
}: ObjectOverlayProps) {
  return (
    <div className="absolute inset-0 pointer-events-none">
      {/* SAM2 Mask Overlays */}
      {showMasks && objects.map((obj, index) => {
        if (!obj.mask_url) return null
        
        const isActive = index === currentObjectIndex
        return (
          <MaskOverlay
            key={`mask-${obj.object_id}`}
            maskUrl={obj.mask_url}
            isActive={isActive}
            opacity={isActive ? 0.7 : 0.4}
          />
        )
      })}
      
      {/* Traditional Bounding Box Overlays */}
      {showBoundingBoxes && objects.map((obj, index) => {
        const [x, y, w, h] = obj.bbox
        const leftPct = (x / imageWidth) * 100
        const topPct = (y / imageHeight) * 100
        const widthPct = (w / imageWidth) * 100
        const heightPct = (h / imageHeight) * 100
        const isActive = index === currentObjectIndex
        
        return (
          <div
            key={`bbox-${obj.object_id}`}
            className={`absolute border-2 rounded ${
              isActive 
                ? 'border-green-400 shadow-[0_0_8px_rgba(34,197,94,0.6)]' 
                : 'border-blue-400 shadow-[0_0_4px_rgba(59,130,246,0.4)]'
            }`}
            style={{ 
              left: `${leftPct}%`, 
              top: `${topPct}%`, 
              width: `${widthPct}%`, 
              height: `${heightPct}%` 
            }}
          />
        )
      })}
      
      {/* Object Labels */}
      {objects.map((obj, index) => {
        const [x, y, w, h] = obj.bbox
        const leftPct = (x / imageWidth) * 100
        const topPct = (y / imageHeight) * 100
        const isActive = index === currentObjectIndex
        
        return (
          <div
            key={`label-${obj.object_id}`}
            className={`absolute text-xs font-semibold px-2 py-1 rounded-full ${
              isActive 
                ? 'bg-green-500 text-white shadow-lg' 
                : 'bg-blue-500 text-white shadow-md'
            }`}
            style={{ 
              left: `${leftPct}%`, 
              top: `${Math.max(0, topPct - 2)}%`,
              transform: 'translateY(-100%)'
            }}
          >
            {obj.category} ({(obj.confidence * 100).toFixed(0)}%)
          </div>
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
  isUpdating?: boolean
  isApprovingScene?: boolean
  isRejectingScene?: boolean
}

export function ReviewInterface({
  scene,
  currentObjectIndex,
  onObjectUpdate,
  onNext,
  onPrevious,
  onApproveScene,
  onRejectScene,
  isLastObject,
  isUpdating = false,
  isApprovingScene = false,
  isRejectingScene = false
}: ReviewInterfaceProps) {
  const [showTagEditor, setShowTagEditor] = useState(false)
  const [showProductMatcher, setShowProductMatcher] = useState(false)
  const [showMasks, setShowMasks] = useState(true)
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(false)
  const [imageDimensions, setImageDimensions] = useState({ width: 1920, height: 1080 })
  
  const currentObject = scene.objects[currentObjectIndex]
  
  // Handle image load to get actual dimensions
  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget
    setImageDimensions({ width: img.naturalWidth, height: img.naturalHeight })
  }
  
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
        
        <div className="ml-auto">
          <SegmentationQualityBadge object={currentObject} />
        </div>
      </div>
      
      {/* Image and Object Overlay */}
      <div className="flex-1 relative bg-black rounded-lg overflow-hidden">
        <img
          src={scene.image_url}
          alt={`Room scene ${scene.scene_id}`}
          className="w-full h-full object-contain"
          onLoad={handleImageLoad}
        />
        
        <ObjectOverlay
          objects={scene.objects}
          currentObjectIndex={currentObjectIndex}
          imageWidth={imageDimensions.width}
          imageHeight={imageDimensions.height}
          showMasks={showMasks}
          showBoundingBoxes={showBoundingBoxes}
        />
      </div>
      
      {/* Object Info Panel */}
      <div className="bg-white rounded-lg p-4 mt-4 border">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold">
              Object {currentObjectIndex + 1} of {scene.objects.length}
            </h3>
            <div className="text-sm text-gray-600 space-y-1">
              <p>
                Category: <span className="font-medium">{currentObject.category}</span>
                {' • '}
                Confidence: <span className="font-medium">{(currentObject.confidence * 100).toFixed(1)}%</span>
              </p>
              <p>
                Segmentation: <span className={`font-medium ${currentObject.mask_url ? 'text-green-600' : 'text-orange-600'}`}>
                  {currentObject.mask_url ? '✅ SAM2 Neural Network' : '⚠️ Bounding Box Fallback'}
                </span>
              </p>
              {currentObject.bbox && (
                <p className="text-xs">
                  Bbox: [{currentObject.bbox.map(n => n.toFixed(0)).join(', ')}]
                  {currentObject.mask_url && ' + Neural Mask'}
                </p>
              )}
            </div>
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
              disabled={isUpdating || isApprovingScene || isRejectingScene}
              className="flex items-center gap-2 px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Reject (R)"
            >
              {isUpdating || isRejectingScene ? <Loader2 size={16} className="animate-spin" /> : <X size={16} />}
              Reject
            </button>
            
            <button
              onClick={handleApprove}
              disabled={isUpdating || isApprovingScene || isRejectingScene}
              className="flex items-center gap-2 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Approve (A)"
            >
              {isUpdating || isApprovingScene ? <Loader2 size={16} className="animate-spin" /> : <Check size={16} />}
              {isLastObject ? 'Complete Scene' : 'Approve'}
            </button>
          </div>
        </div>
        
        {/* Scene Segmentation Statistics */}
        <div className="mt-4 pt-4 border-t">
          <div className="grid grid-cols-3 gap-4 text-center">
            <div className="bg-green-50 rounded-lg p-2 border border-green-200">
              <div className="text-lg font-bold text-green-800">
                {scene.objects.filter(obj => obj.mask_url).length}
              </div>
              <div className="text-xs text-green-600">SAM2 Segmented</div>
            </div>
            <div className="bg-orange-50 rounded-lg p-2 border border-orange-200">
              <div className="text-lg font-bold text-orange-800">
                {scene.objects.filter(obj => !obj.mask_url).length}
              </div>
              <div className="text-xs text-orange-600">Bounding Box Only</div>
            </div>
            <div className="bg-blue-50 rounded-lg p-2 border border-blue-200">
              <div className="text-lg font-bold text-blue-800">
                {((scene.objects.filter(obj => obj.mask_url).length / scene.objects.length) * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-blue-600">Segmentation Rate</div>
            </div>
          </div>
        </div>
        
        {/* Keyboard Shortcuts Help */}
        <div className="mt-3 pt-3 border-t text-xs text-gray-500">
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