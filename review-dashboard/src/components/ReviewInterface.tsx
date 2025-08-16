import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react'
import { useHotkeys } from 'react-hotkeys-hook'
import { Check, X, Edit3, Package, Loader2, Eye, EyeOff, Target } from 'lucide-react'
import { DetectedObject, Scene, Product } from '../types'
import { searchProducts } from '../api/client'
import { MapOverlays, MapControls, SceneMapData } from './MapOverlays'

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
  const [maskLoaded, setMaskLoaded] = useState(false)
  const [maskError, setMaskError] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  
  const drawColoredMask = useCallback(() => {
    const canvas = canvasRef.current
    const image = imageRef.current
    if (!canvas || !image || !maskLoaded) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    // Set canvas size to match container
    const container = canvas.parentElement
    if (container) {
      canvas.width = container.clientWidth
      canvas.height = container.clientHeight
    }
    
    // Clear canvas with transparent background
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    // Calculate scaling to fit mask image
    const aspectRatio = image.naturalWidth / image.naturalHeight
    const containerRatio = canvas.width / canvas.height
    
    let drawWidth, drawHeight, drawX, drawY
    
    if (aspectRatio > containerRatio) {
      drawWidth = canvas.width
      drawHeight = canvas.width / aspectRatio
      drawX = 0
      drawY = (canvas.height - drawHeight) / 2
    } else {
      drawHeight = canvas.height
      drawWidth = canvas.height * aspectRatio
      drawX = (canvas.width - drawWidth) / 2
      drawY = 0
    }
    
    // Create a temporary canvas for processing
    const tempCanvas = document.createElement('canvas')
    const tempCtx = tempCanvas.getContext('2d')
    if (!tempCtx) return
    
    tempCanvas.width = drawWidth
    tempCanvas.height = drawHeight
    
    // Draw mask on temp canvas
    tempCtx.drawImage(image, 0, 0, drawWidth, drawHeight)
    
    // Get pixel data
    const imageData = tempCtx.getImageData(0, 0, drawWidth, drawHeight)
    const data = imageData.data
    
    // Process pixels - only color non-transparent areas
    const color = isActive ? [34, 197, 94, 150] : [59, 130, 246, 100] // Green/Blue with alpha
    
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i]
      const g = data[i + 1] 
      const b = data[i + 2]
      const a = data[i + 3]
      
      // If pixel is not transparent and has some brightness (white/gray parts of mask)
      if (a > 50 && (r > 100 || g > 100 || b > 100)) {
        data[i] = color[0]     // R
        data[i + 1] = color[1] // G  
        data[i + 2] = color[2] // B
        data[i + 3] = color[3] // A
      } else {
        // Make dark/transparent parts fully transparent
        data[i + 3] = 0
      }
    }
    
    // Put processed data back
    tempCtx.putImageData(imageData, 0, 0)
    
    // Draw the colored mask onto main canvas
    ctx.drawImage(tempCanvas, drawX, drawY)
  }, [maskUrl, isActive, maskLoaded])
  
  useEffect(() => {
    if (maskLoaded) {
      drawColoredMask()
    }
  }, [maskLoaded, drawColoredMask])
  
  // Redraw on window resize
  useEffect(() => {
    const handleResize = () => {
      if (maskLoaded) {
        setTimeout(drawColoredMask, 100)
      }
    }
    
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [drawColoredMask, maskLoaded])
  
  return (
    <div className="absolute inset-0 pointer-events-none">
      {/* Hidden image to load the mask */}
      <img
        ref={imageRef}
        src={maskUrl}
        alt="Object mask"
        className="hidden"
        onLoad={() => {
          setMaskLoaded(true)
          setMaskError(false)
          console.log('SAM2 mask loaded successfully:', maskUrl)
          setTimeout(drawColoredMask, 50) // Small delay to ensure canvas is ready
        }}
        onError={(e) => {
          setMaskError(true)
          setMaskLoaded(false)
          console.warn('Failed to load SAM2 mask:', maskUrl)
        }}
      />
      
      {/* Canvas for rendering colored mask */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
        style={{ 
          pointerEvents: 'none',
          mixBlendMode: 'normal'
        }}
      />
      
      {/* Fallback indicator for broken masks */}
      {maskError && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-xs bg-red-100 text-red-800 px-2 py-1 rounded">
            Mask Load Failed
          </div>
        </div>
      )}
      
      {/* Debug info */}
      {maskLoaded && (
        <div className="absolute top-2 left-2 text-xs bg-green-100 text-green-800 px-2 py-1 rounded">
          SAM2 Active
        </div>
      )}
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
  
  // Map state management
  const [sceneMapData, setSceneMapData] = useState<SceneMapData | undefined>(undefined)
  const [mapVisibility, setMapVisibility] = useState({ depth: false, edge: false })
  const [isGeneratingMaps, setIsGeneratingMaps] = useState(false)
  
  // API base URL
  const API_BASE = 'https://ovalay-recruitment-production.up.railway.app'
  
  // Fetch scene map data
  const fetchSceneMapData = useCallback(async (sceneId: string) => {
    try {
      const response = await fetch(`${API_BASE}/scenes/${sceneId}/maps`)
      if (response.ok) {
        const mapData = await response.json()
        setSceneMapData(mapData)
        console.log('Scene map data loaded:', mapData)
      } else {
        console.warn('Failed to fetch scene map data:', response.status)
      }
    } catch (error) {
      console.error('Error fetching scene map data:', error)
    }
  }, [API_BASE])
  
  // Generate maps for scene
  const handleMapGenerate = useCallback(async (sceneId: string, mapTypes: string[]) => {
    try {
      setIsGeneratingMaps(true)
      console.log('Generating maps for scene:', sceneId, mapTypes)
      
      const response = await fetch(`${API_BASE}/generate-maps/${sceneId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ map_types: mapTypes })
      })
      
      if (response.ok) {
        const result = await response.json()
        console.log('Map generation result:', result)
        
        // Refresh scene map data
        await fetchSceneMapData(sceneId)
        
        // Show generated maps
        if (mapTypes.includes('depth')) setMapVisibility(prev => ({ ...prev, depth: true }))
        if (mapTypes.includes('edge')) setMapVisibility(prev => ({ ...prev, edge: true }))
      } else {
        console.error('Map generation failed:', response.status)
      }
    } catch (error) {
      console.error('Error generating maps:', error)
    } finally {
      setIsGeneratingMaps(false)
    }
  }, [API_BASE, fetchSceneMapData])
  
  // Handle map visibility changes
  const handleMapVisibilityChange = useCallback((mapType: string, visible: boolean) => {
    setMapVisibility(prev => ({ ...prev, [mapType]: visible }))
  }, [])
  
  // Load scene map data when scene changes
  useEffect(() => {
    if (scene?.scene_id) {
      fetchSceneMapData(scene.scene_id)
    }
  }, [scene?.scene_id, fetchSceneMapData])
  
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
        
        {/* Map visibility controls */}
        {(sceneMapData?.maps_available?.depth || sceneMapData?.maps_available?.edge) && (
          <>
            {sceneMapData?.maps_available?.depth && (
              <button
                onClick={() => handleMapVisibilityChange('depth', !mapVisibility.depth)}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
                  mapVisibility.depth 
                    ? 'bg-purple-500 text-white shadow-md' 
                    : 'bg-white text-gray-600 border border-gray-300 hover:border-purple-400'
                }`}
              >
                {mapVisibility.depth ? <Eye size={14} /> : <EyeOff size={14} />}
                Depth Map
              </button>
            )}
            
            {sceneMapData?.maps_available?.edge && (
              <button
                onClick={() => handleMapVisibilityChange('edge', !mapVisibility.edge)}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
                  mapVisibility.edge 
                    ? 'bg-orange-500 text-white shadow-md' 
                    : 'bg-white text-gray-600 border border-gray-300 hover:border-orange-400'
                }`}
              >
                {mapVisibility.edge ? <Eye size={14} /> : <EyeOff size={14} />}
                Edge Map
              </button>
            )}
          </>
        )}
        
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
        
        {/* Map Overlays */}
        <MapOverlays
          sceneId={scene.scene_id}
          sceneMapData={sceneMapData}
          mapVisibility={mapVisibility}
        />
        
        {/* Map Controls */}
        <MapControls
          sceneId={scene.scene_id}
          onMapGenerate={handleMapGenerate}
          isGenerating={isGeneratingMaps}
          sceneMapData={sceneMapData}
          onVisibilityChange={handleMapVisibilityChange}
          mapVisibility={mapVisibility}
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