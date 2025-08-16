import React, { useState, useCallback, useRef, useEffect } from 'react'
import { Layers, Eye, EyeOff, Loader2, AlertCircle, Mountain, Zap } from 'lucide-react'

// Types for map data
interface SceneMapData {
  scene_id: string
  maps_available: {
    depth?: {
      r2_key: string
      url: string
    }
    edge?: {
      r2_key: string
      url: string
    }
  }
  maps_generated_at?: string
  maps_metadata?: any
}

interface MapOverlayProps {
  mapUrl: string
  mapType: 'depth' | 'edge'
  isVisible: boolean
  opacity?: number
  blendMode?: string
}

interface MapControlsProps {
  sceneId: string
  onMapGenerate: (sceneId: string, mapTypes: string[]) => void
  isGenerating?: boolean
  sceneMapData?: SceneMapData
  onVisibilityChange: (mapType: string, visible: boolean) => void
  mapVisibility: { depth: boolean; edge: boolean }
}

function DepthMapOverlay({ mapUrl, isVisible, opacity = 0.6 }: MapOverlayProps) {
  const [mapLoaded, setMapLoaded] = useState(false)
  const [mapError, setMapError] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  
  const drawDepthMap = useCallback(() => {
    const canvas = canvasRef.current
    const image = imageRef.current
    if (!canvas || !image || !mapLoaded || !isVisible) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    // Set canvas size to match container
    const container = canvas.parentElement
    if (container) {
      canvas.width = container.clientWidth
      canvas.height = container.clientHeight
    }
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    if (!isVisible) return
    
    // Calculate scaling to fit depth map
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
    
    // Draw depth map with opacity
    ctx.globalAlpha = opacity
    ctx.drawImage(image, drawX, drawY, drawWidth, drawHeight)
    ctx.globalAlpha = 1.0
    
  }, [mapUrl, isVisible, mapLoaded, opacity])
  
  useEffect(() => {
    if (mapLoaded) {
      drawDepthMap()
    }
  }, [mapLoaded, drawDepthMap, isVisible])
  
  // Redraw on window resize
  useEffect(() => {
    const handleResize = () => {
      if (mapLoaded && isVisible) {
        setTimeout(drawDepthMap, 100)
      }
    }
    
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [drawDepthMap, mapLoaded, isVisible])
  
  if (!isVisible) return null
  
  return (
    <div className="absolute inset-0 pointer-events-none">
      {/* Hidden image to load the depth map */}
      <img
        ref={imageRef}
        src={mapUrl}
        alt="Depth map"
        className="hidden"
        onLoad={() => {
          setMapLoaded(true)
          setMapError(false)
          console.log('Depth map loaded successfully:', mapUrl)
          setTimeout(drawDepthMap, 50)
        }}
        onError={(e) => {
          setMapError(true)
          setMapLoaded(false)
          console.warn('Failed to load depth map:', mapUrl)
        }}
      />
      
      {/* Canvas for rendering depth map */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
        style={{ 
          pointerEvents: 'none',
          mixBlendMode: 'multiply' // Blend with underlying image
        }}
      />
      
      {/* Error indicator */}
      {mapError && (
        <div className="absolute top-2 right-2 text-xs bg-red-100 text-red-800 px-2 py-1 rounded flex items-center gap-1">
          <AlertCircle size={12} />
          Depth Map Failed
        </div>
      )}
      
      {/* Success indicator */}
      {mapLoaded && !mapError && (
        <div className="absolute top-2 right-2 text-xs bg-purple-100 text-purple-800 px-2 py-1 rounded flex items-center gap-1">
          <Mountain size={12} />
          Depth Active
        </div>
      )}
    </div>
  )
}

function EdgeMapOverlay({ mapUrl, isVisible, opacity = 0.7 }: MapOverlayProps) {
  const [mapLoaded, setMapLoaded] = useState(false)
  const [mapError, setMapError] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  
  const drawEdgeMap = useCallback(() => {
    const canvas = canvasRef.current
    const image = imageRef.current
    if (!canvas || !image || !mapLoaded || !isVisible) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    // Set canvas size to match container
    const container = canvas.parentElement
    if (container) {
      canvas.width = container.clientWidth
      canvas.height = container.clientHeight
    }
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    if (!isVisible) return
    
    // Calculate scaling to fit edge map
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
    
    // Create temporary canvas for processing edge map
    const tempCanvas = document.createElement('canvas')
    const tempCtx = tempCanvas.getContext('2d')
    if (!tempCtx) return
    
    tempCanvas.width = drawWidth
    tempCanvas.height = drawHeight
    
    // Draw edge map on temp canvas
    tempCtx.drawImage(image, 0, 0, drawWidth, drawHeight)
    
    // Get pixel data and make edges colored
    const imageData = tempCtx.getImageData(0, 0, drawWidth, drawHeight)
    const data = imageData.data
    
    // Process pixels - color the edges
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i]
      const g = data[i + 1] 
      const b = data[i + 2]
      const a = data[i + 3]
      
      // If pixel is bright (edge detected)
      if (r > 100 || g > 100 || b > 100) {
        data[i] = 255     // R - bright edge
        data[i + 1] = 100 // G
        data[i + 2] = 0   // B - orange edges
        data[i + 3] = Math.floor(opacity * 255) // A
      } else {
        // Make non-edge parts transparent
        data[i + 3] = 0
      }
    }
    
    // Put processed data back
    tempCtx.putImageData(imageData, 0, 0)
    
    // Draw the colored edge map onto main canvas
    ctx.drawImage(tempCanvas, drawX, drawY)
    
  }, [mapUrl, isVisible, mapLoaded, opacity])
  
  useEffect(() => {
    if (mapLoaded) {
      drawEdgeMap()
    }
  }, [mapLoaded, drawEdgeMap, isVisible])
  
  // Redraw on window resize
  useEffect(() => {
    const handleResize = () => {
      if (mapLoaded && isVisible) {
        setTimeout(drawEdgeMap, 100)
      }
    }
    
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [drawEdgeMap, mapLoaded, isVisible])
  
  if (!isVisible) return null
  
  return (
    <div className="absolute inset-0 pointer-events-none">
      {/* Hidden image to load the edge map */}
      <img
        ref={imageRef}
        src={mapUrl}
        alt="Edge map"
        className="hidden"
        onLoad={() => {
          setMapLoaded(true)
          setMapError(false)
          console.log('Edge map loaded successfully:', mapUrl)
          setTimeout(drawEdgeMap, 50)
        }}
        onError={(e) => {
          setMapError(true)
          setMapLoaded(false)
          console.warn('Failed to load edge map:', mapUrl)
        }}
      />
      
      {/* Canvas for rendering edge map */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
        style={{ 
          pointerEvents: 'none',
          mixBlendMode: 'normal'
        }}
      />
      
      {/* Error indicator */}
      {mapError && (
        <div className="absolute top-8 right-2 text-xs bg-red-100 text-red-800 px-2 py-1 rounded flex items-center gap-1">
          <AlertCircle size={12} />
          Edge Map Failed
        </div>
      )}
      
      {/* Success indicator */}
      {mapLoaded && !mapError && (
        <div className="absolute top-8 right-2 text-xs bg-orange-100 text-orange-800 px-2 py-1 rounded flex items-center gap-1">
          <Zap size={12} />
          Edges Active
        </div>
      )}
    </div>
  )
}

export function MapControls({
  sceneId,
  onMapGenerate,
  isGenerating = false,
  sceneMapData,
  onVisibilityChange,
  mapVisibility
}: MapControlsProps) {
  const [showControls, setShowControls] = useState(false)
  
  const hasDepthMap = sceneMapData?.maps_available?.depth
  const hasEdgeMap = sceneMapData?.maps_available?.edge
  const hasAnyMaps = hasDepthMap || hasEdgeMap
  
  const handleGenerateMaps = () => {
    const mapTypes = []
    if (!hasDepthMap) mapTypes.push('depth')
    if (!hasEdgeMap) mapTypes.push('edge')
    
    if (mapTypes.length > 0) {
      onMapGenerate(sceneId, mapTypes)
    }
  }
  
  const handleGenerateSpecific = (mapType: string) => {
    onMapGenerate(sceneId, [mapType])
  }
  
  return (
    <div className="absolute bottom-4 left-4 z-10">
      {/* Main toggle button */}
      <button
        onClick={() => setShowControls(!showControls)}
        className={`flex items-center gap-2 px-3 py-2 rounded-lg shadow-lg transition-all ${
          showControls 
            ? 'bg-blue-600 text-white' 
            : 'bg-white text-gray-700 hover:bg-gray-50'
        }`}
      >
        <Layers size={16} />
        <span className="text-sm font-medium">Maps</span>
        {hasAnyMaps && (
          <div className="w-2 h-2 bg-green-400 rounded-full"></div>
        )}
      </button>
      
      {/* Expanded controls */}
      {showControls && (
        <div className="mt-2 bg-white rounded-lg shadow-xl border p-4 min-w-64">
          <h3 className="text-sm font-semibold text-gray-900 mb-3">Map Analysis</h3>
          
          {/* Generation section */}
          <div className="mb-4">
            <h4 className="text-xs font-medium text-gray-700 mb-2">Generate Maps</h4>
            
            {!hasAnyMaps ? (
              <button
                onClick={handleGenerateMaps}
                disabled={isGenerating}
                className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
              >
                {isGenerating ? (
                  <>
                    <Loader2 size={14} className="animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Mountain size={14} />
                    Generate Depth & Edge Maps
                  </>
                )}
              </button>
            ) : (
              <div className="space-y-2">
                {!hasDepthMap && (
                  <button
                    onClick={() => handleGenerateSpecific('depth')}
                    disabled={isGenerating}
                    className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50 text-sm"
                  >
                    {isGenerating ? (
                      <Loader2 size={14} className="animate-spin" />
                    ) : (
                      <Mountain size={14} />
                    )}
                    Generate Depth Map
                  </button>
                )}
                
                {!hasEdgeMap && (
                  <button
                    onClick={() => handleGenerateSpecific('edge')}
                    disabled={isGenerating}
                    className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-orange-600 text-white rounded hover:bg-orange-700 disabled:opacity-50 text-sm"
                  >
                    {isGenerating ? (
                      <Loader2 size={14} className="animate-spin" />
                    ) : (
                      <Zap size={14} />
                    )}
                    Generate Edge Map
                  </button>
                )}
              </div>
            )}
          </div>
          
          {/* Visibility controls */}
          {hasAnyMaps && (
            <div>
              <h4 className="text-xs font-medium text-gray-700 mb-2">Overlay Controls</h4>
              <div className="space-y-2">
                {hasDepthMap && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 flex items-center gap-2">
                      <Mountain size={14} className="text-purple-600" />
                      Depth Map
                    </span>
                    <button
                      onClick={() => onVisibilityChange('depth', !mapVisibility.depth)}
                      className={`p-1 rounded transition-colors ${
                        mapVisibility.depth 
                          ? 'text-purple-600 hover:text-purple-700' 
                          : 'text-gray-400 hover:text-gray-600'
                      }`}
                    >
                      {mapVisibility.depth ? <Eye size={16} /> : <EyeOff size={16} />}
                    </button>
                  </div>
                )}
                
                {hasEdgeMap && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 flex items-center gap-2">
                      <Zap size={14} className="text-orange-600" />
                      Edge Map
                    </span>
                    <button
                      onClick={() => onVisibilityChange('edge', !mapVisibility.edge)}
                      className={`p-1 rounded transition-colors ${
                        mapVisibility.edge 
                          ? 'text-orange-600 hover:text-orange-700' 
                          : 'text-gray-400 hover:text-gray-600'
                      }`}
                    >
                      {mapVisibility.edge ? <Eye size={16} /> : <EyeOff size={16} />}
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}
          
          {/* Map info */}
          {sceneMapData?.maps_generated_at && (
            <div className="mt-3 pt-3 border-t">
              <p className="text-xs text-gray-500">
                Generated: {new Date(sceneMapData.maps_generated_at).toLocaleString()}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// Main map overlays component that combines depth and edge maps
export function MapOverlays({
  sceneId,
  sceneMapData,
  mapVisibility
}: {
  sceneId: string
  sceneMapData?: SceneMapData
  mapVisibility: { depth: boolean; edge: boolean }
}) {
  const API_BASE = import.meta.env.VITE_API_BASE_URL || 'https://ovalay-recruitment-production.up.railway.app'
  
  return (
    <div className="absolute inset-0 pointer-events-none">
      {/* Depth map overlay */}
      {sceneMapData?.maps_available?.depth && (
        <DepthMapOverlay
          mapUrl={`${API_BASE}${sceneMapData.maps_available.depth.url}`}
          mapType="depth"
          isVisible={mapVisibility.depth}
          opacity={0.6}
        />
      )}
      
      {/* Edge map overlay */}
      {sceneMapData?.maps_available?.edge && (
        <EdgeMapOverlay
          mapUrl={`${API_BASE}${sceneMapData.maps_available.edge.url}`}
          mapType="edge"
          isVisible={mapVisibility.edge}
          opacity={0.7}
        />
      )}
    </div>
  )
}

export type { SceneMapData, MapOverlayProps, MapControlsProps }