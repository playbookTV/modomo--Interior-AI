import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Search, Palette, Eye } from 'lucide-react'

interface ColorSearchResult {
  object_id: string
  scene_id: string
  category: string
  confidence: number
  tags: string[]
  colors: any
  similarity: number
}

interface ColorSearchProps {
  onObjectSelect?: (objectId: string, sceneId: string) => void
}

async function searchObjectsByColor(query: string, threshold: number = 0.3, limit: number = 10) {
  const response = await fetch(
    `https://ovalay-recruitment-production.up.railway.app/search/color?query=${encodeURIComponent(query)}&threshold=${threshold}&limit=${limit}`
  )
  if (!response.ok) throw new Error('Color search failed')
  return response.json()
}

async function getColorPalette() {
  const response = await fetch(
    'https://ovalay-recruitment-production.up.railway.app/colors/palette'
  )
  if (!response.ok) throw new Error('Failed to fetch color palette')
  return response.json()
}

export function ColorSearch({ onObjectSelect }: ColorSearchProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<ColorSearchResult[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [selectedThreshold, setSelectedThreshold] = useState(0.3)

  const { data: colorPalette } = useQuery({
    queryKey: ['color-palette'],
    queryFn: getColorPalette
  })

  const handleSearch = async (query: string) => {
    if (!query.trim()) {
      setSearchResults([])
      return
    }

    setIsSearching(true)
    try {
      const results = await searchObjectsByColor(query, selectedThreshold, 20)
      setSearchResults(results.results || [])
    } catch (error) {
      console.error('Color search failed:', error)
      setSearchResults([])
    } finally {
      setIsSearching(false)
    }
  }

  const handleQuickSearch = (colorQuery: string) => {
    setSearchQuery(colorQuery)
    handleSearch(colorQuery)
  }

  const commonQueries = [
    'red sofa', 'blue curtains', 'brown table', 'white cabinet',
    'black chair', 'gray rug', 'green pillow', 'yellow lamp',
    'wood furniture', 'leather sofa', 'dark wood table', 'light colors'
  ]

  return (
    <div className="space-y-6">
      {/* Search Header */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          <Palette className="h-5 w-5 text-purple-600" />
          <h2 className="text-xl font-bold text-gray-900">Color Search</h2>
        </div>
        <p className="text-gray-600">Search for furniture by color using AI-powered semantic matching</p>
      </div>

      {/* Search Input */}
      <div className="space-y-4">
        <div className="flex gap-2">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <input
              type="text"
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
              placeholder="e.g., 'red sofa', 'dark wood table', 'blue curtains'"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSearch(searchQuery)}
            />
          </div>
          <button
            onClick={() => handleSearch(searchQuery)}
            disabled={isSearching}
            className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50"
          >
            {isSearching ? 'Searching...' : 'Search'}
          </button>
        </div>

        {/* Search Options */}
        <div className="flex items-center gap-4 text-sm">
          <label className="flex items-center gap-2">
            <span className="text-gray-600">Similarity Threshold:</span>
            <select 
              className="border border-gray-300 rounded px-2 py-1"
              value={selectedThreshold}
              onChange={(e) => setSelectedThreshold(parseFloat(e.target.value))}
            >
              <option value={0.1}>Low (0.1)</option>
              <option value={0.2}>Medium-Low (0.2)</option>
              <option value={0.3}>Medium (0.3)</option>
              <option value={0.4}>Medium-High (0.4)</option>
              <option value={0.5}>High (0.5)</option>
            </select>
          </label>
        </div>
      </div>

      {/* Quick Search Buttons */}
      <div>
        <p className="text-sm font-medium text-gray-700 mb-2">Quick Searches:</p>
        <div className="flex flex-wrap gap-2">
          {commonQueries.map((query) => (
            <button
              key={query}
              onClick={() => handleQuickSearch(query)}
              className="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm hover:bg-gray-200 transition-colors"
            >
              {query}
            </button>
          ))}
        </div>
      </div>

      {/* Color Palette Reference */}
      {colorPalette?.color_categories && (
        <div className="border border-gray-200 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-700 mb-3">Available Colors:</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(colorPalette.color_categories).map(([category, colors]: [string, any]) => (
              <div key={category}>
                <p className="text-xs font-medium text-gray-600 mb-2 capitalize">
                  {category.replace('_', ' ')}
                </p>
                <div className="flex flex-wrap gap-1">
                  {colors.slice(0, 6).map((color: string) => (
                    <button
                      key={color}
                      onClick={() => handleQuickSearch(`${color} furniture`)}
                      className="px-2 py-1 bg-gray-50 text-gray-600 rounded text-xs hover:bg-gray-100 capitalize"
                      title={`Search for ${color} furniture`}
                    >
                      {color.replace('_', ' ')}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Search Results */}
      {searchResults.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">
              Search Results ({searchResults.length})
            </h3>
            <p className="text-sm text-gray-600">
              Query: "{searchQuery}" | Threshold: {selectedThreshold}
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {searchResults.map((result) => (
              <div 
                key={result.object_id}
                className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
                onClick={() => onObjectSelect?.(result.object_id, result.scene_id)}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-gray-900 capitalize">{result.category}</span>
                  <span className="text-sm text-green-600 font-medium">
                    {(result.similarity * 100).toFixed(1)}% match
                  </span>
                </div>

                <div className="text-sm text-gray-600 mb-2">
                  Confidence: {(result.confidence * 100).toFixed(1)}%
                </div>

                {/* Object Colors */}
                {result.colors?.colors && (
                  <div className="mb-3">
                    <p className="text-xs text-gray-500 mb-1">Colors:</p>
                    <div className="flex gap-1">
                      {result.colors.colors.slice(0, 4).map((color: any, idx: number) => (
                        <div
                          key={idx}
                          className="w-4 h-4 rounded border border-gray-300"
                          style={{ backgroundColor: color.hex }}
                          title={`${color.name} (${color.percentage}%)`}
                        />
                      ))}
                    </div>
                  </div>
                )}

                {/* Tags */}
                {result.tags && result.tags.length > 0 && (
                  <div className="mb-3">
                    <div className="flex flex-wrap gap-1">
                      {result.tags.slice(0, 3).map((tag, idx) => (
                        <span
                          key={idx}
                          className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                <div className="flex items-center justify-between text-xs text-gray-500">
                  <span>Object ID: {result.object_id.slice(0, 8)}...</span>
                  <button className="flex items-center gap-1 text-purple-600 hover:text-purple-800">
                    <Eye className="h-3 w-3" />
                    View
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* No Results */}
      {searchQuery && !isSearching && searchResults.length === 0 && (
        <div className="text-center py-8">
          <div className="text-gray-400 mb-2">
            <Search className="h-8 w-8 mx-auto" />
          </div>
          <p className="text-gray-600">No objects found matching "{searchQuery}"</p>
          <p className="text-sm text-gray-500 mt-1">
            Try adjusting your search terms or lowering the similarity threshold
          </p>
        </div>
      )}
    </div>
  )
}