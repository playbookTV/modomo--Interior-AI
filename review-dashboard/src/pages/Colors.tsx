import React, { useState } from 'react'
import { ColorSearch } from '../components/ColorSearch'
import { ColorStats } from '../components/ColorStats'
import { Search, BarChart3 } from 'lucide-react'

export function Colors() {
  const [activeTab, setActiveTab] = useState<'search' | 'stats'>('search')

  const handleObjectSelect = (objectId: string, sceneId: string) => {
    // Navigate to scene detail or show object details
    console.log('Selected object:', objectId, 'in scene:', sceneId)
    // You could integrate with React Router here
    window.open(`/scene/${sceneId}`, '_blank')
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Color Analysis</h1>
        <p className="text-gray-600">Explore and search furniture by colors using AI-powered analysis</p>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          <button
            onClick={() => setActiveTab('search')}
            className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
              activeTab === 'search'
                ? 'border-purple-500 text-purple-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <div className="flex items-center gap-2">
              <Search className="h-4 w-4" />
              Color Search
            </div>
          </button>
          
          <button
            onClick={() => setActiveTab('stats')}
            className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
              activeTab === 'stats'
                ? 'border-purple-500 text-purple-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <div className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Color Analytics
            </div>
          </button>
        </nav>
      </div>

      {/* Tab Content */}
      <div className="min-h-screen">
        {activeTab === 'search' && (
          <ColorSearch onObjectSelect={handleObjectSelect} />
        )}
        
        {activeTab === 'stats' && (
          <ColorStats />
        )}
      </div>
    </div>
  )
}