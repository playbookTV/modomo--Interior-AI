import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getReviewQueue } from '../api/client'
import { Link } from 'react-router-dom'
import { Search, Filter, ChevronLeft, ChevronRight, Eye, EyeOff, Target } from 'lucide-react'

export function ReviewQueue() {
  // State for filters and pagination
  const [filters, setFilters] = useState({
    search: '',
    room_type: '',
    category: '',
    status: '',
    image_type: '',
    has_masks: null as boolean | null,
    order_by: 'created_at',
    order_dir: 'desc'
  })
  
  const [pagination, setPagination] = useState({
    limit: 20,
    offset: 0
  })

  // Fetch review queue with current filters and pagination
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['review-queue', filters, pagination],
    queryFn: () => getReviewQueue({
      ...filters,
      ...pagination,
      has_masks: filters.has_masks === null ? undefined : filters.has_masks
    }),
    retry: 2,
    retryDelay: 3000,
    refetchOnWindowFocus: true,
    staleTime: 30000, // Cache for 30 seconds
  })

  const handleFilterChange = (key: string, value: any) => {
    setFilters(prev => ({ ...prev, [key]: value }))
    setPagination(prev => ({ ...prev, offset: 0 })) // Reset to first page
  }

  const handlePageChange = (newOffset: number) => {
    setPagination(prev => ({ ...prev, offset: newOffset }))
  }

  const clearFilters = () => {
    setFilters({
      search: '',
      room_type: '',
      category: '',
      status: '',
      image_type: '',
      has_masks: null,
      order_by: 'created_at',
      order_dir: 'desc'
    })
    setPagination({ limit: 20, offset: 0 })
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        <span className="ml-3 text-gray-600">Loading review queue...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <div className="text-red-500 mb-4 text-xl">⚠️ Backend Service Unavailable</div>
        <div className="text-sm text-gray-600 mb-4">
          {error.message.includes('timeout') 
            ? 'The backend service is not responding. This could be due to the service being down or overloaded.'
            : error.message
          }
        </div>
        <button
          onClick={() => refetch()}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          Retry Connection
        </button>
      </div>
    )
  }

  const scenes = data?.scenes || []
  const paginationInfo = data?.pagination || { total: 0, current_page: 1, total_pages: 1, has_more: false }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Review Queue</h1>
        <p className="text-gray-600">Review and validate detected objects in scenes</p>
      </div>

      {/* Filters and Search */}
      <div className="bg-white rounded-lg p-6 shadow-sm border">
        <div className="flex items-center gap-4 mb-4">
          <Filter size={20} className="text-gray-500" />
          <h3 className="text-lg font-medium text-gray-900">Filters & Search</h3>
          <button
            onClick={clearFilters}
            className="ml-auto px-3 py-1 text-sm bg-gray-100 text-gray-600 rounded hover:bg-gray-200"
          >
            Clear All
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Search */}
          <div className="relative">
            <Search size={16} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              placeholder="Search scenes..."
              value={filters.search}
              onChange={(e) => handleFilterChange('search', e.target.value)}
              className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>

          {/* Room Type */}
          <select
            value={filters.room_type}
            onChange={(e) => handleFilterChange('room_type', e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="">All Room Types</option>
            <option value="living_room">Living Room</option>
            <option value="bedroom">Bedroom</option>
            <option value="kitchen">Kitchen</option>
            <option value="bathroom">Bathroom</option>
            <option value="dining_room">Dining Room</option>
            <option value="office">Office</option>
          </select>

          {/* Status */}
          <select
            value={filters.status}
            onChange={(e) => handleFilterChange('status', e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="">Pending Review</option>
            <option value="scraped">Scraped</option>
            <option value="processing">Processing</option>
            <option value="pending_review">Pending Review</option>
            <option value="approved">Approved</option>
            <option value="rejected">Rejected</option>
          </select>

          {/* SAM2 Masks Filter */}
          <select
            value={filters.has_masks === null ? '' : filters.has_masks.toString()}
            onChange={(e) => handleFilterChange('has_masks', e.target.value === '' ? null : e.target.value === 'true')}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="">All Scenes</option>
            <option value="true">With SAM2 Masks</option>
            <option value="false">Bounding Box Only</option>
          </select>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
          {/* Category Filter */}
          <input
            type="text"
            placeholder="Filter by category..."
            value={filters.category}
            onChange={(e) => handleFilterChange('category', e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />

          {/* Order By */}
          <select
            value={filters.order_by}
            onChange={(e) => handleFilterChange('order_by', e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="created_at">Created Date</option>
            <option value="room_type">Room Type</option>
            <option value="status">Status</option>
          </select>

          {/* Order Direction */}
          <select
            value={filters.order_dir}
            onChange={(e) => handleFilterChange('order_dir', e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="desc">Newest First</option>
            <option value="asc">Oldest First</option>
          </select>
        </div>
      </div>

      {/* Results Summary */}
      <div className="flex items-center justify-between">
        <div className="text-sm text-gray-600">
          Showing {scenes.length} of {paginationInfo.total} scenes
          {data?.filters_applied && Object.values(data.filters_applied).some(v => v) && (
            <span className="ml-2 px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs">
              Filtered
            </span>
          )}
        </div>
        
        <div className="text-sm text-gray-500">
          Page {paginationInfo.current_page} of {paginationInfo.total_pages}
        </div>
      </div>

      {/* Scenes Grid */}
      {scenes.length === 0 ? (
        <div className="text-center py-12">
          <div className="text-gray-500 mb-4">
            <svg className="h-12 w-12 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2 2v-5m16 0h-6m-4 0H4" />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No scenes found</h3>
          <p className="text-gray-600">Try adjusting your filters or start by scraping some scenes.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6">
          {scenes.map((scene) => {
            const masksCount = scene.objects?.filter(obj => obj.mask_url).length || 0
            const bboxCount = (scene.objects?.length || 0) - masksCount
            
            return (
              <div key={scene.scene_id} className="bg-white rounded-lg p-6 shadow-sm border hover:shadow-md transition-shadow">
                <div className="flex items-start space-x-4">
                  <img
                    src={scene.image_url}
                    alt="Scene"
                    className="w-32 h-32 object-cover rounded-lg"
                  />
                  <div className="flex-1">
                    <div className="flex items-start justify-between">
                      <div>
                        <h3 className="font-medium text-gray-900">{scene.houzz_id || scene.scene_id}</h3>
                        <p className="text-sm text-gray-600">Room: {scene.room_type || 'Unknown'}</p>
                        <p className="text-sm text-gray-600">Status: {scene.status}</p>
                        {scene.image_type && (
                          <p className="text-sm text-gray-600">Type: {scene.image_type}</p>
                        )}
                      </div>
                      
                      {/* Object Stats */}
                      <div className="text-right">
                        <div className="text-lg font-bold text-gray-900">
                          {scene.objects?.length || 0} objects
                        </div>
                        <div className="flex items-center gap-2 text-sm">
                          {masksCount > 0 && (
                            <span className="flex items-center gap-1 text-green-600">
                              <Target size={12} />
                              {masksCount} SAM2
                            </span>
                          )}
                          {bboxCount > 0 && (
                            <span className="flex items-center gap-1 text-orange-600">
                              <div className="w-3 h-3 border border-orange-600 rounded-sm" />
                              {bboxCount} bbox
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                    
                    {/* Style Tags */}
                    {scene.style_tags && scene.style_tags.length > 0 && (
                      <div className="mt-2">
                        <div className="flex flex-wrap gap-1">
                          {scene.style_tags.slice(0, 3).map((tag, index) => (
                            <span key={index} className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full">
                              {tag}
                            </span>
                          ))}
                          {scene.style_tags.length > 3 && (
                            <span className="px-2 py-1 bg-gray-100 text-gray-500 text-xs rounded-full">
                              +{scene.style_tags.length - 3} more
                            </span>
                          )}
                        </div>
                      </div>
                    )}
                    
                    <div className="mt-4">
                      <Link 
                        to={`/scene/${scene.scene_id}`} 
                        className="inline-block px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                      >
                        Start Review
                      </Link>
                    </div>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Pagination */}
      {paginationInfo.total_pages > 1 && (
        <div className="flex items-center justify-between pt-6">
          <button
            onClick={() => handlePageChange(Math.max(0, pagination.offset - pagination.limit))}
            disabled={pagination.offset === 0}
            className="flex items-center gap-2 px-4 py-2 border border-gray-300 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
          >
            <ChevronLeft size={16} />
            Previous
          </button>
          
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600">
              Page {paginationInfo.current_page} of {paginationInfo.total_pages}
            </span>
          </div>
          
          <button
            onClick={() => handlePageChange(pagination.offset + pagination.limit)}
            disabled={!paginationInfo.has_more}
            className="flex items-center gap-2 px-4 py-2 border border-gray-300 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
          >
            Next
            <ChevronRight size={16} />
          </button>
        </div>
      )}

      {/* Debug Info */}
      {data?.debug && (
        <div className="text-xs text-gray-400 mt-4 p-3 bg-gray-50 rounded">
          Debug: {data.debug.total_scenes_in_db} total scenes, {data.debug.scenes_after_filters} after filters
        </div>
      )}
    </div>
  )
}