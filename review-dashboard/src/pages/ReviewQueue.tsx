import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { getReviewQueue } from '../api/client'

export function ReviewQueue() {
  const { data: scenes, isLoading } = useQuery({
    queryKey: ['review-queue'],
    queryFn: () => getReviewQueue({})
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Review Queue</h1>
        <p className="text-gray-600">Review and validate detected objects in scenes</p>
      </div>

      {!scenes || scenes.length === 0 ? (
        <div className="text-center py-12">
          <div className="text-gray-500 mb-4">
            <svg className="h-12 w-12 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2 2v-5m16 0h-6m-4 0H4" />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No scenes to review</h3>
          <p className="text-gray-600">Start by scraping some scenes to populate the review queue.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6">
          {scenes.map((scene) => (
            <div key={scene.scene_id} className="bg-white rounded-lg p-6 shadow-sm border">
              <div className="flex items-start space-x-4">
                <img
                  src={scene.image_url}
                  alt="Scene"
                  className="w-32 h-32 object-cover rounded-lg"
                />
                <div className="flex-1">
                  <h3 className="font-medium text-gray-900">Scene {scene.scene_id}</h3>
                  <p className="text-sm text-gray-600">Room: {scene.room_type || 'Unknown'}</p>
                  <p className="text-sm text-gray-600">
                    Objects: {scene.objects?.length || 0} detected
                  </p>
                  <div className="mt-2">
                    <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                      Start Review
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}