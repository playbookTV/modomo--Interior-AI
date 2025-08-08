import React from 'react'
import { useParams } from 'react-router-dom'

export function SceneDetail() {
  const { sceneId } = useParams()

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Scene Detail</h1>
        <p className="text-gray-600">Scene ID: {sceneId}</p>
      </div>

      <div className="bg-white rounded-lg p-6 shadow-sm border">
        <p className="text-gray-600">Scene details and review interface will be implemented here.</p>
      </div>
    </div>
  )
}