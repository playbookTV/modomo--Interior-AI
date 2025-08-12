import React, { useEffect, useMemo, useState } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { getReviewQueue, updateReview, approveScene } from '../api/client'
import { ReviewInterface } from '../components/ReviewInterface'
import type { DetectedObject, Scene } from '../types'

export function SceneDetail() {
  const { sceneId } = useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data: scenes, isLoading, isError } = useQuery({
    queryKey: ['review-queue'],
    queryFn: () => getReviewQueue({})
  })

  const initialScene = useMemo<Scene | undefined>(
    () => scenes?.find((s) => s.scene_id === sceneId),
    [scenes, sceneId]
  )

  const [localScene, setLocalScene] = useState<Scene | undefined>(initialScene)

  useEffect(() => {
    // Sync local state when data (or route) changes
    if (initialScene) setLocalScene(initialScene)
  }, [initialScene])

  const firstUnreviewedIndex = useMemo(() => {
    if (!localScene?.objects?.length) return 0
    const idx = localScene.objects.findIndex((o) => o.approved == null)
    return idx >= 0 ? idx : 0
  }, [localScene])

  const [currentObjectIndex, setCurrentObjectIndex] = useState<number>(0)

  useEffect(() => {
    // Initialize to the first unreviewed object when scene loads
    if (localScene) setCurrentObjectIndex(firstUnreviewedIndex)
  }, [localScene, firstUnreviewedIndex])

  const updateReviewMutation = useMutation({
    mutationFn: (payload: { object_id: string; updates: Partial<DetectedObject> }) => {
      const { object_id, updates } = payload
      const sanitized = {
        object_id,
        category: updates.category,
        tags: updates.tags,
        matched_product_id: updates.matched_product_id,
        approved: updates.approved === null ? undefined : updates.approved,
      }
      return updateReview([sanitized])
    },
    onError: (error: any) => {
      console.error('Failed to update review:', error.message)
      // TODO: Add toast notification or error display
    }
  })

  const approveSceneMutation = useMutation({
    mutationFn: (id: string) => approveScene(id),
    onSuccess: () => {
      // Invalidate review queue cache to remove completed scene
      queryClient.invalidateQueries({ queryKey: ['review-queue'] })
      navigate('/review')
    },
    onError: (error: any) => {
      console.error('Failed to approve scene:', error.message)
      // TODO: Add toast notification or error display
    }
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  if (isError || !localScene) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-bold text-gray-900">Scene not found</h1>
        <p className="text-gray-600">We couldn't load the requested scene.</p>
        <Link to="/review" className="inline-block px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
          Back to Review Queue
        </Link>
      </div>
    )
  }

  const handleObjectUpdate = (objectId: string, updates: Partial<DetectedObject>) => {
    // Prevent multiple concurrent requests
    if (updateReviewMutation.isPending) {
      console.log('Update already in progress, skipping...')
      return
    }

    // Optimistic local update
    setLocalScene((prev) => {
      if (!prev) return prev
      const updatedObjects = prev.objects.map((obj) =>
        obj.object_id === objectId ? { ...obj, ...updates } : obj
      )
      return { ...prev, objects: updatedObjects }
    })
    updateReviewMutation.mutate({ object_id: objectId, updates })
  }

  const handleApproveScene = () => {
    if (!localScene || approveSceneMutation.isPending) {
      console.log('Scene approval already in progress, skipping...')
      return
    }
    approveSceneMutation.mutate(localScene.scene_id)
  }

  const handleRejectScene = () => {
    // Currently, rejecting completes the scene similarly to approve flow per existing component logic
    if (!localScene) return
    approveSceneMutation.mutate(localScene.scene_id)
  }

  const handleNext = () => {
    setCurrentObjectIndex((idx) => Math.min(idx + 1, (localScene.objects.length - 1)))
  }

  const handlePrevious = () => {
    setCurrentObjectIndex((idx) => Math.max(idx - 1, 0))
  }

  const isLastObject = currentObjectIndex >= localScene.objects.length - 1

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Scene Detail</h1>
          <p className="text-gray-600">Scene ID: {localScene.scene_id}</p>
        </div>
        <div>
          <button
            onClick={() => navigate('/review')}
            className="px-4 py-2 text-sm border rounded-lg hover:bg-gray-50"
          >
            Back to Queue
          </button>
        </div>
      </div>

      <div className="bg-white rounded-lg p-4 shadow-sm border">
        <ReviewInterface
          scene={localScene}
          currentObjectIndex={currentObjectIndex}
          onObjectUpdate={handleObjectUpdate}
          onNext={handleNext}
          onPrevious={handlePrevious}
          onApproveScene={handleApproveScene}
          onRejectScene={handleRejectScene}
          isLastObject={isLastObject}
          isUpdating={updateReviewMutation.isPending}
          isApprovingScene={approveSceneMutation.isPending}
        />
      </div>
    </div>
  )
}