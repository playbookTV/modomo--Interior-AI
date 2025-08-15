import React from 'react'
import { CategoryStats } from '../types'

interface CategoryChartProps {
  data: CategoryStats[]
}

export function CategoryChart({ data }: CategoryChartProps) {
  // Add safety check for data array
  const safeData = Array.isArray(data) ? data : []
  const maxObjects = safeData.length > 0 ? Math.max(...safeData.map(d => d.total_objects)) : 0

  return (
    <div className="space-y-4">
      {safeData.length > 0 ? safeData.map((category) => {
        const percentage = maxObjects > 0 ? (category.total_objects / maxObjects) * 100 : 0
        const approvalRate = category.total_objects > 0 
          ? (category.approved_objects / category.total_objects) * 100 
          : 0

        return (
          <div key={category.category} className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium capitalize">
                {category.category.replace('_', ' ')}
              </span>
              <div className="text-xs text-gray-500 space-x-4">
                <span>{category.total_objects} total</span>
                <span>{category.approved_objects} approved</span>
                <span>{approvalRate.toFixed(1)}% rate</span>
              </div>
            </div>
            
            <div className="relative">
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full"
                  style={{ width: `${percentage}%` }}
                ></div>
              </div>
              <div className="absolute top-0 left-0 h-2 rounded-full bg-green-600 opacity-70"
                style={{ width: `${(percentage * approvalRate) / 100}%` }}
              ></div>
            </div>
          </div>
        )
      }) : (
        <div className="text-center text-gray-500 py-4">
          No category data available
        </div>
      )}
    </div>
  )
}