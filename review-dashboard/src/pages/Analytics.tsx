import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { getDatasetStats, getCategoryStats } from '../api/client'
import { StatCard } from '../components/StatCard'
import { CategoryChart } from '../components/CategoryChart'
import { BarChart3, Package, CheckCircle, TrendingUp } from 'lucide-react'

export function Analytics() {
  const { data: stats, isLoading: statsLoading } = useQuery('dataset-stats', getDatasetStats)
  const { data: categoryStats } = useQuery('category-stats', getCategoryStats)

  if (statsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  const approvalRate = stats ? (stats.approved_objects / stats.total_objects * 100) : 0

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Analytics</h1>
        <p className="text-gray-600">Dataset statistics and insights</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Scenes"
          value={stats?.total_scenes || 0}
          icon={<BarChart3 className="h-6 w-6" />}
          color="blue"
          subtitle={`${stats?.approved_scenes || 0} approved`}
        />
        
        <StatCard
          title="Detected Objects"
          value={stats?.total_objects || 0}
          icon={<Package className="h-6 w-6" />}
          color="green"
          subtitle={`${stats?.approved_objects || 0} approved`}
        />
        
        <StatCard
          title="Approval Rate"
          value={`${approvalRate.toFixed(1)}%`}
          icon={<CheckCircle className="h-6 w-6" />}
          color="purple"
          subtitle="Object approval rate"
        />
        
        <StatCard
          title="Categories"
          value={stats?.unique_categories || 0}
          icon={<TrendingUp className="h-6 w-6" />}
          color="orange"
          subtitle="Unique categories found"
        />
      </div>

      {/* Category Breakdown */}
      {categoryStats && categoryStats.length > 0 && (
        <div className="bg-white rounded-lg p-6 shadow-sm border">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Category Breakdown</h2>
          <CategoryChart data={categoryStats} />
        </div>
      )}

      {/* Additional Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg p-6 shadow-sm border">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Detection Quality</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-600">Average Confidence</span>
              <span className="font-medium">{((stats?.avg_confidence || 0) * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Objects with Products</span>
              <span className="font-medium">{stats?.objects_with_products || 0}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Product Match Rate</span>
              <span className="font-medium">
                {stats ? ((stats.objects_with_products / stats.total_objects) * 100).toFixed(1) : 0}%
              </span>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg p-6 shadow-sm border">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Progress Overview</h3>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm text-gray-600">Scene Review Progress</span>
                <span className="text-sm font-medium">
                  {stats ? ((stats.approved_scenes / stats.total_scenes) * 100).toFixed(1) : 0}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full"
                  style={{ 
                    width: `${stats ? (stats.approved_scenes / stats.total_scenes) * 100 : 0}%` 
                  }}
                ></div>
              </div>
            </div>
            
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm text-gray-600">Object Review Progress</span>
                <span className="text-sm font-medium">{approvalRate.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-green-600 h-2 rounded-full"
                  style={{ width: `${approvalRate}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}