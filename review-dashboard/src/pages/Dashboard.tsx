import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { 
  BarChart3, 
  Eye, 
  Download, 
  Camera, 
  Package,
  CheckCircle,
  AlertCircle,
  TrendingUp
} from 'lucide-react'
import { getDatasetStats, getCategoryStats, getActiveJobs } from '../api/client'
import { StatCard } from '../components/StatCard'
import { CategoryChart } from '../components/CategoryChart'
import { JobsMonitor } from '../components/JobsMonitor'

export function Dashboard() {
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['dataset-stats'],
    queryFn: getDatasetStats
  })
  const { data: categoryStats } = useQuery({
    queryKey: ['category-stats'],
    queryFn: getCategoryStats
  })
  const { data: activeJobs } = useQuery({
    queryKey: ['active-jobs'],
    queryFn: getActiveJobs,
    refetchInterval: 5000, // Refresh every 5 seconds
  })

  if (statsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  const approvalRate = stats ? (stats.approved_objects / stats.total_objects * 100) : 0
  const sceneCompletionRate = stats ? (stats.approved_scenes / stats.total_scenes * 100) : 0

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Modomo Dataset Dashboard</h1>
        <p className="text-gray-600 mt-2">
          Monitor scraping progress, review scenes, and export training datasets
        </p>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Scenes"
          value={stats?.total_scenes || 0}
          icon={<Camera className="h-6 w-6" />}
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
          title="Avg Confidence"
          value={`${((stats?.avg_confidence || 0) * 100).toFixed(1)}%`}
          icon={<TrendingUp className="h-6 w-6" />}
          color="orange"
          subtitle="Detection confidence"
        />
      </div>

      {/* Action Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Link
          to="/review"
          className="bg-white rounded-lg p-6 border-2 border-dashed border-gray-300 hover:border-blue-500 hover:bg-blue-50 transition-colors group"
        >
          <Eye className="h-12 w-12 text-gray-400 group-hover:text-blue-500 mb-4" />
          <h3 className="text-lg font-semibold text-gray-900 group-hover:text-blue-900">
            Review Queue
          </h3>
          <p className="text-gray-600 group-hover:text-blue-700">
            Review and tag detected objects in scraped scenes
          </p>
          <div className="mt-4 flex items-center text-sm text-blue-600 group-hover:text-blue-700">
            <span>Start reviewing →</span>
          </div>
        </Link>

        <Link
          to="/analytics"
          className="bg-white rounded-lg p-6 border-2 border-dashed border-gray-300 hover:border-purple-500 hover:bg-purple-50 transition-colors group"
        >
          <BarChart3 className="h-12 w-12 text-gray-400 group-hover:text-purple-500 mb-4" />
          <h3 className="text-lg font-semibold text-gray-900 group-hover:text-purple-900">
            Analytics
          </h3>
          <p className="text-gray-600 group-hover:text-purple-700">
            View detailed statistics and category breakdowns
          </p>
          <div className="mt-4 flex items-center text-sm text-purple-600 group-hover:text-purple-700">
            <span>View analytics →</span>
          </div>
        </Link>

        <Link
          to="/export"
          className="bg-white rounded-lg p-6 border-2 border-dashed border-gray-300 hover:border-green-500 hover:bg-green-50 transition-colors group"
        >
          <Download className="h-12 w-12 text-gray-400 group-hover:text-green-500 mb-4" />
          <h3 className="text-lg font-semibold text-gray-900 group-hover:text-green-900">
            Export Dataset
          </h3>
          <p className="text-gray-600 group-hover:text-green-700">
            Export approved data for ML training pipelines
          </p>
          <div className="mt-4 flex items-center text-sm text-green-600 group-hover:text-green-700">
            <span>Export data →</span>
          </div>
        </Link>
      </div>

      {/* Active Jobs */}
      {activeJobs && activeJobs.length > 0 && (
        <div className="bg-white rounded-lg p-6 shadow-sm border">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Active Jobs</h2>
          <JobsMonitor jobs={activeJobs} />
        </div>
      )}

      {/* Category Breakdown */}
      {categoryStats && categoryStats.length > 0 && (
        <div className="bg-white rounded-lg p-6 shadow-sm border">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Category Breakdown</h2>
          <CategoryChart data={categoryStats} />
        </div>
      )}

      {/* Quick Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg p-6 shadow-sm border">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Scene Progress</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Completion Rate</span>
              <span className="text-sm font-medium">{sceneCompletionRate.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${sceneCompletionRate}%` }}
              ></div>
            </div>
            <div className="flex justify-between text-xs text-gray-500">
              <span>{stats?.approved_scenes || 0} approved</span>
              <span>{stats?.total_scenes || 0} total</span>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg p-6 shadow-sm border">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Object Detection</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Product Matching</span>
              <span className="text-sm font-medium">
                {stats ? ((stats.objects_with_products / stats.total_objects) * 100).toFixed(1) : 0}%
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Categories Found</span>
              <span className="text-sm font-medium">{stats?.unique_categories || 0}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Objects with Products</span>
              <span className="text-sm font-medium">{stats?.objects_with_products || 0}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}