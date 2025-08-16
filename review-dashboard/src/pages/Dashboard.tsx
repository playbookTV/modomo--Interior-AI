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
  TrendingUp,
  TestTube
} from 'lucide-react'
import { getDatasetStats, getCategoryStats, getActiveJobs } from '../api/client'
import { StatCard } from '../components/StatCard'
import { CategoryChart } from '../components/CategoryChart'
import { JobsMonitor } from '../components/JobsMonitor'
import { DatasetImporter } from '../components/DatasetImporter'

export function Dashboard() {
  const { data: stats, isLoading: statsLoading, error: statsError } = useQuery({
    queryKey: ['dataset-stats'],
    queryFn: getDatasetStats,
    retry: 2,
    retryDelay: 1000,
    onError: (error) => {
      console.error('Dataset stats fetch failed:', error)
    }
  })
  const { data: categoryStats } = useQuery({
    queryKey: ['category-stats'],
    queryFn: getCategoryStats,
    retry: 2,
    retryDelay: 1000,
    onError: (error) => {
      console.warn('Category stats fetch failed:', error)
    }
  })
  const { data: activeJobs } = useQuery({
    queryKey: ['active-jobs'],
    queryFn: getActiveJobs,
    refetchInterval: 10000, // Refresh every 10 seconds (slower)
    retry: 1, // Retry once on failure
    enabled: true, // Re-enable active jobs monitoring
    onError: (error) => {
      console.warn('Active jobs fetch failed:', error.message)
    }
  })

  if (statsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  // Debug logging
  console.log('Dashboard stats:', stats)
  console.log('Category stats:', categoryStats)
  console.log('Active jobs:', activeJobs)

  if (statsError) {
    console.error('Stats error:', statsError)
  }

  const approvalRate = stats ? (stats.approved_objects / stats.total_objects * 100) : 0
  const sceneCompletionRate = stats ? (stats.approved_scenes / stats.total_scenes * 100) : 0

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="text-center py-8">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-slate-800 to-slate-600 bg-clip-text text-transparent mb-4">
          Modomo Dataset Dashboard
        </h1>
        <p className="text-slate-600 text-lg max-w-2xl mx-auto">
          Monitor scraping progress, review scenes, and export training datasets for AI model development
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

      {/* Dataset Import */}
      <div className="bg-white rounded-xl p-8 shadow-sm border border-slate-200/60">
        <h2 className="text-2xl font-bold text-slate-800 mb-6">Quick Dataset Import</h2>
        <DatasetImporter />
      </div>

      {/* Action Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Link
          to="/review"
          className="group bg-white rounded-xl p-8 border-2 border-dashed border-slate-200 hover:border-blue-400 hover:shadow-lg transition-all duration-300 hover:scale-105"
        >
          <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl w-fit mb-6 group-hover:scale-110 transition-transform">
            <Eye className="h-10 w-10 text-blue-600" />
          </div>
          <h3 className="text-xl font-bold text-slate-800 mb-3 group-hover:text-blue-900">
            Review Queue
          </h3>
          <p className="text-slate-600 group-hover:text-blue-700 leading-relaxed">
            Review and tag detected objects in scraped scenes to build high-quality datasets
          </p>
          <div className="mt-6 flex items-center text-blue-600 group-hover:text-blue-700 font-medium">
            <span>Start reviewing</span>
            <span className="ml-2 group-hover:translate-x-1 transition-transform">→</span>
          </div>
        </Link>

        <Link
          to="/analytics"
          className="group bg-white rounded-xl p-8 border-2 border-dashed border-slate-200 hover:border-purple-400 hover:shadow-lg transition-all duration-300 hover:scale-105"
        >
          <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl w-fit mb-6 group-hover:scale-110 transition-transform">
            <BarChart3 className="h-10 w-10 text-purple-600" />
          </div>
          <h3 className="text-xl font-bold text-slate-800 mb-3 group-hover:text-purple-900">
            Analytics
          </h3>
          <p className="text-slate-600 group-hover:text-purple-700 leading-relaxed">
            View detailed statistics, performance metrics, and category breakdowns
          </p>
          <div className="mt-6 flex items-center text-purple-600 group-hover:text-purple-700 font-medium">
            <span>View analytics</span>
            <span className="ml-2 group-hover:translate-x-1 transition-transform">→</span>
          </div>
        </Link>

        <Link
          to="/classification"
          className="group bg-white rounded-xl p-8 border-2 border-dashed border-slate-200 hover:border-purple-400 hover:shadow-lg transition-all duration-300 hover:scale-105"
        >
          <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl w-fit mb-6 group-hover:scale-110 transition-transform">
            <TestTube className="h-10 w-10 text-purple-600" />
          </div>
          <h3 className="text-xl font-bold text-slate-800 mb-3 group-hover:text-purple-900">
            Classification Tools
          </h3>
          <p className="text-slate-600 group-hover:text-purple-700 leading-relaxed">
            Test and improve scene vs object detection with enhanced AI analysis
          </p>
          <div className="mt-6 flex items-center text-purple-600 group-hover:text-purple-700 font-medium">
            <span>Open tools</span>
            <span className="ml-2 group-hover:translate-x-1 transition-transform">→</span>
          </div>
        </Link>

        <Link
          to="/export"
          className="group bg-white rounded-xl p-8 border-2 border-dashed border-slate-200 hover:border-emerald-400 hover:shadow-lg transition-all duration-300 hover:scale-105"
        >
          <div className="p-4 bg-gradient-to-br from-emerald-50 to-emerald-100 rounded-xl w-fit mb-6 group-hover:scale-110 transition-transform">
            <Download className="h-10 w-10 text-emerald-600" />
          </div>
          <h3 className="text-xl font-bold text-slate-800 mb-3 group-hover:text-emerald-900">
            Export Dataset
          </h3>
          <p className="text-slate-600 group-hover:text-emerald-700 leading-relaxed">
            Export approved data in multiple formats for ML training pipelines
          </p>
          <div className="mt-6 flex items-center text-emerald-600 group-hover:text-emerald-700 font-medium">
            <span>Export data</span>
            <span className="ml-2 group-hover:translate-x-1 transition-transform">→</span>
          </div>
        </Link>
      </div>

      {/* Active Jobs */}
      {activeJobs && activeJobs.length > 0 && (
        <div className="bg-white rounded-xl p-8 shadow-sm border border-slate-200/60 animate-slide-up">
          <h2 className="text-2xl font-bold text-slate-800 mb-6">Active Jobs</h2>
          <JobsMonitor jobs={activeJobs} />
        </div>
      )}

      {/* Category Breakdown */}
      {categoryStats && categoryStats.length > 0 && (
        <div className="bg-white rounded-xl p-8 shadow-sm border border-slate-200/60 animate-slide-up">
          <h2 className="text-2xl font-bold text-slate-800 mb-6">Category Breakdown</h2>
          <CategoryChart data={categoryStats} />
        </div>
      )}

      {/* Quick Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="bg-white rounded-xl p-8 shadow-sm border border-slate-200/60">
          <h3 className="text-xl font-bold text-slate-800 mb-6">Scene Progress</h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-slate-600 font-medium">Completion Rate</span>
              <span className="text-lg font-bold text-slate-800">{sceneCompletionRate.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-slate-100 rounded-full h-3">
              <div 
                className="bg-gradient-to-r from-blue-500 to-blue-600 h-3 rounded-full transition-all duration-500 shadow-sm"
                style={{ width: `${sceneCompletionRate}%` }}
              ></div>
            </div>
            <div className="flex justify-between text-sm text-slate-500">
              <span className="font-medium">{stats?.approved_scenes || 0} approved</span>
              <span className="font-medium">{stats?.total_scenes || 0} total</span>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl p-8 shadow-sm border border-slate-200/60">
          <h3 className="text-xl font-bold text-slate-800 mb-6">Object Detection</h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-slate-600 font-medium">Product Matching</span>
              <span className="text-lg font-bold text-slate-800">
                {stats ? ((stats.objects_with_products / stats.total_objects) * 100).toFixed(1) : 0}%
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-600 font-medium">Categories Found</span>
              <span className="text-lg font-bold text-slate-800">{stats?.unique_categories || 0}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-600 font-medium">Objects with Products</span>
              <span className="text-lg font-bold text-slate-800">{stats?.objects_with_products || 0}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Scene Classification Breakdown */}
      {stats?.scenes_by_type && (
        <div className="bg-white rounded-xl p-8 shadow-sm border border-slate-200/60 animate-slide-up">
          <h2 className="text-2xl font-bold text-slate-800 mb-6">Scene Classification Breakdown</h2>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="text-center p-4 bg-blue-50 rounded-lg border border-blue-200">
              <div className="text-3xl font-bold text-blue-600 mb-2">{stats.scenes_by_type.scene}</div>
              <div className="text-sm font-medium text-blue-800">Full Scenes</div>
              <div className="text-xs text-blue-600 mt-1">3+ diverse objects</div>
            </div>
            
            <div className="text-center p-4 bg-purple-50 rounded-lg border border-purple-200">
              <div className="text-3xl font-bold text-purple-600 mb-2">{stats.scenes_by_type.object}</div>
              <div className="text-sm font-medium text-purple-800">Single Objects</div>
              <div className="text-xs text-purple-600 mt-1">1-2 furniture pieces</div>
            </div>
            
            <div className="text-center p-4 bg-orange-50 rounded-lg border border-orange-200">
              <div className="text-3xl font-bold text-orange-600 mb-2">{stats.scenes_by_type.hybrid}</div>
              <div className="text-sm font-medium text-orange-800">Hybrid Scenes</div>
              <div className="text-xs text-orange-600 mt-1">Focal object + context</div>
            </div>
            
            <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200">
              <div className="text-3xl font-bold text-green-600 mb-2">{stats.scenes_by_type.product}</div>
              <div className="text-sm font-medium text-green-800">Product Images</div>
              <div className="text-xs text-green-600 mt-1">Catalog/studio photos</div>
            </div>
          </div>
          
          {/* Room Types */}
          {stats.room_types && stats.room_types.length > 0 && (
            <div className="mt-6 p-4 bg-slate-50 rounded-lg border border-slate-200">
              <h4 className="font-medium text-slate-800 mb-3">Detected Room Types</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {stats.room_types.slice(0, 8).map((room) => (
                  <div key={room.room_type} className="flex items-center justify-between text-sm">
                    <span className="text-slate-600 capitalize">{room.room_type?.replace(/_/g, ' ')}</span>
                    <span className="font-medium text-slate-800">{room.count}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Detected Styles */}
          {stats.detected_styles && stats.detected_styles.length > 0 && (
            <div className="mt-4 p-4 bg-pink-50 rounded-lg border border-pink-200">
              <h4 className="font-medium text-pink-800 mb-3">Detected Design Styles</h4>
              <div className="flex flex-wrap gap-2">
                {stats.detected_styles.slice(0, 10).map((style) => (
                  <span key={style.style} className="px-3 py-1 bg-pink-100 text-pink-700 rounded-full text-sm border border-pink-300">
                    {style.style} ({style.count})
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Quick Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
      </div>
    </div>
  )
}