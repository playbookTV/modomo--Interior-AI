import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import { Palette, Thermometer, TrendingUp } from 'lucide-react'

async function getColorStats() {
  const response = await fetch(
    'https://ovalay-recruitment-production.up.railway.app/stats/colors'
  )
  if (!response.ok) throw new Error('Failed to fetch color statistics')
  return response.json()
}

const COLORS = {
  warm: '#ff6b6b',
  cool: '#4ecdc4', 
  neutral: '#95a5a6',
  red: '#e74c3c',
  blue: '#3498db',
  green: '#2ecc71',
  brown: '#8b4513',
  white: '#ecf0f1',
  black: '#2c3e50',
  gray: '#7f8c8d',
  beige: '#f5deb3',
  yellow: '#f1c40f'
}

export function ColorStats() {
  const { data: colorStats, isLoading, error } = useQuery({
    queryKey: ['color-stats'],
    queryFn: getColorStats,
    refetchInterval: 30000 // Refresh every 30 seconds
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600"></div>
      </div>
    )
  }

  if (error || !colorStats) {
    return (
      <div className="text-center py-8">
        <p className="text-gray-500">Failed to load color statistics</p>
      </div>
    )
  }

  // Prepare data for charts
  const colorDistributionData = Object.entries(colorStats.color_distribution || {})
    .map(([color, count]) => ({
      name: color.replace('_', ' '),
      value: count,
      color: COLORS[color as keyof typeof COLORS] || '#8884d8'
    }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 10) // Top 10 colors

  const temperatureData = Object.entries(colorStats.color_temperature_distribution || {})
    .map(([temp, count]) => ({
      name: temp,
      value: count,
      color: COLORS[temp as keyof typeof COLORS] || '#8884d8'
    }))

  const dominantColorsData = Object.entries(colorStats.dominant_colors || {})
    .map(([color, count]) => ({
      name: color.replace('_', ' '),
      value: count
    }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 8)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Palette className="h-6 w-6 text-purple-600" />
        <h2 className="text-2xl font-bold text-gray-900">Color Analytics</h2>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg p-6 shadow-sm border">
          <div className="flex items-center">
            <div className="p-2 bg-purple-100 rounded-lg">
              <Palette className="h-5 w-5 text-purple-600" />
            </div>
            <div className="ml-3">
              <p className="text-sm text-gray-600">Objects with Colors</p>
              <p className="text-xl font-bold text-gray-900">
                {colorStats.total_objects_with_colors || 0}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg p-6 shadow-sm border">
          <div className="flex items-center">
            <div className="p-2 bg-blue-100 rounded-lg">
              <TrendingUp className="h-5 w-5 text-blue-600" />
            </div>
            <div className="ml-3">
              <p className="text-sm text-gray-600">Unique Colors</p>
              <p className="text-xl font-bold text-gray-900">
                {Object.keys(colorStats.color_distribution || {}).length}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg p-6 shadow-sm border">
          <div className="flex items-center">
            <div className="p-2 bg-red-100 rounded-lg">
              <Thermometer className="h-5 w-5 text-red-600" />
            </div>
            <div className="ml-3">
              <p className="text-sm text-gray-600">Most Common</p>
              <p className="text-xl font-bold text-gray-900 capitalize">
                {dominantColorsData[0]?.name || 'N/A'}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg p-6 shadow-sm border">
          <div className="flex items-center">
            <div className="p-2 bg-green-100 rounded-lg">
              <div className="w-5 h-5 bg-gradient-to-r from-red-500 via-yellow-500 to-blue-500 rounded"></div>
            </div>
            <div className="ml-3">
              <p className="text-sm text-gray-600">Color Categories</p>
              <p className="text-xl font-bold text-gray-900">
                {Object.keys(colorStats.colors_by_category || {}).length}
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Color Distribution Chart */}
        <div className="bg-white rounded-lg p-6 shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Color Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={colorDistributionData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="name" 
                angle={-45}
                textAnchor="end"
                height={80}
                fontSize={12}
              />
              <YAxis />
              <Tooltip 
                formatter={(value, name) => [value, 'Objects']}
                labelFormatter={(label) => `Color: ${label}`}
              />
              <Bar dataKey="value" fill="#8884d8" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Color Temperature Distribution */}
        <div className="bg-white rounded-lg p-6 shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Color Temperature</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={temperatureData}
                cx="50%"
                cy="50%"
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
              >
                {temperatureData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Colors by Category */}
      <div className="bg-white rounded-lg p-6 shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Colors by Furniture Category</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full table-auto">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-2 px-4 font-medium text-gray-700">Category</th>
                <th className="text-left py-2 px-4 font-medium text-gray-700">Top Colors</th>
                <th className="text-right py-2 px-4 font-medium text-gray-700">Total Objects</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(colorStats.colors_by_category || {}).map(([category, colors]: [string, any]) => {
                const totalObjects = Object.values(colors).reduce((sum: number, count: any) => sum + count, 0)
                const topColors = Object.entries(colors)
                  .sort(([,a], [,b]) => (b as number) - (a as number))
                  .slice(0, 5)

                return (
                  <tr key={category} className="border-b border-gray-100">
                    <td className="py-3 px-4 font-medium text-gray-900 capitalize">
                      {category.replace('_', ' ')}
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex flex-wrap gap-2">
                        {topColors.map(([color, count]) => (
                          <span 
                            key={color}
                            className="inline-flex items-center px-2 py-1 bg-gray-100 text-gray-800 text-xs rounded-full"
                          >
                            <div 
                              className="w-3 h-3 rounded-full mr-1 border border-gray-300"
                              style={{ backgroundColor: COLORS[color as keyof typeof COLORS] || '#ccc' }}
                            />
                            {color.replace('_', ' ')} ({count})
                          </span>
                        ))}
                      </div>
                    </td>
                    <td className="py-3 px-4 text-right text-gray-600">
                      {totalObjects}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Dominant Colors Ranking */}
      <div className="bg-white rounded-lg p-6 shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Most Common Dominant Colors</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {dominantColorsData.map((colorData, index) => (
            <div key={colorData.name} className="flex items-center space-x-3">
              <div className="flex-shrink-0">
                <div 
                  className="w-8 h-8 rounded-full border-2 border-gray-200"
                  style={{ backgroundColor: COLORS[colorData.name.replace(' ', '_') as keyof typeof COLORS] || '#ccc' }}
                />
              </div>
              <div>
                <p className="text-sm font-medium text-gray-900 capitalize">
                  {colorData.name}
                </p>
                <p className="text-sm text-gray-500">
                  {colorData.value} objects
                </p>
              </div>
              <div className="text-xs text-gray-400 ml-auto">
                #{index + 1}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}