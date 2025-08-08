import React, { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { startDatasetExport } from '../api/client'

export function DatasetExport() {
  const [splitRatios, setSplitRatios] = useState({
    train: 0.7,
    val: 0.2,
    test: 0.1
  })

  const exportMutation = useMutation({
    mutationFn: startDatasetExport
  })

  const handleExport = () => {
    exportMutation.mutate({
      train_ratio: splitRatios.train,
      val_ratio: splitRatios.val,
      test_ratio: splitRatios.test,
    })
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dataset Export</h1>
        <p className="text-gray-600">Export approved data for ML training</p>
      </div>

      <div className="bg-white rounded-lg p-6 shadow-sm border max-w-2xl">
        <h2 className="text-lg font-medium text-gray-900 mb-4">Split Configuration</h2>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Training Set Ratio
            </label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.1"
              value={splitRatios.train}
              onChange={(e) => setSplitRatios({...splitRatios, train: parseFloat(e.target.value)})}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Validation Set Ratio
            </label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.1"
              value={splitRatios.val}
              onChange={(e) => setSplitRatios({...splitRatios, val: parseFloat(e.target.value)})}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Test Set Ratio
            </label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.1"
              value={splitRatios.test}
              onChange={(e) => setSplitRatios({...splitRatios, test: parseFloat(e.target.value)})}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
          
          <div className="pt-4">
            <button
              onClick={handleExport}
              disabled={exportMutation.isPending}
              className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
            >
              {exportMutation.isPending ? 'Exporting...' : 'Start Export'}
            </button>
          </div>
          
          {exportMutation.data && (
            <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
              <p className="text-green-800">Export started with ID: {exportMutation.data.export_id}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}