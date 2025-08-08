import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Navbar } from './components/Navbar'
import { Dashboard } from './pages/Dashboard'
import { ReviewQueue } from './pages/ReviewQueue'
import { SceneDetail } from './pages/SceneDetail'
import { DatasetExport } from './pages/DatasetExport'
import { Analytics } from './pages/Analytics'
import './index.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
})

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="min-h-screen bg-gray-50">
          <Navbar />
          <main className="container mx-auto px-4 py-8">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/review" element={<ReviewQueue />} />
              <Route path="/scene/:sceneId" element={<SceneDetail />} />
              <Route path="/export" element={<DatasetExport />} />
              <Route path="/analytics" element={<Analytics />} />
            </Routes>
          </main>
        </div>
      </Router>
    </QueryClientProvider>
  )
}