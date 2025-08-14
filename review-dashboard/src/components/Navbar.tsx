import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { BarChart3, Eye, Download, Database, Target, Palette, Activity, TestTube } from 'lucide-react'

export function Navbar() {
  const location = useLocation()

  const navItems = [
    { path: '/', label: 'Dashboard', icon: <Database size={18} /> },
    { path: '/jobs', label: 'Jobs', icon: <Activity size={18} /> },
    { path: '/review', label: 'Review', icon: <Eye size={18} /> },
    { path: '/classification', label: 'Classification', icon: <TestTube size={18} /> },
    { path: '/colors', label: 'Colors', icon: <Palette size={18} /> },
    { path: '/analytics', label: 'Analytics', icon: <BarChart3 size={18} /> },
    { path: '/export', label: 'Export', icon: <Download size={18} /> }
  ]

  return (
    <nav className="nav-glass sticky top-0 z-50 shadow-sm">
      <div className="container mx-auto px-6">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-10">
            <Link to="/" className="flex items-center space-x-3 group">
              <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg shadow-lg group-hover:scale-105 transition-transform">
                <Target className="h-5 w-5 text-white" />
              </div>
              <span className="text-xl font-bold bg-gradient-to-r from-slate-800 to-slate-600 bg-clip-text text-transparent">
                Modomo
              </span>
            </Link>
            
            <div className="flex space-x-1">
              {navItems.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                    location.pathname === item.path
                      ? 'bg-blue-50 text-blue-700 shadow-sm border border-blue-200/50'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                  }`}
                >
                  {item.icon}
                  <span>{item.label}</span>
                </Link>
              ))}
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <div className="text-sm font-medium text-slate-500 bg-slate-50 px-3 py-1 rounded-full">
              Dataset Creation System
            </div>
          </div>
        </div>
      </div>
    </nav>
  )
}