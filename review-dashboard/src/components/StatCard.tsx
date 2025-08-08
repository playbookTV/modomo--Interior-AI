import React from 'react'

interface StatCardProps {
  title: string
  value: string | number
  icon: React.ReactNode
  color: 'blue' | 'green' | 'purple' | 'orange'
  subtitle?: string
}

const colorClasses = {
  blue: {
    bg: 'bg-gradient-to-br from-blue-50 to-blue-100',
    text: 'text-blue-600',
    border: 'border-blue-200/50',
    shadow: 'shadow-blue-100/50'
  },
  green: {
    bg: 'bg-gradient-to-br from-emerald-50 to-emerald-100',
    text: 'text-emerald-600',
    border: 'border-emerald-200/50',
    shadow: 'shadow-emerald-100/50'
  },
  purple: {
    bg: 'bg-gradient-to-br from-purple-50 to-purple-100',
    text: 'text-purple-600',
    border: 'border-purple-200/50',
    shadow: 'shadow-purple-100/50'
  },
  orange: {
    bg: 'bg-gradient-to-br from-orange-50 to-orange-100',
    text: 'text-orange-600',
    border: 'border-orange-200/50',
    shadow: 'shadow-orange-100/50'
  }
}

export function StatCard({ title, value, icon, color, subtitle }: StatCardProps) {
  const colorClass = colorClasses[color]
  
  return (
    <div className="group bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 stat-card-hover">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-slate-600 mb-2">{title}</p>
          <p className="text-3xl font-bold text-slate-900 mb-1 tracking-tight">{value}</p>
          {subtitle && (
            <p className="text-xs text-slate-500 font-medium">{subtitle}</p>
          )}
        </div>
        <div className={`p-3 rounded-xl border backdrop-blur-sm ${colorClass.bg} ${colorClass.text} ${colorClass.border} ${colorClass.shadow} group-hover:scale-110 transition-transform duration-300`}>
          {icon}
        </div>
      </div>
    </div>
  )
}