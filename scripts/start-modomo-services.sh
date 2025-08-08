#!/bin/bash

# Start Modomo services in the background
set -e

echo "🚀 Starting Modomo Services"
echo "==========================="

# Check if infrastructure is running
if ! docker ps | grep -q postgres; then
    echo "🐳 Starting ReRoom infrastructure..."
    pnpm run docker:up
    sleep 10
fi

# Kill any existing processes on our ports
echo "🧹 Cleaning up existing processes..."
lsof -ti:8001 | xargs kill -9 2>/dev/null || true
lsof -ti:3001 | xargs kill -9 2>/dev/null || true

# Start scraper API in background
echo "🔧 Starting scraper API..."
cd backend/modomo-scraper
if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    echo "📦 Installing minimal dependencies first..."
    pip install -r requirements-minimal.txt
else
    source venv/bin/activate
fi

# Start the API server in background
nohup python main-basic.py > scraper.log 2>&1 &
SCRAPER_PID=$!
echo "✅ Scraper API started (PID: $SCRAPER_PID)"
cd ../..

# Start dashboard in background  
echo "📊 Starting review dashboard..."
cd review-dashboard
if [ ! -d "node_modules" ]; then
    if command -v pnpm &> /dev/null; then
        pnpm install
    else
        npm install
    fi
fi

# Start the dashboard in background
nohup npm run dev > dashboard.log 2>&1 &
DASHBOARD_PID=$!
echo "✅ Dashboard started (PID: $DASHBOARD_PID)"
cd ..

# Save PIDs for later cleanup
echo $SCRAPER_PID > .modomo-scraper.pid
echo $DASHBOARD_PID > .modomo-dashboard.pid

# Wait a moment and check if services are running
sleep 5

echo ""
echo "🎉 Modomo Services Started!"
echo "==========================="

if curl -f http://localhost:8001/health &>/dev/null; then
    echo "✅ Scraper API: http://localhost:8001 (healthy)"
else
    echo "⏳ Scraper API: http://localhost:8001 (starting...)"
fi

echo "📊 Review Dashboard: http://localhost:3001"
echo "📖 API Documentation: http://localhost:8001/docs"
echo ""
echo "📝 View logs:"
echo "   • Scraper: tail -f backend/modomo-scraper/scraper.log"
echo "   • Dashboard: tail -f review-dashboard/dashboard.log"
echo ""
echo "🛑 Stop services:"
echo "   • Run: ./scripts/stop-modomo-services.sh"
echo ""
echo "🌟 System ready for dataset creation!"