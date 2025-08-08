#!/bin/bash

# Stop Modomo services
echo "🛑 Stopping Modomo Services"
echo "============================"

# Kill processes by PID if PID files exist
if [ -f ".modomo-scraper.pid" ]; then
    SCRAPER_PID=$(cat .modomo-scraper.pid)
    if kill -0 $SCRAPER_PID 2>/dev/null; then
        kill $SCRAPER_PID
        echo "✅ Stopped scraper API (PID: $SCRAPER_PID)"
    fi
    rm .modomo-scraper.pid
fi

if [ -f ".modomo-dashboard.pid" ]; then
    DASHBOARD_PID=$(cat .modomo-dashboard.pid)
    if kill -0 $DASHBOARD_PID 2>/dev/null; then
        kill $DASHBOARD_PID
        echo "✅ Stopped dashboard (PID: $DASHBOARD_PID)"
    fi
    rm .modomo-dashboard.pid
fi

# Also kill by port in case PID files are missing
lsof -ti:8001 | xargs kill -9 2>/dev/null || true
lsof -ti:3001 | xargs kill -9 2>/dev/null || true

echo "🎯 All Modomo services stopped"