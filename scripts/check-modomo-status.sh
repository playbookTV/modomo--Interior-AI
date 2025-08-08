#!/bin/bash

# Modomo System Status Check Script
# Verifies all services are running and endpoints are accessible

set -e

echo "🔍 Checking Modomo System Status..."
echo ""

# Check API health
echo "📡 API Service (localhost:8001):"
if curl -s -f http://localhost:8001/health > /dev/null; then
    echo "  ✅ Health endpoint: OK"
    
    # Test specific endpoints
    if curl -s -f http://localhost:8001/stats/dataset > /dev/null; then
        echo "  ✅ Dataset stats: OK"
    else
        echo "  ❌ Dataset stats: Failed"
    fi
    
    if curl -s -f http://localhost:8001/stats/categories > /dev/null; then
        echo "  ✅ Categories stats: OK"
    else
        echo "  ❌ Categories stats: Failed"
    fi
    
    if curl -s -f http://localhost:8001/jobs/active > /dev/null; then
        echo "  ✅ Active jobs: OK"
    else
        echo "  ❌ Active jobs: Failed"
    fi
    
    if curl -s -f http://localhost:8001/taxonomy > /dev/null; then
        echo "  ✅ Taxonomy endpoint: OK"
    else
        echo "  ❌ Taxonomy endpoint: Failed"
    fi
else
    echo "  ❌ API Service: Not responding"
fi

echo ""

# Check Dashboard
echo "🖥️  Review Dashboard (localhost:3001):"
if curl -s -f http://localhost:3001/ > /dev/null; then
    echo "  ✅ Dashboard: Accessible"
else
    echo "  ❌ Dashboard: Not accessible"
fi

echo ""

# Check Database connection
echo "🗄️  Database:"
if curl -s -f http://localhost:8001/health | grep -q "Connected to database"; then
    echo "  ✅ Database: Connected"
else
    echo "  ❌ Database: Not connected"
fi

# Check Redis connection
echo "⚡ Redis:"
if curl -s -f http://localhost:8001/health | grep -q "Connected to Redis"; then
    echo "  ✅ Redis: Connected"
else
    echo "  ❌ Redis: Not connected"
fi

echo ""

# Summary
API_HEALTH=$(curl -s http://localhost:8001/health 2>/dev/null || echo "failed")
DASHBOARD_HEALTH=$(curl -s -I http://localhost:3001/ 2>/dev/null | head -n1 | grep -q "200" && echo "ok" || echo "failed")

if [[ "$API_HEALTH" != "failed" && "$DASHBOARD_HEALTH" == "ok" ]]; then
    echo "🎉 Modomo System Status: ALL GOOD!"
    echo ""
    echo "📊 Dashboard: http://localhost:3001"
    echo "📖 API Docs: http://localhost:8001/docs"
    echo ""
    echo "💡 System is running in basic mode."
    echo "   Install AI dependencies (torch, transformers) for full functionality."
else
    echo "⚠️  Some services are not responding correctly."
    echo "   Check the logs and restart services if needed."
fi