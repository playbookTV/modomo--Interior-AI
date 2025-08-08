#!/bin/bash

# Test script to verify Modomo system is working correctly
set -e

echo "🧪 Testing Modomo System"
echo "========================"

# Test API health
echo "🔍 Testing API health..."
HEALTH_RESPONSE=$(curl -s http://localhost:8001/health)
if [[ $HEALTH_RESPONSE == *"healthy"* ]]; then
    echo "✅ API is healthy"
else
    echo "❌ API health check failed"
    echo "Response: $HEALTH_RESPONSE"
    exit 1
fi

# Test dashboard
echo "🌐 Testing dashboard..."
DASHBOARD_RESPONSE=$(curl -s http://localhost:3001/)
if [[ $DASHBOARD_RESPONSE == *"Modomo Review Dashboard"* ]]; then
    echo "✅ Dashboard is serving content"
else
    echo "❌ Dashboard not responding correctly"
    exit 1
fi

# Test API proxy through dashboard
echo "🔗 Testing API proxy..."
PROXY_RESPONSE=$(curl -s http://localhost:3001/api/health)
if [[ $PROXY_RESPONSE == *"healthy"* ]]; then
    echo "✅ Dashboard → API proxy working"
else
    echo "❌ Dashboard → API proxy failed"
    exit 1
fi

# Test dataset stats endpoint
echo "📊 Testing dataset stats..."
STATS_RESPONSE=$(curl -s http://localhost:8001/stats/dataset)
if [[ $STATS_RESPONSE == *"unique_categories"* ]]; then
    echo "✅ Dataset stats endpoint working"
else
    echo "❌ Dataset stats endpoint failed"
    exit 1
fi

# Test taxonomy endpoint
echo "📝 Testing taxonomy..."
TAXONOMY_RESPONSE=$(curl -s http://localhost:8001/taxonomy)
if [[ $TAXONOMY_RESPONSE == *"seating"* ]]; then
    echo "✅ Taxonomy endpoint working"
else
    echo "❌ Taxonomy endpoint failed"
    exit 1
fi

# Test dummy detection
echo "🤖 Testing dummy detection..."
DETECTION_RESPONSE=$(curl -s -X POST http://localhost:8001/test/detection)
if [[ $DETECTION_RESPONSE == *"sofa"* ]]; then
    echo "✅ Detection endpoint working"
else
    echo "❌ Detection endpoint failed"
    exit 1
fi

echo ""
echo "🎉 All Tests Passed!"
echo "=================="
echo ""
echo "✅ API Server: http://localhost:8001"
echo "✅ Review Dashboard: http://localhost:3001"
echo "✅ API Documentation: http://localhost:8001/docs"
echo ""
echo "🚀 System is fully operational and ready for use!"
echo ""
echo "Next steps:"
echo "  1. Open http://localhost:3001 in your browser"
echo "  2. Explore the dashboard interface"
echo "  3. Check API docs at http://localhost:8001/docs"
echo "  4. Start scraping scenes and building your dataset!"