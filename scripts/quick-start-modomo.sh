#!/bin/bash

# Quick start script for Modomo using Docker with pgvector support
# This is a simpler alternative that uses Docker for everything

set -e

echo "🚀 Quick Start: Modomo Dataset Creation System"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "package.json" ] || [ ! -f "CLAUDE.md" ]; then
    echo "❌ Error: Please run this script from the ReRoom project root directory"
    exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed. Please install Docker first."
    exit 1
fi

echo "✅ Found ReRoom project"

# Stop existing containers to avoid conflicts
echo "🛑 Stopping existing containers..."
docker-compose down 2>/dev/null || true

# Start infrastructure with pgvector support
echo "🐳 Starting infrastructure with pgvector support..."
docker-compose up -d

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
sleep 15

# Check if database is ready
until docker exec $(docker ps -q -f name=postgres) pg_isready -U reroom -d reroom_dev; do
  echo "⏳ Waiting for database connection..."
  sleep 2
done

echo "✅ Database is ready!"

# The schema is automatically applied via docker-entrypoint-initdb.d
echo "📊 Database schema will be applied automatically"

# Install dashboard dependencies
echo "📦 Installing dashboard dependencies..."
cd review-dashboard
if command -v pnpm &> /dev/null; then
    pnpm install
else
    npm install
fi
cd ..

# Build and start Modomo services
echo "🔨 Building Modomo services..."
docker-compose -f docker-compose.modomo.yml build

echo "🚀 Starting Modomo services..."
docker-compose -f docker-compose.modomo.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🔍 Checking service health..."
if curl -f http://localhost:8001/health &>/dev/null; then
    echo "✅ Scraper API is healthy"
else
    echo "⚠️  Scraper API may still be starting..."
fi

echo ""
echo "🎉 Modomo System Started!"
echo "========================"
echo ""
echo "📊 Services:"
echo "   • Scraper API: http://localhost:8001"
echo "   • Review Dashboard: http://localhost:3001"
echo "   • API Docs: http://localhost:8001/docs"
echo ""
echo "🔧 Useful commands:"
echo "   • View logs: docker-compose -f docker-compose.modomo.yml logs -f"
echo "   • Stop services: docker-compose -f docker-compose.modomo.yml down"
echo "   • Restart: docker-compose -f docker-compose.modomo.yml restart"
echo ""
echo "📖 Next steps:"
echo "   1. Open http://localhost:3001 for the review dashboard"
echo "   2. Start scraping scenes from the dashboard"
echo "   3. Review and tag detected objects"
echo "   4. Export datasets for ML training"
echo ""
echo "Happy dataset creation! 🎯"