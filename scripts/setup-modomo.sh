#!/bin/bash

# Modomo Dataset Creation System Setup Script
# Run this script to set up the complete Modomo system

set -e

echo "🎯 Setting up Modomo Dataset Creation System..."
echo "================================================"

# Check if we're in the right directory
if [ ! -f "package.json" ] || [ ! -f "CLAUDE.md" ]; then
    echo "❌ Error: Please run this script from the ReRoom project root directory"
    exit 1
fi

echo "✅ Found ReRoom project structure"

# Check dependencies
echo "🔍 Checking dependencies..."

# Check pnpm
if ! command -v pnpm &> /dev/null; then
    echo "❌ pnpm is required but not installed. Please install pnpm first:"
    echo "   npm install -g pnpm"
    exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed. Please install Docker first."
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is required but not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ All required dependencies found"

# Create database tables
echo "🗄️  Setting up database schema..."

# Check if PostgreSQL is running
if ! docker ps | grep -q postgres; then
    echo "🐳 Starting ReRoom infrastructure..."
    pnpm run docker:up
    sleep 10  # Wait for containers to start
fi

# Install pgvector extension in PostgreSQL
echo "🔧 Installing pgvector extension..."
docker exec $(docker ps -q -f name=postgres) bash -c "
apt-get update && apt-get install -y postgresql-14-pgvector
"

# Restart PostgreSQL to load the extension
echo "🔄 Restarting PostgreSQL..."
docker restart $(docker ps -q -f name=postgres)
sleep 5

# Apply database schema
echo "📊 Creating Modomo database tables..."
docker exec -i $(docker ps -q -f name=postgres) psql -U reroom -d reroom_dev < backend/modomo-scraper/database/schema.sql

if [ $? -eq 0 ]; then
    echo "✅ Database schema applied successfully"
else
    echo "⚠️  Database schema may already exist (this is normal)"
fi

# Install Python dependencies
echo "🐍 Installing Python dependencies for scraper service..."
cd backend/modomo-scraper

# Check if Python virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install dependencies
echo "🔧 Installing Python packages..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate

cd ../..

# Install Node.js dependencies for dashboard
echo "📦 Installing Node.js dependencies for review dashboard..."
cd review-dashboard
pnpm install
cd ..

# Build and start services
echo "🚀 Building and starting Modomo services..."

# Build modomo services
docker-compose -f docker-compose.modomo.yml build

echo "✅ Modomo setup completed!"
echo ""
echo "🎉 Setup Summary:"
echo "=================="
echo "✅ Database schema created"
echo "✅ Python dependencies installed"
echo "✅ Node.js dependencies installed"
echo "✅ Docker services built"
echo ""
echo "🚀 To start the Modomo system:"
echo "   docker-compose -f docker-compose.modomo.yml up"
echo ""
echo "📊 Services will be available at:"
echo "   • Scraper API: http://localhost:8001"
echo "   • Review Dashboard: http://localhost:3001"
echo "   • API Documentation: http://localhost:8001/docs"
echo ""
echo "🔧 To run individual components:"
echo "   • Start scraper only: cd backend/modomo-scraper && uvicorn main:app --reload --port 8001"
echo "   • Start dashboard only: cd review-dashboard && pnpm dev"
echo ""
echo "📖 Next steps:"
echo "   1. Start the services with: docker-compose -f docker-compose.modomo.yml up"
echo "   2. Open http://localhost:3001 to access the review dashboard"
echo "   3. Use the dashboard to start scraping scenes and reviewing objects"
echo ""
echo "Happy dataset creation! 🎯"