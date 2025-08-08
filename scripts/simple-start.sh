#!/bin/bash

# Simple start script for Modomo - works with existing ReRoom infrastructure
set -e

echo "🎯 Starting Modomo with Existing ReRoom Infrastructure"
echo "===================================================="

# Check if we're in the right directory
if [ ! -f "package.json" ] || [ ! -f "CLAUDE.md" ]; then
    echo "❌ Error: Please run this script from the ReRoom project root directory"
    exit 1
fi

# Start existing ReRoom infrastructure first
echo "🐳 Starting ReRoom infrastructure..."
pnpm run docker:up
sleep 10

# Apply database schema (works with or without pgvector)
echo "📊 Setting up Modomo database tables..."
docker exec -i $(docker ps -q -f name=postgres) psql -U reroom -d reroom_dev < backend/modomo-scraper/database/schema.sql

echo "✅ Database schema applied (pgvector warnings are normal)"

# Install Python dependencies in virtual environment
echo "🐍 Setting up Python environment for scraper..."
cd backend/modomo-scraper

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Install dependencies
echo "🔧 Installing Python packages..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate

cd ../..

# Install dashboard dependencies
echo "📦 Installing dashboard dependencies..."
cd review-dashboard
if command -v pnpm &> /dev/null; then
    pnpm install
else
    npm install
fi
cd ..

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "🚀 To start the Modomo system:"
echo ""
echo "Terminal 1 - Start the scraper API:"
echo "   cd backend/modomo-scraper"
echo "   source venv/bin/activate"
echo "   python main.py"
echo ""
echo "Terminal 2 - Start the review dashboard:"
echo "   cd review-dashboard" 
echo "   pnpm dev"
echo ""
echo "📊 Then access:"
echo "   • Review Dashboard: http://localhost:3001"
echo "   • Scraper API: http://localhost:8001"
echo "   • API Documentation: http://localhost:8001/docs"
echo ""
echo "💡 The system will work without pgvector (embeddings stored as JSON)"
echo "   To install pgvector later: ./scripts/install-pgvector.sh"
echo ""
echo "Happy dataset creation! 🎯"