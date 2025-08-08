#!/bin/bash

# Deploy Modomo Scraper to Railway
# This script helps set up the Railway project and environment variables

set -e

echo "🚄 Modomo Scraper - Railway Deployment Setup"
echo ""

# Check if railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found!"
    echo "   Install with: npm install -g @railway/cli"
    echo "   Then run: railway login"
    exit 1
fi

# Check if logged in to Railway
if ! railway whoami &> /dev/null; then
    echo "❌ Not logged in to Railway!"
    echo "   Run: railway login"
    exit 1
fi

echo "✅ Railway CLI ready"
echo ""

# Navigate to modomo-scraper directory
cd backend/modomo-scraper

echo "📁 Current directory: $(pwd)"
echo ""

# Check if already in a Railway project
if railway status &> /dev/null; then
    echo "🚄 Railway project already linked"
    PROJECT_NAME=$(railway status --json | jq -r '.project.name')
    echo "   Project: $PROJECT_NAME"
else
    echo "🆕 Creating new Railway project..."
    
    # Create new project
    railway login
    railway init
    
    echo "✅ Railway project created"
fi

echo ""
echo "🔧 Setting environment variables..."

# Core environment variables for production
railway variables set DATABASE_URL_CLOUD="postgresql://postgres:qYs8XprwrHQoM7n6@db.nyeeewcpexqsqfzzmvyu.supabase.co:5432/postgres"

railway variables set CLOUDFLARE_R2_ENDPOINT="https://9bbdb3861142e65685d23f4955b88ebe.r2.cloudflarestorage.com"
railway variables set CLOUDFLARE_R2_BUCKET="reroom"
railway variables set CLOUDFLARE_R2_ACCESS_KEY_ID="6c8abdff2cdad89323e36b258b1d0f4b"
railway variables set CLOUDFLARE_R2_SECRET_ACCESS_KEY="35b7b4a1f586211d407b246b54898d2e50d13562cba7e7be6293d4b6ccea06c5"
railway variables set CLOUDFLARE_R2_PUBLIC_URL="https://photos.reroom.app"

# API configuration
railway variables set CORS_ORIGINS="*"
railway variables set DEBUG="false"
railway variables set LOG_LEVEL="info"

# Production environment
railway variables set ENVIRONMENT="production"
railway variables set RAILWAY_ENVIRONMENT="production"

echo "✅ Environment variables set"
echo ""

echo "🚀 Deploying to Railway..."

# Deploy the current directory
railway up --detach

echo ""
echo "✅ Deployment initiated!"
echo ""
echo "🔗 Your Railway project:"
railway status

echo ""
echo "📋 Next steps:"
echo "1. Wait for deployment to complete (~2-3 minutes)"
echo "2. Check logs: railway logs"
echo "3. Get your service URL: railway domain"
echo "4. Initialize database schema from Railway console"
echo "5. Update dashboard API URL to point to Railway service"
echo ""

echo "🏁 Railway deployment setup complete!"
echo "   Visit your Railway dashboard to monitor the deployment."