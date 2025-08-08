#!/bin/bash

# Deploy Modomo Scraper with Full AI to Railway
# This enables GroundingDINO, SAM2, and CLIP processing

set -e

echo "🤖 Deploying Modomo Scraper with Full AI to Railway"
echo ""

# Check if railway CLI is available
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found!"
    echo "   Install with: npm install -g @railway/cli"
    exit 1
fi

# Navigate to project directory
cd backend/modomo-scraper

echo "📁 Current directory: $(pwd)"
echo ""

# Set AI mode environment variable for build
echo "🔧 Setting AI mode environment variable..."
railway variables --set AI_MODE=full

echo "🚀 Deploying with full AI capabilities..."
echo "   This will install PyTorch, CLIP, and other AI dependencies"
echo "   Deployment may take 5-10 minutes due to large dependencies"
echo ""

# Deploy with AI mode
railway up --detach

echo ""
echo "✅ AI deployment initiated!"
echo ""
echo "📊 Features enabled:"
echo "   - GroundingDINO object detection"
echo "   - SAM2 segmentation"
echo "   - CLIP embeddings" 
echo "   - Vector similarity search"
echo "   - Real-time processing"
echo ""

echo "⏱️  Expected deployment time: 5-10 minutes"
echo "   Monitor progress: railway logs --follow"
echo ""

echo "🔗 Your Railway project:"
railway status

echo ""
echo "📋 Next steps:"
echo "1. Wait for AI deployment to complete"
echo "2. Check logs: railway logs"
echo "3. Test AI endpoints: /detect/process"
echo "4. Monitor GPU usage in Railway dashboard"
echo ""

echo "🤖 Full AI deployment complete!"
echo "   Your dataset creation pipeline now has real AI processing!"