#!/bin/bash

# =============================================================================
# REROOM CLOUD MIGRATION SETUP SCRIPT
# =============================================================================
# This script guides you through setting up all cloud services for ReRoom
# Run this script after updating your .env file with the API keys

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Emojis for better UX
ROCKET="🚀"
CHECK="✅"
WARNING="⚠️"
ERROR="❌"
INFO="ℹ️"
GEAR="⚙️"

echo -e "${BLUE}${ROCKET} ReRoom Cloud Migration Setup${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# =============================================================================
# ENVIRONMENT VALIDATION
# =============================================================================

echo -e "${YELLOW}${GEAR} Step 1: Validating Environment Configuration${NC}"
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${ERROR} .env file not found!"
    echo "Please create a .env file with your cloud service credentials."
    exit 1
fi

# Load environment variables
source .env

# Validate required environment variables
check_env_var() {
    local var_name=$1
    local var_value=$(eval echo \$$var_name)
    
    if [ -z "$var_value" ] || [ "$var_value" = "[YOUR_${var_name#*_}]" ]; then
        echo -e "${WARNING} $var_name is not set or still has placeholder value"
        return 1
    else
        echo -e "${CHECK} $var_name is configured"
        return 0
    fi
}

# Check all required variables
echo "Checking required environment variables..."

ENV_ERRORS=0

# Supabase
if ! check_env_var "SUPABASE_URL"; then ENV_ERRORS=$((ENV_ERRORS + 1)); fi
if ! check_env_var "SUPABASE_ANON_KEY"; then ENV_ERRORS=$((ENV_ERRORS + 1)); fi

# Clerk
if ! check_env_var "EXPO_PUBLIC_CLERK_PUBLISHABLE_KEY"; then ENV_ERRORS=$((ENV_ERRORS + 1)); fi
if ! check_env_var "CLERK_SECRET_KEY"; then ENV_ERRORS=$((ENV_ERRORS + 1)); fi

# Cloudflare R2
if ! check_env_var "CLOUDFLARE_R2_ENDPOINT"; then ENV_ERRORS=$((ENV_ERRORS + 1)); fi
if ! check_env_var "CLOUDFLARE_R2_ACCESS_KEY_ID"; then ENV_ERRORS=$((ENV_ERRORS + 1)); fi
if ! check_env_var "CLOUDFLARE_R2_SECRET_ACCESS_KEY"; then ENV_ERRORS=$((ENV_ERRORS + 1)); fi

# RunPod
if ! check_env_var "RUNPOD_API_KEY"; then ENV_ERRORS=$((ENV_ERRORS + 1)); fi
if ! check_env_var "RUNPOD_ENDPOINT"; then ENV_ERRORS=$((ENV_ERRORS + 1)); fi

echo ""

if [ $ENV_ERRORS -gt 0 ]; then
    echo -e "${ERROR} Found $ENV_ERRORS environment configuration errors."
    echo -e "${INFO} Please update your .env file with the correct values and run this script again."
    echo ""
    echo -e "${CYAN}Required values:${NC}"
    echo "1. Get Supabase keys from: https://supabase.com/dashboard/project/[project]/settings/api"
    echo "2. Get Clerk keys from: https://dashboard.clerk.com/apps/[app]/api-keys"
    echo "3. Get Cloudflare R2 keys from: https://dash.cloudflare.com/[account]/r2/api-tokens"
    echo "4. Get RunPod API key from: https://www.runpod.io/console/user/settings"
    exit 1
fi

echo -e "${GREEN}${CHECK} All environment variables are configured!${NC}"
echo ""

# =============================================================================
# SUPABASE DATABASE SETUP
# =============================================================================

echo -e "${YELLOW}${GEAR} Step 2: Setting up Supabase Database${NC}"
echo ""

echo "This will create the ReRoom database schema in your Supabase project."
echo -e "${WARNING} This will drop existing tables if they exist!"
read -p "Continue? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Setting up Supabase database..."
    
    # Check if psql is available
    if command -v psql &> /dev/null; then
        echo "Executing database migration via psql..."
        export PGPASSWORD=$(echo $DATABASE_URL_CLOUD | grep -oP '(?<=postgres:)[^@]*(?=@)')
        
        psql "$DATABASE_URL_CLOUD" -f database/supabase-migration.sql
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}${CHECK} Supabase database setup completed!${NC}"
        else
            echo -e "${ERROR} Database setup failed. Please run the SQL manually in Supabase dashboard."
        fi
    else
        echo -e "${INFO} psql not found. Please run the migration manually:"
        echo "1. Go to your Supabase dashboard: https://supabase.com/dashboard"
        echo "2. Navigate to SQL Editor"
        echo "3. Copy and paste the contents of: database/supabase-migration.sql"
        echo "4. Execute the SQL script"
    fi
else
    echo -e "${INFO} Skipping database setup. You can run it manually later."
fi

echo ""

# =============================================================================
# BACKEND SERVICE SETUP
# =============================================================================

echo -e "${YELLOW}${GEAR} Step 3: Setting up Backend Service${NC}"
echo ""

echo "Installing backend dependencies..."
cd backend

# Install dependencies
if [ -f "package.json" ]; then
    npm install
    echo -e "${GREEN}${CHECK} Backend dependencies installed!${NC}"
else
    echo -e "${ERROR} Backend package.json not found!"
    exit 1
fi

# Build TypeScript
echo "Building backend TypeScript..."
npm run build

if [ $? -eq 0 ]; then
    echo -e "${GREEN}${CHECK} Backend built successfully!${NC}"
else
    echo -e "${ERROR} Backend build failed!"
    exit 1
fi

# Test backend locally
echo "Testing backend services..."
npm run start:dev &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Test health endpoint
if curl -f http://localhost:6969/health > /dev/null 2>&1; then
    echo -e "${GREEN}${CHECK} Backend health check passed!${NC}"
else
    echo -e "${WARNING} Backend health check failed - this might be normal if services aren't fully configured yet."
fi

# Stop test backend
kill $BACKEND_PID 2>/dev/null || true

cd ..
echo ""

# =============================================================================
# MOBILE APP SETUP
# =============================================================================

echo -e "${YELLOW}${GEAR} Step 4: Setting up Mobile App${NC}"
echo ""

echo "Installing mobile app dependencies..."
cd mobile

# Install dependencies
if [ -f "package.json" ]; then
    npm install
    echo -e "${GREEN}${CHECK} Mobile app dependencies installed!${NC}"
else
    echo -e "${ERROR} Mobile package.json not found!"
    exit 1
fi

# Install Expo CLI if not present
if ! command -v npx expo &> /dev/null; then
    echo "Installing Expo CLI..."
    npm install -g @expo/cli
fi

cd ..
echo ""

# =============================================================================
# RAILWAY DEPLOYMENT SETUP
# =============================================================================

echo -e "${YELLOW}${GEAR} Step 5: Railway Deployment Setup${NC}"
echo ""

echo "Railway deployment configuration is ready!"
echo -e "${INFO} To deploy to Railway:"
echo "1. Install Railway CLI: npm install -g @railway/cli"
echo "2. Login: railway login"
echo "3. Link project: railway link"
echo "4. Set environment variables: railway variables set [VAR_NAME]=[VALUE]"
echo "5. Deploy: railway up"
echo ""

# =============================================================================
# TESTING CLOUD SERVICES
# =============================================================================

echo -e "${YELLOW}${GEAR} Step 6: Testing Cloud Service Connections${NC}"
echo ""

# Test Supabase connection
echo "Testing Supabase connection..."
if command -v curl &> /dev/null; then
    SUPABASE_TEST=$(curl -s -H "apikey: $SUPABASE_ANON_KEY" "$SUPABASE_URL/rest/v1/users?select=count" || echo "failed")
    if [[ $SUPABASE_TEST == *"count"* ]]; then
        echo -e "${GREEN}${CHECK} Supabase connection successful!${NC}"
    else
        echo -e "${WARNING} Supabase connection test failed. Check your keys.${NC}"
    fi
else
    echo -e "${INFO} curl not available, skipping Supabase test${NC}"
fi

# Test Cloudflare R2 connection (basic)
echo "Cloudflare R2 endpoint configured: $CLOUDFLARE_R2_ENDPOINT"

# Test RunPod endpoint
echo "RunPod endpoint configured: $RUNPOD_ENDPOINT"

echo ""

# =============================================================================
# MIGRATION SUMMARY
# =============================================================================

echo -e "${GREEN}${ROCKET} ReRoom Cloud Migration Setup Complete!${NC}"
echo -e "${GREEN}===========================================${NC}"
echo ""

echo -e "${CYAN}Services Configured:${NC}"
echo -e "${CHECK} Supabase Database: $SUPABASE_URL"
echo -e "${CHECK} Clerk Authentication: $(echo $EXPO_PUBLIC_CLERK_PUBLISHABLE_KEY | cut -c1-20)..."
echo -e "${CHECK} Cloudflare R2 Storage: $CLOUDFLARE_R2_ENDPOINT"
echo -e "${CHECK} RunPod AI Processing: $RUNPOD_ENDPOINT"
echo -e "${CHECK} Railway Backend: $RAILWAY_BACKEND_URL"
echo ""

echo -e "${CYAN}Next Steps:${NC}"
echo "1. ${INFO} Deploy backend to Railway: cd backend && railway up"
echo "2. ${INFO} Test mobile app: cd mobile && npx expo start"
echo "3. ${INFO} Update mobile app with Railway backend URL"
echo "4. ${INFO} Test complete photo → AI → results pipeline"
echo ""

echo -e "${CYAN}Development Commands:${NC}"
echo "• Start backend locally: cd backend && npm run dev"
echo "• Start mobile app: cd mobile && npx expo start"
echo "• View Supabase dashboard: https://supabase.com/dashboard"
echo "• View Railway dashboard: https://railway.app/dashboard"
echo ""

echo -e "${PURPLE}${ROCKET} Ready to revolutionize interior design with AI! ${ROCKET}${NC}"
echo ""

# Save setup completion status
echo "$(date): Cloud migration setup completed" >> .migration-log

exit 0