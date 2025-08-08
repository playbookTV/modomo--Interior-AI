#!/bin/bash

# Script to install pgvector extension in running PostgreSQL container
set -e

echo "🔧 Installing pgvector extension in PostgreSQL..."

# Find the PostgreSQL container
POSTGRES_CONTAINER=$(docker ps -q -f name=postgres)

if [ -z "$POSTGRES_CONTAINER" ]; then
    echo "❌ PostgreSQL container not found. Please start with 'pnpm run docker:up' first."
    exit 1
fi

echo "✅ Found PostgreSQL container: $POSTGRES_CONTAINER"

# Update package list and install build dependencies
echo "📦 Installing build dependencies..."
docker exec $POSTGRES_CONTAINER bash -c "
    apt-get update && 
    apt-get install -y wget build-essential postgresql-server-dev-14 git
"

# Install pgvector from source
echo "🛠️ Building and installing pgvector..."
docker exec $POSTGRES_CONTAINER bash -c "
    cd /tmp &&
    git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git &&
    cd pgvector &&
    make &&
    make install
"

# Restart PostgreSQL to load the extension
echo "🔄 Restarting PostgreSQL..."
docker restart $POSTGRES_CONTAINER
sleep 5

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to restart..."
until docker exec $POSTGRES_CONTAINER pg_isready -U reroom -d reroom_dev; do
  sleep 2
done

# Create the extension
echo "🎯 Creating pgvector extension..."
docker exec $POSTGRES_CONTAINER psql -U reroom -d reroom_dev -c "CREATE EXTENSION IF NOT EXISTS vector;"

if [ $? -eq 0 ]; then
    echo "✅ pgvector extension installed successfully!"
else
    echo "⚠️ Failed to create extension, but continuing..."
fi

echo "🎉 pgvector installation complete!"