#!/bin/bash
# TEMPORARY: Flush Railway volume contents
# Remove this after one deployment

echo "🗑️ FLUSHING VOLUME CONTENTS..."
rm -rf /app/models/*
echo "✅ Volume flushed, will download fresh models"