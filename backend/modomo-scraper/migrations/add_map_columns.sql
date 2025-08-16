-- Migration: Add map URL columns to scenes table
-- Date: 2025-01-16
-- Description: Add columns for storing depth map and edge map R2 keys

-- Add depth map and edge map columns to scenes table
ALTER TABLE scenes ADD COLUMN IF NOT EXISTS depth_map_r2_key TEXT;
ALTER TABLE scenes ADD COLUMN IF NOT EXISTS edge_map_r2_key TEXT;
ALTER TABLE scenes ADD COLUMN IF NOT EXISTS maps_generated_at TIMESTAMP WITH TIME ZONE;
ALTER TABLE scenes ADD COLUMN IF NOT EXISTS maps_metadata JSONB DEFAULT '{}';

-- Add indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_scenes_depth_map ON scenes(depth_map_r2_key) WHERE depth_map_r2_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_scenes_edge_map ON scenes(edge_map_r2_key) WHERE edge_map_r2_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_scenes_maps_generated ON scenes(maps_generated_at) WHERE maps_generated_at IS NOT NULL;

-- Add comments for documentation
COMMENT ON COLUMN scenes.depth_map_r2_key IS 'R2 storage key for depth map (training-data/maps/depth/{scene_id}_depth.png)';
COMMENT ON COLUMN scenes.edge_map_r2_key IS 'R2 storage key for edge map (training-data/maps/edge/{scene_id}_edge.png)';
COMMENT ON COLUMN scenes.maps_generated_at IS 'Timestamp when maps were generated';
COMMENT ON COLUMN scenes.maps_metadata IS 'JSON metadata about map generation (model versions, parameters, etc.)';

-- Optional: Add map URLs for direct access (computed from R2 keys)
-- These can be computed dynamically but stored for performance
ALTER TABLE scenes ADD COLUMN IF NOT EXISTS depth_map_url TEXT;
ALTER TABLE scenes ADD COLUMN IF NOT EXISTS edge_map_url TEXT;

COMMENT ON COLUMN scenes.depth_map_url IS 'Public URL for depth map (computed from R2 key)';
COMMENT ON COLUMN scenes.edge_map_url IS 'Public URL for edge map (computed from R2 key)';

-- Create a view for scenes with maps
CREATE OR REPLACE VIEW scenes_with_maps AS
SELECT 
    s.*,
    (s.depth_map_r2_key IS NOT NULL) AS has_depth_map,
    (s.edge_map_r2_key IS NOT NULL) AS has_edge_map,
    CASE 
        WHEN s.depth_map_r2_key IS NOT NULL AND s.edge_map_r2_key IS NOT NULL THEN 'both'
        WHEN s.depth_map_r2_key IS NOT NULL THEN 'depth_only'
        WHEN s.edge_map_r2_key IS NOT NULL THEN 'edge_only'
        ELSE 'none'
    END AS maps_available
FROM scenes s;

COMMENT ON VIEW scenes_with_maps IS 'View showing scenes with map availability information';

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, UPDATE ON scenes TO your_app_user;
-- GRANT SELECT ON scenes_with_maps TO your_app_user;