-- =============================================================================
-- REROOM SUPABASE CLOUD MIGRATION SCRIPT
-- =============================================================================
-- Execute this script in Supabase SQL Editor (Dashboard → SQL Editor)
-- This migrates from local PostgreSQL to Supabase cloud database

-- Drop existing tables if they exist (for clean migration)
DROP TABLE IF EXISTS product_suggestions CASCADE;
DROP TABLE IF EXISTS makeovers CASCADE;  
DROP TABLE IF EXISTS photos CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- =============================================================================
-- CORE TABLES
-- =============================================================================

-- Users table (Clerk integration)
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  clerk_user_id TEXT UNIQUE NOT NULL,
  email TEXT,
  subscription_tier TEXT DEFAULT 'free' CHECK (subscription_tier IN ('free', 'pro', 'premium')),
  preferences JSONB DEFAULT '{}',
  total_photos INTEGER DEFAULT 0,
  total_makeovers INTEGER DEFAULT 0,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Photos table with Cloudflare R2 integration
CREATE TABLE photos (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  clerk_user_id TEXT NOT NULL, -- Direct reference for RLS
  
  -- Cloudflare R2 storage info
  original_url TEXT NOT NULL,
  optimized_url TEXT,
  cloudflare_key TEXT NOT NULL,
  
  -- File metadata
  original_name TEXT,
  mime_type TEXT DEFAULT 'image/jpeg',
  original_size INTEGER,
  optimized_size INTEGER,
  width INTEGER,
  height INTEGER,
  
  -- Processing metadata  
  metadata JSONB DEFAULT '{}',
  status TEXT DEFAULT 'uploaded' CHECK (status IN ('uploaded', 'processing', 'ready', 'failed')),
  
  -- Timestamps
  taken_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- AI Makeovers table with RunPod integration
CREATE TABLE makeovers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  photo_id UUID REFERENCES photos(id) ON DELETE CASCADE,
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  clerk_user_id TEXT NOT NULL,
  
  -- RunPod job tracking
  runpod_job_id TEXT,
  runpod_status TEXT,
  
  -- Makeover configuration
  style_preference TEXT NOT NULL DEFAULT 'Modern',
  budget_range TEXT,
  room_type TEXT,
  
  -- Results
  makeover_url TEXT,
  before_url TEXT,
  detected_objects JSONB DEFAULT '[]',
  suggested_products JSONB DEFAULT '[]',
  
  -- Status tracking
  status TEXT DEFAULT 'queued' CHECK (status IN ('queued', 'processing', 'completed', 'failed')),
  progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
  error_message TEXT,
  
  -- Timestamps
  processing_started_at TIMESTAMP WITH TIME ZONE,
  completed_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Product suggestions with live pricing
CREATE TABLE product_suggestions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  makeover_id UUID REFERENCES makeovers(id) ON DELETE CASCADE,
  
  -- Product info
  product_name TEXT NOT NULL,
  category TEXT,
  description TEXT,
  brand TEXT,
  
  -- Pricing from multiple retailers
  amazon_price DECIMAL(10,2),
  amazon_url TEXT,
  ikea_price DECIMAL(10,2), 
  ikea_url TEXT,
  wayfair_price DECIMAL(10,2),
  wayfair_url TEXT,
  
  -- Product media
  image_url TEXT,
  product_images JSONB DEFAULT '[]',
  
  -- AI confidence and matching
  confidence_score DECIMAL(3,2), -- 0.00-1.00
  
  -- Timestamps
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- ROW LEVEL SECURITY (RLS)
-- =============================================================================

-- Enable Row Level Security
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE photos ENABLE ROW LEVEL SECURITY;
ALTER TABLE makeovers ENABLE ROW LEVEL SECURITY; 
ALTER TABLE product_suggestions ENABLE ROW LEVEL SECURITY;

-- 🔐 RLS Policies for Clerk Authentication
-- Users can only access their own data
CREATE POLICY "Users can manage their own profile" ON users
  FOR ALL USING (clerk_user_id = auth.jwt() ->> 'sub');

-- Photos access control
CREATE POLICY "Users can manage their own photos" ON photos
  FOR ALL USING (clerk_user_id = auth.jwt() ->> 'sub');

-- Makeovers access control  
CREATE POLICY "Users can manage their own makeovers" ON makeovers
  FOR ALL USING (clerk_user_id = auth.jwt() ->> 'sub');

-- Product suggestions read access
CREATE POLICY "Users can view products for their makeovers" ON product_suggestions
  FOR SELECT USING (
    makeover_id IN (
      SELECT id FROM makeovers WHERE clerk_user_id = auth.jwt() ->> 'sub'
    )
  );

-- =============================================================================
-- PERFORMANCE INDEXES
-- =============================================================================

-- Core user and authentication indexes
CREATE INDEX idx_users_clerk_id ON users(clerk_user_id);

-- Photo query optimization
CREATE INDEX idx_photos_user_created ON photos(clerk_user_id, created_at DESC);
CREATE INDEX idx_photos_status ON photos(status);
CREATE INDEX idx_photos_cloudflare_key ON photos(cloudflare_key);

-- Makeover tracking indexes
CREATE INDEX idx_makeovers_user_status ON makeovers(clerk_user_id, status);
CREATE INDEX idx_makeovers_photo_id ON makeovers(photo_id);
CREATE INDEX idx_makeovers_runpod_job ON makeovers(runpod_job_id);
CREATE INDEX idx_makeovers_status ON makeovers(status);

-- Product suggestion indexes
CREATE INDEX idx_product_suggestions_makeover ON product_suggestions(makeover_id);
CREATE INDEX idx_product_suggestions_confidence ON product_suggestions(confidence_score DESC);
CREATE INDEX idx_product_suggestions_category ON product_suggestions(category);

-- =============================================================================
-- DATABASE FUNCTIONS
-- =============================================================================

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ language 'plpgsql';

-- User statistics increment function
CREATE OR REPLACE FUNCTION increment_user_stats(
  user_clerk_id TEXT,
  photo_increment INTEGER DEFAULT 0,
  makeover_increment INTEGER DEFAULT 0
)
RETURNS VOID AS $$
BEGIN
  UPDATE users 
  SET 
    total_photos = total_photos + photo_increment,
    total_makeovers = total_makeovers + makeover_increment,
    updated_at = NOW()
  WHERE clerk_user_id = user_clerk_id;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- TRIGGERS
-- =============================================================================

-- Auto-update timestamps
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_photos_updated_at BEFORE UPDATE ON photos
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_makeovers_updated_at BEFORE UPDATE ON makeovers  
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_product_suggestions_updated_at BEFORE UPDATE ON product_suggestions
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- REAL-TIME SUBSCRIPTIONS
-- =============================================================================

-- Enable real-time subscriptions for makeover updates
ALTER PUBLICATION supabase_realtime ADD TABLE makeovers;
ALTER PUBLICATION supabase_realtime ADD TABLE photos;

-- =============================================================================
-- INITIAL TEST DATA
-- =============================================================================

-- Insert test user for development
INSERT INTO users (clerk_user_id, email, subscription_tier) VALUES
('user_test_reroom_123', 'test@reroom.app', 'free')
ON CONFLICT (clerk_user_id) DO NOTHING;

-- =============================================================================
-- MIGRATION VERIFICATION
-- =============================================================================

-- Verify tables were created successfully
DO $$
DECLARE
    table_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO table_count 
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND table_name IN ('users', 'photos', 'makeovers', 'product_suggestions');
    
    IF table_count = 4 THEN
        RAISE NOTICE '✅ All 4 tables created successfully';
        RAISE NOTICE '✅ RLS policies enabled';
        RAISE NOTICE '✅ Indexes created for performance';
        RAISE NOTICE '✅ Real-time subscriptions enabled';
        RAISE NOTICE '';
        RAISE NOTICE '🎯 NEXT STEPS:';
        RAISE NOTICE '1. Get your Supabase ANON key from Settings → API';
        RAISE NOTICE '2. Get your Supabase SERVICE ROLE key from Settings → API';
        RAISE NOTICE '3. Update .env file with the keys';
        RAISE NOTICE '4. Test connection from your backend service';
    ELSE
        RAISE EXCEPTION 'Migration failed: Expected 4 tables, found %', table_count;
    END IF;
END $$;

-- Success message
SELECT 
  '🎉 ReRoom Supabase Migration Completed Successfully! 🎉' as status,
  NOW() as completed_at,
  'Ready for cloud integration' as next_step;