-- Modomo Dataset Creation Database Schema
-- Extends existing ReRoom database with tables for scene scraping and object detection

-- Enable pgvector extension for CLIP embeddings (skip if not available)
DO $$ 
BEGIN
    CREATE EXTENSION IF NOT EXISTS vector;
    RAISE NOTICE 'pgvector extension enabled successfully';
EXCEPTION 
    WHEN OTHERS THEN
        RAISE WARNING 'pgvector extension not available - CLIP embeddings will be stored as JSON';
END $$;

-- Scenes table for scraped interior design images
CREATE TABLE IF NOT EXISTS scenes (
    scene_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    houzz_id VARCHAR(255) UNIQUE NOT NULL,
    image_url TEXT NOT NULL,
    image_r2_key TEXT, -- R2 storage key for downloaded image
    room_type VARCHAR(100),
    style_tags TEXT[] DEFAULT '{}',
    color_tags TEXT[] DEFAULT '{}',
    project_url TEXT,
    status VARCHAR(50) DEFAULT 'scraped' CHECK (status IN ('scraped', 'processing', 'pending_review', 'approved', 'rejected')),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    reviewed_at TIMESTAMP WITH TIME ZONE,
    reviewed_by VARCHAR(255)
);

-- Create indexes for scenes
CREATE INDEX IF NOT EXISTS idx_scenes_status ON scenes(status);
CREATE INDEX IF NOT EXISTS idx_scenes_room_type ON scenes(room_type);
CREATE INDEX IF NOT EXISTS idx_scenes_created_at ON scenes(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_scenes_houzz_id ON scenes(houzz_id);

-- Products table for catalog items
CREATE TABLE IF NOT EXISTS products (
    product_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sku VARCHAR(255) UNIQUE NOT NULL,
    name TEXT NOT NULL,
    category VARCHAR(100) NOT NULL,
    price_gbp DECIMAL(10,2),
    brand VARCHAR(255),
    material VARCHAR(255),
    dimensions JSONB DEFAULT '{}', -- {width: 120, height: 80, depth: 45} in mm
    image_url TEXT,
    image_r2_key TEXT, -- R2 storage key for product image
    product_url TEXT,
    description TEXT,
    retailer VARCHAR(100),
    clip_embedding_vector vector(512), -- CLIP embedding vector (if pgvector available)
    clip_embedding_json JSONB, -- CLIP embedding as JSON fallback
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    active BOOLEAN DEFAULT true
);

-- Create indexes for products
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_products_brand ON products(brand);
CREATE INDEX IF NOT EXISTS idx_products_retailer ON products(retailer);
CREATE INDEX IF NOT EXISTS idx_products_sku ON products(sku);
CREATE INDEX IF NOT EXISTS idx_products_active ON products(active) WHERE active = true;

-- HNSW index for CLIP embedding similarity search (only if pgvector available)
DO $$ 
BEGIN
    CREATE INDEX IF NOT EXISTS idx_products_clip_embedding ON products 
    USING hnsw (clip_embedding_vector vector_cosine_ops) 
    WITH (m = 16, ef_construction = 64);
EXCEPTION 
    WHEN OTHERS THEN
        RAISE WARNING 'Skipping vector index - pgvector not available';
END $$;

-- Detected objects table
CREATE TABLE IF NOT EXISTS detected_objects (
    object_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scene_id UUID NOT NULL REFERENCES scenes(scene_id) ON DELETE CASCADE,
    bbox DECIMAL[] NOT NULL CHECK (array_length(bbox, 1) = 4), -- [x, y, width, height]
    mask_r2_key TEXT, -- R2 storage key for segmentation mask
    category VARCHAR(100) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    tags TEXT[] DEFAULT '{}',
    matched_product_id UUID REFERENCES products(product_id) ON DELETE SET NULL,
    clip_embedding_vector vector(512), -- CLIP embedding vector (if pgvector available)
    clip_embedding_json JSONB, -- CLIP embedding as JSON fallback
    approved BOOLEAN DEFAULT NULL, -- NULL = not reviewed, true = approved, false = rejected
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for detected objects
CREATE INDEX IF NOT EXISTS idx_detected_objects_scene_id ON detected_objects(scene_id);
CREATE INDEX IF NOT EXISTS idx_detected_objects_category ON detected_objects(category);
CREATE INDEX IF NOT EXISTS idx_detected_objects_confidence ON detected_objects(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_detected_objects_approved ON detected_objects(approved);
CREATE INDEX IF NOT EXISTS idx_detected_objects_matched_product ON detected_objects(matched_product_id);

-- HNSW index for object CLIP embeddings (only if pgvector available)
DO $$ 
BEGIN
    CREATE INDEX IF NOT EXISTS idx_detected_objects_clip_embedding ON detected_objects 
    USING hnsw (clip_embedding_vector vector_cosine_ops) 
    WITH (m = 16, ef_construction = 64);
EXCEPTION 
    WHEN OTHERS THEN
        RAISE WARNING 'Skipping vector index - pgvector not available';
END $$;

-- Scraping jobs table for tracking crawl progress
CREATE TABLE IF NOT EXISTS scraping_jobs (
    job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_type VARCHAR(50) NOT NULL CHECK (job_type IN ('scenes', 'products', 'detection')),
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    total_items INTEGER DEFAULT 0,
    processed_items INTEGER DEFAULT 0,
    parameters JSONB DEFAULT '{}', -- Job-specific parameters
    error_message TEXT,
    results JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for jobs
CREATE INDEX IF NOT EXISTS idx_scraping_jobs_status ON scraping_jobs(status);
CREATE INDEX IF NOT EXISTS idx_scraping_jobs_type ON scraping_jobs(job_type);
CREATE INDEX IF NOT EXISTS idx_scraping_jobs_created_at ON scraping_jobs(created_at DESC);

-- Dataset exports table
CREATE TABLE IF NOT EXISTS dataset_exports (
    export_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    export_name VARCHAR(255),
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    split_ratios JSONB NOT NULL, -- {train: 0.7, val: 0.2, test: 0.1}
    scene_count INTEGER,
    object_count INTEGER,
    manifest_r2_keys JSONB, -- {train: "key1", val: "key2", test: "key3"}
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    created_by VARCHAR(255)
);

-- Create index for exports
CREATE INDEX IF NOT EXISTS idx_dataset_exports_status ON dataset_exports(status);
CREATE INDEX IF NOT EXISTS idx_dataset_exports_created_at ON dataset_exports(created_at DESC);

-- Product similarities table for caching CLIP matches
CREATE TABLE IF NOT EXISTS product_similarities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    object_id UUID NOT NULL REFERENCES detected_objects(object_id) ON DELETE CASCADE,
    product_id UUID NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
    similarity_score DECIMAL(6,5) NOT NULL CHECK (similarity_score >= 0 AND similarity_score <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(object_id, product_id)
);

-- Create indexes for similarities
CREATE INDEX IF NOT EXISTS idx_product_similarities_object_id ON product_similarities(object_id);
CREATE INDEX IF NOT EXISTS idx_product_similarities_product_id ON product_similarities(product_id);
CREATE INDEX IF NOT EXISTS idx_product_similarities_score ON product_similarities(similarity_score DESC);

-- Review statistics table
CREATE TABLE IF NOT EXISTS review_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reviewer_id VARCHAR(255) NOT NULL,
    review_date DATE DEFAULT CURRENT_DATE,
    objects_reviewed INTEGER DEFAULT 0,
    objects_approved INTEGER DEFAULT 0,
    objects_rejected INTEGER DEFAULT 0,
    avg_review_time_seconds DECIMAL(8,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(reviewer_id, review_date)
);

-- Create index for review stats
CREATE INDEX IF NOT EXISTS idx_review_stats_reviewer ON review_stats(reviewer_id);
CREATE INDEX IF NOT EXISTS idx_review_stats_date ON review_stats(review_date DESC);

-- Functions for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_scenes_updated_at BEFORE UPDATE ON scenes 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_products_updated_at BEFORE UPDATE ON products 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_detected_objects_updated_at BEFORE UPDATE ON detected_objects 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to calculate similarity scores
CREATE OR REPLACE FUNCTION calculate_object_product_similarities(
    target_object_id UUID,
    similarity_threshold DECIMAL DEFAULT 0.5,
    max_results INTEGER DEFAULT 10
)
RETURNS TABLE (
    product_id UUID,
    similarity_score DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p.product_id,
        (1 - (o.clip_embedding <=> p.clip_embedding))::DECIMAL(6,5) as similarity_score
    FROM detected_objects o
    CROSS JOIN products p
    WHERE o.object_id = target_object_id 
        AND o.clip_embedding IS NOT NULL 
        AND p.clip_embedding IS NOT NULL
        AND p.active = true
        AND (1 - (o.clip_embedding <=> p.clip_embedding)) >= similarity_threshold
    ORDER BY similarity_score DESC
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Function to get review queue
CREATE OR REPLACE FUNCTION get_review_queue(
    queue_limit INTEGER DEFAULT 10,
    filter_room_type VARCHAR DEFAULT NULL,
    filter_category VARCHAR DEFAULT NULL
)
RETURNS TABLE (
    scene_id UUID,
    image_url TEXT,
    room_type VARCHAR,
    style_tags TEXT[],
    objects JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        s.scene_id,
        s.image_url,
        s.room_type,
        s.style_tags,
        jsonb_agg(
            jsonb_build_object(
                'object_id', o.object_id,
                'bbox', array_to_json(o.bbox),
                'category', o.category,
                'confidence', o.confidence,
                'tags', o.tags,
                'approved', o.approved,
                'matched_product_id', o.matched_product_id
            )
        ) as objects
    FROM scenes s
    JOIN detected_objects o ON s.scene_id = o.scene_id
    WHERE s.status = 'pending_review'
        AND (filter_room_type IS NULL OR s.room_type = filter_room_type)
        AND (filter_category IS NULL OR o.category = filter_category)
    GROUP BY s.scene_id, s.image_url, s.room_type, s.style_tags
    ORDER BY s.created_at ASC
    LIMIT queue_limit;
END;
$$ LANGUAGE plpgsql;

-- View for dataset statistics
CREATE OR REPLACE VIEW dataset_stats AS
SELECT 
    COUNT(DISTINCT s.scene_id) as total_scenes,
    COUNT(DISTINCT CASE WHEN s.status = 'approved' THEN s.scene_id END) as approved_scenes,
    COUNT(o.object_id) as total_objects,
    COUNT(CASE WHEN o.approved = true THEN o.object_id END) as approved_objects,
    COUNT(DISTINCT o.category) as unique_categories,
    AVG(o.confidence) as avg_confidence,
    COUNT(CASE WHEN o.matched_product_id IS NOT NULL THEN o.object_id END) as objects_with_products
FROM scenes s
LEFT JOIN detected_objects o ON s.scene_id = o.scene_id;

-- View for category statistics
CREATE OR REPLACE VIEW category_stats AS
SELECT 
    o.category,
    COUNT(*) as total_objects,
    COUNT(CASE WHEN o.approved = true THEN 1 END) as approved_objects,
    AVG(o.confidence) as avg_confidence,
    COUNT(CASE WHEN o.matched_product_id IS NOT NULL THEN 1 END) as matched_objects
FROM detected_objects o
GROUP BY o.category
ORDER BY total_objects DESC;