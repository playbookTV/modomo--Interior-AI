-- ReRoom Database Initialization Script
-- This script creates the initial database structure

-- Create database if it doesn't exist (handled by Docker)
-- CREATE DATABASE reroom_dev;

-- Connect to the database
\c reroom_dev;

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create ENUM types
CREATE TYPE subscription_type AS ENUM ('free', 'premium');
CREATE TYPE subscription_status AS ENUM ('active', 'cancelled', 'expired');
CREATE TYPE room_type AS ENUM ('living_room', 'bedroom', 'kitchen', 'bathroom', 'dining_room', 'office', 'other');
CREATE TYPE style_type AS ENUM ('modern', 'scandinavian', 'boho', 'industrial', 'minimalist', 'traditional', 'eclectic', 'farmhouse', 'mid-century', 'contemporary', 'coastal', 'rustic');
CREATE TYPE product_category AS ENUM ('sofa', 'chair', 'table', 'lamp', 'rug', 'artwork', 'decor', 'storage', 'bed', 'dresser', 'mirror', 'plant', 'cushion', 'curtain', 'other');
CREATE TYPE processing_stage AS ENUM ('analyzing', 'styling', 'matching', 'optimizing', 'complete', 'failed');
CREATE TYPE availability_status AS ENUM ('in_stock', 'low_stock', 'out_of_stock');

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    avatar VARCHAR(500),
    preferences JSONB DEFAULT '{}',
    email_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Subscriptions table
CREATE TABLE subscriptions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    type subscription_type NOT NULL DEFAULT 'free',
    status subscription_status NOT NULL DEFAULT 'active',
    current_period_start TIMESTAMP WITH TIME ZONE,
    current_period_end TIMESTAMP WITH TIME ZONE,
    cancel_at_period_end BOOLEAN DEFAULT FALSE,
    stripe_customer_id VARCHAR(255),
    stripe_subscription_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Retailers table
CREATE TABLE retailers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    logo VARCHAR(500),
    base_url VARCHAR(500) NOT NULL,
    shipping_info JSONB DEFAULT '{}',
    return_policy JSONB DEFAULT '{}',
    trust_score DECIMAL(3,2) DEFAULT 5.00,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Products table
CREATE TABLE products (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    retailer_id UUID NOT NULL REFERENCES retailers(id),
    external_id VARCHAR(255),
    title VARCHAR(500) NOT NULL,
    description TEXT,
    price DECIMAL(10,2) NOT NULL,
    original_price DECIMAL(10,2),
    currency VARCHAR(3) DEFAULT 'GBP',
    category product_category NOT NULL,
    brand VARCHAR(100),
    image_urls TEXT[],
    affiliate_url VARCHAR(1000) NOT NULL,
    availability availability_status DEFAULT 'in_stock',
    rating DECIMAL(3,2),
    review_count INTEGER DEFAULT 0,
    dimensions JSONB DEFAULT '{}',
    features TEXT[],
    is_active BOOLEAN DEFAULT TRUE,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Rooms table
CREATE TABLE rooms (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    type room_type NOT NULL,
    original_photo_url VARCHAR(1000) NOT NULL,
    total_budget DECIMAL(10,2),
    actual_spent DECIMAL(10,2) DEFAULT 0,
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- AI Renders table
CREATE TABLE ai_renders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    room_id UUID NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    original_photo_url VARCHAR(1000) NOT NULL,
    styled_image_url VARCHAR(1000),
    style style_type NOT NULL,
    confidence DECIMAL(5,4),
    processing_time DECIMAL(8,2),
    total_cost DECIMAL(10,2) DEFAULT 0,
    estimated_savings DECIMAL(10,2) DEFAULT 0,
    processing_stage processing_stage DEFAULT 'analyzing',
    processing_progress INTEGER DEFAULT 0,
    processing_message TEXT,
    processing_started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Render Products (many-to-many relationship)
CREATE TABLE render_products (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    render_id UUID NOT NULL REFERENCES ai_renders(id) ON DELETE CASCADE,
    product_id UUID NOT NULL REFERENCES products(id),
    visual_similarity DECIMAL(5,4),
    position_x DECIMAL(5,4),
    position_y DECIMAL(5,4),
    confidence DECIMAL(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(render_id, product_id)
);

-- User Favorites table
CREATE TABLE user_favorites (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    product_id UUID REFERENCES products(id) ON DELETE CASCADE,
    room_id UUID REFERENCES rooms(id) ON DELETE CASCADE,
    render_id UUID REFERENCES ai_renders(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT user_favorites_check CHECK (
        (product_id IS NOT NULL AND room_id IS NULL AND render_id IS NULL) OR
        (product_id IS NULL AND room_id IS NOT NULL AND render_id IS NULL) OR
        (product_id IS NULL AND room_id IS NULL AND render_id IS NOT NULL)
    )
);

-- Shopping Cart table
CREATE TABLE shopping_cart (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    product_id UUID NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    quantity INTEGER NOT NULL DEFAULT 1,
    added_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, product_id)
);

-- Price History table (for tracking price changes)
CREATE TABLE price_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    product_id UUID NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    price DECIMAL(10,2) NOT NULL,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Analytics Events table
CREATE TABLE analytics_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    event_name VARCHAR(100) NOT NULL,
    properties JSONB DEFAULT '{}',
    session_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_subscriptions_user_id ON subscriptions(user_id);
CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_products_retailer_id ON products(retailer_id);
CREATE INDEX idx_products_price ON products(price);
CREATE INDEX idx_rooms_user_id ON rooms(user_id);
CREATE INDEX idx_ai_renders_room_id ON ai_renders(room_id);
CREATE INDEX idx_ai_renders_user_id ON ai_renders(user_id);
CREATE INDEX idx_ai_renders_style ON ai_renders(style);
CREATE INDEX idx_ai_renders_processing_stage ON ai_renders(processing_stage);
CREATE INDEX idx_render_products_render_id ON render_products(render_id);
CREATE INDEX idx_render_products_product_id ON render_products(product_id);
CREATE INDEX idx_user_favorites_user_id ON user_favorites(user_id);
CREATE INDEX idx_shopping_cart_user_id ON shopping_cart(user_id);
CREATE INDEX idx_price_history_product_id ON price_history(product_id);
CREATE INDEX idx_price_history_recorded_at ON price_history(recorded_at);
CREATE INDEX idx_analytics_events_user_id ON analytics_events(user_id);
CREATE INDEX idx_analytics_events_event_name ON analytics_events(event_name);
CREATE INDEX idx_analytics_events_created_at ON analytics_events(created_at);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at trigger to relevant tables
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_subscriptions_updated_at BEFORE UPDATE ON subscriptions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_retailers_updated_at BEFORE UPDATE ON retailers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_rooms_updated_at BEFORE UPDATE ON rooms
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default retailers
INSERT INTO retailers (name, logo, base_url, shipping_info, return_policy, trust_score) VALUES 
('Amazon UK', 'https://logo.clearbit.com/amazon.co.uk', 'https://amazon.co.uk', 
 '{"free_shipping_threshold": 25, "standard_delivery_days": 2, "express_delivery_days": 1}',
 '{"days": 30, "conditions": ["Original packaging", "Unused condition"]}', 9.2),
 
('Argos', 'https://logo.clearbit.com/argos.co.uk', 'https://argos.co.uk',
 '{"free_shipping_threshold": 30, "standard_delivery_days": 3, "express_delivery_days": 1}',
 '{"days": 30, "conditions": ["Original packaging", "Receipt required"]}', 8.5),
 
('IKEA', 'https://logo.clearbit.com/ikea.com', 'https://ikea.com',
 '{"free_shipping_threshold": 50, "standard_delivery_days": 5, "express_delivery_days": 2}',
 '{"days": 365, "conditions": ["Original packaging", "Assembly instructions"]}', 8.8),
 
('Wayfair', 'https://logo.clearbit.com/wayfair.co.uk', 'https://wayfair.co.uk',
 '{"free_shipping_threshold": 40, "standard_delivery_days": 7, "express_delivery_days": 3}',
 '{"days": 30, "conditions": ["Original packaging", "Return shipping fee applies"]}', 8.1),
 
('John Lewis', 'https://logo.clearbit.com/johnlewis.com', 'https://johnlewis.com',
 '{"free_shipping_threshold": 50, "standard_delivery_days": 3, "express_delivery_days": 1}',
 '{"days": 35, "conditions": ["Never knowingly undersold", "Original packaging"]}', 9.0);

-- Create a function to automatically create user subscription
CREATE OR REPLACE FUNCTION create_user_subscription()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO subscriptions (user_id, type, status)
    VALUES (NEW.id, 'free', 'active');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to create subscription when user is created
CREATE TRIGGER create_user_subscription_trigger
    AFTER INSERT ON users
    FOR EACH ROW
    EXECUTE FUNCTION create_user_subscription();

-- Add some sample data for development
-- This will be handled by seed scripts in production 