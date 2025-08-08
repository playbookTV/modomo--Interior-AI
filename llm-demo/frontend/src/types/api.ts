export interface DetectedObject {
  object_type: string;
  confidence: number;
  bounding_box: number[];
  description: string;
}

export interface ProductPrice {
  retailer: string;
  price: number;
  currency: string;
  url: string;
  availability: string;
  shipping?: string;
}

export interface SuggestedProduct {
  product_id: string;
  name: string;
  category: string;
  description: string;
  coordinates: number[];
  prices: ProductPrice[];
  image_url: string;
  confidence: number;
}

export interface StyleTransformation {
  style_name: string;
  before_image_url: string;
  after_image_url: string;
  detected_objects: DetectedObject[];
  suggested_products: SuggestedProduct[];
  total_estimated_cost: number;
  savings_amount: number;
}

export interface RoomMakeoverRequest {
  photo_url: string;
  photo_id: string;
  style_preference: string;
  budget_range?: 'low' | 'medium' | 'high';
  user_id?: string;
}

export interface EnhancedMakeoverRequest extends RoomMakeoverRequest {
  use_multi_controlnet: boolean;
  quality_level: 'standard' | 'high' | 'premium';
  strength: number;
  guidance_scale: number;
  num_inference_steps: number;
}

export interface RoomMakeoverResponse {
  makeover_id: string;
  photo_id: string;
  status: string;
  transformation: StyleTransformation | null;
  processing_time_ms: number;
  created_at: string;
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  version: string;
  components: Record<string, {
    status: string;
    details?: any;
    error?: string;
  }>;
}

export interface StyleComparisonRequest {
  image_data: string;
  styles: string[];
  include_original: boolean;
}

export interface StyleComparisonResult {
  comparison_id: string;
  original_included: boolean;
  styles_processed: number;
  results: Record<string, {
    style: string;
    image_data?: string;
    suggested_products?: Array<{
      name: string;
      category: string;
      description: string;
      confidence: number;
      price_range: string;
    }>;
    detected_objects?: Array<{
      object_type: string;
      confidence: number;
      description: string;
    }>;
    error?: string;
  }>;
  generation_timestamp: string;
}