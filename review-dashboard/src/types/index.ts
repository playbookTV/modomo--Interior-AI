export interface Scene {
  scene_id: string
  houzz_id: string
  image_url: string
  room_type: string | null
  style_tags: string[]
  color_tags: string[]
  project_url: string | null
  status: 'scraped' | 'processing' | 'pending_review' | 'approved' | 'rejected'
  created_at: string
  objects: DetectedObject[]
}

export interface DetectedObject {
  object_id: string
  scene_id: string
  bbox: [number, number, number, number] // [x, y, width, height]
  mask_url?: string
  category: string
  confidence: number
  tags?: string[]
  matched_product_id?: string
  approved?: boolean | null
  created_at: string
}

export interface Product {
  product_id: string
  sku: string
  name: string
  category: string
  price_gbp: number
  brand: string
  material: string
  dimensions: {
    width?: number
    height?: number
    depth?: number
  }
  image_url: string
  product_url: string
  description: string
  retailer: string
  active: boolean
}

export interface ProductSimilarity {
  product_id: string
  product: Product
  similarity_score: number
}

export interface ScrapingJob {
  job_id: string
  job_type?: 'scenes' | 'products' | 'detection' | 'import' | 'export' | 'processing'
  status: 'pending' | 'running' | 'processing' | 'completed' | 'failed' | 'error'
  progress: number
  total: number
  processed: number
  total_items?: number  // backwards compatibility
  processed_items?: number  // backwards compatibility
  message: string
  created_at?: string
  updated_at?: string
  error_message?: string
  dataset?: string
  features?: string[]
}

export interface DatasetExportJob {
  export_id: string
  export_name?: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  split_ratios: {
    train: number
    val: number
    test: number
  }
  scene_count?: number
  object_count?: number
  manifest_urls?: {
    train: string
    val: string
    test: string
  }
  created_at: string
  completed_at?: string
}

export interface DatasetStats {
  total_scenes: number
  approved_scenes: number
  total_objects: number
  approved_objects: number
  unique_categories: number
  avg_confidence: number
  objects_with_products: number
}

export interface CategoryStats {
  category: string
  total_objects: number
  approved_objects: number
  avg_confidence: number
  matched_objects: number
}

export interface ReviewUpdate {
  object_id: string
  category?: string
  tags?: string[]
  approved?: boolean
  matched_product_id?: string
}

export const FURNITURE_TAXONOMY = {
  seating: ["sofa", "sectional", "armchair", "dining_chair", "stool", "bench"],
  tables: ["coffee_table", "side_table", "dining_table", "console_table", "desk"],
  storage: ["bookshelf", "cabinet", "dresser", "wardrobe"],
  lighting: ["pendant_light", "floor_lamp", "table_lamp", "wall_sconce"],
  soft_furnishings: ["rug", "curtains", "pillow", "blanket"],
  decor: ["wall_art", "mirror", "plant", "decorative_object"],
  bed_bath: ["bed_frame", "mattress", "headboard", "nightstand", "bathtub", "sink_vanity"]
} as const

export type FurnitureCategory = keyof typeof FURNITURE_TAXONOMY
export type FurnitureItem = typeof FURNITURE_TAXONOMY[FurnitureCategory][number]