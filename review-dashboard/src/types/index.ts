export interface Scene {
  scene_id: string
  houzz_id: string
  image_url: string
  image_r2_key?: string
  image_type?: 'scene' | 'object' | 'product' | 'hybrid'
  is_primary_object?: boolean
  primary_category?: string
  room_type: string | null
  style_tags: string[]
  color_tags: string[]
  project_url: string | null
  status: 'scraped' | 'processing' | 'pending_review' | 'approved' | 'rejected'
  created_at: string
  updated_at?: string
  objects: DetectedObject[]
  object_count?: number
  metadata?: {
    classification_confidence?: number
    classification_reason?: string
    detected_room_type?: string
    detected_styles?: string[]
    scores?: {
      object: number
      scene: number
      hybrid: number
      style: number
    }
    reclassification?: {
      job_id: string
      timestamp: string
      confidence: number
      reason: string
      previous_type: string
    }
  }
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
  metadata?: {
    colors?: {
      colors?: Array<{
        rgb: [number, number, number]
        hex: string
        name: string
        percentage: number
      }>
      dominant_color?: {
        rgb: [number, number, number]
        hex: string
        name: string
      }
      properties?: {
        brightness: number
        is_neutral: boolean
        color_temperature?: string
      }
    }
  }
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
  started_at?: string
  completed_at?: string
  error_message?: string
  dataset?: string
  features?: string[]
  parameters?: Record<string, any>
  duration_seconds?: number
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
  scenes_by_type?: {
    scene: number
    object: number
    hybrid: number
    product: number
  }
  room_types?: Array<{
    room_type: string
    count: number
  }>
  detected_styles?: Array<{
    style: string
    count: number
  }>
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
  // Primary Furniture Categories
  seating: ["sofa", "sectional", "armchair", "dining_chair", "stool", "bench", "loveseat", "recliner", "chaise_lounge", "bar_stool", "office_chair", "accent_chair", "ottoman", "pouffe"],
  
  tables: ["coffee_table", "side_table", "dining_table", "console_table", "desk", "nightstand", "end_table", "accent_table", "writing_desk", "computer_desk", "bar_table", "bistro_table", "nesting_tables", "dressing_table"],
  
  storage: ["bookshelf", "cabinet", "dresser", "wardrobe", "armoire", "chest_of_drawers", "credenza", "sideboard", "buffet", "china_cabinet", "display_cabinet", "tv_stand", "media_console", "shoe_cabinet", "pantry_cabinet"],
  
  bedroom: ["bed_frame", "mattress", "headboard", "footboard", "bed_base", "platform_bed", "bunk_bed", "daybed", "murphy_bed", "crib", "bassinet", "changing_table"],
  
  // Lighting & Electrical
  lighting: ["pendant_light", "floor_lamp", "table_lamp", "wall_sconce", "chandelier", "ceiling_light", "track_lighting", "recessed_light", "under_cabinet_light", "desk_lamp", "reading_light", "accent_lighting", "string_lights"],
  
  ceiling_fixtures: ["ceiling_fan", "smoke_detector", "air_vent", "skylight", "beam", "molding", "medallion"],
  
  // Kitchen & Appliances
  kitchen_cabinets: ["upper_cabinet", "lower_cabinet", "kitchen_island", "breakfast_bar", "pantry", "spice_rack", "wine_rack"],
  
  kitchen_appliances: ["refrigerator", "stove", "oven", "microwave", "dishwasher", "range_hood", "garbage_disposal", "coffee_maker", "toaster", "blender"],
  
  kitchen_fixtures: ["kitchen_sink", "faucet", "backsplash", "countertop", "kitchen_island_top"],
  
  // Bathroom & Fixtures
  bathroom_fixtures: ["toilet", "shower", "bathtub", "sink_vanity", "bathroom_sink", "shower_door", "shower_curtain", "medicine_cabinet", "towel_rack", "toilet_paper_holder"],
  
  bathroom_storage: ["linen_closet", "bathroom_cabinet", "vanity_cabinet", "over_toilet_storage"],
  
  // Textiles & Soft Furnishings
  window_treatments: ["curtains", "drapes", "blinds", "shades", "shutters", "valance", "cornice", "window_film"],
  
  soft_furnishings: ["rug", "carpet", "pillow", "cushion", "throw_pillow", "blanket", "throw", "bedding", "duvet", "comforter", "sheets", "pillowcase"],
  
  upholstery: ["sofa_cushions", "chair_cushions", "seat_cushions", "back_cushions"],
  
  // Decor & Accessories
  wall_decor: ["wall_art", "painting", "photograph", "poster", "wall_sculpture", "wall_clock", "decorative_plate", "wall_shelf", "floating_shelf"],
  
  decor_accessories: ["mirror", "vase", "candle", "sculpture", "decorative_bowl", "picture_frame", "clock", "lamp_shade", "decorative_object"],
  
  plants_planters: ["potted_plant", "hanging_plant", "planter", "flower_pot", "garden_planter", "herb_garden"],
  
  // Architectural Elements
  doors_windows: ["door", "window", "french_doors", "sliding_door", "bifold_door", "pocket_door", "window_frame", "door_frame"],
  
  architectural_features: ["fireplace", "mantle", "column", "pillar", "archway", "niche", "built_in_shelf", "wainscoting", "chair_rail"],
  
  flooring: ["hardwood_floor", "tile_floor", "carpet_floor", "laminate_floor", "vinyl_floor", "stone_floor"],
  
  wall_features: ["accent_wall", "brick_wall", "stone_wall", "wood_paneling", "wallpaper"],
  
  // Electronics & Technology
  entertainment: ["tv", "television", "stereo", "speakers", "gaming_console", "dvd_player", "sound_bar"],
  
  home_office: ["computer", "monitor", "printer", "desk_accessories", "filing_cabinet", "desk_organizer"],
  
  smart_home: ["smart_speaker", "security_camera", "thermostat", "smart_switch", "home_hub"],
  
  // Outdoor & Patio
  outdoor_furniture: ["patio_chair", "outdoor_table", "patio_umbrella", "outdoor_sofa", "deck_chair", "garden_bench", "outdoor_dining_set"],
  
  outdoor_decor: ["outdoor_plant", "garden_sculpture", "outdoor_lighting", "wind_chime", "bird_feeder"],
  
  // Specialty Items
  exercise_equipment: ["treadmill", "exercise_bike", "weights", "yoga_mat", "exercise_ball"],
  
  children_furniture: ["toy_chest", "kids_table", "kids_chair", "high_chair", "play_table", "toy_storage"],
  
  office_furniture: ["conference_table", "office_desk", "executive_chair", "meeting_chair", "whiteboard", "bulletin_board"],
  
  // Miscellaneous
  room_dividers: ["screen", "room_divider", "partition", "bookcase_divider"],
  
  seasonal_decor: ["christmas_tree", "holiday_decoration", "seasonal_pillow", "seasonal_wreath"],
  
  hardware_fixtures: ["door_handle", "cabinet_hardware", "light_switch", "outlet", "vent_cover"]
} as const

export type FurnitureCategory = keyof typeof FURNITURE_TAXONOMY
export type FurnitureItem = typeof FURNITURE_TAXONOMY[FurnitureCategory][number]

export interface ImageClassification {
  image_type: 'scene' | 'object' | 'product' | 'hybrid'
  is_primary_object: boolean
  primary_category?: string
  confidence: number
  reason: string
  metadata: {
    scores: {
      object: number
      scene: number
      hybrid: number
      style: number
    }
    detected_room_type?: string
    detected_styles: string[]
    keyword_matches: {
      object_matches: string[]
      scene_matches: string[]
    }
  }
}

export interface ClassificationTestResult {
  image_url: string
  caption?: string
  classification: ImageClassification
  status: 'success' | 'failed'
  error?: string
}