"""
Comprehensive taxonomy from main_full.py - Enhanced furniture categories
"""

# Enhanced Configuration for better object detection (from main_full.py)
COMPREHENSIVE_TAXONOMY = {
    # Primary Furniture Categories
    "seating": ["sofa", "sectional", "armchair", "dining_chair", "stool", "bench", "loveseat", "recliner", "chaise_lounge", "bar_stool", "office_chair", "accent_chair", "ottoman", "pouffe"],
    
    "tables": ["coffee_table", "side_table", "dining_table", "console_table", "desk", "nightstand", "end_table", "accent_table", "writing_desk", "computer_desk", "bar_table", "bistro_table", "nesting_tables", "dressing_table"],
    
    "storage": ["bookshelf", "cabinet", "dresser", "wardrobe", "armoire", "chest_of_drawers", "credenza", "sideboard", "buffet", "china_cabinet", "display_cabinet", "tv_stand", "media_console", "shoe_cabinet", "pantry_cabinet"],
    
    "bedroom": ["bed_frame", "mattress", "headboard", "footboard", "bed_base", "platform_bed", "bunk_bed", "daybed", "murphy_bed", "crib", "bassinet", "changing_table"],
    
    # Lighting & Electrical
    "lighting": ["pendant_light", "floor_lamp", "table_lamp", "wall_sconce", "chandelier", "ceiling_light", "track_lighting", "recessed_light", "under_cabinet_light", "desk_lamp", "reading_light", "accent_lighting", "string_lights"],
    
    "ceiling_fixtures": ["ceiling_fan", "smoke_detector", "air_vent", "skylight", "beam", "molding", "medallion"],
    
    # Kitchen & Appliances
    "kitchen_cabinets": ["upper_cabinet", "lower_cabinet", "kitchen_island", "breakfast_bar", "pantry", "spice_rack", "wine_rack"],
    
    "kitchen_appliances": ["refrigerator", "stove", "oven", "microwave", "dishwasher", "range_hood", "garbage_disposal", "coffee_maker", "toaster", "blender"],
    
    "kitchen_fixtures": ["kitchen_sink", "faucet", "backsplash", "countertop", "kitchen_island_top"],
    
    # Bathroom & Fixtures
    "bathroom_fixtures": ["toilet", "shower", "bathtub", "sink_vanity", "bathroom_sink", "shower_door", "shower_curtain", "medicine_cabinet", "towel_rack", "toilet_paper_holder"],
    
    "bathroom_storage": ["linen_closet", "bathroom_cabinet", "vanity_cabinet", "over_toilet_storage"],
    
    # Textiles & Soft Furnishings
    "window_treatments": ["curtains", "drapes", "blinds", "shades", "shutters", "valance", "cornice", "window_film"],
    
    "soft_furnishings": ["rug", "carpet", "pillow", "cushion", "throw_pillow", "blanket", "throw", "bedding", "duvet", "comforter", "sheets", "pillowcase"],
    
    "upholstery": ["sofa_cushions", "chair_cushions", "seat_cushions", "back_cushions"],
    
    # Decor & Accessories
    "wall_decor": ["wall_art", "painting", "photograph", "poster", "wall_sculpture", "wall_clock", "decorative_plate", "wall_shelf", "floating_shelf"],
    
    "decor_accessories": ["mirror", "plant", "vase", "candle", "sculpture", "decorative_bowl", "picture_frame", "clock", "lamp_shade", "decorative_object"],
    
    "plants_planters": ["potted_plant", "hanging_plant", "planter", "flower_pot", "garden_planter", "herb_garden"],
    
    # Architectural Elements
    "doors_windows": ["door", "window", "french_doors", "sliding_door", "bifold_door", "pocket_door", "window_frame", "door_frame"],
    
    "architectural_features": ["fireplace", "mantle", "column", "pillar", "archway", "niche", "built_in_shelf", "wainscoting", "chair_rail"],
    
    "flooring": ["hardwood_floor", "tile_floor", "carpet_floor", "laminate_floor", "vinyl_floor", "stone_floor"],
    
    "wall_features": ["accent_wall", "brick_wall", "stone_wall", "wood_paneling", "wallpaper"],
    
    # Electronics & Technology
    "entertainment": ["tv", "television", "stereo", "speakers", "gaming_console", "dvd_player", "sound_bar"],
    
    "home_office": ["computer", "monitor", "printer", "desk_accessories", "filing_cabinet", "desk_organizer"],
    
    "smart_home": ["smart_speaker", "security_camera", "thermostat", "smart_switch", "home_hub"],
    
    # Outdoor & Patio
    "outdoor_furniture": ["patio_chair", "outdoor_table", "patio_umbrella", "outdoor_sofa", "deck_chair", "garden_bench", "outdoor_dining_set"],
    
    "outdoor_decor": ["outdoor_plant", "garden_sculpture", "outdoor_lighting", "wind_chime", "bird_feeder"],
    
    # Specialty Items
    "exercise_equipment": ["treadmill", "exercise_bike", "weights", "yoga_mat", "exercise_ball"],
    
    "children_furniture": ["toy_chest", "kids_table", "kids_chair", "high_chair", "play_table", "toy_storage"],
    
    "office_furniture": ["conference_table", "office_desk", "executive_chair", "meeting_chair", "whiteboard", "bulletin_board"],
    
    # Miscellaneous
    "room_dividers": ["screen", "room_divider", "partition", "bookcase_divider"],
    
    "seasonal_decor": ["christmas_tree", "holiday_decoration", "seasonal_pillow", "seasonal_wreath"],
    
    "hardware_fixtures": ["door_handle", "cabinet_hardware", "light_switch", "outlet", "vent_cover"]
}


def get_comprehensive_keywords():
    """
    Comprehensive keyword system for robust image classification.
    Covers furniture, decor, room types, styles, and contextual indicators.
    """
    return {
        # OBJECT-ONLY INDICATORS (Single furniture/decor pieces)
        "object": [
            # === SEATING ===
            "sofa", "couch", "sectional", "loveseat", "settee", "chesterfield", "daybed",
            "chair", "armchair", "accent chair", "lounge chair", "dining chair", "desk chair",
            "office chair", "swivel chair", "recliner", "wingback", "bergere", "club chair",
            "stool", "bar stool", "counter stool", "ottoman", "pouf", "footstool",
            "bench", "storage bench", "entryway bench", "window bench", "piano bench",
            
            # === TABLES ===
            "table", "coffee table", "cocktail table", "end table", "side table", "accent table",
            "dining table", "kitchen table", "breakfast table", "console table", "entry table",
            "desk", "writing desk", "computer desk", "standing desk", "secretary desk",
            "nightstand", "bedside table", "night table", "bedstand",
            "nesting tables", "tv stand", "media console", "entertainment center",
            
            # === STORAGE ===
            "bookshelf", "bookcase", "shelf", "shelving unit", "etagere", "ladder shelf",
            "cabinet", "storage cabinet", "display cabinet", "china cabinet", "curio cabinet",
            "dresser", "chest of drawers", "tall dresser", "low dresser", "bachelor chest",
            "wardrobe", "armoire", "closet", "credenza", "sideboard", "buffet", "hutch",
            "filing cabinet", "storage unit", "modular storage", "cube organizer",
            
            # === LIGHTING ===
            "lamp", "table lamp", "desk lamp", "task lamp", "reading lamp", "accent lamp",
            "floor lamp", "torchiere", "arc lamp", "tripod lamp", "tree lamp",
            "pendant light", "hanging light", "chandelier", "ceiling light", "flush mount",
            "wall sconce", "wall light", "vanity light", "picture light", "under cabinet light",
            "track lighting", "recessed light", "can light", "spotlight", "downlight",
            
            # === BEDROOM ===
            "bed", "bed frame", "platform bed", "sleigh bed", "canopy bed", "four poster",
            "headboard", "footboard", "mattress", "box spring", "bedding", "pillows",
            "comforter", "duvet", "blanket", "throw", "bed skirt", "mattress pad",
            
            # === BATHROOM ===
            "bathtub", "tub", "freestanding tub", "clawfoot tub", "soaking tub", "jacuzzi",
            "shower", "walk-in shower", "shower stall", "shower door", "shower curtain",
            "vanity", "bathroom vanity", "sink vanity", "double vanity", "floating vanity",
            "toilet", "water closet", "bidet", "pedestal sink", "vessel sink",
            
            # === DECOR & ACCESSORIES ===
            "mirror", "wall mirror", "floor mirror", "vanity mirror", "decorative mirror",
            "artwork", "wall art", "painting", "print", "poster", "canvas", "framed art",
            "sculpture", "statue", "figurine", "decorative object", "vase", "bowl",
            "plant", "houseplant", "planter", "pot", "artificial plant", "tree", "fern",
            "rug", "area rug", "runner", "carpet", "mat", "doormat", "bath mat",
            "curtains", "drapes", "blinds", "shades", "window treatments", "valance",
            "pillow", "throw pillow", "accent pillow", "cushion", "bolster",
            "clock", "wall clock", "mantel clock", "desk clock", "alarm clock",
            
            # === PRODUCT/CATALOG INDICATORS ===
            "product", "item", "piece", "furniture piece", "accent piece",
            "single", "individual", "standalone", "isolated", "solo",
            "catalog", "listing", "for sale", "available", "buy now", "purchase",
            "studio", "white background", "neutral background", "clean background",
            "product photo", "catalog image", "stock photo", "commercial photo",
            "furniture store", "showroom", "retail", "brand new", "unused",
        ],
        
        # SCENE INDICATORS (Complete rooms/environments)
        "scene": [
            # === ROOM TYPES ===
            "living room", "family room", "great room", "sitting room", "lounge",
            "bedroom", "master bedroom", "guest bedroom", "kids bedroom", "nursery",
            "kitchen", "galley kitchen", "eat-in kitchen", "chef's kitchen", "kitchenette",
            "dining room", "breakfast nook", "dining area", "formal dining", "casual dining",
            "bathroom", "master bath", "guest bath", "powder room", "half bath", "en suite",
            "home office", "study", "den", "library", "workspace", "craft room",
            "entryway", "foyer", "mudroom", "hallway", "corridor", "landing",
            "basement", "finished basement", "recreation room", "game room", "media room",
            "attic", "loft", "studio apartment", "open concept", "great room",
            "laundry room", "utility room", "pantry", "walk-in closet", "dressing room",
            "porch", "patio", "deck", "balcony", "sunroom", "conservatory",
            "garage", "workshop", "shed", "outdoor kitchen", "pool house",
            
            # === ARCHITECTURAL/DESIGN TERMS ===
            "interior", "interior design", "room design", "home decor", "decorating",
            "furnished", "staged", "model home", "show home", "display home",
            "renovation", "remodel", "makeover", "before and after", "transformation",
            "layout", "floor plan", "room layout", "furniture arrangement",
            "color scheme", "design scheme", "decorating style", "design aesthetic",
            "coordinated", "matching", "cohesive", "pulled together", "designed",
            
            # === STYLE DESCRIPTORS ===
            "modern", "contemporary", "traditional", "transitional", "rustic",
            "industrial", "scandinavian", "minimalist", "maximalist", "eclectic",
            "bohemian", "boho", "farmhouse", "country", "coastal", "nautical",
            "mediterranean", "southwestern", "art deco", "mid-century", "vintage",
            "shabby chic", "french country", "english country", "colonial", "craftsman",
            
            # === CONTEXTUAL SCENE INDICATORS ===
            "room", "space", "area", "corner", "nook", "alcove", "zone",
            "complete", "furnished", "decorated", "designed", "styled", "arranged",
            "entire", "whole", "full", "comprehensive", "total", "overall",
            "environment", "setting", "atmosphere", "ambiance", "mood", "feel",
            "lifestyle", "living", "home", "house", "residence", "dwelling",
            "real home", "actual home", "lived-in", "occupied", "inhabited",
            "daily life", "everyday", "functional", "practical", "usable",
            
            # === MULTIPLE ITEM INDICATORS ===
            "furniture set", "room set", "collection", "ensemble", "grouping",
            "multiple", "several", "various", "many", "numerous", "collection of",
            "arrangement", "composition", "vignette", "display", "setup",
            "coordinated pieces", "matching set", "furniture grouping",
        ]
    }