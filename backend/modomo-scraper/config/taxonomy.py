"""
Modomo Furniture Taxonomy
Comprehensive furniture and decor categorization for object detection
"""

# Enhanced Configuration for better object detection
MODOMO_TAXONOMY = {
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


def get_all_categories() -> list:
    """Get a flat list of all furniture categories"""
    return [item for items in MODOMO_TAXONOMY.values() for item in items]


def get_category_group(category: str) -> str:
    """Get the group name for a specific category"""
    for group_name, items in MODOMO_TAXONOMY.items():
        if category in items:
            return group_name
    return "other"


def get_category_count() -> int:
    """Get total number of categories"""
    return len(get_all_categories())