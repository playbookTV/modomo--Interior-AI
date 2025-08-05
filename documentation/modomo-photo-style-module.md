# Modomo: Photo-to-Style AI Module - Technical Specification

## 🎯 Module Overview

The **Photo-to-Style AI Module** is the core component of Modomo that transforms user-uploaded room photos into styled interior designs while maintaining room structure and providing shoppable product recommendations.

**Key Requirements:**
- **Processing Time**: <15 seconds average (mobile-optimized)
- **Quality Score**: >85% user satisfaction
- **Style Accuracy**: Match selected design aesthetic with >90% consistency
- **Room Preservation**: Maintain architectural elements (walls, windows, doors)
- **Product Integration**: Generate shoppable furniture recommendations for every suggested item

---

## 🧠 AI Architecture Stack

### **Recommended Model Pipeline: SD 1.5 + ControlNet (Budget-Optimized)**

Based on analysis, **Stable Diffusion 1.5 + ControlNet** is the optimal choice for Modomo:

```python
# Core AI Pipeline Architecture - SD 1.5 Optimized
class ModomoStyleTransferPipeline:
    def __init__(self):
        # Stage 1: Room Analysis
        self.depth_estimator = pipeline("depth-estimation", 
            model="Intel/dpt-large")
        self.segmentation_model = pipeline("image-segmentation",
            model="openmmlab/upernet-convnext-small") 
        self.object_detector = pipeline("object-detection",
            model="facebook/detr-resnet-50")
        
        # Stage 2: Style Transfer (SD 1.5 + Multiple ControlNets)
        self.sd15_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        
        # Multiple ControlNet models for different use cases
        self.controlnet_models = {
            'interior_segmentation': ControlNetModel.from_pretrained(
                "BertChristiaens/controlnet-seg-room"),  # Interior-specific
            'depth_control': ControlNetModel.from_pretrained(
                "lllyasviel/control_v11f1p_sd15_depth"),  # Room structure
            'mlsd_lines': ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_mlsd"),     # Interior lines
            'canny_edges': ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_canny")     # Edge preservation
        }
        
        # Stage 3: Product Matching
        self.clip_model = SentenceTransformer('clip-ViT-B-32')
        self.product_embeddings = self.load_product_database()
        
    async def transform_room(self, image_path: str, style: str) -> StyleResult:
        # Multi-ControlNet processing pipeline
        return await self.apply_multi_controlnet_transform(image_path, style)
```

### **Model Selection Analysis:**

#### **🏆 Primary Choice: Stable Diffusion 1.5 + ControlNet (RECOMMENDED)**
**Why SD 1.5 + ControlNet is Perfect for Modomo:**

**Cost Advantages:**
- **100% Free**: All models are open source with commercial rights
- **Low GPU Requirements**: Minimum 4GB GPU RAM, Recommended 8GB - works on consumer GPUs like GTX 1650 or RTX 3060
- **Processing Cost**: £0.03-0.05 per image vs £0.13 for FLUX
- **Mature Ecosystem**: Massive library of LoRAs, ControlNets, and fine-tuned models

**Technical Advantages:**
- **Proven Interior Design**: ControlNet with SD 1.5 specifically supports M-LSD (Mobile Line Segment Detection) for interior design and depth models for 3D composition
- **Multiple Control Methods**: Segmentation, depth, lines, edges - perfect for room structure preservation
- **Fast Processing**: SD 1.5 is lightweight with smaller parameters, making it manageable even for consumer hardware
- **Stable & Reliable**: 2+ years of production use, well-documented issues and solutions

**Interior Design Specific:**
- **BertChristiaens/controlnet-seg-room**: Custom trained on 130K interior images
- **M-LSD ControlNet**: Perfect for "extracting outlines with straight edges like interior designs, buildings" - exactly what rooms need
- **Depth Control**: Preserves room 3D structure while changing style
- **Multi-ControlNet**: Can combine segmentation + depth + lines for maximum control

#### **🥈 Alternative: FLUX.1-Kontext (If Budget Allows)**
**Advantages:**
- **Cutting-Edge Quality**: Latest 2025 technology
- **In-Context Editing**: Designed specifically for image modification
- **Better Results**: Higher quality, more photorealistic outputs

**Disadvantages:**
- **Higher Cost**: £0.13+ per image vs £0.05 for SD 1.5
- **Commercial License**: Additional licensing fees required
- **Less Mature**: Fewer ControlNet options, less community support
- **Higher GPU Requirements**: Needs more powerful hardware

#### **🥉 Rejected: SDXL + ControlNet**
While SDXL offers better quality, "the main ControlNet models are based on the SD 1.5 version" and "it's not possible to use the same model files directly" with newer versions. This creates compatibility issues and reduces available resources.

---

## 🏗️ Detailed Technical Implementation

### **Stage 1: Room Analysis & Understanding**

#### **1.1 Photo Quality Assessment**
```python
async def assess_photo_quality(image_path: str) -> QualityScore:
    """Real-time photo quality analysis for user feedback"""
    
    # Load and preprocess image
    image = load_image(image_path)
    
    # Multi-factor quality assessment
    quality_checks = {
        'lighting': assess_lighting_quality(image),      # 0-1 score
        'focus': assess_image_sharpness(image),          # 0-1 score  
        'composition': assess_room_composition(image),    # 0-1 score
        'resolution': check_minimum_resolution(image),    # boolean
        'furniture_visible': count_furniture_objects(image) # count
    }
    
    # Calculate composite quality score
    composite_score = (
        quality_checks['lighting'] * 0.3 +
        quality_checks['focus'] * 0.2 + 
        quality_checks['composition'] * 0.3 +
        (1.0 if quality_checks['resolution'] else 0.0) * 0.1 +
        min(quality_checks['furniture_visible'] / 3, 1.0) * 0.1
    )
    
    return QualityScore(
        overall=composite_score,
        recommendations=generate_improvement_tips(quality_checks),
        acceptable=composite_score >= 0.7
    )
```

#### **1.2 Room Dimension Detection**
```python
async def detect_room_dimensions(image_path: str) -> RoomDimensions:
    """Extract room dimensions using depth estimation and perspective geometry"""
    
    # Generate depth map
    depth_map = self.depth_estimator(image_path)['depth']
    
    # Detect room boundaries using line detection
    lines = detect_room_lines(image_path, method='hough')
    room_corners = find_room_corners(lines)
    
    # Calculate dimensions using perspective geometry
    dimensions = calculate_room_size(depth_map, room_corners)
    
    return RoomDimensions(
        length_meters=dimensions['length'],
        width_meters=dimensions['width'], 
        height_meters=dimensions['height'],
        area_sqm=dimensions['area'],
        confidence=dimensions['confidence']
    )
```

#### **1.3 Object Detection & Segmentation**
```python
async def analyze_room_objects(image_path: str) -> RoomAnalysis:
    """Comprehensive room content analysis"""
    
    # Object detection for furniture identification
    objects = self.object_detector(image_path)
    furniture_objects = filter_furniture_objects(objects)
    
    # Semantic segmentation for precise boundaries
    segmentation = self.segmentation_model(image_path)
    room_segments = process_interior_segments(segmentation)
    
    # Room type classification
    room_type = classify_room_type(furniture_objects, room_segments)
    
    return RoomAnalysis(
        room_type=room_type,  # living_room, bedroom, kitchen, etc.
        furniture_items=furniture_objects,
        architectural_elements=room_segments['architecture'],
        color_palette=extract_color_palette(image_path),
        lighting_conditions=analyze_lighting(image_path),
        style_hints=detect_existing_style(furniture_objects)
    )
```

### **Stage 2: Style Transfer & Generation**

#### **2.1 Style Prompt Engineering**
```python
def generate_style_prompt(room_analysis: RoomAnalysis, target_style: str) -> str:
    """Create optimized prompts for each interior design style"""
    
    style_templates = {
        'scandinavian': {
            'base': "Scandinavian interior design, light wood furniture, white walls, minimalist decor, cozy hygge atmosphere",
            'colors': "white, light gray, natural wood tones, soft pastels",
            'materials': "light oak, white paint, natural textiles, wool, linen"
        },
        'modern_minimalist': {
            'base': "Modern minimalist interior, clean lines, neutral colors, uncluttered space, contemporary furniture",
            'colors': "white, black, gray, occasional accent color",
            'materials': "steel, glass, polished concrete, sleek surfaces"
        },
        'bohemian': {
            'base': "Bohemian interior design, eclectic mix, rich textures, warm colors, vintage furniture, plants",
            'colors': "earth tones, jewel tones, warm oranges, deep reds",
            'materials': "wood, rattan, vintage textiles, brass, natural fibers"
        },
        'industrial': {
            'base': "Industrial interior design, exposed elements, raw materials, urban loft style, metal fixtures",
            'colors': "charcoal, black, rust, raw steel, weathered wood",
            'materials': "exposed brick, steel, concrete, reclaimed wood, iron"
        },
        'traditional': {
            'base': "Traditional interior design, classic furniture, elegant details, timeless style, formal arrangement",
            'colors': "warm neutrals, navy, burgundy, gold accents",
            'materials': "dark wood, leather, silk, wool, brass hardware"
        }
    }
    
    style_config = style_templates[target_style]
    room_context = f"{room_analysis.room_type} with {', '.join(room_analysis.furniture_items[:3])}"
    
    # Construct comprehensive prompt
    prompt = f"""
    {style_config['base']}, {room_context}, 
    professional interior photography, high quality, detailed,
    color palette: {style_config['colors']},
    materials: {style_config['materials']},
    well-lit, realistic proportions, no distortion
    """
    
    return prompt.strip()
```

#### **2.2 SD 1.5 + Multi-ControlNet Implementation**
```python
async def apply_sd15_multi_controlnet(
    image_path: str, 
    style_prompt: str, 
    room_analysis: RoomAnalysis
) -> StyleResult:
    """Advanced SD 1.5 + Multi-ControlNet for superior room control"""
    
    try:
        # Load input image
        input_image = load_image(image_path)
        
        # Generate multiple control maps
        control_maps = await self.generate_control_maps(input_image, room_analysis)
        
        # Select best ControlNet combination based on room type
        controlnet_config = self.select_optimal_controlnet_combination(
            room_analysis.room_type, room_analysis.complexity_score
        )
        
        # Multi-ControlNet pipeline
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet_config['models'],
            torch_dtype=torch.float16,
            safety_checker=None,  # Disable for speed
            requires_safety_checker=False
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()  # Memory optimization
        
        # Enhanced prompt with style-specific LoRA
        enhanced_prompt = self.enhance_prompt_with_lora(style_prompt, controlnet_config['style'])
        
        # Generate with multiple controls
        result = pipe(
            prompt=enhanced_prompt,
            image=input_image,
            control_image=control_maps,
            num_inference_steps=20,  # Optimal for SD 1.5
            guidance_scale=7.5,      # Good balance for interior design
            controlnet_conditioning_scale=controlnet_config['scales'],
            generator=torch.manual_seed(42)  # Reproducible results
        )
        
        styled_image = result.images[0]
        
        # Quality assessment
        quality_score = await self.assess_sd15_quality(styled_image, style_prompt)
        
        return StyleResult(
            styled_image=styled_image,
            confidence=quality_score,
            processing_time=12.0,  # SD 1.5 is fast
            model_used='sd15_multi_controlnet',
            control_methods=controlnet_config['methods']
        )
        
    except Exception as e:
        # Fallback to single ControlNet
        return await self.apply_single_controlnet_fallback(image_path, style_prompt, room_analysis)

def select_optimal_controlnet_combination(self, room_type: str, complexity: float) -> dict:
    """Dynamic ControlNet selection based on room characteristics"""
    
    combinations = {
        'living_room': {
            'models': [self.controlnet_models['interior_segmentation'], 
                      self.controlnet_models['depth_control']],
            'scales': [0.8, 0.6],  # Strong segmentation, moderate depth
            'methods': ['segmentation', 'depth'],
            'style': 'living_room_optimized'
        },
        'bedroom': {
            'models': [self.controlnet_models['interior_segmentation'],
                      self.controlnet_models['mlsd_lines']],
            'scales': [0.7, 0.5],  # Moderate control for intimate spaces
            'methods': ['segmentation', 'lines'],
            'style': 'bedroom_cozy'
        },
        'kitchen': {
            'models': [self.controlnet_models['depth_control'],
                      self.controlnet_models['canny_edges']],
            'scales': [0.8, 0.4],  # Strong depth, light edge control
            'methods': ['depth', 'edges'],
            'style': 'kitchen_functional'
        }
    }
    
    # Default to living room if room type not recognized
    return combinations.get(room_type, combinations['living_room'])

async def generate_control_maps(self, image: Image, room_analysis: RoomAnalysis) -> List[Image]:
    """Generate optimized control maps for interior design"""
    
    control_maps = []
    
    # 1. Segmentation map (furniture boundaries)
    segmentation = self.segmentation_model(image)
    seg_map = self.process_interior_segmentation(segmentation, room_analysis.furniture_items)
    control_maps.append(seg_map)
    
    # 2. Depth map (3D room structure)
    depth_map = self.depth_estimator(image)['depth']
    depth_image = self.process_depth_for_interior(depth_map)
    control_maps.append(depth_image)
    
    # 3. Line detection (architectural elements) - if needed
    if room_analysis.architectural_complexity > 0.6:
        line_map = self.detect_interior_lines(image, method='mlsd')
        control_maps.append(line_map)
    
    return control_maps

def enhance_prompt_with_lora(self, base_prompt: str, style_type: str) -> str:
    """Add style-specific LoRA and prompt enhancements"""
    
    lora_styles = {
        'living_room_optimized': '<lora:interior_living:0.8>',
        'bedroom_cozy': '<lora:cozy_bedroom:0.7>',
        'kitchen_functional': '<lora:modern_kitchen:0.8>'
    }
    
    style_enhancers = {
        'scandinavian': 'light wood, white walls, hygge, minimalist Nordic design',
        'modern_minimalist': 'clean lines, neutral palette, uncluttered space',
        'bohemian': 'warm textures, eclectic mix, plants, vintage elements',
        'industrial': 'exposed brick, metal fixtures, raw materials, urban loft',
        'traditional': 'classic furniture, elegant details, formal arrangement'
    }
    
    lora_tag = lora_styles.get(style_type, '')
    enhanced_prompt = f"{lora_tag} {base_prompt}, professional interior photography, 8k, high quality"
    
    return enhanced_prompt
```

#### **2.3 ControlNet Fallback Implementation**
```python
async def apply_controlnet_fallback(
    image_path: str,
    style_prompt: str, 
    room_analysis: RoomAnalysis
) -> StyleResult:
    """Fallback using interior-specific ControlNet"""
    
    # Generate segmentation map for control
    segmentation_map = self.segmentation_model(image_path)
    control_image = process_segmentation_for_controlnet(segmentation_map)
    
    # Apply ControlNet style transfer
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=self.controlnet_interior,
        torch_dtype=torch.float16
    )
    
    result = pipe(
        prompt=style_prompt,
        image=load_image(image_path),
        control_image=control_image,
        num_inference_steps=20,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.8  # Strong structural control
    )
    
    return StyleResult(
        styled_image=result.images[0],
        confidence=0.75,  # Default confidence for ControlNet
        processing_time=15.0,
        model_used='controlnet_interior'
    )
```

### **Stage 3: Product Matching & Shopping Integration**

#### **3.1 Furniture Object Extraction**
```python
async def extract_furniture_objects(
    original_image: str,
    styled_image: Image,
    room_analysis: RoomAnalysis
) -> List[FurnitureObject]:
    """Extract individual furniture items from styled image"""
    
    # Detect objects in the styled image
    styled_objects = self.object_detector(styled_image)
    
    # Segment individual furniture pieces
    segmentation = self.segmentation_model(styled_image)
    
    furniture_objects = []
    for obj in styled_objects:
        if obj['label'] in FURNITURE_CATEGORIES:
            # Extract object from image using segmentation mask
            object_mask = extract_object_mask(segmentation, obj['box'])
            object_image = apply_mask(styled_image, object_mask)
            
            # Generate product search query
            search_query = generate_product_query(obj, object_image)
            
            furniture_objects.append(FurnitureObject(
                category=obj['label'],
                confidence=obj['score'],
                bounding_box=obj['box'],
                extracted_image=object_image,
                search_query=search_query,
                estimated_price_range=estimate_price_range(obj['label'])
            ))
    
    return furniture_objects
```

#### **3.2 Multi-Retailer Product Matching**
```python
async def find_matching_products(
    furniture_object: FurnitureObject,
    user_preferences: dict
) -> List[ProductMatch]:
    """Find matching products across multiple retailers"""
    
    # Generate visual embedding for the furniture object
    visual_embedding = self.clip_model.encode(furniture_object.extracted_image)
    
    # Search across retailer databases
    search_tasks = []
    for retailer in self.retailers:
        search_tasks.append(
            retailer.search_similar_products(
                visual_embedding=visual_embedding,
                category=furniture_object.category,
                price_range=furniture_object.estimated_price_range,
                user_location=user_preferences.get('location', 'UK')
            )
        )
    
    # Execute searches in parallel
    all_results = await asyncio.gather(*search_tasks)
    
    # Combine and rank results
    combined_results = []
    for retailer_results in all_results:
        for product in retailer_results:
            similarity_score = calculate_visual_similarity(
                visual_embedding, product['embedding']
            )
            
            if similarity_score > 0.7:  # Minimum similarity threshold
                combined_results.append(ProductMatch(
                    product=product,
                    similarity_score=similarity_score,
                    price_score=calculate_price_score(product['price'], user_preferences),
                    availability_score=1.0 if product['in_stock'] else 0.0,
                    retailer=product['retailer']
                ))
    
    # Sort by composite score
    return sorted(combined_results, key=lambda x: x.composite_score, reverse=True)[:3]
```

---

## 📊 Performance Optimization

### **Mobile-First Architecture**

#### **Image Preprocessing Pipeline**
```python
class MobileOptimizer:
    def __init__(self):
        self.target_resolution = (1024, 768)  # Optimal for AI processing
        self.compression_quality = 85
        
    async def optimize_for_processing(self, image_path: str) -> str:
        """Optimize image for fast AI processing"""
        
        # Load and analyze image
        image = Image.open(image_path)
        original_size = image.size
        
        # Smart resize maintaining aspect ratio
        if max(original_size) > 1024:
            image = image.resize(self.target_resolution, Image.LANCZOS)
        
        # Apply mobile-specific optimizations
        if self.is_mobile_capture(image):
            image = self.correct_mobile_artifacts(image)
        
        # Compress for faster upload
        optimized_path = f"{image_path}_optimized.jpg"
        image.save(optimized_path, "JPEG", quality=self.compression_quality, optimize=True)
        
        return optimized_path
        
    def is_mobile_capture(self, image: Image) -> bool:
        """Detect if image was captured on mobile device"""
        # Check EXIF data for mobile camera signatures
        exif = image.getexif()
        if exif:
            make = exif.get(272, '').lower()  # Camera make
            model = exif.get(273, '').lower()  # Camera model
            return any(brand in make for brand in ['apple', 'samsung', 'google', 'huawei'])
        return False
        
    def correct_mobile_artifacts(self, image: Image) -> Image:
        """Apply corrections for common mobile photography issues"""
        # Auto-rotate based on EXIF orientation
        image = ImageOps.exif_transpose(image)
        
        # Enhance contrast for interior lighting
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Reduce noise common in mobile captures
        image_array = np.array(image)
        denoised = cv2.bilateralFilter(image_array, 9, 75, 75)
        image = Image.fromarray(denoised)
        
        return image
```

#### **Batch Processing Queue**
```python
class ProcessingQueue:
    def __init__(self, max_concurrent=3):
        self.queue = asyncio.Queue()
        self.max_concurrent = max_concurrent
        self.processing_jobs = {}
        
    async def submit_job(self, user_id: str, image_path: str, style: str) -> str:
        """Submit style transfer job to processing queue"""
        
        job_id = generate_job_id()
        job = ProcessingJob(
            id=job_id,
            user_id=user_id,
            image_path=image_path,
            style=style,
            status='queued',
            created_at=datetime.now()
        )
        
        await self.queue.put(job)
        self.processing_jobs[job_id] = job
        
        # Start processing if capacity available
        if len([j for j in self.processing_jobs.values() if j.status == 'processing']) < self.max_concurrent:
            asyncio.create_task(self.process_next_job())
        
        return job_id
        
    async def get_job_status(self, job_id: str) -> JobStatus:
        """Get current status of processing job"""
        job = self.processing_jobs.get(job_id)
        if not job:
            return JobStatus(status='not_found')
            
        return JobStatus(
            status=job.status,
            progress=job.progress,
            estimated_time_remaining=job.estimated_time_remaining,
            result=job.result if job.status == 'completed' else None
        )
```

---

## 💰 Cost Analysis & Budget Optimization

### **Processing Cost Breakdown (SD 1.5 Optimized)**
```python
# Cost estimates per image processing - SD 1.5 Focus
COST_BREAKDOWN = {
    'sd15_multi_controlnet': {
        'model_license': 0.00,      # 100% Free & Open Source
        'gpu_compute': 0.03,        # £0.03 per image on Modal (lightweight)
        'storage_costs': 0.005,     # £0.005 for temporary image storage
        'total_per_image': 0.035
    },
    'sd15_single_controlnet': {
        'model_license': 0.00,      # Free
        'gpu_compute': 0.02,        # £0.02 per image (faster processing)
        'storage_costs': 0.005,     # £0.005 for storage
        'total_per_image': 0.025
    },
    'flux_kontext': {
        'model_license': 0.05,      # £0.05 per image (estimated)
        'gpu_compute': 0.08,        # £0.08 per image on Modal
        'total_per_image': 0.13
    },
    'product_matching': {
        'clip_inference': 0.01,     # £0.01 per image
        'retailer_api_calls': 0.02, # £0.02 per image (multiple retailers)
        'total_per_image': 0.03
    }
}

# Total cost per successful transformation
ESTIMATED_COST_SD15_MULTI = 0.065      # £0.065 using SD 1.5 Multi-ControlNet
ESTIMATED_COST_SD15_BASIC = 0.055      # £0.055 using SD 1.5 Single-ControlNet  
ESTIMATED_COST_FLUX = 0.16             # £0.16 using FLUX

# Revenue model (unchanged)
AVERAGE_AFFILIATE_COMMISSION = 15.00   # £15 per conversion
TARGET_CONVERSION_RATE = 0.08         # 8% of users make purchases
EXPECTED_REVENUE_PER_USER = 1.20      # £15 × 8% = £1.20

# Unit economics with SD 1.5
PROFIT_PER_USER_SD15 = 1.135          # £1.20 - £0.065 = £1.135 (94.6% profit margin!)
PROFIT_PER_USER_FLUX = 1.04           # £1.20 - £0.16 = £1.04 (86.7% profit margin)

# Scale economics
MONTHLY_PROCESSING_BUDGET = 500        # £500/month for AI processing
MAX_IMAGES_SD15 = 7692                 # 500 / 0.065 = 7,692 images/month
MAX_IMAGES_FLUX = 3125                 # 500 / 0.16 = 3,125 images/month

# Break-even analysis
USERS_TO_BREAK_EVEN_SD15 = 441        # £500 / £1.135 = 441 converting users
USERS_TO_BREAK_EVEN_FLUX = 481        # £500 / £1.04 = 481 converting users
```

### **Tiered Processing Strategy (Cost-Optimized)**
```python
class AdaptiveProcessing:
    def __init__(self):
        self.usage_thresholds = {
            'free_tier': {
                'daily_limit': 3, 
                'model': 'sd15_single_controlnet',
                'cost_per_user': 0.055
            },
            'premium': {
                'daily_limit': 20, 
                'model': 'sd15_multi_controlnet',
                'cost_per_user': 0.065  
            },
            'pro': {
                'daily_limit': 100, 
                'model': 'flux_kontext',  # Premium users get best quality
                'cost_per_user': 0.16
            }
        }
        
    def select_processing_model(self, user_tier: str, daily_usage: int, quality_required: str) -> str:
        """Intelligent model selection balancing cost and quality"""
        
        tier_config = self.usage_thresholds[user_tier]
        
        if daily_usage >= tier_config['daily_limit']:
            return 'rate_limited'
        
        # Free tier: Always use basic SD 1.5
        if user_tier == 'free_tier':
            return 'sd15_single_controlnet'
            
        # Premium: Multi-ControlNet for complex rooms
        elif user_tier == 'premium':
            if quality_required == 'high_complexity':
                return 'sd15_multi_controlnet'
            else:
                return 'sd15_single_controlnet'  # Save costs on simple rooms
                
        # Pro: Best available model
        else:
            return 'flux_kontext'

# Budget allocation for 16-week development
DEVELOPMENT_AI_BUDGET = {
    'weeks_1_8': {
        'sd15_development': 200,      # £200 for model setup and testing
        'test_processing': 300,       # £300 for development testing (4600 test images)
        'total': 500
    },
    'weeks_9_16': {
        'beta_testing': 400,          # £400 for beta user processing (6000+ images)
        'optimization': 200,          # £200 for A/B testing different approaches  
        'launch_buffer': 400,         # £400 buffer for launch volume
        'total': 1000
    },
    'total_16_weeks': 1500           # £1,500 total AI processing budget
}
```

### **Scaling Strategy**
```python
class AdaptiveProcessing:
    def __init__(self):
        self.usage_thresholds = {
            'free_tier': {'daily_limit': 3, 'model': 'controlnet'},
            'premium': {'daily_limit': 50, 'model': 'flux_kontext'},
            'pro': {'daily_limit': 200, 'model': 'flux_kontext'}
        }
        
    def select_processing_model(self, user_tier: str, daily_usage: int) -> str:
        """Dynamic model selection based on user tier and usage"""
        
        tier_config = self.usage_thresholds[user_tier]
        
        if daily_usage >= tier_config['daily_limit']:
            return 'rate_limited'
        elif user_tier == 'free_tier':
            return 'controlnet_interior'  # Lower cost for free users
        else:
            return 'flux_kontext'         # Premium quality for paid users
```

---

## 🎯 Quality Assurance & Validation

### **Automated Quality Checking**
```python
class QualityValidator:
    def __init__(self):
        self.min_confidence_score = 0.75
        self.similarity_threshold = 0.6
        
    async def validate_style_result(
        self, 
        original_image: str,
        styled_result: StyleResult,
        target_style: str
    ) -> ValidationResult:
        """Comprehensive quality validation"""
        
        validations = {
            'structural_integrity': await self.check_room_structure_preserved(
                original_image, styled_result.styled_image),
            'style_consistency': await self.validate_style_application(
                styled_result.styled_image, target_style),
            'visual_quality': await self.assess_visual_quality(
                styled_result.styled_image),
            'object_coherence': await self.check_object_coherence(
                styled_result.styled_image)
        }
        
        overall_score = sum(validations.values()) / len(validations)
        
        return ValidationResult(
            passed=overall_score >= self.min_confidence_score,
            overall_score=overall_score,
            individual_scores=validations,
            retry_recommended=overall_score < 0.6
        )
```

---

## 🚀 Integration Points

### **React Native Integration**
```typescript
// React Native service integration
class ModomoAIService {
  private baseURL = 'https://api.modomo.com/v1'
  
  async processRoomPhoto(
    imageUri: string,
    style: string,
    preferences: UserPreferences
  ): Promise<ProcessingResult> {
    
    // Optimize image for processing
    const optimizedImage = await this.optimizeImage(imageUri)
    
    // Submit to processing queue
    const jobId = await this.submitProcessingJob({
      image: optimizedImage,
      style: style,
      preferences: preferences
    })
    
    // Poll for results with progress updates
    return this.pollForResults(jobId, (progress) => {
      // Real-time progress updates to UI
      this.onProgressUpdate?.(progress)
    })
  }
  
  private async pollForResults(
    jobId: string, 
    onProgress: (progress: number) => void
  ): Promise<ProcessingResult> {
    
    while (true) {
      const status = await fetch(`${this.baseURL}/jobs/${jobId}/status`)
      const statusData = await status.json()
      
      onProgress(statusData.progress)
      
      if (statusData.status === 'completed') {
        return statusData.result
      } else if (statusData.status === 'failed') {
        throw new Error(statusData.error)
      }
      
      // Wait 2 seconds before next poll
      await new Promise(resolve => setTimeout(resolve, 2000))
    }
  }
}
```

This comprehensive technical specification provides Modomo with a production-ready AI module that combines cutting-edge technology (FLUX.1-Kontext) with proven fallbacks (ControlNet), optimized for mobile performance and commercial viability within the £5,000 budget.