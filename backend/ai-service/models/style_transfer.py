import torch
import gc
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionInpaintPipeline
from diffusers.utils import load_image
import cv2
import structlog
from controlnet_aux import CannyDetector, OpenposeDetector
import io
import base64
import contextlib

logger = structlog.get_logger()

class StyleTransferModel:
    """
    Real style transfer using Stable Diffusion + ControlNet for room makeovers
    """
    
    # Enhanced style prompts with more detailed descriptions
    STYLE_PROMPTS = {
        'modern': {
            'prompt': 'modern minimalist interior design, clean lines, neutral colors, contemporary furniture, sleek aesthetic, bright lighting, open space, uncluttered, geometric shapes',
            'negative': 'cluttered, ornate, traditional, dark, cramped, outdated, busy patterns, vintage furniture',
            'colors': 'white, black, gray, occasional accent color',
            'materials': 'steel, glass, polished concrete, sleek surfaces, chrome',
            'lora_weight': 0.8
        },
        'scandinavian': {
            'prompt': 'scandinavian interior design, light wood, white walls, cozy textiles, natural materials, hygge aesthetic, minimalist furniture, nordic style, bright and airy',
            'negative': 'dark colors, heavy furniture, ornate details, cluttered, industrial, stark',
            'colors': 'white, light gray, natural wood tones, soft pastels',
            'materials': 'light oak, white paint, natural textiles, wool, linen',
            'lora_weight': 0.7
        },
        'industrial': {
            'prompt': 'industrial interior design, exposed brick, metal fixtures, concrete floors, dark colors, urban loft aesthetic, raw materials, Edison bulbs, steel beams',
            'negative': 'soft colors, delicate furniture, ornate decorations, pastel colors, floral patterns',
            'colors': 'charcoal, black, rust, raw steel, weathered wood',
            'materials': 'exposed brick, steel, concrete, reclaimed wood, iron',
            'lora_weight': 0.8
        },
        'bohemian': {
            'prompt': 'bohemian interior design, colorful textiles, plants, eclectic furniture, warm lighting, artistic decorations, cozy atmosphere, vintage rugs, layered textures',
            'negative': 'minimalist, sterile, monochrome, corporate, stark, empty walls',
            'colors': 'earth tones, jewel tones, warm oranges, deep reds',
            'materials': 'wood, rattan, vintage textiles, brass, natural fibers',
            'lora_weight': 0.7
        },
        'traditional': {
            'prompt': 'traditional interior design, classic furniture, warm colors, elegant details, comfortable seating, timeless aesthetic, formal arrangement, rich fabrics',
            'negative': 'ultra modern, stark, industrial, minimalist, concrete, steel',
            'colors': 'warm neutrals, navy, burgundy, gold accents',
            'materials': 'dark wood, leather, silk, wool, brass hardware',
            'lora_weight': 0.6
        }
    }
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        self.model_id = model_id
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.controlnet_pipeline = None
        self.multi_controlnet_pipeline = None
        self.inpaint_pipeline = None
        self.canny_detector = None
        self.depth_detector = None
        self.mlsd_detector = None
        self._models_loaded = False
        
        # ControlNet models dictionary
        self.controlnet_models = {}
        self.available_controlnets = [
            'canny',
            'depth',
            'mlsd',
            'interior_segmentation'
        ]
        
        logger.info(f"Enhanced style transfer will use device: {self.device}")
        
    def _clear_gpu_memory(self):
        """Clear GPU memory and run garbage collection"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
    @contextlib.contextmanager
    def _gpu_memory_context(self):
        """Context manager for GPU memory management during inference"""
        try:
            # Clear memory before inference
            self._clear_gpu_memory()
            yield
        finally:
            # Clear memory after inference
            self._clear_gpu_memory()
            
    def unload_models(self):
        """Unload models from memory to free up GPU/CPU resources"""
        try:
            if self.controlnet_pipeline is not None:
                # Move pipeline to CPU first to free GPU memory
                if hasattr(self.controlnet_pipeline, 'to'):
                    self.controlnet_pipeline.to('cpu')
                del self.controlnet_pipeline
                self.controlnet_pipeline = None
                
            if self.inpaint_pipeline is not None:
                if hasattr(self.inpaint_pipeline, 'to'):
                    self.inpaint_pipeline.to('cpu')
                del self.inpaint_pipeline
                self.inpaint_pipeline = None
                
            if self.canny_detector is not None:
                del self.canny_detector
                self.canny_detector = None
                
            self._models_loaded = False
            self._clear_gpu_memory()
            logger.info("Style transfer models unloaded successfully")
            
        except Exception as e:
            logger.error(f"Error unloading models: {e}")
            
    def __del__(self):
        """Cleanup on object destruction"""
        self.unload_models()
        
    async def load_models(self, load_multi_controlnet: bool = True):
        """Load Stable Diffusion and ControlNet models with multi-ControlNet support"""
        if self._models_loaded:
            logger.info("Style transfer models already loaded")
            return
            
        try:
            logger.info("Loading enhanced ControlNet and Stable Diffusion models...")
            
            # Clear any existing memory first
            self._clear_gpu_memory()
            
            # Load individual ControlNet models
            await self._load_controlnet_models()
            
            # Load single ControlNet pipeline (fallback)
            canny_controlnet = self.controlnet_models.get('canny')
            if canny_controlnet:
                self.controlnet_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                    self.model_id,
                    controlnet=canny_controlnet,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=True
                )
            
            # Load multi-ControlNet pipeline if requested and supported
            if load_multi_controlnet and len(self.controlnet_models) > 1:
                try:
                    await self._load_multi_controlnet_pipeline()
                except Exception as e:
                    logger.warning(f"Failed to load multi-ControlNet pipeline: {e}")
                    logger.info("Continuing with single ControlNet support")
            
            # Load inpainting pipeline for product additions
            self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True
            )
            
            # Move pipelines to device
            if self.controlnet_pipeline:
                self.controlnet_pipeline = self.controlnet_pipeline.to(self.device)
            if self.multi_controlnet_pipeline:
                self.multi_controlnet_pipeline = self.multi_controlnet_pipeline.to(self.device)
            self.inpaint_pipeline = self.inpaint_pipeline.to(self.device)
            
            # Load preprocessing detectors
            await self._load_preprocessing_detectors()
            
            # Enable optimizations
            await self._enable_optimizations()
                    
            self._models_loaded = True
            logger.info(f"Enhanced style transfer models loaded successfully "
                       f"({len(self.controlnet_models)} ControlNets available)")
            
        except Exception as e:
            logger.error(f"Failed to load style transfer models: {e}")
            self.unload_models()  # Clean up on failure
            raise
    
    async def _load_controlnet_models(self):
        """Load individual ControlNet models for flexible use"""
        controlnet_configs = {
            'canny': "lllyasviel/sd-controlnet-canny",
            'depth': "lllyasviel/sd-controlnet-depth", 
            'mlsd': "lllyasviel/sd-controlnet-mlsd",
            # Note: interior_segmentation would need a custom model
        }
        
        for name, model_path in controlnet_configs.items():
            try:
                logger.info(f"Loading ControlNet: {name}")
                controlnet = ControlNetModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    use_safetensors=True
                )
                self.controlnet_models[name] = controlnet
                logger.info(f"✅ ControlNet {name} loaded successfully")
                
            except Exception as e:
                logger.warning(f"Failed to load ControlNet {name}: {e}")
                # Continue loading other models
                continue
    
    async def _load_multi_controlnet_pipeline(self):
        """Load multi-ControlNet pipeline for advanced control"""
        try:
            from diffusers import MultiControlNetModel
            
            # Create MultiControlNet from available models
            available_models = list(self.controlnet_models.values())
            if len(available_models) < 2:
                logger.warning("Not enough ControlNet models for multi-ControlNet pipeline")
                return
            
            multi_controlnet = MultiControlNetModel(available_models[:3])  # Limit to 3 for memory
            
            self.multi_controlnet_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                self.model_id,
                controlnet=multi_controlnet,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True
            )
            
            logger.info("✅ Multi-ControlNet pipeline loaded successfully")
            
        except ImportError:
            logger.warning("MultiControlNetModel not available, using single ControlNet")
        except Exception as e:
            logger.warning(f"Failed to create multi-ControlNet pipeline: {e}")
    
    async def _load_preprocessing_detectors(self):
        """Load preprocessing detectors for control image generation"""
        try:
            from controlnet_aux import CannyDetector, MidasDetector, MLSDdetector
            
            self.canny_detector = CannyDetector()
            logger.info("✅ Canny detector loaded")
            
            try:
                self.depth_detector = MidasDetector.from_pretrained('valhalla/t2iadapter-aux-models')
                logger.info("✅ Depth detector loaded")
            except Exception as e:
                logger.warning(f"Could not load depth detector: {e}")
            
            try:
                self.mlsd_detector = MLSDdetector.from_pretrained('valhalla/t2iadapter-aux-models')
                logger.info("✅ MLSD detector loaded")
            except Exception as e:
                logger.warning(f"Could not load MLSD detector: {e}")
                
        except ImportError as e:
            logger.warning(f"Some preprocessing detectors not available: {e}")
    
    async def _enable_optimizations(self):
        """Enable memory and performance optimizations"""
        pipelines = [self.controlnet_pipeline, self.multi_controlnet_pipeline, self.inpaint_pipeline]
        
        for pipeline in pipelines:
            if pipeline is None:
                continue
                
            # Enable memory efficient attention
            if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                    logger.debug("XFormers enabled for pipeline")
                except Exception as e:
                    logger.debug(f"Could not enable XFormers: {e}")
            
            # Enable CPU offloading for memory efficiency if CUDA is available
            if self.device == 'cuda':
                try:
                    pipeline.enable_sequential_cpu_offload()
                    logger.debug("Sequential CPU offloading enabled for pipeline")
                except Exception as e:
                    logger.debug(f"Could not enable CPU offloading: {e}")
        
        logger.info("Performance optimizations applied to all pipelines")
            
    def generate_room_makeover(
        self, 
        original_image: Image.Image, 
        style: str, 
        detected_objects: List[Dict],
        strength: float = 0.75,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        use_multi_controlnet: bool = True
    ) -> Image.Image:
        """
        Generate a room makeover using advanced ControlNet to preserve structure
        
        Args:
            original_image: Original room image
            style: Style name (modern, scandinavian, etc.)
            detected_objects: List of detected objects for context
            strength: How much to change (0.0 = no change, 1.0 = complete change)
            guidance_scale: How closely to follow the prompt
            num_inference_steps: Quality vs speed tradeoff
            use_multi_controlnet: Whether to use multi-ControlNet if available
            
        Returns:
            Generated makeover image
        """
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
            
        with self._gpu_memory_context():
            try:
                # Resize image if too large
                max_size = 512
                if max(original_image.size) > max_size:
                    original_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Get style configuration
                style_config = self.STYLE_PROMPTS.get(style.lower(), self.STYLE_PROMPTS['modern'])
                
                # Build context-aware prompt
                prompt = self._build_contextual_prompt(style_config, detected_objects)
                negative_prompt = self._build_negative_prompt(style_config, detected_objects)
                
                logger.info(f"Generating {style} makeover with enhanced ControlNet")
                
                # Try multi-ControlNet first if available and requested
                if use_multi_controlnet and self.multi_controlnet_pipeline is not None:
                    try:
                        return await self._generate_with_multi_controlnet(
                            original_image, prompt, negative_prompt, style_config,
                            strength, guidance_scale, num_inference_steps
                        )
                    except Exception as e:
                        logger.warning(f"Multi-ControlNet failed, falling back to single: {e}")
                
                # Fallback to single ControlNet
                return await self._generate_with_single_controlnet(
                    original_image, prompt, negative_prompt, style_config,
                    strength, guidance_scale, num_inference_steps
                )
                
            except Exception as e:
                logger.error(f"Room makeover generation failed: {e}")
                raise
    
    async def _generate_with_multi_controlnet(
        self, 
        original_image: Image.Image,
        prompt: str,
        negative_prompt: str, 
        style_config: Dict,
        strength: float,
        guidance_scale: float,
        num_inference_steps: int
    ) -> Image.Image:
        """Generate using multi-ControlNet for maximum control"""
        
        # Generate multiple control images
        control_images = []
        conditioning_scales = []
        
        # Canny edges (primary control) 
        if self.canny_detector:
            canny_image = self.canny_detector(original_image)
            control_images.append(canny_image)
            conditioning_scales.append(strength * 0.8)  # Strong canny control
            
        # Depth map (structure control)
        if self.depth_detector:
            try:
                depth_image = self.depth_detector(original_image)
                control_images.append(depth_image)
                conditioning_scales.append(strength * 0.6)  # Moderate depth control
            except Exception as e:
                logger.debug(f"Depth detection failed: {e}")
        
        # MLSD lines (architectural control)
        if self.mlsd_detector and len(control_images) < 3:  # Limit to 3 total
            try:
                mlsd_image = self.mlsd_detector(original_image)
                control_images.append(mlsd_image)
                conditioning_scales.append(strength * 0.4)  # Light line control
            except Exception as e:
                logger.debug(f"MLSD detection failed: {e}")
        
        if len(control_images) < 2:
            raise RuntimeError("Not enough control images for multi-ControlNet")
        
        # Generate with multi-ControlNet
        with torch.inference_mode():
            result = self.multi_controlnet_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=conditioning_scales,
                generator=torch.Generator(device=self.device).manual_seed(42)
            )
        
        makeover_image = result.images[0]
        del result
        
        logger.info(f"Multi-ControlNet generation successful ({len(control_images)} controls)")
        return makeover_image
    
    async def _generate_with_single_controlnet(
        self,
        original_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        style_config: Dict,
        strength: float,
        guidance_scale: float,
        num_inference_steps: int
    ) -> Image.Image:
        """Generate using single ControlNet (fallback)"""
        
        if self.controlnet_pipeline is None:
            raise RuntimeError("No ControlNet pipeline available")
        
        # Generate Canny edge map as primary control
        if self.canny_detector is None:
            raise RuntimeError("Canny detector not available")
            
        canny_image = self.canny_detector(original_image)
        
        # Generate with single ControlNet
        with torch.inference_mode():
            result = self.controlnet_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=canny_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=strength,
                generator=torch.Generator(device=self.device).manual_seed(42)
            )
        
        makeover_image = result.images[0]
        del result
        
        logger.info("Single ControlNet generation successful")
        return makeover_image
            
    def add_suggested_products(
        self, 
        base_image: Image.Image, 
        product_suggestions: List[Dict],
        room_layout: Dict
    ) -> Image.Image:
        """
        Add suggested products to the room using inpainting
        
        Args:
            base_image: Base room image
            product_suggestions: List of products to add
            room_layout: Room layout analysis
            
        Returns:
            Image with added products
        """
        if self.inpaint_pipeline is None or not self._models_loaded:
            raise RuntimeError("Inpainting model not loaded.")
            
        with self._gpu_memory_context():
            try:
                result_image = base_image.copy()
                
                for i, product in enumerate(product_suggestions):
                    # Create mask for product placement
                    mask = self._create_product_mask(
                        base_image.size, 
                        product['coordinates'], 
                        product['category']
                    )
                    
                    # Generate product prompt
                    product_prompt = self._create_product_prompt(product, room_layout)
                    
                    # Inpaint product with memory management
                    with torch.inference_mode():
                        inpaint_result = self.inpaint_pipeline(
                            prompt=product_prompt,
                            image=result_image,
                            mask_image=mask,
                            num_inference_steps=15,
                            guidance_scale=7.5,
                            generator=torch.Generator(device=self.device).manual_seed(42)
                        )
                    
                    result_image = inpaint_result.images[0]
                    
                    # Clear intermediate results
                    del inpaint_result
                    del mask
                    
                    # Clear GPU cache between products if using CUDA
                    if self.device == 'cuda' and i < len(product_suggestions) - 1:
                        torch.cuda.empty_cache()
                    
                logger.info(f"Added {len(product_suggestions)} products to room")
                return result_image
                
            except Exception as e:
                logger.error(f"Product addition failed: {e}")
                return base_image  # Return original on failure
            
    def _build_contextual_prompt(self, style_config: Dict, detected_objects: List[Dict]) -> str:
        """Build enhanced context-aware prompt based on detected objects and style"""
        
        # Extract room type and object context
        object_types = [obj.get('type', obj.get('object_type', '')) for obj in detected_objects]
        
        # Determine room type
        room_context = self._determine_room_type(object_types)
        
        # Get base style prompt
        base_prompt = style_config['prompt']
        colors = style_config.get('colors', '')
        materials = style_config.get('materials', '')
        
        # Build furniture context
        furniture_context = self._build_furniture_context(object_types)
        
        # Construct comprehensive prompt
        enhanced_prompt = f"""
        {room_context}, {base_prompt}, 
        {furniture_context},
        color palette: {colors},
        materials: {materials},
        professional interior photography, 8k, high quality, detailed, realistic lighting,
        photorealistic, sharp focus, well composed
        """.strip().replace('\n', ' ').replace('  ', ' ')
        
        logger.debug(f"Enhanced prompt: {enhanced_prompt[:100]}...")
        return enhanced_prompt
    
    def _build_negative_prompt(self, style_config: Dict, detected_objects: List[Dict]) -> str:
        """Build comprehensive negative prompt"""
        base_negative = style_config['negative']
        
        # Add general negative terms for interior photography
        general_negatives = [
            "blurry", "low quality", "distorted", "deformed", "ugly", 
            "bad anatomy", "watermark", "signature", "text", "logo",
            "oversaturated", "underexposed", "noise", "artifacts"
        ]
        
        # Add style-specific negatives
        style_negatives = []
        object_types = [obj.get('type', obj.get('object_type', '')) for obj in detected_objects]
        
        if 'bed' in object_types:
            style_negatives.extend(["messy bed", "unmade bed"])
        if any(obj in object_types for obj in ['couch', 'sofa']):
            style_negatives.extend(["worn furniture", "stained upholstery"])
            
        # Combine all negatives
        all_negatives = [base_negative] + general_negatives + style_negatives
        negative_prompt = ", ".join(all_negatives)
        
        return negative_prompt
    
    def _determine_room_type(self, object_types: List[str]) -> str:
        """Determine room type from detected objects"""
        if 'bed' in object_types:
            return "bedroom interior"
        elif any(obj in object_types for obj in ['couch', 'sofa']):
            return "living room interior"  
        elif 'dining table' in object_types:
            return "dining room interior"
        elif any(obj in object_types for obj in ['toilet', 'sink']):
            return "bathroom interior"
        elif any(obj in object_types for obj in ['refrigerator', 'microwave', 'oven']):
            return "kitchen interior"
        else:
            return "interior room"
    
    def _build_furniture_context(self, object_types: List[str]) -> str:
        """Build furniture-specific context for prompt"""
        if not object_types:
            return "well furnished room"
            
        # Group similar furniture
        seating = [obj for obj in object_types if obj in ['chair', 'couch', 'sofa']]
        tables = [obj for obj in object_types if 'table' in obj]
        storage = [obj for obj in object_types if obj in ['bookshelf', 'dresser', 'cabinet']]
        decor = [obj for obj in object_types if obj in ['plant', 'vase', 'artwork']]
        
        context_parts = []
        if seating:
            context_parts.append(f"comfortable seating ({', '.join(set(seating))})")
        if tables:
            context_parts.append(f"functional surfaces ({', '.join(set(tables))})")
        if storage:
            context_parts.append(f"organized storage")
        if decor:
            context_parts.append(f"tasteful decorations")
            
        if context_parts:
            return "featuring " + ", ".join(context_parts)
        else:
            return f"with {', '.join(object_types[:3])}" if object_types else "well appointed space"
        
    def _create_product_mask(self, image_size: Tuple[int, int], coordinates: List[int], category: str) -> Image.Image:
        """Create mask for product placement"""
        width, height = image_size
        mask = Image.new('RGB', (width, height), color='black')
        draw = ImageDraw.Draw(mask)
        
        x, y = coordinates
        
        # Size based on category
        if category == 'lighting':
            mask_size = (width // 8, height // 4)
        elif category == 'plants':
            mask_size = (width // 10, height // 6)
        elif category == 'decor':
            mask_size = (width // 12, height // 8)
        else:
            mask_size = (width // 10, height // 8)
            
        # Draw white circle for inpainting area
        left = max(0, x - mask_size[0] // 2)
        top = max(0, y - mask_size[1] // 2)
        right = min(width, x + mask_size[0] // 2)
        bottom = min(height, y + mask_size[1] // 2)
        
        draw.ellipse([left, top, right, bottom], fill='white')
        
        return mask
        
    def _create_product_prompt(self, product: Dict, room_layout: Dict) -> str:
        """Create prompt for specific product inpainting"""
        
        category_prompts = {
            'lighting': f"{product['name']}, elegant floor lamp, warm lighting, modern design",
            'plants': f"{product['name']}, green plant in decorative pot, natural, fresh",
            'decor': f"{product['name']}, wall art, framed artwork, decorative piece",
            'furniture': f"{product['name']}, stylish furniture piece, complementary design"
        }
        
        base_prompt = category_prompts.get(product['category'], product['name'])
        room_type = room_layout.get('room_type', 'room')
        
        return f"{base_prompt} in {room_type}, photorealistic, high quality, natural lighting"
        
    def generate_before_after_composite(
        self, 
        before_image: Image.Image, 
        after_image: Image.Image
    ) -> Image.Image:
        """Create before/after comparison image"""
        
        # Resize images to same size
        target_size = (512, 384)
        before_resized = before_image.resize(target_size, Image.Resampling.LANCZOS)
        after_resized = after_image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Create composite image
        composite_width = target_size[0] * 2 + 20  # 20px gap
        composite_height = target_size[1] + 60  # 60px for labels
        
        composite = Image.new('RGB', (composite_width, composite_height), color='white')
        
        # Paste before and after images
        composite.paste(before_resized, (0, 30))
        composite.paste(after_resized, (target_size[0] + 20, 30))
        
        # Add labels
        draw = ImageDraw.Draw(composite)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
            
        # Before label
        draw.text((target_size[0]//2 - 30, 5), "BEFORE", fill='black', font=font)
        # After label  
        draw.text((target_size[0] + 20 + target_size[0]//2 - 25, 5), "AFTER", fill='black', font=font)
        
        return composite
        
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}" 