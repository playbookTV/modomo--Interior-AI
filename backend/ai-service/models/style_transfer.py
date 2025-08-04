import torch
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

logger = structlog.get_logger()

class StyleTransferModel:
    """
    Real style transfer using Stable Diffusion + ControlNet for room makeovers
    """
    
    STYLE_PROMPTS = {
        'modern': {
            'prompt': 'modern minimalist interior design, clean lines, neutral colors, contemporary furniture, sleek aesthetic, bright lighting, open space',
            'negative': 'cluttered, ornate, traditional, dark, cramped, outdated'
        },
        'scandinavian': {
            'prompt': 'scandinavian interior design, light wood, white walls, cozy textiles, natural materials, hygge aesthetic, minimalist furniture',
            'negative': 'dark colors, heavy furniture, ornate details, cluttered'
        },
        'industrial': {
            'prompt': 'industrial interior design, exposed brick, metal fixtures, concrete floors, dark colors, urban loft aesthetic, raw materials',
            'negative': 'soft colors, delicate furniture, ornate decorations'
        },
        'bohemian': {
            'prompt': 'bohemian interior design, colorful textiles, plants, eclectic furniture, warm lighting, artistic decorations, cozy atmosphere',
            'negative': 'minimalist, sterile, monochrome, corporate'
        },
        'traditional': {
            'prompt': 'traditional interior design, classic furniture, warm colors, elegant details, comfortable seating, timeless aesthetic',
            'negative': 'ultra modern, stark, industrial, minimalist'
        }
    }
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        self.model_id = model_id
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.controlnet_pipeline = None
        self.inpaint_pipeline = None
        self.canny_detector = None
        logger.info(f"Style transfer will use device: {self.device}")
        
    async def load_models(self):
        """Load Stable Diffusion and ControlNet models"""
        try:
            logger.info("Loading ControlNet and Stable Diffusion models...")
            
            # Load ControlNet for structure preservation
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            
            # Load ControlNet pipeline
            self.controlnet_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                self.model_id,
                controlnet=controlnet,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Load inpainting pipeline for product additions
            self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Move to device
            self.controlnet_pipeline = self.controlnet_pipeline.to(self.device)
            self.inpaint_pipeline = self.inpaint_pipeline.to(self.device)
            
            # Load edge detector
            self.canny_detector = CannyDetector()
            
            # Enable memory efficient attention
            if hasattr(self.controlnet_pipeline, 'enable_xformers_memory_efficient_attention'):
                self.controlnet_pipeline.enable_xformers_memory_efficient_attention()
                self.inpaint_pipeline.enable_xformers_memory_efficient_attention()
                
            logger.info("Style transfer models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load style transfer models: {e}")
            raise
            
    def generate_room_makeover(
        self, 
        original_image: Image.Image, 
        style: str, 
        detected_objects: List[Dict],
        strength: float = 0.75,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20
    ) -> Image.Image:
        """
        Generate a room makeover using ControlNet to preserve structure
        
        Args:
            original_image: Original room image
            style: Style name (modern, scandinavian, etc.)
            detected_objects: List of detected objects for context
            strength: How much to change (0.0 = no change, 1.0 = complete change)
            guidance_scale: How closely to follow the prompt
            num_inference_steps: Quality vs speed tradeoff
            
        Returns:
            Generated makeover image
        """
        if self.controlnet_pipeline is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
            
        try:
            # Resize image if too large
            max_size = 512
            if max(original_image.size) > max_size:
                original_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
            # Generate Canny edge map to preserve structure
            canny_image = self.canny_detector(original_image)
            
            # Get style prompt
            style_config = self.STYLE_PROMPTS.get(style.lower(), self.STYLE_PROMPTS['modern'])
            
            # Build context-aware prompt
            prompt = self._build_contextual_prompt(style_config['prompt'], detected_objects)
            negative_prompt = style_config['negative']
            
            logger.info(f"Generating {style} makeover with prompt: {prompt[:100]}...")
            
            # Generate makeover
            result = self.controlnet_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=canny_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=strength,
                generator=torch.Generator(device=self.device).manual_seed(42)  # Reproducible results
            )
            
            makeover_image = result.images[0]
            
            logger.info("Room makeover generated successfully")
            return makeover_image
            
        except Exception as e:
            logger.error(f"Room makeover generation failed: {e}")
            raise
            
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
        if self.inpaint_pipeline is None:
            raise RuntimeError("Inpainting model not loaded.")
            
        try:
            result_image = base_image.copy()
            
            for product in product_suggestions:
                # Create mask for product placement
                mask = self._create_product_mask(
                    base_image.size, 
                    product['coordinates'], 
                    product['category']
                )
                
                # Generate product prompt
                product_prompt = self._create_product_prompt(product, room_layout)
                
                # Inpaint product
                inpaint_result = self.inpaint_pipeline(
                    prompt=product_prompt,
                    image=result_image,
                    mask_image=mask,
                    num_inference_steps=15,
                    guidance_scale=7.5,
                    generator=torch.Generator(device=self.device).manual_seed(42)
                )
                
                result_image = inpaint_result.images[0]
                
            logger.info(f"Added {len(product_suggestions)} products to room")
            return result_image
            
        except Exception as e:
            logger.error(f"Product addition failed: {e}")
            return base_image  # Return original on failure
            
    def _build_contextual_prompt(self, base_prompt: str, detected_objects: List[Dict]) -> str:
        """Build context-aware prompt based on detected objects"""
        
        # Extract room type context
        object_types = [obj['type'] for obj in detected_objects]
        
        room_context = ""
        if 'bed' in object_types:
            room_context = "bedroom interior"
        elif any(obj in object_types for obj in ['couch', 'sofa']):
            room_context = "living room interior"
        elif 'dining table' in object_types:
            room_context = "dining room interior"
        elif any(obj in object_types for obj in ['toilet', 'sink']):
            room_context = "bathroom interior"
        else:
            room_context = "room interior"
            
        # Build enhanced prompt
        enhanced_prompt = f"{room_context}, {base_prompt}, photorealistic, high quality, well lit, professional interior photography"
        
        return enhanced_prompt
        
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