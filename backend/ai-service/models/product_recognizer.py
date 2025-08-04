import torch
import clip
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import requests
from dataclasses import dataclass
import structlog
import asyncio
import aiohttp
from urllib.parse import quote_plus

logger = structlog.get_logger()

@dataclass
class Product:
    name: str
    category: str
    description: str
    retail_search_terms: List[str]
    confidence: float
    visual_features: Optional[np.ndarray] = None

class ProductRecognizer:
    """
    Real product recognition using CLIP and transformers for identifying furniture and decor
    """
    
    # Product categories with their typical items
    PRODUCT_CATEGORIES = {
        'lighting': [
            'floor lamp', 'table lamp', 'pendant light', 'chandelier', 'desk lamp',
            'ceiling light', 'wall sconce', 'track lighting', 'string lights'
        ],
        'plants': [
            'monstera plant', 'fiddle leaf fig', 'snake plant', 'pothos', 'peace lily',
            'rubber plant', 'spider plant', 'palm plant', 'succulents', 'fern'
        ],
        'decor': [
            'wall art', 'canvas print', 'framed picture', 'mirror', 'wall clock',
            'vase', 'sculpture', 'decorative pillow', 'throw blanket', 'candles'
        ],
        'furniture': [
            'coffee table', 'side table', 'bookshelf', 'ottoman', 'accent chair',
            'storage bench', 'console table', 'bar stool', 'desk chair'
        ],
        'storage': [
            'storage basket', 'decorative box', 'shelving unit', 'wardrobe',
            'dresser', 'nightstand', 'storage ottoman', 'bookcase'
        ]
    }
    
    # Style-specific product mappings
    STYLE_PRODUCTS = {
        'modern': {
            'lighting': ['sleek floor lamp', 'minimalist pendant light', 'LED strip lights'],
            'plants': ['monstera in modern planter', 'snake plant in geometric pot'],
            'decor': ['abstract wall art', 'geometric mirror', 'minimalist vase'],
            'furniture': ['glass coffee table', 'modern accent chair', 'floating shelf']
        },
        'scandinavian': {
            'lighting': ['wood table lamp', 'pendant light with natural materials'],
            'plants': ['fiddle leaf fig in woven basket', 'small potted plants'],
            'decor': ['nature photography print', 'wooden wall clock', 'white ceramic vase'],
            'furniture': ['light wood coffee table', 'sheepskin rug', 'wooden stool']
        },
        'industrial': {
            'lighting': ['metal floor lamp', 'exposed bulb fixtures', 'track lighting'],
            'plants': ['plants in metal planters', 'hanging plants'],
            'decor': ['metal wall art', 'vintage posters', 'industrial mirror'],
            'furniture': ['metal and wood coffee table', 'leather accent chair']
        },
        'bohemian': {
            'lighting': ['moroccan pendant light', 'fairy string lights', 'beaded lamp'],
            'plants': ['hanging plants', 'variety of potted plants'],
            'decor': ['tapestry wall hanging', 'colorful throw pillows', 'patterned rug'],
            'furniture': ['rattan chair', 'vintage coffee table', 'floor cushions']
        }
    }
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip_model = None
        self.clip_processor = None
        self.blip_model = None
        self.blip_processor = None
        logger.info(f"Product recognizer will use device: {self.device}")
        
    async def load_models(self):
        """Load CLIP and BLIP models for product recognition"""
        try:
            logger.info("Loading CLIP and BLIP models...")
            
            # Load CLIP for visual similarity
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # Load BLIP for image captioning
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            
            # Move to device
            self.clip_model = self.clip_model.to(self.device)
            self.blip_model = self.blip_model.to(self.device)
            
            logger.info("Product recognition models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load product recognition models: {e}")
            raise
            
    def identify_products_in_room(
        self, 
        room_image: Image.Image, 
        detected_objects: List[Dict],
        room_style: str = 'modern'
    ) -> List[Product]:
        """
        Identify specific products that would fit in the room
        
        Args:
            room_image: Room image
            detected_objects: Objects detected by YOLO
            room_style: Desired room style
            
        Returns:
            List of identified products
        """
        if self.clip_model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
            
        try:
            # Generate room caption for context
            room_caption = self._generate_room_caption(room_image)
            
            # Analyze room for missing/needed items
            suggested_products = self._suggest_products_for_room(
                detected_objects, 
                room_style, 
                room_caption
            )
            
            # Use CLIP to refine product suggestions based on visual similarity
            refined_products = self._refine_with_visual_similarity(
                room_image, 
                suggested_products
            )
            
            logger.info(f"Identified {len(refined_products)} products for {room_style} room")
            return refined_products
            
        except Exception as e:
            logger.error(f"Product identification failed: {e}")
            return []
            
    def _generate_room_caption(self, image: Image.Image) -> str:
        """Generate descriptive caption for the room"""
        try:
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            out = self.blip_model.generate(**inputs, max_length=50)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            logger.warning(f"Caption generation failed: {e}")
            return "interior room"
            
    def _suggest_products_for_room(
        self, 
        detected_objects: List[Dict], 
        style: str, 
        room_caption: str
    ) -> List[Product]:
        """Suggest products based on room analysis and style"""
        
        existing_objects = [obj['type'].lower() for obj in detected_objects]
        suggested_products = []
        
        # Get style-specific products
        style_products = self.STYLE_PRODUCTS.get(style.lower(), self.STYLE_PRODUCTS['modern'])
        
        # Suggest lighting if none detected
        if not any('lamp' in obj or 'light' in obj for obj in existing_objects):
            lighting_options = style_products['lighting']
            for light in lighting_options[:2]:  # Top 2 lighting options
                product = Product(
                    name=light.title(),
                    category='lighting',
                    description=f"{light} perfect for {style} style rooms",
                    retail_search_terms=[light, f"{style} {light}", "floor lamp"],
                    confidence=0.85
                )
                suggested_products.append(product)
                
        # Suggest plants for life and color
        if not any('plant' in obj for obj in existing_objects):
            plant_options = style_products['plants']
            for plant in plant_options[:2]:  # Top 2 plant options
                product = Product(
                    name=plant.title(),
                    category='plants',
                    description=f"{plant} to bring natural elements to the space",
                    retail_search_terms=[plant, "indoor plant", "houseplant"],
                    confidence=0.75
                )
                suggested_products.append(product)
                
        # Suggest decor items
        decor_options = style_products['decor']
        for decor in decor_options[:2]:  # Top 2 decor options
            product = Product(
                name=decor.title(),
                category='decor',
                description=f"{decor} to enhance the {style} aesthetic",
                retail_search_terms=[decor, f"{style} decor", "wall decor"],
                confidence=0.70
            )
            suggested_products.append(product)
            
        return suggested_products
        
    def _refine_with_visual_similarity(
        self, 
        room_image: Image.Image, 
        products: List[Product]
    ) -> List[Product]:
        """Use CLIP to refine product suggestions based on visual compatibility"""
        
        try:
            # Generate room embedding
            room_inputs = self.clip_processor(images=room_image, return_tensors="pt").to(self.device)
            room_features = self.clip_model.get_image_features(**room_inputs)
            room_features = room_features / room_features.norm(p=2, dim=-1, keepdim=True)
            
            refined_products = []
            
            for product in products:
                # Create text descriptions for the product in context
                product_texts = [
                    f"a {product.name.lower()} in a room",
                    f"{product.name.lower()} interior design",
                    f"room with {product.name.lower()}"
                ]
                
                # Get text embeddings
                text_inputs = self.clip_processor(text=product_texts, return_tensors="pt", padding=True).to(self.device)
                text_features = self.clip_model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                
                # Calculate similarity
                similarities = torch.cosine_similarity(room_features, text_features)
                max_similarity = similarities.max().item()
                
                # Update confidence based on visual similarity
                visual_confidence = max_similarity * 0.5 + 0.5  # Scale to 0.5-1.0 range
                product.confidence = product.confidence * visual_confidence
                
                # Only keep products with reasonable confidence
                if product.confidence > 0.4:
                    refined_products.append(product)
                    
            return sorted(refined_products, key=lambda p: p.confidence, reverse=True)
            
        except Exception as e:
            logger.warning(f"Visual refinement failed, using original suggestions: {e}")
            return products
            
    async def search_products_online(
        self, 
        products: List[Product], 
        retailers: List[str] = ['amazon', 'ikea', 'wayfair']
    ) -> Dict[str, List[Dict]]:
        """
        Search for products across multiple retailers
        
        Args:
            products: List of products to search for
            retailers: List of retailer names
            
        Returns:
            Dictionary mapping product names to retailer results
        """
        
        search_results = {}
        
        async with aiohttp.ClientSession() as session:
            for product in products:
                product_results = {}
                
                # Search each retailer
                search_tasks = []
                for retailer in retailers:
                    task = self._search_retailer(session, product, retailer)
                    search_tasks.append(task)
                    
                # Wait for all searches to complete
                retailer_results = await asyncio.gather(*search_tasks, return_exceptions=True)
                
                # Process results
                for retailer, result in zip(retailers, retailer_results):
                    if not isinstance(result, Exception) and result:
                        product_results[retailer] = result
                        
                search_results[product.name] = product_results
                
        return search_results
        
    async def _search_retailer(
        self, 
        session: aiohttp.ClientSession, 
        product: Product, 
        retailer: str
    ) -> Optional[Dict]:
        """Search a specific retailer for a product"""
        
        try:
            search_term = product.retail_search_terms[0] if product.retail_search_terms else product.name
            
            # Mock search URLs (in production, use real APIs)
            search_urls = {
                'amazon': f"https://www.amazon.com/s?k={quote_plus(search_term)}",
                'ikea': f"https://www.ikea.com/gb/en/search/products/?q={quote_plus(search_term)}",
                'wayfair': f"https://www.wayfair.com/keyword.php?keyword={quote_plus(search_term)}"
            }
            
            if retailer not in search_urls:
                return None
                
            # For now, return mock data (replace with real scraping/API calls)
            return {
                'url': search_urls[retailer],
                'price_range': self._estimate_price_range(product.category),
                'availability': 'In Stock',
                'shipping': 'Free delivery' if retailer == 'ikea' else None
            }
            
        except Exception as e:
            logger.warning(f"Retailer search failed for {retailer}: {e}")
            return None
            
    def _estimate_price_range(self, category: str) -> Tuple[float, float]:
        """Estimate price range for product category"""
        
        price_ranges = {
            'lighting': (25.0, 150.0),
            'plants': (10.0, 60.0),
            'decor': (15.0, 80.0),
            'furniture': (50.0, 300.0),
            'storage': (30.0, 200.0)
        }
        
        return price_ranges.get(category, (20.0, 100.0))
        
    def generate_shopping_coordinates(
        self, 
        products: List[Product], 
        room_layout: Dict,
        image_size: Tuple[int, int]
    ) -> List[Dict]:
        """
        Generate coordinates for placing shopping indicators on room image
        
        Args:
            products: List of products
            room_layout: Room layout analysis
            image_size: (width, height) of room image
            
        Returns:
            List of product dictionaries with coordinates
        """
        
        width, height = image_size
        empty_zones = room_layout.get('layout_analysis', {}).get('empty_zones', ['center'])
        
        product_placements = []
        
        for i, product in enumerate(products):
            # Distribute products across empty zones
            if empty_zones:
                zone = empty_zones[i % len(empty_zones)]
                if zone == 'left':
                    x = width * 0.2
                elif zone == 'right':
                    x = width * 0.8
                else:  # center or other
                    x = width * 0.5
            else:
                # Fallback: distribute evenly
                x = width * (0.2 + 0.6 * i / max(1, len(products) - 1))
                
            # Y coordinate based on product type
            if product.category == 'lighting':
                y = height * 0.3  # Upper area for lamps
            elif product.category == 'plants':
                y = height * 0.7  # Floor level
            elif product.category == 'decor':
                y = height * 0.25  # Wall level
            else:
                y = height * 0.6  # Mid-level
                
            product_placements.append({
                'product': product,
                'coordinates': [int(x), int(y)],
                'zone': zone if empty_zones else 'distributed'
            })
            
        return product_placements 