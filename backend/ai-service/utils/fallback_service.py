"""
Fallback service for AI operations when models fail or are unavailable
"""
import uuid
from typing import List, Dict, Any, Tuple
from PIL import Image
import structlog

logger = structlog.get_logger()

class FallbackService:
    """
    Provides fallback implementations when AI models are unavailable
    """
    
    # Mock furniture objects for fallback detection
    FALLBACK_OBJECTS = [
        {"type": "chair", "confidence": 0.8, "bbox": [100, 150, 80, 120], "description": "Dining chair"},
        {"type": "table", "confidence": 0.7, "bbox": [200, 200, 150, 100], "description": "Coffee table"},
        {"type": "couch", "confidence": 0.9, "bbox": [50, 250, 300, 150], "description": "Living room sofa"},
        {"type": "plant", "confidence": 0.6, "bbox": [350, 100, 50, 80], "description": "Potted plant"},
    ]
    
    # Style-specific product suggestions
    STYLE_PRODUCTS = {
        "modern": [
            {
                "product_id": "modern_lamp_001",
                "name": "Minimalist Floor Lamp",
                "category": "lighting",
                "description": "Clean lines, contemporary style",
                "coordinates": [150, 100],
                "confidence": 0.8,
                "image_url": "https://example.com/modern_lamp.jpg",
                "prices": [
                    {
                        "retailer": "Amazon",
                        "price": 89.99,
                        "currency": "GBP",
                        "url": "https://amazon.co.uk/modern-lamp",
                        "availability": "In Stock"
                    }
                ]
            },
            {
                "product_id": "modern_art_001", 
                "name": "Abstract Wall Art",
                "category": "decor",
                "description": "Modern geometric artwork",
                "coordinates": [250, 80],
                "confidence": 0.7,
                "image_url": "https://example.com/modern_art.jpg",
                "prices": [
                    {
                        "retailer": "IKEA",
                        "price": 45.00,
                        "currency": "GBP", 
                        "url": "https://ikea.com/uk/modern-art",
                        "availability": "In Stock"
                    }
                ]
            }
        ],
        "scandinavian": [
            {
                "product_id": "scandi_plant_001",
                "name": "Nordic Plant Pot",
                "category": "plants",
                "description": "White ceramic with wooden stand",
                "coordinates": [180, 200],
                "confidence": 0.9,
                "image_url": "https://example.com/scandi_plant.jpg",
                "prices": [
                    {
                        "retailer": "IKEA",
                        "price": 25.00,
                        "currency": "GBP",
                        "url": "https://ikea.com/uk/plant-pot",
                        "availability": "In Stock"
                    }
                ]
            }
        ],
        "industrial": [
            {
                "product_id": "industrial_light_001",
                "name": "Edison Bulb Pendant",
                "category": "lighting", 
                "description": "Exposed bulb industrial style",
                "coordinates": [200, 50],
                "confidence": 0.8,
                "image_url": "https://example.com/industrial_light.jpg",
                "prices": [
                    {
                        "retailer": "Amazon",
                        "price": 65.00,
                        "currency": "GBP",
                        "url": "https://amazon.co.uk/edison-pendant",
                        "availability": "In Stock"
                    }
                ]
            }
        ]
    }
    
    @staticmethod
    def get_fallback_object_detection(image: Image.Image) -> List[Dict[str, Any]]:
        """
        Return fallback object detection results
        
        Args:
            image: Input image (used to determine appropriate objects)
            
        Returns:
            List of detected objects
        """
        logger.info("Using fallback object detection")
        
        # Analyze image dimensions to provide contextually relevant objects
        width, height = image.size
        aspect_ratio = width / height
        
        # Adjust bounding boxes based on image size
        objects = []
        for obj in FallbackService.FALLBACK_OBJECTS:
            # Scale bounding box to image dimensions
            scaled_bbox = [
                int(obj["bbox"][0] * width / 400),  # Scale x
                int(obj["bbox"][1] * height / 300), # Scale y
                int(obj["bbox"][2] * width / 400),  # Scale width
                int(obj["bbox"][3] * height / 300)  # Scale height
            ]
            
            objects.append({
                "object_type": obj["type"],
                "confidence": obj["confidence"],
                "bounding_box": scaled_bbox,
                "description": obj["description"]
            })
            
        return objects
    
    @staticmethod
    def get_fallback_style_transfer(
        image: Image.Image, 
        style: str, 
        detected_objects: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """
        Return fallback style transfer results (mock URLs)
        
        Args:
            image: Original image
            style: Desired style
            detected_objects: Previously detected objects
            
        Returns:
            Tuple of (before_url, after_url)
        """
        logger.info(f"Using fallback style transfer for style: {style}")
        
        # Generate deterministic URLs based on style
        style_id = style.lower().replace(" ", "_")
        image_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"reroom_fallback_{style}"))
        
        before_url = f"https://api.reroom.app/fallback/before_{image_id}.jpg"
        after_url = f"https://api.reroom.app/fallback/{style_id}_{image_id}.jpg"
        
        return before_url, after_url
    
    @staticmethod
    def get_fallback_product_suggestions(
        image: Image.Image,
        detected_objects: List[Dict[str, Any]], 
        style: str
    ) -> List[Dict[str, Any]]:
        """
        Return fallback product suggestions
        
        Args:
            image: Room image
            detected_objects: Objects in the room
            style: Desired style
            
        Returns:
            List of suggested products
        """
        logger.info(f"Using fallback product suggestions for style: {style}")
        
        # Get products for the requested style, fallback to modern
        style_key = style.lower()
        products = FallbackService.STYLE_PRODUCTS.get(
            style_key, 
            FallbackService.STYLE_PRODUCTS["modern"]
        )
        
        # Adjust coordinates based on image size
        width, height = image.size
        adjusted_products = []
        
        for product in products:
            adjusted_product = product.copy()
            adjusted_product["coordinates"] = [
                int(product["coordinates"][0] * width / 400),
                int(product["coordinates"][1] * height / 300)
            ]
            adjusted_products.append(adjusted_product)
            
        return adjusted_products
    
    @staticmethod
    def get_fallback_health_status() -> Dict[str, Any]:
        """
        Return fallback health status when models are unavailable
        """
        return {
            "object_detector": False,
            "style_transfer": False, 
            "product_recognizer": False,
            "fallback_mode": True,
            "message": "AI models unavailable, using fallback responses"
        }
    
    @staticmethod
    def create_fallback_makeover_response(
        makeover_id: str,
        photo_id: str,
        style: str,
        processing_time_ms: int
    ) -> Dict[str, Any]:
        """
        Create a complete fallback makeover response
        
        Args:
            makeover_id: Unique makeover ID
            photo_id: Original photo ID
            style: Requested style
            processing_time_ms: Processing time
            
        Returns:
            Complete makeover response
        """
        logger.info(f"Creating fallback makeover response for style: {style}")
        
        # Mock image for fallback
        mock_image = Image.new('RGB', (512, 384), color='white')
        
        # Get fallback components
        detected_objects = FallbackService.get_fallback_object_detection(mock_image)
        before_url, after_url = FallbackService.get_fallback_style_transfer(
            mock_image, style, detected_objects
        )
        suggested_products = FallbackService.get_fallback_product_suggestions(
            mock_image, detected_objects, style
        )
        
        return {
            "makeover_id": makeover_id,
            "photo_id": photo_id,
            "status": "completed",
            "fallback_mode": True,
            "transformation": {
                "style_name": style,
                "before_image_url": before_url,
                "after_image_url": after_url,
                "detected_objects": detected_objects,
                "suggested_products": suggested_products,
                "confidence_score": 0.6  # Lower confidence for fallback
            },
            "processing_time_ms": processing_time_ms,
            "created_at": "2025-01-20T12:00:00Z"
        }