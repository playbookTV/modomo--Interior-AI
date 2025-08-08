"""
CLIP embeddings for image-text similarity matching
"""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from typing import List, Optional, Union
import structlog

logger = structlog.get_logger()

class CLIPEmbedder:
    """Generate CLIP embeddings for images and text"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing CLIP embedder on {self.device}")
        
        # Load CLIP model and processor
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Set model to eval mode
        self.model.eval()
        
    async def embed_image(self, image_path: str) -> List[float]:
        """
        Generate CLIP embedding for a full image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Normalized CLIP embedding vector
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            
            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                image_features = self.model.get_image_features(**inputs)
                
                # Normalize the embedding
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
            return image_features.cpu().numpy().flatten().tolist()
            
        except Exception as e:
            logger.error(f"Image embedding failed for {image_path}", error=str(e))
            return []
    
    async def embed_object(self, image_path: str, bbox: List[float]) -> List[float]:
        """
        Generate CLIP embedding for a cropped object region
        
        Args:
            image_path: Path to the full image
            bbox: Bounding box [x, y, width, height]
            
        Returns:
            Normalized CLIP embedding for the cropped region
        """
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Crop to bounding box
            x, y, w, h = bbox
            cropped_image = image.crop((x, y, x + w, y + h))
            
            # Resize if too small
            if cropped_image.size[0] < 32 or cropped_image.size[1] < 32:
                cropped_image = cropped_image.resize((224, 224), Image.Resampling.LANCZOS)
            
            with torch.no_grad():
                inputs = self.processor(images=cropped_image, return_tensors="pt").to(self.device)
                image_features = self.model.get_image_features(**inputs)
                
                # Normalize the embedding
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
            return image_features.cpu().numpy().flatten().tolist()
            
        except Exception as e:
            logger.error(f"Object embedding failed for {image_path}, bbox {bbox}", error=str(e))
            return []
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate CLIP embedding for text
        
        Args:
            text: Input text string
            
        Returns:
            Normalized CLIP embedding vector
        """
        try:
            with torch.no_grad():
                inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
                text_features = self.model.get_text_features(**inputs)
                
                # Normalize the embedding
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
            return text_features.cpu().numpy().flatten().tolist()
            
        except Exception as e:
            logger.error(f"Text embedding failed for '{text}'", error=str(e))
            return []
    
    async def embed_text_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate CLIP embeddings for multiple texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of normalized embedding vectors
        """
        try:
            embeddings = []
            
            # Process in batches to avoid memory issues
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                with torch.no_grad():
                    inputs = self.processor(text=batch_texts, return_tensors="pt", 
                                          padding=True, truncation=True).to(self.device)
                    text_features = self.model.get_text_features(**inputs)
                    
                    # Normalize embeddings
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    # Convert to list format
                    batch_embeddings = text_features.cpu().numpy().tolist()
                    embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Batch text embedding failed", error=str(e))
            return []
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            
            # Clamp to [0, 1] range
            return max(0.0, min(1.0, float(similarity)))
            
        except Exception as e:
            logger.error("Similarity calculation failed", error=str(e))
            return 0.0
    
    async def find_similar_products(self, object_embedding: List[float], 
                                  product_embeddings: List[List[float]], 
                                  product_ids: List[str], 
                                  top_k: int = 5) -> List[dict]:
        """
        Find most similar products to an object embedding
        
        Args:
            object_embedding: CLIP embedding of detected object
            product_embeddings: List of product CLIP embeddings
            product_ids: List of corresponding product IDs
            top_k: Number of top matches to return
            
        Returns:
            List of matches with product_id and similarity score
        """
        try:
            similarities = []
            
            for product_id, product_embedding in zip(product_ids, product_embeddings):
                similarity = self.calculate_similarity(object_embedding, product_embedding)
                similarities.append({
                    'product_id': product_id,
                    'similarity': similarity
                })
            
            # Sort by similarity (descending) and return top_k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error("Product similarity search failed", error=str(e))
            return []
    
    async def embed_product_descriptions(self, descriptions: List[str]) -> List[List[float]]:
        """
        Generate embeddings for product descriptions
        
        Args:
            descriptions: List of product description texts
            
        Returns:
            List of normalized embedding vectors
        """
        # Preprocess descriptions to create better embeddings
        processed_descriptions = []
        for desc in descriptions:
            # Add context to improve matching
            processed_desc = f"furniture item: {desc}"
            processed_descriptions.append(processed_desc)
        
        return await self.embed_text_batch(processed_descriptions)