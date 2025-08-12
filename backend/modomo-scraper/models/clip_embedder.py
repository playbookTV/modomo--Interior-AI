"""
CLIP embeddings for image-text similarity matching
"""

import torch, clip
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from typing import List, Optional, Union, Dict
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
    
    async def embed_color_query(self, query: str) -> List[float]:
        """
        Generate CLIP embedding optimized for color-based furniture queries
        
        Args:
            query: Color-based query like "red sofa" or "blue curtains"
            
        Returns:
            Normalized CLIP embedding for the query
        """
        # Enhance query with furniture context for better matching
        enhanced_query = f"furniture item that is {query}"
        return await self.embed_text(enhanced_query)
    
    async def search_objects_by_color(self, color_query: str, object_embeddings: List[List[float]], 
                                    object_ids: List[str], threshold: float = 0.3, 
                                    limit: int = 10) -> List[dict]:
        """
        Search for objects matching a color-based query using CLIP embeddings
        
        Args:
            color_query: Color-based search query (e.g., "red sofa", "dark wood table")
            object_embeddings: List of object CLIP embeddings
            object_ids: Corresponding object IDs
            threshold: Minimum similarity threshold
            limit: Maximum results to return
            
        Returns:
            List of matching objects with similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = await self.embed_color_query(color_query)
            
            if not query_embedding:
                return []
            
            # Calculate similarities
            matches = []
            for obj_id, obj_embedding in zip(object_ids, object_embeddings):
                if obj_embedding:  # Skip empty embeddings
                    similarity = self.calculate_similarity(query_embedding, obj_embedding)
                    if similarity >= threshold:
                        matches.append({
                            'object_id': obj_id,
                            'similarity': similarity,
                            'query': color_query
                        })
            
            # Sort by similarity (descending) and limit results
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            return matches[:limit]
            
        except Exception as e:
            logger.error(f"Color-based search failed for query '{color_query}'", error=str(e))
            return []
    
    def generate_color_search_queries(self, colors: List[str], categories: List[str] = None) -> List[str]:
        """
        Generate comprehensive search queries for color + furniture combinations
        
        Args:
            colors: List of color names
            categories: Optional list of furniture categories
            
        Returns:
            List of search query strings
        """
        if not categories:
            categories = [
                "sofa", "chair", "table", "lamp", "rug", "curtains", 
                "cabinet", "bookshelf", "bed", "dresser"
            ]
        
        queries = []
        
        # Basic color + category combinations
        for color in colors:
            for category in categories:
                queries.extend([
                    f"{color} {category}",
                    f"{category} in {color}",
                    f"{color} colored {category}"
                ])
        
        # Color-only queries
        for color in colors:
            queries.extend([
                f"{color} furniture",
                f"furniture in {color}",
                f"{color} items"
            ])
        
        # Descriptive color queries
        color_descriptors = {
            "red": ["burgundy", "crimson", "cherry red"],
            "blue": ["navy blue", "sky blue", "royal blue"],
            "green": ["forest green", "sage green", "mint green"],
            "brown": ["dark brown", "light brown", "chocolate brown"],
            "white": ["off white", "cream", "ivory"],
            "black": ["charcoal", "dark gray", "jet black"],
            "gray": ["light gray", "dark gray", "silver"],
            "yellow": ["golden", "pale yellow", "bright yellow"]
        }
        
        for color in colors:
            if color in color_descriptors:
                for descriptor in color_descriptors[color]:
                    queries.append(f"{descriptor} furniture")
        
        return queries
    
    async def batch_color_search(self, queries: List[str], object_embeddings: List[List[float]], 
                                object_ids: List[str], threshold: float = 0.3) -> Dict[str, List[dict]]:
        """
        Perform batch color-based searches for multiple queries
        
        Args:
            queries: List of search query strings
            object_embeddings: List of object CLIP embeddings
            object_ids: Corresponding object IDs
            threshold: Minimum similarity threshold
            
        Returns:
            Dictionary mapping queries to their search results
        """
        results = {}
        
        for query in queries:
            matches = await self.search_objects_by_color(
                query, object_embeddings, object_ids, threshold
            )
            if matches:  # Only store non-empty results
                results[query] = matches
        
        return results
    
    class ForegroundClipEmbedder:
    def __init__(self, model_name="ViT-L/14@336px", device="cuda"):
        self.device = device if torch.cuda.is_available() and device=="cuda" else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def embed_rgba(self, rgba_path: str):
        img = Image.open(rgba_path).convert("RGBA")
        np_img = np.array(img)
        alpha = np_img[:,:,3:4] / 255.0
        # Composite on neutral gray so backgrounds don’t bias features
        bg = np.full_like(np_img[:,:,:3], 128, dtype=np.uint8)
        comp = (alpha * np_img[:,:,:3] + (1-alpha) * bg).astype(np.uint8)
        comp_pil = Image.fromarray(comp, "RGB")
        with torch.no_grad():
            x = self.preprocess(comp_pil).unsqueeze(0).to(self.device)
            z = self.model.encode_image(x).float()
            z = z / (z.norm(dim=-1, keepdim=True)+1e-8)
        return z.squeeze(0).cpu().numpy()