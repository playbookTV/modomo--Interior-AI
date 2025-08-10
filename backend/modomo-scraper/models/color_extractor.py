"""
Color extraction for detected objects in interior design scenes
"""

import cv2
import numpy as np
from PIL import Image, ImageStat
import webcolors
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple, Optional
import structlog

logger = structlog.get_logger()


class ColorExtractor:
    """Extract dominant colors from image crops using multiple methods"""
    
    def __init__(self, max_colors: int = 5):
        self.max_colors = max_colors
        
        # Common interior design color mappings
        self.color_mappings = {
            # Neutrals
            'white': [(255, 255, 255), (250, 250, 250), (245, 245, 245)],
            'black': [(0, 0, 0), (25, 25, 25), (50, 50, 50)],
            'gray': [(128, 128, 128), (105, 105, 105), (169, 169, 169)],
            'beige': [(245, 245, 220), (255, 228, 196), (222, 184, 135)],
            'cream': [(255, 253, 208), (255, 255, 240), (253, 245, 230)],
            
            # Warm colors  
            'red': [(255, 0, 0), (220, 20, 60), (178, 34, 34)],
            'orange': [(255, 165, 0), (255, 140, 0), (255, 69, 0)],
            'yellow': [(255, 255, 0), (255, 215, 0), (255, 218, 185)],
            'pink': [(255, 192, 203), (255, 20, 147), (219, 112, 147)],
            
            # Cool colors
            'blue': [(0, 0, 255), (30, 144, 255), (70, 130, 180)],
            'green': [(0, 128, 0), (34, 139, 34), (107, 142, 35)],
            'teal': [(0, 128, 128), (32, 178, 170), (72, 209, 204)],
            'purple': [(128, 0, 128), (75, 0, 130), (147, 112, 219)],
            
            # Warm earth tones
            'brown': [(165, 42, 42), (139, 69, 19), (160, 82, 45)],
            'tan': [(210, 180, 140), (205, 133, 63), (244, 164, 96)],
            'gold': [(255, 215, 0), (184, 134, 11), (218, 165, 32)],
            
            # Wood tones
            'light_wood': [(222, 184, 135), (245, 222, 179), (238, 203, 173)],
            'medium_wood': [(205, 133, 63), (160, 82, 45), (139, 69, 19)],
            'dark_wood': [(101, 67, 33), (62, 39, 35), (83, 53, 10)]
        }
    
    async def extract_colors(self, image_path: str, bbox: List[float] = None) -> Dict:
        """
        Extract dominant colors from image or image crop
        
        Args:
            image_path: Path to the image
            bbox: Optional bounding box [x, y, width, height] for cropping
            
        Returns:
            Dictionary with color information
        """
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Crop to bounding box if provided
            if bbox:
                x, y, w, h = [int(coord) for coord in bbox]
                # Ensure bbox is within image bounds
                x = max(0, min(x, image.width - 1))
                y = max(0, min(y, image.height - 1))
                w = min(w, image.width - x)
                h = min(h, image.height - y)
                
                if w > 0 and h > 0:
                    image = image.crop((x, y, x + w, y + h))
                else:
                    logger.warning(f"Invalid bbox for color extraction: {bbox}")
                    return {"colors": [], "dominant_color": None}
            
            # Extract colors using multiple methods
            dominant_colors = await self._extract_dominant_colors_kmeans(image)
            color_names = self._map_colors_to_names(dominant_colors)
            average_color = self._get_average_color(image)
            
            # Analyze color properties
            brightness = self._calculate_brightness(dominant_colors[0] if dominant_colors else average_color)
            color_temperature = self._estimate_color_temperature(dominant_colors[0] if dominant_colors else average_color)
            
            result = {
                "colors": [
                    {
                        "rgb": color,
                        "hex": self._rgb_to_hex(color),
                        "name": color_names[i],
                        "percentage": round(100 / len(dominant_colors), 1)
                    }
                    for i, color in enumerate(dominant_colors[:self.max_colors])
                ],
                "dominant_color": {
                    "rgb": dominant_colors[0] if dominant_colors else average_color,
                    "hex": self._rgb_to_hex(dominant_colors[0] if dominant_colors else average_color),
                    "name": color_names[0] if color_names else "unknown"
                },
                "average_color": {
                    "rgb": average_color,
                    "hex": self._rgb_to_hex(average_color)
                },
                "properties": {
                    "brightness": brightness,
                    "color_temperature": color_temperature,
                    "is_neutral": self._is_neutral_color(dominant_colors[0] if dominant_colors else average_color)
                }
            }
            
            logger.info(f"Extracted {len(result['colors'])} colors from {'crop' if bbox else 'image'}")
            return result
            
        except Exception as e:
            logger.error(f"Color extraction failed for {image_path}", error=str(e))
            return {"colors": [], "dominant_color": None}
    
    async def _extract_dominant_colors_kmeans(self, image: Image.Image, k: int = None) -> List[Tuple[int, int, int]]:
        """Extract dominant colors using K-means clustering"""
        if k is None:
            k = min(self.max_colors, 8)  # Don't over-cluster
        
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Reshape to list of pixels
            pixels = img_array.reshape(-1, 3)
            
            # Remove very dark and very bright pixels (likely shadows/highlights)
            pixels = pixels[~((pixels < 30).all(axis=1) | (pixels > 225).all(axis=1))]
            
            if len(pixels) == 0:
                return [(128, 128, 128)]  # Fallback gray
            
            # Sample pixels if too many (for performance)
            if len(pixels) > 10000:
                indices = np.random.choice(len(pixels), 10000, replace=False)
                pixels = pixels[indices]
            
            # K-means clustering
            kmeans = KMeans(n_clusters=min(k, len(pixels)), random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get cluster centers (dominant colors)
            colors = kmeans.cluster_centers_.astype(int)
            
            # Sort by cluster size (most dominant first)
            labels = kmeans.labels_
            color_counts = [(np.sum(labels == i), tuple(color)) for i, color in enumerate(colors)]
            color_counts.sort(reverse=True)
            
            return [color for _, color in color_counts]
            
        except Exception as e:
            logger.error(f"K-means color extraction failed", error=str(e))
            return [(128, 128, 128)]  # Fallback gray
    
    def _get_average_color(self, image: Image.Image) -> Tuple[int, int, int]:
        """Get average color of the image"""
        try:
            # Use PIL's ImageStat for efficient average calculation
            stat = ImageStat.Stat(image)
            return tuple(int(x) for x in stat.mean)
        except Exception:
            return (128, 128, 128)  # Fallback gray
    
    def _map_colors_to_names(self, colors: List[Tuple[int, int, int]]) -> List[str]:
        """Map RGB colors to human-readable names"""
        color_names = []
        
        for color in colors:
            # Try to find closest match in our color mappings
            closest_name = self._find_closest_color_name(color)
            color_names.append(closest_name)
        
        return color_names
    
    def _find_closest_color_name(self, target_color: Tuple[int, int, int]) -> str:
        """Find the closest color name for an RGB color"""
        min_distance = float('inf')
        closest_name = 'unknown'
        
        for name, reference_colors in self.color_mappings.items():
            for ref_color in reference_colors:
                distance = self._color_distance(target_color, ref_color)
                if distance < min_distance:
                    min_distance = distance
                    closest_name = name
        
        # Try webcolors library as fallback
        try:
            if min_distance > 100:  # If our mapping isn't close enough
                closest_name = webcolors.rgb_to_name(target_color)
        except ValueError:
            pass
        
        return closest_name
    
    def _color_distance(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
        """Calculate Euclidean distance between two RGB colors"""
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5
    
    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB tuple to hex string"""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    def _calculate_brightness(self, color: Tuple[int, int, int]) -> float:
        """Calculate perceptual brightness of a color (0-1)"""
        r, g, b = color
        # Use standard luminance formula
        brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return round(brightness, 3)
    
    def _estimate_color_temperature(self, color: Tuple[int, int, int]) -> str:
        """Estimate if color is warm, cool, or neutral"""
        r, g, b = color
        
        # Simple heuristic based on RGB values
        if r > g and r > b:
            if r - g > 30 or r - b > 30:
                return "warm"
        elif b > r and b > g:
            if b - r > 30 or b - g > 30:
                return "cool"
        elif g > r and g > b:
            if abs(g - r) < 20 and abs(g - b) < 20:
                return "neutral"
            else:
                return "cool"
        
        return "neutral"
    
    def _is_neutral_color(self, color: Tuple[int, int, int]) -> bool:
        """Check if color is neutral (grayscale-ish)"""
        r, g, b = color
        # If RGB values are close to each other, it's neutral
        max_diff = max(abs(r - g), abs(g - b), abs(r - b))
        return max_diff < 30
    
    async def extract_colors_histogram(self, image_path: str, bbox: List[float] = None) -> Dict:
        """
        Alternative color extraction using color histogram analysis
        Useful for comparison with K-means results
        """
        try:
            # Load and crop image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if bbox:
                x, y, w, h = [int(coord) for coord in bbox]
                image = image[y:y+h, x:x+w]
            
            # Calculate color histogram
            hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])
            
            # Find peaks in histogram
            r_peaks = self._find_histogram_peaks(hist_r.flatten())
            g_peaks = self._find_histogram_peaks(hist_g.flatten())
            b_peaks = self._find_histogram_peaks(hist_b.flatten())
            
            # Combine peaks to form colors
            colors = []
            for r_peak in r_peaks[:2]:  # Top 2 peaks
                for g_peak in g_peaks[:2]:
                    for b_peak in b_peaks[:2]:
                        colors.append((r_peak, g_peak, b_peak))
            
            # Remove duplicates and limit
            unique_colors = list(set(colors))[:self.max_colors]
            
            return {
                "method": "histogram",
                "colors": [{"rgb": color, "hex": self._rgb_to_hex(color)} for color in unique_colors]
            }
            
        except Exception as e:
            logger.error(f"Histogram color extraction failed", error=str(e))
            return {"colors": []}
    
    def _find_histogram_peaks(self, histogram: np.ndarray) -> List[int]:
        """Find peaks in color histogram"""
        # Simple peak detection
        peaks = []
        for i in range(1, len(histogram) - 1):
            if histogram[i] > histogram[i-1] and histogram[i] > histogram[i+1]:
                if histogram[i] > np.max(histogram) * 0.1:  # Only significant peaks
                    peaks.append(i)
        
        # Sort by peak height
        peaks.sort(key=lambda x: histogram[x], reverse=True)
        return peaks[:5]  # Top 5 peaks
    
    def generate_color_keywords(self, color_data: Dict) -> List[str]:
        """Generate searchable keywords from color data"""
        keywords = []
        
        # Add color names
        for color_info in color_data.get("colors", []):
            if color_info.get("name"):
                keywords.append(color_info["name"])
        
        # Add dominant color
        dominant = color_data.get("dominant_color", {})
        if dominant.get("name"):
            keywords.append(f"dominant_{dominant['name']}")
        
        # Add properties
        props = color_data.get("properties", {})
        if props.get("color_temperature"):
            keywords.append(f"{props['color_temperature']}_tone")
        
        if props.get("brightness", 0) > 0.7:
            keywords.append("light")
        elif props.get("brightness", 1) < 0.3:
            keywords.append("dark")
        
        if props.get("is_neutral"):
            keywords.append("neutral")
        
        return list(set(keywords))  # Remove duplicates