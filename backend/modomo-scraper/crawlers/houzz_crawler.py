"""
Houzz scene crawler using Scrapy + Playwright for JS-rendered content
"""

import asyncio
import aiohttp
import aiofiles
from playwright.async_api import async_playwright
from typing import List, Dict, Optional, Any
import json
import re
from urllib.parse import urljoin, urlparse
import tempfile
import hashlib
import structlog

logger = structlog.get_logger()

class SceneData:
    """Data structure for scraped scene information"""
    def __init__(self, houzz_id: str, image_url: str, room_type: str = None, 
                 style_tags: List[str] = None, color_tags: List[str] = None, 
                 project_url: str = None):
        self.houzz_id = houzz_id
        self.image_url = image_url
        self.room_type = room_type
        self.style_tags = style_tags or []
        self.color_tags = color_tags or []
        self.project_url = project_url

class HouzzCrawler:
    """Crawler for Houzz UK interior design scenes"""
    
    def __init__(self):
        self.base_url = "https://www.houzz.co.uk"
        self.photos_url = f"{self.base_url}/photos/"
        self.session = None
        
        # User agents to rotate
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
        ]
        
        # Room type mappings
        self.room_types = {
            'living-room': 'living_room',
            'bedroom': 'bedroom',
            'kitchen': 'kitchen',
            'bathroom': 'bathroom',
            'dining-room': 'dining_room',
            'home-office': 'office',
            'kids-room': 'kids_room',
            'outdoor': 'outdoor'
        }
        
    async def scrape_scenes(self, limit: int = 100, room_types: Optional[List[str]] = None) -> List[SceneData]:
        """
        Scrape interior design scenes from Houzz
        
        Args:
            limit: Maximum number of scenes to scrape
            room_types: Filter by specific room types
            
        Returns:
            List of SceneData objects
        """
        scenes = []
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
                )
                
                context = await browser.new_context(
                    user_agent=self.user_agents[0],
                    viewport={'width': 1920, 'height': 1080}
                )
                
                page = await context.new_page()
                
                # Handle different room types or scrape all
                target_room_types = room_types if room_types else list(self.room_types.keys())
                
                for room_type in target_room_types:
                    if len(scenes) >= limit:
                        break
                        
                    room_scenes = await self._scrape_room_type(page, room_type, limit - len(scenes))
                    scenes.extend(room_scenes)
                    
                    # Add delay between room types
                    await asyncio.sleep(2)
                
                await browser.close()
                
        except Exception as e:
            logger.error("Scene scraping failed", error=str(e))
        
        logger.info(f"Scraped {len(scenes)} scenes from Houzz")
        return scenes[:limit]
    
    async def _scrape_room_type(self, page, room_type: str, limit: int) -> List[SceneData]:
        """Scrape scenes for a specific room type"""
        scenes = []
        
        try:
            # Navigate to room type page
            room_url = f"{self.photos_url}{room_type}/"
            await page.goto(room_url, wait_until='networkidle')
            
            # Wait for images to load
            await page.wait_for_selector('img[data-testid="photo-grid-image"]', timeout=10000)
            
            # Scroll to load more images
            await self._scroll_for_more_images(page, limit)
            
            # Extract image data
            image_elements = await page.query_selector_all('img[data-testid="photo-grid-image"]')
            
            for element in image_elements[:limit]:
                if len(scenes) >= limit:
                    break
                    
                try:
                    # Get image URL
                    img_src = await element.get_attribute('src')
                    if not img_src:
                        continue
                    
                    # Get high-res version
                    high_res_url = self._get_high_res_url(img_src)
                    
                    # Get parent link for more metadata
                    parent_link = await element.evaluate('el => el.closest("a")')
                    project_url = None
                    if parent_link:
                        href = await parent_link.get_attribute('href')
                        if href:
                            project_url = urljoin(self.base_url, href)
                    
                    # Extract Houzz ID from URL
                    houzz_id = self._extract_houzz_id(img_src, project_url)
                    
                    if houzz_id and high_res_url:
                        scene = SceneData(
                            houzz_id=houzz_id,
                            image_url=high_res_url,
                            room_type=self.room_types.get(room_type, room_type),
                            project_url=project_url
                        )
                        
                        # Try to extract additional metadata
                        await self._enrich_scene_metadata(page, scene, element)
                        
                        scenes.append(scene)
                        
                except Exception as e:
                    logger.warning(f"Failed to extract scene data from element", error=str(e))
                    continue
            
        except Exception as e:
            logger.error(f"Failed to scrape room type {room_type}", error=str(e))
        
        return scenes
    
    async def _scroll_for_more_images(self, page, target_count: int):
        """Scroll page to trigger infinite loading"""
        current_count = 0
        scroll_attempts = 0
        max_scroll_attempts = 10
        
        while current_count < target_count and scroll_attempts < max_scroll_attempts:
            # Scroll to bottom
            await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            
            # Wait for new images to load
            await asyncio.sleep(2)
            
            # Count current images
            image_elements = await page.query_selector_all('img[data-testid="photo-grid-image"]')
            new_count = len(image_elements)
            
            if new_count == current_count:
                scroll_attempts += 1
            else:
                scroll_attempts = 0
                current_count = new_count
            
            logger.debug(f"Loaded {current_count} images, target: {target_count}")
    
    async def _enrich_scene_metadata(self, page, scene: SceneData, element):
        """Extract additional metadata like style and color tags"""
        try:
            # Look for style and color tags in nearby elements
            parent_container = await element.evaluate('el => el.closest("[data-testid]")')
            
            if parent_container:
                # Try to extract style information from surrounding text
                container_html = await parent_container.inner_html()
                
                # Extract style tags using regex
                style_matches = re.findall(r'style["\']?\s*:\s*["\']?([^"\']+)', container_html, re.IGNORECASE)
                if style_matches:
                    scene.style_tags = [style.strip().lower() for style in style_matches[:3]]
                
                # Extract color information
                color_matches = re.findall(r'color["\']?\s*:\s*["\']?([^"\']+)', container_html, re.IGNORECASE)
                if color_matches:
                    scene.color_tags = [color.strip().lower() for color in color_matches[:3]]
                    
        except Exception as e:
            logger.debug("Failed to enrich metadata", error=str(e))
    
    def _get_high_res_url(self, img_src: str) -> str:
        """Convert thumbnail URL to high-resolution version"""
        if not img_src:
            return img_src
            
        # Houzz image URL patterns
        # Replace size parameters with high-res versions
        high_res_url = img_src
        
        # Remove size constraints
        high_res_url = re.sub(r'/w_\d+,h_\d+,c_fill/', '/w_1920,h_1080,c_fill/', high_res_url)
        high_res_url = re.sub(r'/rs_\d+x\d+/', '/rs_1920x1080/', high_res_url)
        
        # Ensure HTTPS
        if high_res_url.startswith('//'):
            high_res_url = 'https:' + high_res_url
        
        return high_res_url
    
    def _extract_houzz_id(self, img_src: str, project_url: str = None) -> str:
        """Extract unique Houzz ID from image URL or project URL"""
        # Try to extract from image URL first
        if img_src:
            # Look for patterns like /v1234567890/ or similar
            match = re.search(r'/v(\d+)/', img_src)
            if match:
                return f"houzz_{match.group(1)}"
            
            # Try other patterns
            match = re.search(r'/([\w-]+)_\d+/', img_src)
            if match:
                return f"houzz_{match.group(1)}"
        
        # Try to extract from project URL
        if project_url:
            match = re.search(r'/(\d+)/?$', project_url)
            if match:
                return f"houzz_{match.group(1)}"
        
        # Generate from image URL hash as fallback
        if img_src:
            url_hash = hashlib.md5(img_src.encode()).hexdigest()[:8]
            return f"houzz_{url_hash}"
        
        return None
    
    async def download_image(self, image_url: str, max_size_mb: int = 10) -> str:
        """
        Download image to temporary file
        
        Args:
            image_url: URL of the image to download
            max_size_mb: Maximum file size in MB
            
        Returns:
            Path to downloaded image file
        """
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            headers = {
                'User-Agent': self.user_agents[0],
                'Referer': self.base_url
            }
            
            async with self.session.get(image_url, headers=headers) as response:
                if response.status != 200:
                    logger.warning(f"Failed to download image: {response.status}")
                    return None
                
                # Check content length
                content_length = response.headers.get('Content-Length')
                if content_length and int(content_length) > max_size_mb * 1024 * 1024:
                    logger.warning(f"Image too large: {content_length} bytes")
                    return None
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                    temp_path = f.name
                
                # Download image
                async with aiofiles.open(temp_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
                
                logger.debug(f"Downloaded image to {temp_path}")
                return temp_path
                
        except Exception as e:
            logger.error(f"Image download failed for {image_url}", error=str(e))
            return None
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None