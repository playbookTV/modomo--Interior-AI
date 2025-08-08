"""
Product catalog parser for affiliate feeds and retailer JSON-LD data
"""

import asyncio
import aiohttp
import aiofiles
from typing import List, Dict, Optional, Any
import json
import xml.etree.ElementTree as ET
import csv
import re
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()

@dataclass
class ProductData:
    """Product information extracted from catalogs"""
    sku: str
    name: str
    category: str
    price_gbp: float
    brand: str
    material: str
    dimensions: Dict[str, float]  # {'width': 120, 'height': 80, 'depth': 45} in mm
    image_url: str
    product_url: str
    description: str
    retailer: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sku': self.sku,
            'name': self.name,
            'category': self.category,
            'price_gbp': self.price_gbp,
            'brand': self.brand,
            'material': self.material,
            'dimensions': self.dimensions,
            'image_url': self.image_url,
            'product_url': self.product_url,
            'description': self.description,
            'retailer': self.retailer
        }

class CatalogParser:
    """Parse product catalogs from various sources"""
    
    def __init__(self):
        self.session = None
        
        # Category mappings from external to Modomo taxonomy
        self.category_mappings = {
            # IKEA/common mappings
            'sofas': 'sofa',
            'sofa': 'sofa',
            'armchairs': 'armchair',
            'armchair': 'armchair',
            'chairs': 'dining_chair',
            'dining chairs': 'dining_chair',
            'office chairs': 'armchair',
            'tables': 'coffee_table',
            'coffee tables': 'coffee_table',
            'dining tables': 'dining_table',
            'side tables': 'side_table',
            'console tables': 'console_table',
            'desks': 'desk',
            'bookcases': 'bookshelf',
            'shelving': 'bookshelf',
            'wardrobes': 'wardrobe',
            'cabinets': 'cabinet',
            'storage': 'cabinet',
            'beds': 'bed_frame',
            'bed frames': 'bed_frame',
            'mattresses': 'mattress',
            'bedside tables': 'nightstand',
            'lamps': 'table_lamp',
            'floor lamps': 'floor_lamp',
            'table lamps': 'table_lamp',
            'pendant lights': 'pendant_light',
            'rugs': 'rug',
            'carpets': 'rug',
            'curtains': 'curtains',
            'blinds': 'curtains',
            'cushions': 'pillow',
            'pillows': 'pillow',
            'throws': 'blanket',
            'mirrors': 'mirror',
            'wall art': 'wall_art',
            'plants': 'plant'
        }
        
        # Supported retailers
        self.retailers = {
            'ikea': {
                'base_url': 'https://www.ikea.com/gb/en/',
                'json_ld_selector': 'script[type="application/ld+json"]'
            },
            'made': {
                'base_url': 'https://www.made.com/',
                'json_ld_selector': 'script[type="application/ld+json"]'
            },
            'wayfair': {
                'base_url': 'https://www.wayfair.co.uk/',
                'json_ld_selector': 'script[type="application/ld+json"]'
            }
        }
    
    async def parse_affiliate_feed(self, feed_url: str, feed_format: str = 'csv') -> List[ProductData]:
        """
        Parse affiliate network product feed
        
        Args:
            feed_url: URL to the product feed
            feed_format: 'csv', 'xml', or 'json'
            
        Returns:
            List of parsed product data
        """
        products = []
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(feed_url) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch feed: {response.status}")
                    return products
                
                content = await response.text()
                
                if feed_format == 'csv':
                    products = self._parse_csv_feed(content)
                elif feed_format == 'xml':
                    products = self._parse_xml_feed(content)
                elif feed_format == 'json':
                    products = self._parse_json_feed(content)
                
        except Exception as e:
            logger.error(f"Feed parsing failed for {feed_url}", error=str(e))
        
        logger.info(f"Parsed {len(products)} products from {feed_url}")
        return products
    
    def _parse_csv_feed(self, content: str) -> List[ProductData]:
        """Parse CSV affiliate feed"""
        products = []
        
        try:
            # Parse CSV content
            lines = content.strip().split('\n')
            if not lines:
                return products
            
            # Get header row
            header = [col.strip().lower() for col in lines[0].split(',')]
            
            for line in lines[1:]:
                try:
                    values = [val.strip().strip('"') for val in line.split(',')]
                    row = dict(zip(header, values))
                    
                    # Extract product data
                    product = self._extract_product_from_row(row, 'csv')
                    if product:
                        products.append(product)
                        
                except Exception as e:
                    logger.debug(f"Failed to parse CSV row", error=str(e))
                    continue
                    
        except Exception as e:
            logger.error("CSV parsing failed", error=str(e))
        
        return products
    
    def _parse_xml_feed(self, content: str) -> List[ProductData]:
        """Parse XML affiliate feed"""
        products = []
        
        try:
            root = ET.fromstring(content)
            
            # Common XML structures for affiliate feeds
            product_elements = root.findall('.//product') or root.findall('.//item')
            
            for element in product_elements:
                try:
                    product_data = {}
                    
                    # Extract all child elements
                    for child in element:
                        product_data[child.tag.lower()] = child.text
                    
                    product = self._extract_product_from_row(product_data, 'xml')
                    if product:
                        products.append(product)
                        
                except Exception as e:
                    logger.debug("Failed to parse XML product", error=str(e))
                    continue
                    
        except Exception as e:
            logger.error("XML parsing failed", error=str(e))
        
        return products
    
    def _parse_json_feed(self, content: str) -> List[ProductData]:
        """Parse JSON affiliate feed"""
        products = []
        
        try:
            data = json.loads(content)
            
            # Handle different JSON structures
            product_list = data
            if isinstance(data, dict):
                if 'products' in data:
                    product_list = data['products']
                elif 'items' in data:
                    product_list = data['items']
                else:
                    product_list = list(data.values())
            
            for item in product_list:
                if isinstance(item, dict):
                    product = self._extract_product_from_row(item, 'json')
                    if product:
                        products.append(product)
                        
        except Exception as e:
            logger.error("JSON parsing failed", error=str(e))
        
        return products
    
    def _extract_product_from_row(self, row: Dict[str, str], format_type: str) -> Optional[ProductData]:
        """Extract ProductData from a data row"""
        try:
            # Common field mappings
            field_mappings = {
                'sku': ['sku', 'id', 'product_id', 'productid', 'item_id'],
                'name': ['name', 'title', 'product_name', 'productname', 'description'],
                'category': ['category', 'category_name', 'product_category', 'type'],
                'price': ['price', 'cost', 'amount', 'retail_price', 'sale_price'],
                'brand': ['brand', 'manufacturer', 'vendor', 'brand_name'],
                'image_url': ['image_url', 'image', 'img_url', 'picture', 'photo'],
                'product_url': ['url', 'link', 'product_url', 'permalink'],
                'description': ['description', 'summary', 'details', 'product_description'],
                'material': ['material', 'materials', 'fabric', 'finish']
            }
            
            extracted = {}
            
            # Extract fields using mappings
            for field, possible_keys in field_mappings.items():
                for key in possible_keys:
                    if key in row and row[key]:
                        extracted[field] = row[key]
                        break
            
            # Validate required fields
            if not all(k in extracted for k in ['sku', 'name', 'price', 'image_url']):
                return None
            
            # Parse and validate price
            try:
                price_str = re.sub(r'[^\d.]', '', extracted['price'])
                price_gbp = float(price_str)
            except (ValueError, KeyError):
                return None
            
            # Map category to taxonomy
            category = self._map_category(extracted.get('category', ''))
            if not category:
                return None  # Skip products not in our taxonomy
            
            # Parse dimensions if available
            dimensions = self._parse_dimensions(row)
            
            # Create product data
            product = ProductData(
                sku=extracted['sku'],
                name=extracted['name'],
                category=category,
                price_gbp=price_gbp,
                brand=extracted.get('brand', 'Unknown'),
                material=extracted.get('material', ''),
                dimensions=dimensions,
                image_url=extracted['image_url'],
                product_url=extracted.get('product_url', ''),
                description=extracted.get('description', ''),
                retailer=self._extract_retailer(extracted.get('product_url', ''))
            )
            
            return product
            
        except Exception as e:
            logger.debug("Product extraction failed", error=str(e))
            return None
    
    def _map_category(self, category: str) -> Optional[str]:
        """Map external category to Modomo taxonomy"""
        if not category:
            return None
            
        category_lower = category.lower().strip()
        
        # Direct mapping
        if category_lower in self.category_mappings:
            return self.category_mappings[category_lower]
        
        # Partial matching
        for external_cat, internal_cat in self.category_mappings.items():
            if external_cat in category_lower or category_lower in external_cat:
                return internal_cat
        
        return None
    
    def _parse_dimensions(self, row: Dict[str, str]) -> Dict[str, float]:
        """Parse product dimensions from various fields"""
        dimensions = {}
        
        # Look for dimension fields
        dimension_fields = ['width', 'height', 'depth', 'length', 'diameter', 'dimensions']
        
        for field in dimension_fields:
            if field in row and row[field]:
                try:
                    # Extract numeric value (assuming cm, convert to mm)
                    value_str = re.sub(r'[^\d.]', '', row[field])
                    if value_str:
                        value_cm = float(value_str)
                        dimensions[field] = value_cm * 10  # Convert to mm
                except (ValueError, TypeError):
                    continue
        
        # Try to parse combined dimensions string like "120x80x45cm"
        for key, value in row.items():
            if 'dimension' in key.lower() and isinstance(value, str):
                dim_match = re.search(r'(\d+\.?\d*)\s*x\s*(\d+\.?\d*)\s*x\s*(\d+\.?\d*)', value)
                if dim_match:
                    dimensions['width'] = float(dim_match.group(1)) * 10
                    dimensions['height'] = float(dim_match.group(2)) * 10
                    dimensions['depth'] = float(dim_match.group(3)) * 10
                    break
        
        return dimensions
    
    def _extract_retailer(self, product_url: str) -> str:
        """Extract retailer name from product URL"""
        if not product_url:
            return 'Unknown'
        
        try:
            domain = urlparse(product_url).netloc.lower()
            
            if 'ikea' in domain:
                return 'IKEA'
            elif 'made' in domain:
                return 'Made.com'
            elif 'wayfair' in domain:
                return 'Wayfair'
            elif 'amazon' in domain:
                return 'Amazon'
            else:
                return domain.replace('www.', '').split('.')[0].title()
                
        except Exception:
            return 'Unknown'
    
    async def scrape_json_ld_products(self, retailer: str, product_urls: List[str]) -> List[ProductData]:
        """
        Scrape product data using JSON-LD structured data
        
        Args:
            retailer: Retailer name ('ikea', 'made', 'wayfair')
            product_urls: List of product page URLs
            
        Returns:
            List of parsed products
        """
        products = []
        
        if retailer not in self.retailers:
            logger.error(f"Unsupported retailer: {retailer}")
            return products
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            for url in product_urls:
                try:
                    async with self.session.get(url) as response:
                        if response.status != 200:
                            continue
                        
                        html = await response.text()
                        product = await self._extract_json_ld_product(html, retailer)
                        
                        if product:
                            products.append(product)
                        
                        # Rate limiting
                        await asyncio.sleep(1)
                        
                except Exception as e:
                    logger.warning(f"Failed to scrape {url}", error=str(e))
                    continue
                    
        except Exception as e:
            logger.error("JSON-LD scraping failed", error=str(e))
        
        logger.info(f"Scraped {len(products)} products via JSON-LD from {retailer}")
        return products
    
    async def _extract_json_ld_product(self, html: str, retailer: str) -> Optional[ProductData]:
        """Extract product data from JSON-LD structured data"""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find JSON-LD scripts
            json_scripts = soup.find_all('script', type='application/ld+json')
            
            for script in json_scripts:
                try:
                    data = json.loads(script.string)
                    
                    # Handle different JSON-LD structures
                    if isinstance(data, list):
                        for item in data:
                            product = self._parse_json_ld_item(item, retailer)
                            if product:
                                return product
                    else:
                        product = self._parse_json_ld_item(data, retailer)
                        if product:
                            return product
                            
                except json.JSONDecodeError:
                    continue
                    
        except Exception as e:
            logger.debug("JSON-LD extraction failed", error=str(e))
        
        return None
    
    def _parse_json_ld_item(self, data: Dict, retailer: str) -> Optional[ProductData]:
        """Parse individual JSON-LD item"""
        try:
            # Look for Product schema
            if data.get('@type') != 'Product':
                return None
            
            name = data.get('name', '')
            sku = data.get('sku', data.get('productID', ''))
            
            if not name or not sku:
                return None
            
            # Extract price
            price_gbp = 0.0
            offers = data.get('offers', {})
            if isinstance(offers, list):
                offers = offers[0]
            
            if offers:
                price_str = str(offers.get('price', '0'))
                price_gbp = float(re.sub(r'[^\d.]', '', price_str))
            
            # Extract category
            category = data.get('category', '')
            mapped_category = self._map_category(category)
            if not mapped_category:
                return None
            
            # Extract other fields
            brand = data.get('brand', {})
            if isinstance(brand, dict):
                brand = brand.get('name', 'Unknown')
            
            image_url = ''
            image = data.get('image', [])
            if isinstance(image, list) and image:
                image_url = image[0]
            elif isinstance(image, str):
                image_url = image
            
            description = data.get('description', '')
            product_url = data.get('url', '')
            
            product = ProductData(
                sku=sku,
                name=name,
                category=mapped_category,
                price_gbp=price_gbp,
                brand=brand,
                material='',  # Usually not in JSON-LD
                dimensions={},  # Would need additional parsing
                image_url=image_url,
                product_url=product_url,
                description=description,
                retailer=retailer.title()
            )
            
            return product
            
        except Exception as e:
            logger.debug("JSON-LD item parsing failed", error=str(e))
            return None
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None