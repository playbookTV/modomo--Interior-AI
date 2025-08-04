# ReRoom: Technical Implementation Guide

## 🏗️ React Native Architecture

### Core Stack
```
React Native 0.73+ (New Architecture)
├── Expo SDK 50+ (Managed workflow)
├── TypeScript (Strict mode)
├── Zustand (State management)
├── React Query (Server state)
├── React Navigation 6 (Navigation)
├── Reanimated 3 (Animations)
└── FlashList (Performance lists)
```

### Key Native Libraries
```javascript
// Camera & Media
"react-native-vision-camera": "^3.8.0",
"react-native-image-picker": "^7.1.0",
"react-native-image-resizer": "^3.0.7",
"@react-native-camera-roll/camera-roll": "^7.4.0",

// AI/ML Processing
"react-native-pytorch-core": "^0.2.4",
"@tensorflow/tfjs-react-native": "^0.8.0",
"react-native-fast-image": "^8.6.3",

// Networking & Storage
"@react-native-async-storage/async-storage": "^1.21.0",
"react-native-mmkv": "^2.11.0", // Fast storage
"@react-native-community/netinfo": "^11.2.1",

// Payments & Analytics
"react-native-purchases": "^7.17.0",
"@react-native-firebase/analytics": "^19.0.1",
"react-native-branch": "^6.6.0" // Deep linking
```

## 📱 App Architecture Pattern

### File Structure
```
src/
├── components/          # Reusable UI components
│   ├── camera/         # Camera controls
│   ├── ai/            # AI result displays
│   └── shopping/      # Product components
├── screens/            # Screen components
├── services/          # API & business logic
│   ├── camera.ts      # Camera utilities
│   ├── ai-pipeline.ts # AI processing
│   ├── product-api.ts # Product matching
│   └── analytics.ts   # Event tracking
├── stores/            # Zustand stores
├── types/             # TypeScript definitions
└── utils/             # Helper functions
```

### State Management Pattern
```typescript
// stores/app-store.ts
import { create } from 'zustand'
import { createJSONStorage, persist } from 'zustand/middleware'
import { MMKV } from 'react-native-mmkv'

const storage = new MMKV()

interface AppState {
  // Photo state
  currentPhoto: string | null
  photoQuality: 'low' | 'medium' | 'high'
  
  // AI processing
  isProcessing: boolean
  currentRender: AIRender | null
  renderQueue: string[]
  
  // Shopping
  selectedStyle: StyleType
  cartItems: CartItem[]
  savedRooms: SavedRoom[]
  
  // Actions
  setPhoto: (uri: string) => void
  startAIProcessing: (photo: string, style: StyleType) => void
  addToCart: (item: CartItem) => void
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      // State
      currentPhoto: null,
      photoQuality: 'medium',
      isProcessing: false,
      currentRender: null,
      renderQueue: [],
      selectedStyle: 'modern',
      cartItems: [],
      savedRooms: [],
      
      // Actions
      setPhoto: (uri) => set({ currentPhoto: uri }),
      
      startAIProcessing: async (photo, style) => {
        set({ isProcessing: true })
        try {
          const result = await aiPipeline.processPhoto(photo, style)
          set({ currentRender: result, isProcessing: false })
        } catch (error) {
          set({ isProcessing: false })
          throw error
        }
      },
      
      addToCart: (item) => set(state => ({
        cartItems: [...state.cartItems, item]
      }))
    }),
    {
      name: 'reroom-storage',
      storage: createJSONStorage(() => ({
        setItem: (name, value) => storage.set(name, value),
        getItem: (name) => storage.getString(name) ?? null,
        removeItem: (name) => storage.delete(name),
      })),
      partialize: (state) => ({
        savedRooms: state.savedRooms,
        cartItems: state.cartItems,
      }),
    }
  )
)
```

## 📸 Camera Implementation

### Smart Photo Capture
```typescript
// services/camera.ts
import { Camera, useCameraDevice } from 'react-native-vision-camera'
import { runOnJS, useSharedValue } from 'react-native-reanimated'

interface PhotoQualityCheck {
  isWellLit: boolean
  hasGoodFocus: boolean
  containsFurniture: boolean
  confidence: number
}

class SmartCamera {
  async captureOptimizedPhoto(): Promise<string> {
    const device = useCameraDevice('back')
    
    // Configure for interior photography
    const photo = await this.camera.takePhoto({
      quality: 90,
      enableAutoStabilization: true,
      enableAutoRedEyeReduction: false,
      flash: 'auto',
      // HDR for better dynamic range in interiors
      enableAutoHDR: true,
    })
    
    // Real-time quality assessment
    const quality = await this.assessPhotoQuality(photo.path)
    
    if (quality.confidence < 0.7) {
      throw new Error('Photo quality too low. Try better lighting.')
    }
    
    return this.optimizeForAI(photo.path)
  }
  
  private async assessPhotoQuality(photoPath: string): Promise<PhotoQualityCheck> {
    // Use on-device ML for real-time feedback
    const model = await tf.loadLayersModel('bundled://models/photo-quality.json')
    
    const image = await this.loadImageTensor(photoPath)
    const prediction = model.predict(image) as tf.Tensor
    
    return {
      isWellLit: prediction.dataSync()[0] > 0.5,
      hasGoodFocus: prediction.dataSync()[1] > 0.5,
      containsFurniture: prediction.dataSync()[2] > 0.7,
      confidence: Math.min(...prediction.dataSync())
    }
  }
  
  private async optimizeForAI(photoPath: string): Promise<string> {
    // Resize to optimal dimensions for AI processing
    const optimized = await ImageResizer.createResizedImage(
      photoPath,
      1024, // Width
      768,  // Height
      'JPEG',
      85,   // Quality
      0,    // Rotation
      undefined,
      false,
      {
        mode: 'cover',
        onlyScaleDown: true,
      }
    )
    
    return optimized.uri
  }
}
```

### Real-time Camera Guidance
```typescript
// components/camera/GuidedCamera.tsx
import React, { useState, useCallback } from 'react'
import { Camera, useCameraDevice } from 'react-native-vision-camera'
import Animated, { useSharedValue, useAnimatedStyle } from 'react-native-reanimated'

interface CameraGuideOverlay {
  message: string
  type: 'info' | 'warning' | 'success'
  confidence: number
}

export const GuidedCamera: React.FC = () => {
  const device = useCameraDevice('back')
  const [guidance, setGuidance] = useState<CameraGuideOverlay>({
    message: 'Point camera at your room',
    type: 'info',
    confidence: 0
  })
  
  const confidenceValue = useSharedValue(0)
  
  const frameProcessor = useFrameProcessor((frame) => {
    'worklet'
    
    // Run lightweight object detection on-device
    const objects = detectFurniture(frame)
    const lighting = assessLighting(frame)
    
    runOnJS(updateGuidance)({
      objects: objects.length,
      lighting: lighting.score,
      confidence: Math.min(objects.length / 3, lighting.score)
    })
  }, [])
  
  const updateGuidance = useCallback(({ objects, lighting, confidence }) => {
    confidenceValue.value = confidence
    
    if (lighting < 0.3) {
      setGuidance({
        message: 'Need more light - open curtains or turn on lights',
        type: 'warning',
        confidence
      })
    } else if (objects < 2) {
      setGuidance({
        message: 'Move closer to furniture',
        type: 'warning',
        confidence
      })
    } else if (confidence > 0.8) {
      setGuidance({
        message: 'Perfect! Tap to capture',
        type: 'success',
        confidence
      })
    }
  }, [])
  
  const confidenceStyle = useAnimatedStyle(() => ({
    width: `${confidenceValue.value * 100}%`,
    backgroundColor: confidenceValue.value > 0.8 ? '#10B981' : '#EF4444'
  }))
  
  return (
    <View style={styles.container}>
      <Camera
        device={device}
        isActive={true}
        frameProcessor={frameProcessor}
        style={StyleSheet.absoluteFill}
      />
      
      {/* Guidance Overlay */}
      <View style={styles.guidanceOverlay}>
        <Text style={styles.guidanceText}>{guidance.message}</Text>
        <Animated.View style={[styles.confidenceBar, confidenceStyle]} />
      </View>
      
      {/* Room Frame Guide */}
      <View style={styles.frameGuide} />
    </View>
  )
}
```

## 🤖 AI Pipeline Implementation

### Backend AI Service Architecture
```python
# ai-service/main.py
from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from typing import List, Dict
import uuid

app = FastAPI(title="ReRoom AI Service")

class AIOrchestrator:
    def __init__(self):
        self.depth_model = self.load_depth_model()
        self.segmentation_model = self.load_segmentation_model()
        self.style_model = self.load_style_model()
        self.product_matcher = ProductMatcher()
        
    async def process_room_async(
        self, 
        image_data: bytes, 
        style: str,
        user_preferences: Dict
    ) -> Dict:
        """Full AI pipeline for room transformation"""
        
        # 1. Scene Understanding (parallel processing)
        depth_task = asyncio.create_task(
            self.extract_depth(image_data)
        )
        segmentation_task = asyncio.create_task(
            self.segment_objects(image_data)
        )
        
        depth_map, object_masks = await asyncio.gather(
            depth_task, segmentation_task
        )
        
        # 2. Style Transfer
        styled_image = await self.apply_style_transfer(
            image_data, depth_map, object_masks, style
        )
        
        # 3. Product Matching (parallel)
        products = await self.match_products_parallel(
            object_masks, styled_image, user_preferences
        )
        
        # 4. Quality Assessment
        quality_score = await self.assess_quality(styled_image)
        
        if quality_score < 0.7:
            # Retry with different parameters
            return await self.retry_with_fallback(
                image_data, style, user_preferences
            )
        
        return {
            "render_id": str(uuid.uuid4()),
            "styled_image": styled_image,
            "products": products,
            "confidence": quality_score,
            "processing_time": time.time() - start_time
        }

# Advanced depth estimation
class DepthEstimator:
    def __init__(self):
        self.model = pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Large",
            device="cuda"
        )
    
    async def extract_depth(self, image: PIL.Image) -> np.ndarray:
        """Extract high-quality depth map"""
        
        # Preprocess for better interior depth
        image = self.enhance_for_interiors(image)
        
        depth = self.model(image)["depth"]
        
        # Post-process depth map
        depth = self.smooth_depth(depth)
        depth = self.enhance_furniture_boundaries(depth)
        
        return depth.numpy()
    
    def enhance_for_interiors(self, image: PIL.Image) -> PIL.Image:
        """Optimize image for indoor depth estimation"""
        # Increase contrast in furniture regions
        # Reduce noise in wall areas
        # Enhance edge definition
        pass
```

### Advanced Style Transfer Pipeline
```python
# ai-service/style_transfer.py
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
import torch

class StyleTransferEngine:
    def __init__(self):
        # Multi-ControlNet setup for precise control
        self.controlnet_depth = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            torch_dtype=torch.float16
        )
        self.controlnet_canny = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0", 
            torch_dtype=torch.float16
        )
        
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=[self.controlnet_depth, self.controlnet_canny],
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")
        
        # Load custom interior design LoRAs
        self.pipe.load_lora_weights("./models/interior-design-lora")
        
    async def transform_room(
        self,
        image: PIL.Image,
        depth_map: np.ndarray,
        style: str,
        objects_to_keep: List[str]
    ) -> PIL.Image:
        
        # Generate style-specific prompt
        prompt = self.build_style_prompt(style, objects_to_keep)
        
        # Prepare control inputs
        canny_image = self.extract_canny_edges(image)
        depth_image = self.prepare_depth_controlnet(depth_map)
        
        # Multi-stage generation for better quality
        result = await self.multi_stage_generation(
            prompt=prompt,
            image=image,
            control_images=[depth_image, canny_image],
            style=style
        )
        
        return result
    
    def build_style_prompt(self, style: str, keep_objects: List[str]) -> str:
        """Generate contextual prompts for each style"""
        
        style_prompts = {
            "modern": "modern minimalist interior, clean lines, neutral colors, contemporary furniture",
            "japandi": "japandi style interior, natural wood, wabi-sabi aesthetic, minimal decoration",
            "boho": "bohemian interior, warm textiles, plants, eclectic vintage furniture",
            "scandinavian": "scandinavian interior, light wood, white walls, cozy hygge atmosphere"
        }
        
        base_prompt = style_prompts.get(style, "stylish interior design")
        
        # Add object preservation instructions
        keep_instruction = f", keep existing {', '.join(keep_objects)}"
        
        return f"{base_prompt}{keep_instruction}, professional interior photography, high quality, detailed"
    
    async def multi_stage_generation(self, **kwargs) -> PIL.Image:
        """Two-stage generation for higher quality"""
        
        # Stage 1: Layout and major elements
        rough_result = self.pipe(
            **kwargs,
            num_inference_steps=20,
            guidance_scale=7.5,
            controlnet_conditioning_scale=[0.8, 0.6]
        ).images[0]
        
        # Stage 2: Refinement with img2img
        refined_result = self.refine_with_img2img(
            rough_result, kwargs["prompt"]
        )
        
        return refined_result
```

### Enhanced Product Matching System with Multi-Retailer Optimization

```python
# ai-service/product_matcher.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict

class IntelligentProductMatcher:
    def __init__(self):
        # Multi-modal embedding model
        self.clip_model = SentenceTransformer('clip-ViT-B-32')
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load pre-computed product embeddings
        self.retail_index = faiss.read_index("./indices/retail_products.faiss")
        self.alternative_index = faiss.read_index("./indices/alternative_marketplace.faiss")
        
        # Multi-retailer API integrations
        self.amazon_api = AmazonProductAPI()
        self.ebay_api = eBayMarketplaceAPI()
        self.gumtree_api = GumtreeAPI()
        
    async def find_comprehensive_matches(
        self,
        object_masks: Dict,
        styled_image: PIL.Image,
        user_location: Dict,
        user_prefs: Dict
    ) -> Dict:
        """Find best options across all available retailers and marketplaces"""
        
        all_matches = {}
        
        for object_type, mask in object_masks.items():
            if object_type in ['sofa', 'chair', 'table', 'lamp', 'rug']:
                # Get retail matches (new items)
                retail_matches = await self.match_retail_products(
                    object_type, mask, styled_image, user_prefs
                )
                
                # Get alternative marketplace matches
                alternative_matches = await self.match_alternative_marketplaces(
                    object_type, mask, styled_image, user_location, user_prefs
                )
                
                # Combine and optimize for best value
                all_matches[object_type] = self.create_optimized_recommendations(
                    retail_matches, alternative_matches, user_prefs
                )
        
        return all_matches
    
    async def match_alternative_marketplaces(
        self,
        object_type: str,
        mask: np.ndarray,
        styled_image: PIL.Image,
        user_location: Dict,
        user_prefs: Dict
    ) -> List[Dict]:
        """Search alternative marketplaces for matching furniture"""
        
        # Extract object from styled image
        object_image = self.extract_masked_object(styled_image, mask)
        
        # Generate search query from visual analysis
        visual_description = self.generate_marketplace_query(object_image)
        
        # Search multiple alternative marketplaces in parallel
        search_tasks = []
        
        # eBay local pickup search
        search_tasks.append(self.ebay_api.search_local_furniture(
            query=visual_description,
            category=object_type,
            location=user_location,
            max_distance=user_prefs.get('max_distance', 25),
            local_pickup_only=True
        ))
        
        # Gumtree search
        search_tasks.append(self.gumtree_api.search_furniture(
            query=visual_description,
            category=object_type,
            location=user_location,
            max_distance=user_prefs.get('max_distance', 25)
        ))
        
        # Vinted search (for smaller furniture items)
        if object_type in ['lamp', 'decor', 'small_table']:
            search_tasks.append(self.vinted_api.search_home_goods(
                query=visual_description,
                location=user_location
            ))
        
        marketplace_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Filter and rank by style matching
        filtered_results = []
        for results in marketplace_results:
            if isinstance(results, Exception):
                continue
                
            for item in results:
                style_score = await self.calculate_style_match_score(
                    object_image, item['images']
                )
                
                if style_score > 0.65:  # Good style match threshold
                    filtered_results.append({
                        'item': item,
                        'source': item['marketplace'],
                        'style_score': style_score,
                        'price': item['price'],
                        'location': item.get('location'),
                        'distance_miles': item.get('distance', 0),
                        'estimated_savings': self.calculate_savings_vs_retail(item),
                        'pickup_effort': self.calculate_pickup_effort(item, user_location),
                        'seller_rating': item.get('seller_rating', 0),
                        'condition_score': self.assess_condition(item)
                    })
        
        return sorted(filtered_results, 
                     key=lambda x: x['estimated_savings'], reverse=True)
    
    def create_optimized_recommendations(
        self,
        retail_matches: List[Dict],
        alternative_matches: List[Dict],
        user_prefs: Dict
    ) -> Dict:
        """Create comprehensive shopping options optimized for user preferences"""
        
        # Option 1: Best retail (new) option
        best_retail = retail_matches[0] if retail_matches else None
        
        # Option 2: Best alternative marketplace option  
        best_alternative = alternative_matches[0] if alternative_matches else None
        
        # Option 3: Balanced option (best value considering convenience vs savings)
        balanced_option = self.find_balanced_option(
            retail_matches, alternative_matches, user_prefs
        )
        
        return {
            'retail_option': best_retail,
            'alternative_option': best_alternative,
            'balanced_option': balanced_option,
            'total_retail_savings': sum(m['savings'] for m in retail_matches[:3]),
            'total_alternative_savings': sum(m['estimated_savings'] for m in alternative_matches[:3]),
            'recommendation': self.recommend_best_approach(
                best_retail, best_alternative, balanced_option, user_prefs
            )
        }

class eBayMarketplaceAPI:
    def __init__(self):
        self.api_key = settings.EBAY_API_KEY
        self.base_url = "https://api.ebay.com/buy/browse/v1"
        
    async def search_local_furniture(
        self,
        query: str,
        category: str,
        location: Dict,
        max_distance: int = 25,
        local_pickup_only: bool = True
    ) -> List[Dict]:
        """Search eBay for local pickup furniture items"""
        
        # eBay browse API search
        search_params = {
            'q': query,
            'category_ids': self.get_furniture_category_id(category),
            'filter': f'conditionIds:{{1000|1500|2000|2500|3000}},itemLocationCountry:GB,maxDistance:{max_distance}mi,pickupCountryCodes:GB',
            'sort': 'price',
            'limit': 50
        }
        
        if local_pickup_only:
            search_params['filter'] += ',pickupOptions:{IN_STORE_PICKUP}'
        
        headers = {
            'Authorization': f'Bearer {self.get_access_token()}',
            'Content-Type': 'application/json'
        }
        
        try:
            response = await self.make_request(
                f"{self.base_url}/item_summary/search",
                params=search_params,
                headers=headers
            )
            return self.process_ebay_results(response.get('itemSummaries', []))
        except Exception as e:
            logger.error(f"eBay API error: {e}")
            return []
    
    def process_ebay_results(self, raw_results: List[Dict]) -> List[Dict]:
        """Process and enrich eBay search results"""
        
        processed = []
        for item in raw_results:
            processed_item = {
                'id': item['itemId'],
                'title': item['title'],
                'price': float(item['price']['value']),
                'currency': item['price']['currency'],
                'images': [img['imageUrl'] for img in item.get('image', [])],
                'location': self.extract_location_info(item.get('itemLocation', {})),
                'seller': self.extract_seller_info(item.get('seller', {})),
                'condition': item.get('condition', 'Used'),
                'marketplace': 'ebay',
                'item_url': item['itemWebUrl'],
                'pickup_available': 'IN_STORE_PICKUP' in item.get('pickupOptions', [])
            }
            
            processed.append(processed_item)
            
        return processed

class GumtreeAPI:
    def __init__(self):
        # Gumtree doesn't have official API, use web scraping with proper rate limiting
        self.scraper = GumtreeScraper()
        
    async def search_furniture(
        self,
        query: str,
        category: str,
        location: Dict,
        max_distance: int = 25
    ) -> List[Dict]:
        """Search Gumtree for furniture items"""
        
        try:
            # Use ethical web scraping with rate limiting
            results = await self.scraper.search_furniture_listings(
                query=query,
                location=location,
                max_distance=max_distance,
                category=category
            )
            
            return self.process_gumtree_results(results)
        except Exception as e:
            logger.error(f"Gumtree search error: {e}")
            return []
    
    def process_gumtree_results(self, raw_results: List[Dict]) -> List[Dict]:
        """Process Gumtree search results"""
        
        processed = []
        for item in raw_results:
            processed_item = {
                'id': item['id'],
                'title': item['title'],
                'price': item['price'],
                'currency': 'GBP',
                'images': item['images'],
                'location': item['location'],
                'seller': item['seller_info'],
                'marketplace': 'gumtree',
                'item_url': item['url'],
                'posted_date': item['posted_date']
            }
            
            processed.append(processed_item)
            
        return processed
```

### Multi-Retailer Price Optimization Engine

```python
# services/price_optimizer.py
import asyncio
from typing import List, Dict, Tuple
from itertools import combinations

class MultiRetailerOptimizer:
    def __init__(self):
        self.retailers = {
            'amazon': AmazonAPI(),
            'temu': TemuAPI(), 
            'argos': ArgosAPI(),
            'ikea': IkeaAPI(),
            'wayfair': WayfairAPI(),
            'john_lewis': JohnLewisAPI(),
            'ebay': eBayMarketplaceAPI(),
            'gumtree': GumtreeAPI()
        }
        
    async def optimize_room_purchase(
        self,
        room_items: List[Dict],
        user_preferences: Dict
    ) -> Dict:
        """Find optimal combination of retailers for complete room purchase"""
        
        # Get all possible sources for each item
        item_sources = {}
        search_tasks = []
        
        for item in room_items:
            search_tasks.append(
                self.find_all_sources_for_item(item, user_preferences)
            )
        
        all_sources = await asyncio.gather(*search_tasks)
        
        for i, item in enumerate(room_items):
            item_sources[item['id']] = all_sources[i]
        
        # Find optimal combination
        optimal_combination = self.find_optimal_retailer_combination(
            item_sources, user_preferences
        )
        
        # Calculate logistics and total cost
        logistics_analysis = self.analyze_purchase_logistics(
            optimal_combination, user_preferences
        )
        
        return {
            'optimal_combination': optimal_combination,
            'total_cost': logistics_analysis['total_cost'],
            'total_savings': logistics_analysis['total_savings'],
            'delivery_analysis': logistics_analysis['delivery'],
            'payment_options': logistics_analysis['payment'],
            'alternative_combinations': self.generate_alternatives(item_sources)
        }
    
    async def find_all_sources_for_item(
        self,
        item: Dict,
        user_preferences: Dict
    ) -> List[Dict]:
        """Find all available sources for a single item"""
        
        search_tasks = []
        
        # Search each retailer
        for retailer_name, retailer_api in self.retailers.items():
            search_tasks.append(
                retailer_api.search_product(
                    query=item['search_query'],
                    category=item['category'],
                    style_requirements=item['style_match'],
                    user_prefs=user_preferences
                )
            )
        
        all_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Process and rank results
        valid_sources = []
        for i, results in enumerate(all_results):
            if isinstance(results, Exception):
                continue
                
            retailer_name = list(self.retailers.keys())[i]
            for result in results[:3]:  # Top 3 from each retailer
                result['retailer'] = retailer_name
                result['style_match_score'] = self.calculate_style_match(
                    result, item['target_style']
                )
                valid_sources.append(result)
        
        # Sort by best combination of price, style match, and retailer reliability
        return sorted(valid_sources, key=self.calculate_source_score, reverse=True)
    
    def find_optimal_retailer_combination(
        self,
        item_sources: Dict[str, List[Dict]],
        user_preferences: Dict
    ) -> Dict:
        """Use optimization algorithm to find best retailer combination"""
        
        # Extract user priorities
        priority_savings = user_preferences.get('priority_savings', 0.7)
        priority_convenience = user_preferences.get('priority_convenience', 0.2)
        priority_quality = user_preferences.get('priority_quality', 0.1)
        
        best_combination = None
        best_score = 0
        
        # Generate all possible combinations (limited to reasonable size)
        item_ids = list(item_sources.keys())
        for combination in self.generate_combinations(item_sources):
            
            # Calculate total cost and logistics
            total_cost = sum(source['price'] for source in combination.values())
            delivery_cost = self.calculate_delivery_cost(combination)
            convenience_score = self.calculate_convenience_score(combination)
            quality_score = self.calculate_quality_score(combination)
            
            # Calculate weighted score
            total_with_delivery = total_cost + delivery_cost
            savings_score = 1 - (total_with_delivery / self.calculate_single_retailer_cost(item_sources))
            
            combined_score = (
                priority_savings * savings_score +
                priority_convenience * convenience_score +
                priority_quality * quality_score
            )
            
            if combined_score > best_score:
                best_score = combined_score
                best_combination = combination
        
        return {
            'selected_sources': best_combination,
            'total_score': best_score,
            'breakdown': {
                'total_product_cost': sum(s['price'] for s in best_combination.values()),
                'delivery_cost': self.calculate_delivery_cost(best_combination),
                'estimated_delivery_time': self.calculate_delivery_time(best_combination)
            }
        }
``` {
            'optimized_route': optimized_route,
            'total_distance_miles': route_analysis['total_distance'],
            'total_time_minutes': route_analysis['total_time'],
            'estimated_fuel_cost': route_analysis['fuel_cost'],
            'pickup_schedule': self.generate_pickup_schedule(optimized_route),
            'net_savings': self.calculate_net_savings(
                marketplace_items, route_analysis['total_cost']
            )
        }
    
    async def coordinate_seller_communication(
        self,
        user_id: str,
        marketplace_items: List[Dict],
        pickup_schedule: Dict
    ) -> Dict:
        """Manage communication with multiple sellers"""
        
        coordination_results = []
        
        for item in marketplace_items:
            # Generate pickup request message
            message = self.generate_pickup_message(
                item, pickup_schedule[item['id']]
            )
            
            # Send message through Facebook Messenger API
            try:
                message_result = await self.send_facebook_message(
                    seller_id=item['seller']['id'],
                    message=message,
                    item_context=item
                )
                
                coordination_results.append({
                    'item_id': item['id'],
                    'seller_id': item['seller']['id'],
                    'message_sent': True,
                    'message_id': message_result['id'],
                    'expected_response_time': '2-4 hours'
                })
                
            except Exception as e:
                coordination_results.append({
                    'item_id': item['id'],
                    'seller_id': item['seller']['id'],
                    'message_sent': False,
                    'error': str(e),
                    'fallback_action': 'manual_contact_required'
                })
        
        return {
            'coordination_results': coordination_results,
            'success_rate': sum(1 for r in coordination_results if r['message_sent']) / len(coordination_results),
            'next_steps': self.generate_next_steps(coordination_results)
        }
```
```

## 📦 React Native Integration

### AI Processing Hook
```typescript
// hooks/useAIProcessing.ts
import { useMutation, useQuery } from '@tanstack/react-query'
import { useAppStore } from '../stores/app-store'

interface AIProcessingOptions {
  style: StyleType
  keepObjects?: string[]
  budget?: { min: number; max: number }
}

export const useAIProcessing = () => {
  const { currentPhoto, setCurrentRender } = useAppStore()
  
  const processingMutation = useMutation({
    mutationFn: async (options: AIProcessingOptions) => {
      if (!currentPhoto) throw new Error('No photo selected')
      
      // Upload photo and start processing
      const formData = new FormData()
      formData.append('image', {
        uri: currentPhoto,
        type: 'image/jpeg',
        name: 'room.jpg',
      } as any)
      formData.append('style', options.style)
      formData.append('preferences', JSON.stringify(options))
      
      const response = await fetch(`${API_BASE}/ai/process`, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      
      if (!response.ok) {
        throw new Error('AI processing failed')
      }
      
      return response.json()
    },
    
    onSuccess: (result) => {
      setCurrentRender(result)
      
      // Track successful render
      analytics.track('AI_Render_Success', {
        style: result.style,
        confidence: result.confidence,
        processing_time: result.processing_time,
        product_count: result.products.length,
      })
    },
    
    onError: (error) => {
      analytics.track('AI_Render_Failed', {
        error: error.message,
      })
    },
  })
  
  // Poll for results if processing is async
  const { data: processingStatus } = useQuery({
    queryKey: ['processing-status', processingMutation.data?.render_id],
    queryFn: async () => {
      if (!processingMutation.data?.render_id) return null
      
      const response = await fetch(
        `${API_BASE}/ai/status/${processingMutation.data.render_id}`
      )
      return response.json()
    },
    enabled: !!processingMutation.data?.render_id && processingMutation.isLoading,
    refetchInterval: 2000, // Poll every 2 seconds
  })
  
  return {
    processPhoto: processingMutation.mutate,
    isProcessing: processingMutation.isLoading,
    result: processingMutation.data,
    error: processingMutation.error,
    progress: processingStatus?.progress || 0,
  }
}
```

### Optimistic UI Updates
```typescript
// components/ai/RenderProgress.tsx
import React from 'react'
import Animated, { 
  useSharedValue, 
  useAnimatedStyle, 
  withTiming,
  withSequence,
  withDelay
} from 'react-native-reanimated'

interface RenderProgressProps {
  progress: number
  isProcessing: boolean
}

export const RenderProgress: React.FC<RenderProgressProps> = ({
  progress,
  isProcessing
}) => {
  const progressValue = useSharedValue(0)
  const glowOpacity = useSharedValue(0)
  
  React.useEffect(() => {
    if (isProcessing) {
      // Animate progress bar
      progressValue.value = withTiming(progress, { duration: 500 })
      
      // Pulse glow effect
      glowOpacity.value = withSequence(
        withTiming(1, { duration: 800 }),
        withTiming(0.3, { duration: 800 }),
        withDelay(200, withTiming(1, { duration: 800 }))
      )
    }
  }, [progress, isProcessing])
  
  const progressStyle = useAnimatedStyle(() => ({
    width: `${progressValue.value}%`,
  }))
  
  const glowStyle = useAnimatedStyle(() => ({
    opacity: glowOpacity.value,
  }))
  
  const getStageText = (progress: number) => {
    if (progress < 25) return 'Analyzing your room...'
    if (progress < 50) return 'Understanding layout...'
    if (progress < 75) return 'Applying style...'
    if (progress < 95) return 'Finding products...'
    return 'Almost done!'
  }
  
  return (
    <View style={styles.container}>
      <Text style={styles.stageText}>
        {getStageText(progress)}
      </Text>
      
      <View style={styles.progressContainer}>
        <Animated.View style={[styles.progressBar, progressStyle]} />
        <Animated.View style={[styles.progressGlow, glowStyle]} />
      </View>
      
      <Text style={styles.progressText}>
        {Math.round(progress)}% complete
      </Text>
    </View>
  )
}
```

## 🛒 Shopping Integration

### Product API Integration
```typescript
// services/product-api.ts
interface ProductSearchParams {
  category: string
  color?: string
  style?: string
  priceRange?: { min: number; max: number }
  region: 'UK' | 'US' | 'EU'
}

class ProductAPIService {
  private async searchAmazon(params: ProductSearchParams) {
    // Amazon Product Advertising API
    const amazonParams = {
      Keywords: `${params.category} ${params.color} ${params.style}`,
      SearchIndex: 'Home',
      MinPrice: params.priceRange?.min,
      MaxPrice: params.priceRange?.max,
      ResponseGroup: 'Images,ItemAttributes,Offers',
    }
    
    const response = await this.makeAmazonRequest(amazonParams)
    return this.normalizeAmazonResults(response)
  }
  
  private async searchTemu(params: ProductSearchParams) {
    // Temu API (if available) or web scraping
    const searchQuery = `${params.category} ${params.style} home furniture`
    
    // Use Puppeteer for reliable scraping
    const results = await this.scrapeTemuSearch(searchQuery, params.priceRange)
    return this.normalizeTemuResults(results)
  }
  
  async searchAllSources(params: ProductSearchParams): Promise<Product[]> {
    const [amazonResults, temuResults, wayfairResults] = await Promise.allSettled([
      this.searchAmazon(params),
      this.searchTemu(params),
      this.searchWayfair(params),
    ])
    
    const allResults = [
      ...this.getResults(amazonResults),
      ...this.getResults(temuResults),
      ...this.getResults(wayfairResults),
    ]
    
    // Sort by relevance + price + availability
    return this.rankProducts(allResults, params)
  }
  
  private rankProducts(products: Product[], params: ProductSearchParams): Product[] {
    return products
      .map(product => ({
        ...product,
        relevanceScore: this.calculateRelevanceScore(product, params)
      }))
      .sort((a, b) => b.relevanceScore - a.relevanceScore)
      .slice(0, 20) // Top 20 results
  }
  
  private calculateRelevanceScore(product: Product, params: ProductSearchParams): number {
    let score = 0
    
    // Visual similarity (from AI matching)
    score += product.visualSimilarity * 0.4
    
    // Price within range
    if (params.priceRange) {
      const priceScore = this.calculatePriceScore(product.price, params.priceRange)
      score += priceScore * 0.2
    }
    
    // Availability
    score += product.inStock ? 0.2 : 0
    
    // Reviews
    score += (product.rating / 5) * 0.1
    
    // Shipping speed
    score += product.fastShipping ? 0.1 : 0
    
    return score
  }
}
```

## 📊 Performance Optimization

### Image Optimization Pipeline
```typescript
// utils/image-optimization.ts
import { Image } from 'react-native'
import ImageResizer from 'react-native-image-resizer'

interface OptimizationOptions {
  maxWidth: number
  maxHeight: number
  quality: number
  format: 'JPEG' | 'PNG' | 'WEBP'
}

class ImageOptimizer {
  // Optimize for AI processing
  static async optimizeForAI(imageUri: string): Promise<string> {
    return ImageResizer.createResizedImage(
      imageUri,
      1024, // AI models work best with 1024px
      768,
      'JPEG',
      85, // Good quality/size balance
      0,
      undefined,
      false,
      {
        mode: 'cover',
        onlyScaleDown: true,
      }
    ).then(response => response.uri)
  }
  
  // Optimize for display
  static async optimizeForDisplay(imageUri: string): Promise<string> {
    const { width: screenWidth } = Dimensions.get('window')
    
    return ImageResizer.createResizedImage(
      imageUri,
      screenWidth * 2, // 2x for retina
      (screenWidth * 2) * 0.75, // 4:3 aspect ratio
      'JPEG',
      90,
      0,
      undefined,
      false,
      {
        mode: 'contain',
        onlyScaleDown: true,
      }
    ).then(response => response.uri)
  }
  
  // Progressive loading
  static async createThumbnail(imageUri: string): Promise<string> {
    return ImageResizer.createResizedImage(
      imageUri,
      200,
      150,
      'JPEG',
      70,
      0,
      undefined,
      false
    ).then(response => response.uri)
  }
}
```

### Memory Management
```typescript
// utils/memory-manager.ts
import { AppState, DeviceEventEmitter } from 'react-native'

class MemoryManager {
  private imageCache = new Map<string, string>()
  private maxCacheSize = 50 // Max cached images
  
  constructor() {
    AppState.addEventListener('change', this.handleAppStateChange)
    DeviceEventEmitter.addListener('memoryWarning', this.clearCache)
  }
  
  cacheImage(key: string, uri: string) {
    if (this.imageCache.size >= this.maxCacheSize) {
      // Remove oldest entries
      const firstKey = this.imageCache.keys().next().value
      this.imageCache.delete(firstKey)
    }
    
    this.imageCache.set(key, uri)
  }
  
  getCachedImage(key: string): string | undefined {
    return this.imageCache.get(key)
  }
  
  private handleAppStateChange = (nextAppState: string) => {
    if (nextAppState === 'background') {
      // Clear non-essential cache when app goes to background
      this.clearNonEssentialCache()
    }
  }
  
  private clearCache = () => {
    this.imageCache.clear()
    // Force garbage collection if available
    if (global.gc) {
      global.gc()
    }
  }
  
  private clearNonEssentialCache() {
    // Keep only the last 10 images
    const entries = Array.from(this.imageCache.entries())
    if (entries.length > 10) {
      const toKeep = entries.slice(-10)
      this.imageCache.clear()
      toKeep.forEach(([key, value]) => this.imageCache.set(key, value))
    }
  }
}
```

## 🚀 Deployment & Infrastructure

### Backend Deployment (Docker + Kubernetes)
```yaml
# k8s/ai-service-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: reroom-ai-service
  labels:
    app: reroom-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: reroom-ai
  template:
    metadata:
      labels:
        app: reroom-ai
    spec:
      containers:
      - name: ai-service
        image: reroom/ai-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: MODEL_CACHE_DIR
          value: "/models"
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: model-cache
          mountPath: /models
        - name: temp-storage
          mountPath: /tmp
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      - name: temp-storage
        emptyDir:
          sizeLimit: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: reroom-ai-service
spec:
  selector:
    app: reroom-ai
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Auto-scaling Configuration
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: reroom-ai-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: reroom-ai-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: gpu_utilization
      target:
        type: AverageValue
        averageValue: "75"
```

### CI/CD Pipeline
```yaml
# .github/workflows/deploy.yml
name: Deploy ReRoom
on:
  push:
    branches: [main]

jobs:
  test-mobile:
    runs-on: macos-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-node@v3
      with:
        node-version: '18'
    
    - name: Install dependencies
      run: |
        cd mobile
        npm ci
    
    - name: Run tests
      run: |
        cd mobile
        npm run test
    
    - name: Build iOS
      run: |
        cd mobile
        npx eas build --platform ios --non-interactive
        
    - name: Build Android
      run: |
        cd mobile
        npx eas build --platform android --non-interactive

  deploy-backend:
    runs-on: ubuntu-latest
    needs: test-mobile
    steps:
    - uses: actions/checkout@v3
    
    - name: Build AI Service Docker
      run: |
        docker build -t reroom/ai-service:${{ github.sha }} ./ai-service
        docker tag reroom/ai-service:${{ github.sha }} reroom/ai-service:latest
    
    - name: Push to Registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push reroom/ai-service:${{ github.sha }}
        docker push reroom/ai-service:latest
    
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/reroom-ai-service ai-service=reroom/ai-service:${{ github.sha }}
        kubectl rollout status deployment/reroom-ai-service
```

## 📱 React Native App Structure

### Navigation Setup
```typescript
// navigation/AppNavigator.tsx
import React from 'react'
import { NavigationContainer } from '@react-navigation/native'
import { createNativeStackNavigator } from '@react-navigation/native-stack'
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs'

// Screens
import { CameraScreen } from '../screens/CameraScreen'
import { StyleSelectionScreen } from '../screens/StyleSelectionScreen'
import { ProcessingScreen } from '../screens/ProcessingScreen'
import { ResultScreen } from '../screens/ResultScreen'
import { ShoppingScreen } from '../screens/ShoppingScreen'
import { SavedRoomsScreen } from '../screens/SavedRoomsScreen'

const Stack = createNativeStackNavigator()
const Tab = createBottomTabNavigator()

const HomeStack = () => (
  <Stack.Navigator 
    initialRouteName="Camera"
    screenOptions={{
      headerShown: false,
      animation: 'slide_from_right'
    }}
  >
    <Stack.Screen name="Camera" component={CameraScreen} />
    <Stack.Screen name="StyleSelection" component={StyleSelectionScreen} />
    <Stack.Screen name="Processing" component={ProcessingScreen} />
    <Stack.Screen name="Result" component={ResultScreen} />
    <Stack.Screen name="Shopping" component={ShoppingScreen} />
  </Stack.Navigator>
)

export const AppNavigator = () => (
  <NavigationContainer>
    <Tab.Navigator
      screenOptions={{
        tabBarStyle: {
          backgroundColor: '#000',
          borderTopColor: '#333',
        },
        tabBarActiveTintColor: '#fff',
        tabBarInactiveTintColor: '#666',
      }}
    >
      <Tab.Screen 
        name="Home" 
        component={HomeStack}
        options={{
          tabBarIcon: ({ color }) => <CameraIcon color={color} />,
          headerShown: false
        }}
      />
      <Tab.Screen 
        name="Saved" 
        component={SavedRoomsScreen}
        options={{
          tabBarIcon: ({ color }) => <BookmarkIcon color={color} />
        }}
      />
    </Tab.Navigator>
  </NavigationContainer>
)
```

### Core Screens Implementation
```typescript
// screens/CameraScreen.tsx
import React, { useState, useRef } from 'react'
import { View, Text, StyleSheet, Alert } from 'react-native'
import { Camera, useCameraDevice } from 'react-native-vision-camera'
import { GuidedCamera } from '../components/camera/GuidedCamera'
import { useAppStore } from '../stores/app-store'

export const CameraScreen: React.FC = ({ navigation }) => {
  const device = useCameraDevice('back')
  const cameraRef = useRef<Camera>(null)
  const { setPhoto } = useAppStore()
  
  const [hasPermission, setHasPermission] = useState(false)
  
  React.useEffect(() => {
    checkCameraPermission()
  }, [])
  
  const checkCameraPermission = async () => {
    const permission = await Camera.getCameraPermissionStatus()
    if (permission === 'authorized') {
      setHasPermission(true)
    } else {
      const newPermission = await Camera.requestCameraPermission()
      setHasPermission(newPermission === 'authorized')
    }
  }
  
  const capturePhoto = async () => {
    try {
      if (!cameraRef.current) return
      
      const photo = await cameraRef.current.takePhoto({
        quality: 90,
        enableAutoHDR: true,
        enableAutoStabilization: true,
      })
      
      // Optimize image for AI processing
      const optimizedUri = await ImageOptimizer.optimizeForAI(photo.path)
      setPhoto(optimizedUri)
      
      // Navigate to style selection
      navigation.navigate('StyleSelection')
      
    } catch (error) {
      Alert.alert('Error', 'Failed to capture photo. Please try again.')
    }
  }
  
  if (!hasPermission) {
    return (
      <View style={styles.permissionContainer}>
        <Text style={styles.permissionText}>
          Camera permission required to take photos of your room
        </Text>
      </View>
    )
  }
  
  if (!device) {
    return (
      <View style={styles.errorContainer}>
        <Text style={styles.errorText}>No camera device found</Text>
      </View>
    )
  }
  
  return (
    <View style={styles.container}>
      <GuidedCamera
        ref={cameraRef}
        device={device}
        onCapture={capturePhoto}
      />
      
      {/* UI Overlays */}
      <View style={styles.topOverlay}>
        <Text style={styles.instructionText}>
          Point your camera at the room you want to redesign
        </Text>
      </View>
      
      <View style={styles.bottomOverlay}>
        <TouchableOpacity 
          style={styles.captureButton}
          onPress={capturePhoto}
        >
          <View style={styles.captureButtonInner} />
        </TouchableOpacity>
      </View>
    </View>
  )
}
```

### Processing Screen with Real-time Updates
```typescript
// screens/ProcessingScreen.tsx
import React, { useEffect } from 'react'
import { View, Text, StyleSheet } from 'react-native'
import { useAIProcessing } from '../hooks/useAIProcessing'
import { RenderProgress } from '../components/ai/RenderProgress'
import { useAppStore } from '../stores/app-store'

export const ProcessingScreen: React.FC = ({ navigation, route }) => {
  const { style } = route.params
  const { currentPhoto } = useAppStore()
  const { processPhoto, isProcessing, result, error, progress } = useAIProcessing()
  
  useEffect(() => {
    if (currentPhoto) {
      processPhoto({ style })
    }
  }, [currentPhoto, style])
  
  useEffect(() => {
    if (result) {
      // Navigate to result screen with a slight delay for UX
      setTimeout(() => {
        navigation.replace('Result', { result })
      }, 1000)
    }
  }, [result])
  
  useEffect(() => {
    if (error) {
      Alert.alert(
        'Processing Failed', 
        'We had trouble processing your photo. Please try again.',
        [
          { text: 'Try Again', onPress: () => navigation.goBack() },
          { text: 'Cancel', onPress: () => navigation.navigate('Camera') }
        ]
      )
    }
  }, [error])
  
  return (
    <View style={styles.container}>
      <View style={styles.contentContainer}>
        {/* Show original photo while processing */}
        <View style={styles.imageContainer}>
          <Image source={{ uri: currentPhoto }} style={styles.originalImage} />
          <View style={styles.processingOverlay}>
            <ActivityIndicator size="large" color="#fff" />
          </View>
        </View>
        
        {/* Progress indicator */}
        <RenderProgress 
          progress={progress} 
          isProcessing={isProcessing}
        />
        
        {/* Style preview */}
        <View style={styles.stylePreview}>
          <Text style={styles.styleText}>
            Creating your {style} style room...
          </Text>
        </View>
      </View>
    </View>
  )
}
```

## 🔄 Real-time Updates & WebSockets

### WebSocket Integration for Live Updates
```typescript
// services/websocket.ts
import { io, Socket } from 'socket.io-client'

class WebSocketService {
  private socket: Socket | null = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  
  connect(userId: string) {
    this.socket = io(process.env.WEBSOCKET_URL!, {
      auth: {
        userId,
      },
      transports: ['websocket'],
    })
    
    this.socket.on('connect', () => {
      console.log('WebSocket connected')
      this.reconnectAttempts = 0
    })
    
    this.socket.on('disconnect', () => {
      console.log('WebSocket disconnected')
      this.handleReconnect()
    })
    
    // AI processing updates
    this.socket.on('processing_update', (data) => {
      this.handleProcessingUpdate(data)
    })
    
    // Product price updates
    this.socket.on('price_update', (data) => {
      this.handlePriceUpdate(data)
    })
  }
  
  private handleProcessingUpdate(data: any) {
    // Update processing progress in real-time
    useAppStore.getState().updateProcessingProgress(data.progress)
    
    if (data.stage) {
      useAppStore.getState().setProcessingStage(data.stage)
    }
  }
  
  private handlePriceUpdate(data: any) {
    // Update product prices in real-time
    useAppStore.getState().updateProductPrice(data.productId, data.newPrice)
  }
  
  subscribeToRender(renderId: string) {
    this.socket?.emit('subscribe_render', { renderId })
  }
  
  subscribeToProducts(productIds: string[]) {
    this.socket?.emit('subscribe_products', { productIds })
  }
  
  private handleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      setTimeout(() => {
        this.connect(useAppStore.getState().userId)
      }, 1000 * this.reconnectAttempts)
    }
  }
  
  disconnect() {
    this.socket?.disconnect()
    this.socket = null
  }
}

export const wsService = new WebSocketService()
```

## 📊 Analytics & Monitoring

### Analytics Implementation
```typescript
// services/analytics.ts
import analytics from '@react-native-firebase/analytics'
import crashlytics from '@react-native-firebase/crashlytics'

interface AnalyticsEvent {
  event_name: string
  parameters: Record<string, any>
}

class AnalyticsService {
  // Track user journey
  async trackPhotoCapture(metadata: {
    lighting_quality: number
    room_type: string
    photo_size: number
  }) {
    await analytics().logEvent('photo_captured', {
      lighting_quality: metadata.lighting_quality,
      room_type: metadata.room_type,
      photo_size_mb: Math.round(metadata.photo_size / 1024 / 1024),
    })
  }
  
  async trackAIProcessing(data: {
    style: string
    processing_time: number
    success: boolean
    confidence?: number
  }) {
    await analytics().logEvent('ai_processing_complete', {
      style: data.style,
      processing_time_seconds: Math.round(data.processing_time),
      success: data.success,
      confidence_score: data.confidence || 0,
    })
  }
  
  async trackProductInteraction(action: 'view' | 'click' | 'add_to_cart', product: {
    id: string
    price: number
    category: string
    source: string
  }) {
    await analytics().logEvent(`product_${action}`, {
      product_id: product.id,
      price: product.price,
      category: product.category,
      source: product.source,
      currency: 'GBP',
    })
  }
  
  async trackConversion(data: {
    total_value: number
    item_count: number
    style: string
    session_duration: number
  }) {
    await analytics().logEvent('purchase', {
      value: data.total_value,
      currency: 'GBP',
      items: data.item_count,
      style: data.style,
      session_duration_minutes: Math.round(data.session_duration / 60),
    })
  }
  
  // Performance monitoring
  async trackPerformance(metric: 'app_start' | 'ai_render' | 'image_load', duration: number) {
    await analytics().logEvent('performance_metric', {
      metric_type: metric,
      duration_ms: Math.round(duration),
    })
  }
  
  // Error tracking
  async trackError(error: Error, context: string) {
    await crashlytics().recordError(error)
    await analytics().logEvent('error_occurred', {
      error_type: error.name,
      context,
      error_message: error.message.substring(0, 100), // Truncate for privacy
    })
  }
  
  // User properties
  async setUserProperties(properties: {
    preferred_style?: string
    avg_budget?: number
    room_count?: number
  }) {
    if (properties.preferred_style) {
      await analytics().setUserProperty('preferred_style', properties.preferred_style)
    }
    if (properties.avg_budget) {
      await analytics().setUserProperty('avg_budget', properties.avg_budget.toString())
    }
    if (properties.room_count) {
      await analytics().setUserProperty('room_count', properties.room_count.toString())
    }
  }
}

export const analyticsService = new AnalyticsService()
```

### Performance Monitoring
```typescript
// utils/performance-monitor.ts
import { InteractionManager, AppState } from 'react-native'
import { PerformanceObserver, performance } from 'react-native-performance'

class PerformanceMonitor {
  private startTimes = new Map<string, number>()
  
  startTimer(label: string) {
    this.startTimes.set(label, performance.now())
  }
  
  endTimer(label: string): number {
    const startTime = this.startTimes.get(label)
    if (!startTime) return 0
    
    const duration = performance.now() - startTime
    this.startTimes.delete(label)
    
    // Report to analytics
    analyticsService.trackPerformance(label as any, duration)
    
    return duration
  }
  
  // Monitor app startup performance
  measureAppStart() {
    this.startTimer('app_start')
    
    InteractionManager.runAfterInteractions(() => {
      const duration = this.endTimer('app_start')
      console.log(`App startup took ${duration}ms`)
    })
  }
  
  // Monitor image processing performance
  async measureImageProcessing<T>(operation: () => Promise<T>): Promise<T> {
    this.startTimer('image_processing')
    try {
      const result = await operation()
      this.endTimer('image_processing')
      return result
    } catch (error) {
      this.endTimer('image_processing')
      throw error
    }
  }
  
  // Monitor memory usage
  monitorMemoryUsage() {
    if (__DEV__) {
      setInterval(() => {
        const memoryInfo = performance.memory
        if (memoryInfo) {
          console.log('Memory usage:', {
            used: Math.round(memoryInfo.usedJSHeapSize / 1024 / 1024),
            total: Math.round(memoryInfo.totalJSHeapSize / 1024 / 1024),
            limit: Math.round(memoryInfo.jsHeapSizeLimit / 1024 / 1024),
          })
        }
      }, 10000) // Every 10 seconds
    }
  }
}

export const performanceMonitor = new PerformanceMonitor()
```

## 🧪 Testing Strategy

### Unit Tests for Core Logic
```typescript
// __tests__/ai-processing.test.ts
import { renderHook, act } from '@testing-library/react-native'
import { useAIProcessing } from '../hooks/useAIProcessing'
import { useAppStore } from '../stores/app-store'

jest.mock('../services/api', () => ({
  processImage: jest.fn(),
}))

describe('useAIProcessing', () => {
  beforeEach(() => {
    useAppStore.getState().setPhoto('mock-photo-uri')
  })
  
  it('should process photo successfully', async () => {
    const mockResult = {
      render_id: 'test-123',
      styled_image: 'styled-uri',
      products: [],
      confidence: 0.85,
    }
    
    require('../services/api').processImage.mockResolvedValue(mockResult)
    
    const { result } = renderHook(() => useAIProcessing())
    
    await act(async () => {
      result.current.processPhoto({ style: 'modern' })
    })
    
    expect(result.current.result).toEqual(mockResult)
    expect(result.current.isProcessing).toBe(false)
  })
  
  it('should handle processing errors', async () => {
    require('../services/api').processImage.mockRejectedValue(
      new Error('Processing failed')
    )
    
    const { result } = renderHook(() => useAIProcessing())
    
    await act(async () => {
      result.current.processPhoto({ style: 'modern' })
    })
    
    expect(result.current.error).toBeTruthy()
    expect(result.current.isProcessing).toBe(false)
  })
})
```

### Integration Tests
```typescript
// __tests__/integration/camera-to-result.test.ts
import { render, fireEvent, waitFor } from '@testing-library/react-native'
import { NavigationContainer } from '@react-navigation/native'
import { AppNavigator } from '../navigation/AppNavigator'

jest.mock('react-native-vision-camera', () => ({
  Camera: 'Camera',
  useCameraDevice: () => ({ id: 'mock-device' }),
}))

describe('Camera to Result Flow', () => {
  it('should complete full user journey', async () => {
    const { getByTestId, getByText } = render(
      <NavigationContainer>
        <AppNavigator />
      </NavigationContainer>
    )
    
    // 1. Capture photo
    const captureButton = getByTestId('capture-button')
    fireEvent.press(captureButton)
    
    // 2. Select style
    await waitFor(() => {
      expect(getByText('Choose your style')).toBeTruthy()
    })
    
    const modernStyle = getByTestId('style-modern')
    fireEvent.press(modernStyle)
    
    // 3. Wait for processing
    await waitFor(() => {
      expect(getByText('Processing complete!')).toBeTruthy()
    }, { timeout: 30000 })
    
    // 4. Verify result screen
    expect(getByTestId('result-image')).toBeTruthy()
    expect(getByTestId('product-list')).toBeTruthy()
  })
})
```

### Performance Tests
```typescript
// __tests__/performance/image-processing.test.ts
import { ImageOptimizer } from '../utils/image-optimization'

describe('Image Processing Performance', () => {
  it('should optimize images under 2 seconds', async () => {
    const mockImageUri = 'file://test-image.jpg'
    
    const startTime = performance.now()
    await ImageOptimizer.optimizeForAI(mockImageUri)
    const endTime = performance.now()
    
    const duration = endTime - startTime
    expect(duration).toBeLessThan(2000) // Under 2 seconds
  })
  
  it('should handle large images without memory issues', async () => {
    const largeImageUri = 'file://large-test-image.jpg' // 12MB image
    
    const initialMemory = performance.memory?.usedJSHeapSize || 0
    await ImageOptimizer.optimizeForAI(largeImageUri)
    const finalMemory = performance.memory?.usedJSHeapSize || 0
    
    const memoryIncrease = finalMemory - initialMemory
    expect(memoryIncrease).toBeLessThan(50 * 1024 * 1024) // Under 50MB increase
  })
})
```

This technical implementation provides a complete, production-ready foundation for ReRoom. The architecture emphasizes performance, user experience, and scalability while maintaining clean, maintainable code.

**Key Technical Decisions:**
1. **React Native with Expo** for faster development and deployment
2. **Microservices backend** for scalability and independent deployments  
3. **Real-time WebSocket updates** for better user experience
4. **Comprehensive analytics** for data-driven product decisions
5. **Performance monitoring** to catch issues early
6. **Progressive image optimization** to handle varying device capabilities

The next step would be setting up the development environment and building the MVP following this technical blueprint.