Got it — you want a proper, in-depth, nothing-left-hanging breakdown so we can go from “Modomo scraping idea” to “full system in motion” without guessing later.
Let’s redo this the right way, like a full internal spec your dev team could pick up and run with tomorrow.

⸻

A. ARCHITECTURE MAP – Modomo Scraping & Tagging System

Overall Pipeline

[ Scene Source: Houzz UK ]        [ Product Source: Catalog feeds/APIs ]
             ↓                                      ↓
  [ Scrapy + Playwright Crawler ]     [ Scrapy/Feed Parser ]
             ↓                                      ↓
[ Raw Image + Meta → R2 Storage ]   [ Product Images + Meta → R2 Storage ]
             ↓                                      ↓
[ Postgres (metadata) + pgvector embeddings for both scene & product images ]
             ↓
[ Detection Worker (GroundingDINO + SAM2) on scenes ]
             ↓
[ Auto-classify objects → match to products via CLIP similarity ]
             ↓
[ Preact Tagging Dashboard → Human-in-the-loop review & correction ]
             ↓
[ Approved Dataset in R2 + Final Export Manifest in JSON/CSV for training ]

Key Services
	•	Crawler service – Houzz scenes scraper (JS rendered)
	•	Catalog service – Product feed/API parser
	•	Detection service – AI-based bounding boxes, masks, and CLIP embeddings
	•	Review service – Human UI for validation & tagging
	•	Export service – Dataset generation for ML pipelines

⸻

B. PRD – Modomo Dataset Scraping, Tagging & Catalog Integration v1.0

1. Purpose

Create a scalable system to:
	•	Scrape styled interior scenes (Houzz UK)
	•	Detect & label objects according to Modomo’s taxonomy
	•	Cross-match detected objects to known product catalog items
	•	Provide a human review interface to maintain dataset quality
	•	Output a clean, structured dataset for ML training

⸻

2. Goals
	•	Scene scraping capacity: 5k+ images/week from Houzz
	•	Product ingestion: 50k+ catalog items in initial load
	•	Detection accuracy after review: ≥85%
	•	Review throughput: ≥300 objects/hour per reviewer
	•	Dataset export format compatible with training pipeline (YOLOv8, Detectron2, etc.)

⸻

3. Core Features

Scene Scraping
	•	Target URL: https://www.houzz.co.uk/photos/
	•	Extract:
	•	High-res image URLs
	•	Room type
	•	Style tags
	•	Colour tags
	•	Project link
	•	Store:
	•	Images in Cloudflare R2
	•	Metadata in Postgres

Product Catalog
	•	Sources:
	•	Affiliate feeds (Awin, CJ, Rakuten)
	•	JSON-LD scraping from retailers
	•	Extract:
	•	Product name, SKU, category, price, brand, material, dimensions, image, URL
	•	Normalize categories to Modomo taxonomy
	•	Store images + compute embeddings

Detection & Tagging
	•	Run GroundingDINO with prompt set from taxonomy
	•	Refine masks with SAM2
	•	Compute CLIP embeddings for each detected object
	•	Auto-match against product embeddings
	•	Store detection results for review

Review Dashboard (Preact SPA)
	•	Image-by-image review mode with object overlays
	•	Tag editing (taxonomy dropdown + free tags)
	•	Bulk accept/reject mode
	•	Filters: by room type, category, detection confidence
	•	Keyboard shortcuts for speed

Export
	•	Dataset split: train, val, test
	•	Manifest includes:

{
  "image_id": "...",
  "objects": [
    {
      "bbox": [x,y,w,h],
      "mask": "r2://path/to/mask.png",
      "category": "sofa",
      "tags": ["modern", "grey"],
      "matched_product_id": "prod_123",
      "confidence": 0.91
    }
  ],
  "source": "houzz",
  "license": "per houzz ToS"
}


	•	Images and masks zipped & uploaded to R2

⸻

4. Constraints
	•	No Next.js; Preact + Vite for UI
	•	No storage in local FS — all assets go to R2
	•	Adhere to robots.txt/ToS
	•	Scrapers must handle JS rendering, pagination, and infinite scroll

⸻

5. Success Metrics
	•	Scene ingestion rate stable with <2% failure
	•	Detection false positive rate <15% post-review
	•	Reviewer productivity ≥ target
	•	Catalog match rate ≥60% for common furniture categories

⸻

C. TECH DOC – Implementation Details

1. Stack & Infra
	•	Backend/API – Python + FastAPI
	•	Crawling – Scrapy + scrapy-playwright
	•	Detection – PyTorch (GroundingDINO, SAM2, CLIP)
	•	Database – Postgres + pgvector (Neon.tech)
	•	Storage – Cloudflare R2 via boto3 S3 API
	•	UI – Preact + Vite, deployed to Cloudflare Pages
	•	Workers – Celery or RQ for detection and processing jobs

⸻

2. Data Flow – Scenes
	1.	Crawler fetches scene HTML → extracts meta + image URLs
	2.	Images saved to R2 /raw/scenes/
	3.	Metadata inserted into Postgres scenes table
	4.	Worker detects objects → saves masks to /masks/
	5.	CLIP embeddings saved in objects.clip_vector
	6.	Scene status moves to pending_review

⸻

3. Data Flow – Catalog
	1.	Feed parser reads affiliate CSV/XML or scrapes product pages
	2.	Normalizes categories to taxonomy
	3.	Downloads product images to /raw/catalog/
	4.	Computes CLIP embeddings for product image
	5.	Inserts into products table

⸻

4. Review UI → API
	•	GET /queue?limit=N – Fetch next N pending scenes
	•	POST /update-object – Edit category/tags
	•	POST /approve – Mark reviewed; move to /approved/
	•	POST /reject – Mark for retraining/discard

⸻

5. Taxonomy v0.1 (Detection Classes)

Seating
	•	Sofa, Sectional, Armchair, Dining Chair, Stool, Bench

Tables
	•	Coffee Table, Side Table, Dining Table, Console Table, Desk

Storage
	•	Bookshelf, Cabinet, Dresser, Wardrobe

Lighting
	•	Pendant Light, Floor Lamp, Table Lamp, Wall Sconce

Soft Furnishings
	•	Rug, Curtains, Pillow, Blanket

Décor
	•	Wall Art, Mirror, Plant, Decorative Object

Bed & Bath
	•	Bed Frame, Mattress, Headboard, Nightstand, Bathtub, Sink/Vanity

⸻

6. Deployment
	•	Crawler workers: Docker container with Playwright Chromium
	•	Detection workers: GPU-enabled container (Runpod or AWS G4dn)
	•	API: FastAPI app on Fly.io or Railway
	•	UI: Cloudflare Pages static site
	•	Storage: Cloudflare R2
	•	DB: Neon.tech Postgres with pgvector

⸻

D. PRODUCT CATALOG DATASET CREATION PLAN

Step 1 – Source Identification
	•	Start with Awin + CJ affiliate networks for UK furniture brands
	•	Supplement with JSON-LD scraping from IKEA, Made.com, Wayfair UK

Step 2 – Parsing & Normalization
	•	Parse feeds/pages → extract fields
	•	Map external categories → Modomo taxonomy
	•	Coerce prices to GBP; dimensions to mm

Step 3 – Image Handling
	•	Download, hash, store in /raw/catalog/
	•	Compute CLIP embeddings for matching

Step 4 – Linking to Scene Objects
	•	When reviewing scenes, UI shows top-N closest products (embedding similarity)
	•	Reviewer can link/unlink product to object

