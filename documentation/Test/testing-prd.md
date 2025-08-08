Perfect. Here’s a breakdown of all three deliverables tailored for your current Modomo test phase:

⸻

🧪 1. Modomo Testing Webapp – PRD (Product Requirements Document)

Objective

Build an internal webapp for testing AI models that generate enhanced interior designs from uploaded images. This tool will help evaluate the performance of model training, object generation, and tagging accuracy.

Core Flow
	1.	User uploads an image of a real room.
	2.	Model processes the image to enhance it — e.g., by adding decor, furniture, lighting, or stylistic changes.
	3.	Output image is displayed, showing new visual elements added by the model.
	4.	User clicks on objects in the image to:
	•	View object metadata (name, type).
	•	See price and purchase options (dummy data for now).

Features

🔹 MVP
	•	Secure image upload (drag & drop or file picker)
	•	Display model-enhanced image
	•	Clickable bounding boxes or regions
	•	Metadata tooltip or sidebar (name, style, price)
	•	Log inference details for debugging

🔹 Admin-only Tools
	•	View model parameters used
	•	Re-run model with different styles/prompts
	•	Toggle layers (original / generated / mask / tags)

Tech Stack (Suggested)
	•	Frontend: React + TailwindCSS (or Preact)
	•	Backend: FastAPI or Node (with Docker)
	•	Storage: S3 / Cloudflare R2 / local dev
	•	Model Inference: Runpod or Lambda Labs
	•	Model Backend: ControlNet + Stable Diffusion or custom LoRA

⸻

🖼️ 2. Dataset Strategy for Interior Image Scraping

Goal

Gather high-quality, diverse interior design images for:
	•	Training the generation model
	•	Object detection and segmentation
	•	Style transfer accuracy

Sources

Source	Type	Notes
Pinterest	High-res lifestyle shots	Use Puppeteer + proxy rotation
Houzz	Styled room photos	Requires dynamic scraping
Behance	Designer portfolios	Can use open API or headless scrape
IKEA, Wayfair	Product scenes	Good for labeled room setups
ArchDaily	Architecture-focused	Include minimal and modern interiors
Unsplash/Pexels	Free stock	Use for generic spaces
Reddit	r/InteriorDesign, r/malelivingspace	Raw user-uploaded setups

Scraping Strategy
	•	Use Scrapy or Puppeteer + rotating proxies (Zyte, ScraperAPI)
	•	Store:
	•	image_url
	•	style_tag (if available or inferred later)
	•	room_type (living, kitchen, bedroom)
	•	objects_in_scene (optional)
	•	Avoid watermarked/duplicate images

⸻

🧠 3. Model Training Plan (Phase 1 – Object Identification + Generation)

Training Goal

Train a model (or chain of models) to:
	1.	Understand indoor scenes
	2.	Replace/enhance objects
	3.	Generate click-tag data with bounding boxes and metadata

⸻

Model Stack (Suggested)

🔹 A. Scene Understanding
	•	Model: YOLOv8, Grounding DINO or SAM (Segment Anything)
	•	Task: Detect layout + existing furniture

🔹 B. Image Generation
	•	Model: ControlNet + SDXL or Style-LoRA
	•	Task: Add furniture based on style prompt
	•	Training: Fine-tune LoRA on scraped datasets (with ControlNet hints)

🔹 C. Object Metadata Extraction
	•	Bounding Box Mapping: Store object ID + coordinates + metadata
	•	Use heatmaps or feature attribution for clickable mapping

⸻

Training Tools
	•	Annotation: Label Studio or CVAT
	•	Training: 🤗 Diffusers + DreamBooth/LoRA for SD
	•	Infrastructure: Runpod (GPU 3090/4090), Dockerized Jupyter setup
	•	Logging: Weights & Biases or TensorBoard

⸻

📈 Evaluation
	•	Track image enhancement quality (side-by-side comparisons)
	•	Measure:
	•	Object diversity
	•	Tagging accuracy
	•	Visual appeal (manual + CLIP scores)

⸻
