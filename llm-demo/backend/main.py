from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import io
import base64
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Demo Backend", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock AI model response for testing
def mock_ai_process(image_data):
    """Mock AI processing - replace with actual model inference"""
    
    # Simulate object detection results
    mock_objects = [
        {
            "id": "obj_1",
            "name": "Modern Sofa",
            "type": "furniture",
            "style": "contemporary",
            "price": 1299.99,
            "bbox": {"x": 100, "y": 150, "width": 200, "height": 120},
            "confidence": 0.92
        },
        {
            "id": "obj_2", 
            "name": "Floor Lamp",
            "type": "lighting",
            "style": "industrial",
            "price": 249.99,
            "bbox": {"x": 320, "y": 80, "width": 60, "height": 180},
            "confidence": 0.88
        },
        {
            "id": "obj_3",
            "name": "Coffee Table",
            "type": "furniture", 
            "style": "minimalist",
            "price": 399.99,
            "bbox": {"x": 150, "y": 280, "width": 120, "height": 80},
            "confidence": 0.85
        }
    ]
    
    return {
        "enhanced_image": image_data,  # In real implementation, this would be the AI-enhanced image
        "objects": mock_objects,
        "processing_time": 2.3,
        "model_version": "test-v1.0"
    }

@app.get("/")
async def root():
    return {"message": "LLM Demo Backend - Interior Design AI Testing"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    """Process uploaded image with AI model"""
    
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to base64 for frontend display
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        logger.info(f"Processing image: {file.filename}, size: {image.size}")
        
        # Process with mock AI model
        result = mock_ai_process(img_base64)
        
        return JSONResponse({
            "success": True,
            "data": result,
            "metadata": {
                "filename": file.filename,
                "image_size": image.size,
                "file_size": len(image_data)
            }
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@app.get("/models")
async def get_models():
    """Get available AI models"""
    return {
        "models": [
            {
                "id": "sdxl-controlnet-v1",
                "name": "Stable Diffusion XL + ControlNet",
                "description": "Enhanced interior design generation",
                "status": "active"
            },
            {
                "id": "yolo-v8-furniture",
                "name": "YOLOv8 Furniture Detection", 
                "description": "Object detection for furniture items",
                "status": "active"
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)