import asyncio
import os
from typing import Dict, List, Optional
import uuid
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import structlog
from PIL import Image
import numpy as np
import requests
import json

# Setup logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# FastAPI app
app = FastAPI(
    title="ReRoom AI Service",
    description="AI-powered interior design analysis and recommendations",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class PhotoAnalysisRequest(BaseModel):
    photo_url: str
    photo_id: str
    user_id: Optional[str] = None
    analysis_type: str = "room_analysis"

class RoomAnalysis(BaseModel):
    room_type: str
    style: str
    colors: List[str]
    lighting: str
    furniture_detected: List[str]
    improvement_suggestions: List[str]
    confidence_score: float

class PhotoAnalysisResponse(BaseModel):
    analysis_id: str
    photo_id: str
    status: str
    room_analysis: Optional[RoomAnalysis] = None
    processing_time_ms: int
    created_at: str

# In-memory storage (in production, use proper database)
analysis_results = {}

@app.get("/")
async def root():
    return {
        "service": "ReRoom AI Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "analyze_photo": "POST /analyze/photo",
            "get_analysis": "GET /analysis/{analysis_id}",
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ai-service",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

def analyze_room_basic(image: Image.Image) -> RoomAnalysis:
    """
    Basic room analysis using image properties
    In production, this would use advanced AI models
    """
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Get image dimensions and basic properties
    width, height = image.size
    
    # Convert to numpy array for basic analysis
    img_array = np.array(image)
    
    # Basic color analysis
    avg_color = np.mean(img_array, axis=(0, 1))
    dominant_colors = []
    
    if avg_color[0] > avg_color[1] and avg_color[0] > avg_color[2]:
        dominant_colors.append("red")
    elif avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]:
        dominant_colors.append("green")
    elif avg_color[2] > avg_color[0] and avg_color[2] > avg_color[1]:
        dominant_colors.append("blue")
    
    # Brightness analysis
    brightness = np.mean(avg_color)
    lighting = "bright" if brightness > 150 else "moderate" if brightness > 100 else "dim"
    
    # Basic room type detection based on aspect ratio and brightness
    aspect_ratio = width / height
    if aspect_ratio > 1.5:
        room_type = "living_room"
    elif aspect_ratio < 0.8:
        room_type = "bathroom"
    else:
        room_type = "bedroom"
    
    # Mock furniture detection (in production, use object detection models)
    furniture_detected = ["chair", "table"] if brightness > 120 else ["bed", "nightstand"]
    
    # Generate style based on color analysis
    if brightness > 180:
        style = "modern_minimalist"
    elif len(dominant_colors) > 1:
        style = "eclectic"
    else:
        style = "traditional"
    
    # Generate improvement suggestions
    suggestions = []
    if brightness < 100:
        suggestions.append("Add more lighting to brighten the space")
    if aspect_ratio > 2:
        suggestions.append("Consider area rugs to define different zones")
    suggestions.append("Add plants to bring life to the room")
    suggestions.append("Consider accent pillows in complementary colors")
    
    return RoomAnalysis(
        room_type=room_type,
        style=style,
        colors=dominant_colors,
        lighting=lighting,
        furniture_detected=furniture_detected,
        improvement_suggestions=suggestions,
        confidence_score=0.75  # Mock confidence score
    )

@app.post("/analyze/photo", response_model=PhotoAnalysisResponse)
async def analyze_photo(request: PhotoAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Analyze a photo for room type, style, and improvement suggestions
    """
    start_time = datetime.utcnow()
    analysis_id = str(uuid.uuid4())
    
    try:
        logger.info("Starting photo analysis", 
                   photo_id=request.photo_id, 
                   analysis_id=analysis_id)
        
        # Download the image
        response = requests.get(request.photo_url, timeout=30)
        response.raise_for_status()
        
        # Open image with PIL
        image = Image.open(requests.get(request.photo_url, stream=True).raw)
        
        # Perform analysis
        room_analysis = analyze_room_basic(image)
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Create response
        result = PhotoAnalysisResponse(
            analysis_id=analysis_id,
            photo_id=request.photo_id,
            status="completed",
            room_analysis=room_analysis,
            processing_time_ms=int(processing_time),
            created_at=start_time.isoformat()
        )
        
        # Store result
        analysis_results[analysis_id] = result
        
        logger.info("Photo analysis completed", 
                   analysis_id=analysis_id,
                   processing_time_ms=int(processing_time))
        
        return result
        
    except Exception as e:
        logger.error("Photo analysis failed", 
                    analysis_id=analysis_id, 
                    error=str(e))
        
        # Return error response
        error_result = PhotoAnalysisResponse(
            analysis_id=analysis_id,
            photo_id=request.photo_id,
            status="failed",
            room_analysis=None,
            processing_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
            created_at=start_time.isoformat()
        )
        
        analysis_results[analysis_id] = error_result
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/analysis/{analysis_id}", response_model=PhotoAnalysisResponse)
async def get_analysis(analysis_id: str):
    """
    Get analysis results by ID
    """
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis_results[analysis_id]

@app.get("/analysis")
async def list_analyses():
    """
    List all analyses (for debugging)
    """
    return {
        "total": len(analysis_results),
        "analyses": list(analysis_results.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 