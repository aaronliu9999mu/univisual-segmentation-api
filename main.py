"""
Univisual Cell Segmentation Backend
Local FastAPI server for Cellpose 3 and StarDist
"""

import io
import os
import logging
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set cache directories to avoid encoding issues
os.environ['STARDIST_CACHEDIR'] = os.path.expanduser('~/.stardist_models')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

# Global model cache
models = {}


def load_models():
    """Load segmentation models into memory"""
    global models
    
    # Load Cellpose
    try:
        logger.info("Loading Cellpose model...")
        from cellpose import models as cp_models
        models["cellpose"] = cp_models.Cellpose(model_type="nuclei", gpu=False)
        logger.info("✓ Cellpose loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load Cellpose: {e}")
        models["cellpose"] = None

    # Load StarDist (Disabled for now)
    models["stardist"] = None
    logger.info("StarDist model disabled (Cellpose-only mode)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    logger.info("Starting Univisual Segmentation Server (Cellpose Only)...")
    load_models()
    yield
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Univisual Segmentation API (Cellpose Only)",
    description="Local backend for Cellpose cell segmentation",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://uni-visual.com", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint - redirect to docs"""
    return {"message": "Univisual Segmentation API (Cellpose Only)", "docs": "/docs", "health": "/health"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models": {
            "cellpose": models.get("cellpose") is not None,
            "stardist": False,  # Disabled
        }
    }


@app.get("/models")
async def list_models():
    """List available models and their status"""
    return {
        "models": [
            {
                "id": "cellpose",
                "name": "Cellpose 3 (Cell Nucleus)",
                "available": models.get("cellpose") is not None,
                "description": "General cell/nucleus segmentation using Cellpose 3"
            },
            {
                "id": "stardist",
                "name": "StarDist (Dense Cells)",
                "available": False,
                "description": "Currently unavailable (Disk space limited)"
            }
        ]
    }


def image_to_array(image_bytes: bytes) -> tuple[np.ndarray, float]:
    """Convert image bytes to numpy array, downsizing if needed for memory.
    
    Returns (image_array, scale_factor) where scale_factor is used to
    map coordinates back to the original image size.
    """
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if needed
    if image.mode == "RGBA":
        image = image.convert("RGB")
    elif image.mode == "L":
        pass  # Grayscale - keep as is
    elif image.mode != "RGB":
        image = image.convert("RGB")
    
    # Downsize large images to fit in 512MB RAM
    MAX_DIM = 1024
    w, h = image.size
    scale = 1.0
    
    if max(w, h) > MAX_DIM:
        scale = MAX_DIM / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        logger.info(f"Downsizing image from {w}x{h} to {new_w}x{new_h} (scale={scale:.3f})")
        image = image.resize((new_w, new_h), Image.LANCZOS)
    
    return np.array(image), scale


def run_cellpose(image: np.ndarray, scale: float, diameter: Optional[float] = None) -> dict:
    """Run Cellpose segmentation"""
    model = models.get("cellpose")
    if model is None:
        raise HTTPException(status_code=503, detail="Cellpose model not loaded")
    
    # Adjust diameter for the scale
    effective_diameter = (diameter or 30) * scale
    
    # Run segmentation
    masks, flows, styles, diams = model.eval(
        image,
        diameter=effective_diameter,
        channels=[0, 0] if len(image.shape) == 2 else [0, 0],
        flow_threshold=0.4,
        cellprob_threshold=0.0,
    )
    
    # Count cells
    cell_count = len(np.unique(masks)) - 1  # Subtract background
    
    # Get outlines (contours) for each cell
    from cellpose import utils
    outlines = utils.outlines_list(masks)
    
    # Inverse scale to map coordinates back to original image size
    inv_scale = 1.0 / scale
    
    # Process results into a structured format
    cells = []
    for i, outline in enumerate(outlines):
        # outline is (N, 2) array — Cellpose returns (col, row) = (x, y) format
        if len(outline) == 0:
            continue
            
        # Downsample points for performance (take every 4th point)
        step = 4 if len(outline) > 20 else 1
        sampled_outline = outline[::step]
        
        # Convert to list of {x, y} points — scaled back to original image size
        # pt[0] = col = x,  pt[1] = row = y
        points = [{"x": int(pt[0] * inv_scale), "y": int(pt[1] * inv_scale)} for pt in sampled_outline]
        
        # Calculate bounding box from FULL outline for accuracy
        all_x = [int(pt[0] * inv_scale) for pt in outline]
        all_y = [int(pt[1] * inv_scale) for pt in outline]
        
        if not all_x or not all_y:
            continue
            
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        cells.append({
            "id": i,
            "outline": points,
            "bbox": {
                "minX": min_x,
                "maxX": max_x,
                "minY": min_y,
                "maxY": max_y,
            },
            "centroid": {
                "x": float(np.mean(all_x)),
                "y": float(np.mean(all_y)),
            }
        })
    
    return {
        "cell_count": len(cells),
        "cells": cells,
    }


def run_stardist(image: np.ndarray, prob_thresh: float = 0.5, nms_thresh: float = 0.4) -> dict:
    """Run StarDist segmentation (Disabled)"""
    raise HTTPException(status_code=503, detail="StarDist model is currently disabled to save disk space")


@app.post("/segment")
async def segment_image(
    file: UploadFile = File(...),
    model: str = Form("cellpose"),
    diameter: Optional[float] = Form(None),
    prob_thresh: float = Form(0.5),
    nms_thresh: float = Form(0.4),
):
    """
    Segment cells in uploaded image
    
    - **file**: Image file (TIFF, PNG, JPG supported)
    - **model**: Either "cellpose" or "stardist"
    - **diameter**: Cell diameter for Cellpose (optional)
    - **prob_thresh**: Probability threshold for StarDist (0-1)
    - **nms_thresh**: NMS threshold for StarDist (0-1)
    """
    try:
        contents = await file.read()
        logger.info(f"Received image: {file.filename}, size: {len(contents)} bytes")
        
        image, scale = image_to_array(contents)
        logger.info(f"Image shape: {image.shape} (scale={scale:.3f})")
        
        if model == "cellpose":
            result = run_cellpose(image, scale, diameter)
        elif model == "stardist":
            result = run_stardist(image, prob_thresh, nms_thresh)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")
        
        logger.info(f"Segmentation complete: {result['cell_count']} cells detected")
        
        return JSONResponse({
            "success": True,
            "model": model,
            "filename": file.filename,
            **result,
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("🔬 Univisual Cell Segmentation Server")
    print("=" * 50)
    print("Starting server at http://localhost:8000")
    print("API docs available at http://localhost:8000/docs")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
