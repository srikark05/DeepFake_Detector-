#!/usr/bin/env python3
"""
FastAPI Backend for DeepFake Detection
Provides REST API endpoints for image upload and classification
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import io
import pickle
import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# Add module paths to sys.path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "Filtration"))
sys.path.insert(0, str(Path(__file__).parent / "Homology and Topology"))
sys.path.insert(0, str(Path(__file__).parent / "ML Models"))
sys.path.insert(0, str(Path(__file__).parent / "Preprocessing"))

# Import modules (lazy import for TDA modules to handle missing giotto-tda)
from Filtration import Filtration
from SVM import TDASVMClassifier

# Try to import TDA modules, but make them optional
try:
    from T_Diagrams import Persistence
    from Entropy import TDA_FeatureExtractor
    from pipeline import (
        load_image,
        image_to_point_cloud,
        apply_filtrations_to_image,
        extract_tda_features_for_image
    )
    TDA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TDA modules not available: {e}")
    print("Please install giotto-tda to use the prediction features.")
    TDA_AVAILABLE = False
    Persistence = None
    TDA_FeatureExtractor = None

app = FastAPI(title="DeepFake Detection API", version="1.0.0")

# Create directories for static files and templates
static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"
static_dir.mkdir(exist_ok=True)
templates_dir.mkdir(exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

# Model path
MODEL_PATH = Path(__file__).parent / "trained_model.pkl"
PERSISTENCE_MODULE = None

# Initialize persistence module (reusable)
def init_persistence():
    """Initialize persistence module with dummy image"""
    global PERSISTENCE_MODULE
    if not TDA_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="TDA modules not available. Please install giotto-tda. See INSTALL_GIOTTO.md for instructions."
        )
    if PERSISTENCE_MODULE is None:
        dummy_image = np.zeros((128, 128))
        PERSISTENCE_MODULE = Persistence(dummy_image)
    return PERSISTENCE_MODULE

# Load trained model
def load_model():
    """Load the trained model from disk"""
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="Model not found. Please train the model first by running: python train_model.py"
        )
    
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data['classifier'], model_data['target_size']

# Process uploaded image
def process_uploaded_image(file_content: bytes) -> np.ndarray:
    """Process uploaded image file"""
    try:
        # Open image from bytes
        img = Image.open(io.BytesIO(file_content))
        
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

# Extract features from image
def extract_features_from_image(image: np.ndarray, target_size=(128, 128)) -> np.ndarray:
    """Extract TDA features from a single image"""
    if not TDA_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="TDA modules not available. Please install giotto-tda. See INSTALL_GIOTTO.md for instructions."
        )
    try:
        # Resize image to target size
        img_pil = Image.fromarray(image.astype(np.uint8))
        img_resized = img_pil.resize(target_size, Image.LANCZOS)
        img_array = np.array(img_resized, dtype=np.float32)
        
        # Initialize persistence module
        persistence_module = init_persistence()
        
        # Extract features
        features = extract_tda_features_for_image(img_array, persistence_module)
        
        # Reshape to 2D if needed (for single sample)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        return features
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting features: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict if uploaded image is fake or real
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Process image
        image = process_uploaded_image(file_content)
        
        # Load model
        classifier, target_size = load_model()
        
        # Extract features
        features = extract_features_from_image(image, target_size)
        
        # Make prediction
        prediction = classifier.predict(features)[0]
        probabilities = classifier.predict_proba(features)[0]
        
        # Format response
        result = {
            "prediction": "Real" if prediction == 1 else "Fake",
            "confidence": float(probabilities[prediction]),
            "probabilities": {
                "fake": float(probabilities[0]),
                "real": float(probabilities[1])
            }
        }
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    model_exists = MODEL_PATH.exists()
    return {
        "status": "healthy" if model_exists else "model_not_found",
        "model_loaded": model_exists,
        "tda_available": TDA_AVAILABLE,
        "message": "Install giotto-tda to enable predictions" if not TDA_AVAILABLE else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
