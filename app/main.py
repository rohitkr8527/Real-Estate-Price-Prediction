from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd
import json
import os

app = FastAPI(title="Bangalore House Price Prediction API")

# CORRECTED PATH CONFIGURATION
BASE_DIR = Path(__file__).resolve().parent.parent  # Points to project root
MODEL_DIR = BASE_DIR / "model_artifacts"  # Correct artifacts directory
MODEL_PATH = MODEL_DIR / "best_model.pkl"  # Using your actual model name
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"  # Using your actual metadata name
LOG_PATH = BASE_DIR / "logs" / "prediction_logs.jsonl"

# Ensure directories exist
os.makedirs(LOG_PATH.parent, exist_ok=True)

# Debug print paths
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"PREPROCESSOR_PATH: {PREPROCESSOR_PATH}")
print(f"METADATA_PATH: {METADATA_PATH}")
print(f"LOG_PATH: {LOG_PATH}")

# Load artifacts with error handling
try:
    # Verify files exist before loading
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor file not found at {PREPROCESSOR_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")
    
    # Load artifacts
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    with open(METADATA_PATH) as f:
        model_metadata = json.load(f)
    
    print("✅ Artifacts loaded successfully")
    
except Exception as e:
    error_msg = f"Artifact loading failed: {str(e)}"
    print(error_msg)
    raise RuntimeError(error_msg)

# Input schema
class HouseRequest(BaseModel):
    location: str
    total_sqft: float
    bath: float
    bhk: int

@app.post("/predict", summary="Predict house price in lakhs")
async def predict(request: HouseRequest):
    """Predict house price based on location, size, and features"""
    try:
        # Create input DataFrame
        input_data = pd.DataFrame([{
            "location": request.location,
            "total_sqft": request.total_sqft,
            "bath": request.bath,
            "bhk": request.bhk
        }])
        
        # Preprocess input
        input_transformed = preprocessor.transform(input_data)
        
        # Handle sparse matrices
        if hasattr(input_transformed, "toarray"):
            input_transformed = input_transformed.toarray()
        
        # Generate prediction
        prediction = model.predict(input_transformed)[0]
        
        # Prepare response
        timestamp = datetime.utcnow().isoformat()
        
        # Flexible metadata handling
        response_metadata = {
            "model_type": model_metadata.get("model", "unknown"),
            "training_date": model_metadata.get("date", "unknown"),
        }
        
        # Add performance metrics if available
        if "performance_metrics" in model_metadata:
            response_metadata["performance"] = {
                "test_r2": model_metadata["performance_metrics"].get("test_r2"),
                "test_rmse": model_metadata["performance_metrics"].get("test_rmse")
            }
            # Calculate prediction interval
            pred_interval = 2 * model_metadata["performance_metrics"].get("test_rmse", 0)
        else:
            pred_interval = 0
        
        response = {
            "predicted_price_lakhs": round(float(prediction), 2),
            "prediction_interval_95": round(pred_interval, 2),
            "input": request.dict(),
            "model_metadata": response_metadata,
            "timestamp": timestamp
        }

        # Log prediction
        with open(LOG_PATH, "a") as log_file:
            log_file.write(json.dumps(response) + "\n")

        return response

    except Exception as e:
        import traceback
        error_detail = f"Prediction failed: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)

@app.get("/metadata", summary="Get model metadata")
async def get_metadata():
    """Retrieve model training metadata and performance"""
    try:
        return model_metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metadata retrieval failed: {str(e)}")

@app.get("/health", summary="Service health check")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": bool(model),
        "preprocessor_loaded": bool(preprocessor),
        "metadata_loaded": bool(model_metadata)
    }
