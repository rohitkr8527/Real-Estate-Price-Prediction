from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd
import json

app = FastAPI()

# Paths
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "model_artifacts" / "best_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "model_artifacts" / "preprocessor.pkl"
METADATA_PATH = BASE_DIR / "model_artifacts" / "model_metadata.json"
LOG_PATH = BASE_DIR / "logs" / "prediction_logs.jsonl"

# Ensure logs directory exists
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load model and preprocessor
try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    with open(METADATA_PATH) as f:
        model_metadata = json.load(f)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load artifacts: {e}")

# Input schema
class HouseRequest(BaseModel):
    location: str
    total_sqft: float
    bath: int
    bhk: int

@app.post("/predict")
def predict(request: HouseRequest):
    try:
        input_df = pd.DataFrame([request.model_dump()])
        input_transformed = preprocessor.transform(input_df)
        prediction = float(model.predict(input_transformed)[0])
        timestamp = datetime.utcnow().isoformat()

        # Prepare response
        response = {
            "predicted_price_lakhs": round(prediction, 2),
            "input": request.model_dump(),
            "metadata": {
                "model_name": model_metadata.get("model_name", "unknown"),
                "timestamp": timestamp,
                "features": model_metadata.get("features", []),
                "test_rmse": model_metadata.get("test_rmse"),
                "test_r2": model_metadata.get("test_r2")
            }
        }

        # Log the request and prediction
        with open(LOG_PATH, "a") as log_file:
            log_file.write(json.dumps(response) + "\n")

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/metadata")
def get_metadata():
    try:
        return model_metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metadata retrieval failed: {str(e)}")
