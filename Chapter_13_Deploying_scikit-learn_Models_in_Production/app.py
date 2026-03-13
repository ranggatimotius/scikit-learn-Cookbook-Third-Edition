
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Cancer Classifier API")

# Load model at startup
model = joblib.load("production_model.pkl")

class PredictionRequest(BaseModel):
    features: list[float]  # 30 features for breast cancer dataset

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    label: str

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    X = np.array(request.features).reshape(1, -1)
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0].max()
    label = "malignant" if prediction == 0 else "benign"
    return PredictionResponse(
        prediction=int(prediction),
        probability=float(probability),
        label=label
    )

@app.get("/health")
def health():
    return {"status": "healthy"}

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
