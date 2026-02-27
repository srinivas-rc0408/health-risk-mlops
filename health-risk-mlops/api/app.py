"""
api/app.py
FastAPI server that serves predictions from all 5 trained models.

Run: uvicorn api.app:app --reload
Docs: http://localhost:8000/docs
"""

import os
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="🏥 Health Risk Prediction API",
    description="Predicts health risks using 5 ML models: Logistic Regression, Random Forest, XGBoost, K-Means, Neural Network.",
    version="1.0.0",
)

# ─────────────────────────────────────────────
# Load Models
# ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def load_models():
    try:
        models = {
            "scaler": joblib.load(os.path.join(MODEL_DIR, "scaler.pkl")),
            "lr":     joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl")),
            "rf":     joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl")),
            "xgb":    joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl")),
            "kmeans": joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl")),
        }
        # Load neural network separately (TF)
        try:
            import tensorflow as tf
            models["nn"] = tf.keras.models.load_model(os.path.join(MODEL_DIR, "neural_network.h5"))
        except Exception as e:
            print(f"Warning: Could not load Neural Network: {e}")
            models["nn"] = None
        return models
    except Exception as e:
        print(f"Warning: Models not loaded yet — run training first. ({e})")
        return {}

MODELS = load_models()

SEGMENT_LABELS = {0: "High Risk", 1: "Low Risk", 2: "Medium Risk"}

# ─────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────
class PatientData(BaseModel):
    age:               int   = Field(..., ge=1,  le=120, example=62)
    bmi:               float = Field(..., ge=10, le=60,  example=31.5)
    glucose:           int   = Field(..., ge=50, le=400, example=155)
    blood_pressure:    int   = Field(..., ge=40, le=200, example=130)
    cholesterol:       int   = Field(..., ge=100,le=400, example=265)
    heart_rate:        int   = Field(..., ge=30, le=200, example=88)
    smoking:           int   = Field(..., ge=0,  le=1,   example=1)
    alcohol:           int   = Field(..., ge=0,  le=1,   example=0)
    physical_activity: int   = Field(..., ge=0,  le=1,   example=0)
    family_history:    int   = Field(..., ge=0,  le=1,   example=1)


class PredictionResponse(BaseModel):
    heart_disease_risk:    str
    diabetes_risk:         str
    stroke_risk:           str
    patient_segment:       str
    overall_health_score:  str
    risk_summary:          str


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "models_loaded": list(MODELS.keys()),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    if not MODELS:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run: python training/train_all_models.py first."
        )

    features = np.array([[
        patient.age, patient.bmi, patient.glucose, patient.blood_pressure,
        patient.cholesterol, patient.heart_rate, patient.smoking,
        patient.alcohol, patient.physical_activity, patient.family_history,
    ]])

    scaled = MODELS["scaler"].transform(features)

    heart_risk    = int(MODELS["lr"].predict(scaled)[0])
    diabetes_risk = int(MODELS["rf"].predict(scaled)[0])
    stroke_risk   = int(MODELS["xgb"].predict(scaled)[0])
    segment       = int(MODELS["kmeans"].predict(scaled)[0])

    nn_score = 0.0
    if MODELS.get("nn") is not None:
        nn_score = float(MODELS["nn"].predict(scaled, verbose=0)[0][0])

    total_risks = heart_risk + diabetes_risk + stroke_risk
    if total_risks == 0:
        summary = "✅ Low overall risk. Maintain healthy habits."
    elif total_risks == 1:
        summary = "⚠️ Moderate risk detected. Consult a doctor."
    else:
        summary = "🚨 High risk detected across multiple areas. Seek medical attention."

    return PredictionResponse(
        heart_disease_risk   = "YES ⚠️" if heart_risk    else "NO ✅",
        diabetes_risk        = "YES ⚠️" if diabetes_risk else "NO ✅",
        stroke_risk          = "YES ⚠️" if stroke_risk   else "NO ✅",
        patient_segment      = SEGMENT_LABELS.get(segment, "Unknown"),
        overall_health_score = f"{nn_score:.2%} risk",
        risk_summary         = summary,
    )


@app.get("/")
def root():
    return {
        "message": "🏥 Health Risk Prediction API",
        "docs": "/docs",
        "predict": "/predict",
    }
