🏥 Health Risk Prediction — MLOps Project
A multi-model ML platform that predicts health risks using 5 different ML models, all tracked and served under one MLOps pipeline.

📌 Project Overview
ModelTaskTypeLogistic RegressionHeart Disease DetectionSupervisedRandom ForestDiabetes Risk + Feature ImportanceSupervisedXGBoostStroke Risk PredictionSupervisedK-Means ClusteringPatient SegmentationUnsupervisedNeural Network (MLP)Overall Health Risk ScoreSupervised

🗂️ Project Structure
health-risk-mlops/
├── data/                    # Dataset
├── models/                  # Saved trained models (.pkl, .h5)
├── training/
│   └── train_all_models.py  # Train all 5 models at once
├── api/
│   └── app.py               # FastAPI prediction server
├── reports/                 # Auto-generated plots & charts
├── .github/workflows/
│   └── ci.yml               # GitHub Actions CI pipeline
├── health_risk_mlops.ipynb  # Main Colab notebook
├── requirements.txt
└── README.md

⚙️ MLOps Stack

MLflow — Experiment tracking & model registry
FastAPI — REST API for serving all 5 model predictions
GitHub Actions — CI pipeline (lint + train on push)
Joblib / TensorFlow — Model serialization
Docker (optional) — Containerization


🚀 How to Run
1. Install dependencies
bashpip install -r requirements.txt
2. Train all models
bashpython training/train_all_models.py
3. Start the prediction API
bashuvicorn api.app:app --reload
4. Open API docs → http://localhost:8000/docs

📊 API Endpoints
EndpointMethodDescription/predictPOSTRun all 5 models on patient data/healthGETAPI health check
Sample Request:
json{
  "age": 62, "bmi": 31.5, "glucose": 155,
  "blood_pressure": 130, "cholesterol": 265,
  "heart_rate": 88, "smoking": 1,
  "alcohol": 0, "physical_activity": 0, "family_history": 1
}
Sample Response:
json{
  "heart_disease_risk": "YES ⚠️",
  "diabetes_risk": "YES ⚠️",
  "stroke_risk": "YES ⚠️",
  "patient_segment": "High Risk",
  "overall_health_score": "78.43% risk",
  "risk_summary": "🚨 High risk detected. Seek medical attention."
}

📦 Dataset
Synthetic health dataset auto-generated in the notebook. Can be replaced with the real Heart Failure Prediction Dataset from Kaggle.

👨‍💻 Subject
MLOps — Multi-Model Health Risk Prediction Platform
