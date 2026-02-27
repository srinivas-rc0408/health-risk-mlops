# 🏥 Health Risk Prediction — MLOps Project

A multi-model ML platform that predicts health risks using **5 different ML models**, all tracked and served under one MLOps pipeline.

---

## 📌 Project Overview

| Model | Task | Type |
|---|---|---|
| Logistic Regression | Heart Disease Detection | Supervised |
| Random Forest | Diabetes Risk + Feature Importance | Supervised |
| XGBoost | Stroke Risk Prediction | Supervised |
| K-Means Clustering | Patient Segmentation | Unsupervised |
| Neural Network (MLP) | Overall Health Risk Score | Supervised |

---

## 🗂️ Project Structure

```
health-risk-mlops/
├── data/                    # Dataset (generated or from Kaggle)
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
```

---

## ⚙️ MLOps Stack

- **MLflow** — Experiment tracking & model registry
- **FastAPI** — REST API for serving all 5 model predictions
- **GitHub Actions** — CI pipeline (lint + train on push)
- **Joblib / TensorFlow** — Model serialization
- **Docker** *(optional)* — Containerization

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train all models
```bash
python training/train_all_models.py
```

### 3. Start the prediction API
```bash
uvicorn api.app:app --reload
```

### 4. Open API docs
Visit: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Run all 5 models on patient data |
| `/health` | GET | API health check |

### Sample Request
```json
{
  "age": 62,
  "bmi": 31.5,
  "glucose": 155,
  "blood_pressure": 130,
  "cholesterol": 265,
  "heart_rate": 88,
  "smoking": 1,
  "alcohol": 0,
  "physical_activity": 0,
  "family_history": 1
}
```

### Sample Response
```json
{
  "heart_disease_risk": "YES",
  "diabetes_risk": "YES",
  "stroke_risk": "YES",
  "patient_segment": "High Risk",
  "overall_health_score": "78.43% risk"
}
```

---

## 📓 Google Colab

Open `health_risk_mlops.ipynb` in Google Colab to run the full pipeline interactively — trains all 5 models, generates charts, tracks with MLflow, and exports a ZIP for GitHub.

---

## 📦 Dataset

Using a synthetic health dataset (auto-generated in the notebook). You can also replace it with the real [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) from Kaggle.

---

## 👨‍💻 Subject
MLOps — Multi-Model Health Risk Prediction Platform
