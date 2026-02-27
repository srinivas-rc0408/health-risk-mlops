# 🏥 Health Risk Prediction - MLOps Project

A multi-model ML platform for health risk prediction using 5 different ML models.

## Models Used
| Model | Task | Type |
|---|---|---|
| Logistic Regression | Heart Disease | Supervised |
| Random Forest | Diabetes Risk | Supervised |
| XGBoost | Stroke Risk | Supervised |
| K-Means | Patient Segmentation | Unsupervised |
| Neural Network (MLP) | Overall Health Score | Supervised |

## MLOps Tools
- MLflow - Experiment tracking
- Joblib / TensorFlow - Model serialization
- FastAPI - REST API serving
- Docker - Containerization

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook health_risk_mlops.ipynb
```
