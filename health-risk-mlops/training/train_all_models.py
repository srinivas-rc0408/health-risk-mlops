"""
train_all_models.py
Trains all 5 ML models and saves them to the /models directory.
Run: python training/train_all_models.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

for d in [DATA_DIR, MODEL_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────
# 1. Generate / Load Dataset
# ─────────────────────────────────────────────
def generate_dataset():
    print("📊 Generating synthetic health dataset...")
    np.random.seed(42)
    n = 1000

    data = pd.DataFrame({
        "age":               np.random.randint(20, 80, n),
        "bmi":               np.round(np.random.normal(27, 5, n), 1),
        "glucose":           np.random.randint(70, 200, n),
        "blood_pressure":    np.random.randint(60, 140, n),
        "cholesterol":       np.random.randint(150, 300, n),
        "heart_rate":        np.random.randint(55, 110, n),
        "smoking":           np.random.randint(0, 2, n),
        "alcohol":           np.random.randint(0, 2, n),
        "physical_activity": np.random.randint(0, 2, n),
        "family_history":    np.random.randint(0, 2, n),
    })

    data["heart_disease"] = (
        ((data["age"] > 50) & (data["cholesterol"] > 240)) | (data["blood_pressure"] > 120)
    ).astype(int)
    data["diabetes"] = (
        (data["glucose"] > 140) | (data["bmi"] > 30)
    ).astype(int)
    data["stroke_risk"] = (
        ((data["age"] > 55) & (data["blood_pressure"] > 115)) | (data["smoking"] == 1)
    ).astype(int)

    path = os.path.join(DATA_DIR, "health_data.csv")
    data.to_csv(path, index=False)
    print(f"   Saved to {path}  shape={data.shape}")
    return data


# ─────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────
FEATURES = [
    "age", "bmi", "glucose", "blood_pressure", "cholesterol",
    "heart_rate", "smoking", "alcohol", "physical_activity", "family_history",
]

def preprocess(data):
    print("🔧 Preprocessing data...")
    X = data[FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    splits = {}
    for target in ["heart_disease", "diabetes", "stroke_risk"]:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_scaled, data[target], test_size=0.2, random_state=42
        )
        splits[target] = (X_tr, X_te, y_tr, y_te)

    # Combined target for NN
    y_combined = ((data["heart_disease"] + data["diabetes"] + data["stroke_risk"]) >= 1).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_combined, test_size=0.2, random_state=42)
    splits["combined"] = (X_tr, X_te, y_tr, y_te)

    return X_scaled, splits


def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename), dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# Model 1 — Logistic Regression
# ─────────────────────────────────────────────
def train_logistic_regression(splits):
    print("\n🤖 MODEL 1: Logistic Regression (Heart Disease)...")
    X_tr, X_te, y_tr, y_te = splits["heart_disease"]
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    acc = accuracy_score(y_te, preds)
    print(f"   Accuracy: {acc:.4f}")
    print(classification_report(y_te, preds))
    save_confusion_matrix(y_te, preds, "Logistic Regression — Confusion Matrix", "lr_confusion_matrix.png")
    joblib.dump(model, os.path.join(MODEL_DIR, "logistic_regression.pkl"))
    return model, acc


# ─────────────────────────────────────────────
# Model 2 — Random Forest
# ─────────────────────────────────────────────
def train_random_forest(splits):
    print("\n🤖 MODEL 2: Random Forest (Diabetes Risk)...")
    X_tr, X_te, y_tr, y_te = splits["diabetes"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    acc = accuracy_score(y_te, preds)
    print(f"   Accuracy: {acc:.4f}")
    print(classification_report(y_te, preds))

    # Feature importance plot
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    plt.figure(figsize=(9, 5))
    importances.plot(kind="bar", color="forestgreen", edgecolor="white")
    plt.title("Random Forest — Feature Importance")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "rf_feature_importance.png"), dpi=150)
    plt.close()

    joblib.dump(model, os.path.join(MODEL_DIR, "random_forest.pkl"))
    return model, acc


# ─────────────────────────────────────────────
# Model 3 — XGBoost
# ─────────────────────────────────────────────
def train_xgboost(splits):
    print("\n🤖 MODEL 3: XGBoost (Stroke Risk)...")
    X_tr, X_te, y_tr, y_te = splits["stroke_risk"]
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    acc = accuracy_score(y_te, preds)
    print(f"   Accuracy: {acc:.4f}")
    print(classification_report(y_te, preds))
    save_confusion_matrix(y_te, preds, "XGBoost — Confusion Matrix", "xgb_confusion_matrix.png")
    joblib.dump(model, os.path.join(MODEL_DIR, "xgboost.pkl"))
    return model, acc


# ─────────────────────────────────────────────
# Model 4 — K-Means Clustering
# ─────────────────────────────────────────────
def train_kmeans(X_scaled):
    print("\n🤖 MODEL 4: K-Means Clustering (Patient Segmentation)...")
    from sklearn.decomposition import PCA
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = model.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    labels_map = {0: "High Risk", 1: "Low Risk", 2: "Medium Risk"}

    plt.figure(figsize=(8, 6))
    for i, (color, label) in enumerate(zip(colors, labels_map.values())):
        mask = clusters == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=label, alpha=0.6, s=40)
    plt.title("K-Means — Patient Segments (PCA View)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "kmeans_clusters.png"), dpi=150)
    plt.close()

    dist = pd.Series(clusters).value_counts().rename(labels_map)
    print(f"   Cluster Distribution:\n{dist.to_string()}")
    joblib.dump(model, os.path.join(MODEL_DIR, "kmeans.pkl"))
    return model


# ─────────────────────────────────────────────
# Model 5 — Neural Network (MLP)
# ─────────────────────────────────────────────
def train_neural_network(splits):
    print("\n🤖 MODEL 5: Neural Network MLP (Overall Health Score)...")
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    X_tr, X_te, y_tr, y_te = splits["combined"]

    model = keras.Sequential([
        layers.Input(shape=(len(FEATURES),)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(X_tr, y_tr, epochs=30, batch_size=32, validation_split=0.1, verbose=0)
    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    print(f"   Accuracy: {acc:.4f} | Loss: {loss:.4f}")

    # Training history plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("NN — Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("NN — Loss")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "nn_training_history.png"), dpi=150)
    plt.close()

    model.save(os.path.join(MODEL_DIR, "neural_network.h5"))
    return model, acc


# ─────────────────────────────────────────────
# MLflow Logging
# ─────────────────────────────────────────────
def log_to_mlflow(models_info):
    print("\n📊 Logging to MLflow...")
    mlflow.set_experiment("health-risk-mlops")
    for name, model, acc in models_info:
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type", name)
            mlflow.log_metric("accuracy", acc)
            if hasattr(model, "predict"):
                try:
                    mlflow.sklearn.log_model(model, name)
                except Exception:
                    pass
            print(f"   ✅ {name}: {acc:.4f}")


# ─────────────────────────────────────────────
# Final Comparison Chart
# ─────────────────────────────────────────────
def plot_comparison(results):
    df = pd.DataFrame(results, columns=["Model", "Accuracy"])
    plt.figure(figsize=(9, 5))
    bars = plt.bar(df["Model"], df["Accuracy"] * 100,
                   color=["#3498db", "#2ecc71", "#e67e22", "#9b59b6"],
                   edgecolor="white", width=0.5)
    for bar, val in zip(bars, df["Accuracy"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val*100:.1f}%", ha="center", fontweight="bold")
    plt.ylim(0, 115)
    plt.title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "model_comparison.png"), dpi=150)
    plt.close()
    print(f"\n   Chart saved to reports/model_comparison.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  HEALTH RISK MLOPS — TRAINING PIPELINE")
    print("=" * 50)

    data = generate_dataset()
    X_scaled, splits = preprocess(data)

    lr_model,  lr_acc  = train_logistic_regression(splits)
    rf_model,  rf_acc  = train_random_forest(splits)
    xgb_model, xgb_acc = train_xgboost(splits)
    km_model           = train_kmeans(X_scaled)
    nn_model,  nn_acc  = train_neural_network(splits)

    log_to_mlflow([
        ("Logistic_Regression", lr_model,  lr_acc),
        ("Random_Forest",       rf_model,  rf_acc),
        ("XGBoost",             xgb_model, xgb_acc),
        ("Neural_Network",      nn_model,  nn_acc),
    ])

    results = [
        ("Logistic Regression", lr_acc),
        ("Random Forest",       rf_acc),
        ("XGBoost",             xgb_acc),
        ("Neural Network",      nn_acc),
    ]
    plot_comparison(results)

    print("\n" + "=" * 50)
    print("  ✅ ALL MODELS TRAINED & SAVED SUCCESSFULLY")
    print("=" * 50)
    print(f"  LR  Accuracy : {lr_acc:.4f}")
    print(f"  RF  Accuracy : {rf_acc:.4f}")
    print(f"  XGB Accuracy : {xgb_acc:.4f}")
    print(f"  NN  Accuracy : {nn_acc:.4f}")
    print("=" * 50)
