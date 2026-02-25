"""
Model Training Module for Customer Churn Prediction.

Trains a Random Forest classifier, evaluates it, and saves the model + artifacts.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Columns used for training
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

TARGET_COL = "Churn"
DROP_COLS = ["customerID"]


def load_and_preprocess(filepath):
    """Load CSV and preprocess for training."""
    df = pd.read_csv(filepath)

    # Drop ID column
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # Convert TotalCharges to numeric (may have blanks in real Kaggle data)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing numeric values with median
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Encode target
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    df[TARGET_COL] = df[TARGET_COL].fillna(0).astype(int)

    return df


def encode_features(df, encoders=None, fit=True):
    """Label-encode categorical columns. Returns encoded df and encoders dict."""
    if encoders is None:
        encoders = {}
    df = df.copy()

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le is None:
                raise ValueError(f"No encoder found for column: {col}")
            # Handle unseen labels gracefully
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    return df, encoders


def train_model(data_path=None, test_size=0.2, random_state=42):
    """Full training pipeline: load, preprocess, train, evaluate, save."""
    if data_path is None:
        data_path = os.path.join(DATA_DIR, "raw", "churn_train.csv")

    print(f"[*] Loading data from {data_path}")
    df = load_and_preprocess(data_path)
    print(f"    Shape: {df.shape}")

    # Encode categoricals
    df_encoded, encoders = encode_features(df, fit=True)

    feature_cols = [c for c in df_encoded.columns if c != TARGET_COL]
    X = df_encoded[feature_cols]
    y = df_encoded[TARGET_COL]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"    Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("[+] Model trained successfully")

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    print("\n[+] Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"    {k:>12}: {v}")

    print("\n[+] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    cm = confusion_matrix(y_test, y_pred)
    print(f"[+] Confusion Matrix:\n{cm}")

    # Save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "churn_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n[+] Model saved: {model_path}")

    encoder_path = os.path.join(MODEL_DIR, "encoders.pkl")
    with open(encoder_path, "wb") as f:
        pickle.dump(encoders, f)
    print(f"[+] Encoders saved: {encoder_path}")

    feature_path = os.path.join(MODEL_DIR, "feature_columns.pkl")
    with open(feature_path, "wb") as f:
        pickle.dump(feature_cols, f)
    print(f"[+] Feature columns saved: {feature_path}")

    metrics_path = os.path.join(REPORTS_DIR, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[+] Metrics saved: {metrics_path}")

    # Save reference data stats for drift detection
    ref_stats = {}
    for col in NUMERIC_COLS:
        if col in df.columns:
            ref_stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }
    ref_stats_path = os.path.join(DATA_DIR, "processed", "reference_stats.json")
    with open(ref_stats_path, "w") as f:
        json.dump(ref_stats, f, indent=2)
    print(f"[+] Reference stats saved: {ref_stats_path}")

    # Save processed reference data for drift comparison
    ref_data_path = os.path.join(DATA_DIR, "processed", "reference_data.csv")
    df.to_csv(ref_data_path, index=False)
    print(f"[+] Reference data saved: {ref_data_path}")

    return model, encoders, feature_cols, metrics


if __name__ == "__main__":
    train_model()
