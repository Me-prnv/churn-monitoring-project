"""
Prediction Module for Customer Churn.

Loads the trained model and makes predictions on new data.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
TARGET_COL = "Churn"
DROP_COLS = ["customerID"]


def load_model():
    """Load trained model, encoders, and feature columns."""
    model_path = os.path.join(MODEL_DIR, "churn_model.pkl")
    encoder_path = os.path.join(MODEL_DIR, "encoders.pkl")
    feature_path = os.path.join(MODEL_DIR, "feature_columns.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        encoders = pickle.load(f)
    with open(feature_path, "rb") as f:
        feature_cols = pickle.load(f)

    return model, encoders, feature_cols


def preprocess_new_data(df):
    """Preprocess new data for prediction."""
    df = df.copy()
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Encode target if present
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
        df[TARGET_COL] = df[TARGET_COL].fillna(0).astype(int)

    return df


def predict(df, model=None, encoders=None, feature_cols=None):
    """Make predictions on a DataFrame.

    Returns:
        df with added 'prediction' and 'churn_probability' columns,
        and performance metrics dict if true labels are available.
    """
    if model is None:
        model, encoders, feature_cols = load_model()

    df = preprocess_new_data(df)

    # Encode categorical features
    from model.train import encode_features, CATEGORICAL_COLS
    df_encoded, _ = encode_features(df, encoders=encoders, fit=False)

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    X = df_encoded[feature_cols]

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    df["prediction"] = predictions
    df["churn_probability"] = np.round(probabilities, 4)

    # Compute metrics if true labels available
    metrics = None
    if TARGET_COL in df.columns and df[TARGET_COL].notna().all():
        y_true = df[TARGET_COL]
        metrics = {
            "accuracy": round(accuracy_score(y_true, predictions), 4),
            "precision": round(precision_score(y_true, predictions, zero_division=0), 4),
            "recall": round(recall_score(y_true, predictions, zero_division=0), 4),
            "f1_score": round(f1_score(y_true, predictions, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_true, probabilities), 4),
        }
        print("\n[+] New Data Performance Metrics:")
        for k, v in metrics.items():
            print(f"    {k:>12}: {v}")

        # Save prediction metrics
        os.makedirs(REPORTS_DIR, exist_ok=True)
        metrics_path = os.path.join(REPORTS_DIR, "prediction_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[+] Prediction metrics saved: {metrics_path}")

    return df, metrics


def predict_from_file(filepath):
    """Load CSV and predict."""
    print(f"[*] Loading new data from {filepath}")
    df = pd.read_csv(filepath)
    print(f"    Shape: {df.shape}")
    return predict(df)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        BASE_DIR, "data", "raw", "churn_new_clean.csv"
    )
    result_df, metrics = predict_from_file(path)
    print(f"\n[+] Predictions complete. Churn rate: "
          f"{result_df['prediction'].mean():.2%}")
