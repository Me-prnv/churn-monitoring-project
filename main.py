"""
Main Pipeline Orchestrator.

Automates the full monitoring pipeline:
1. Load new data
2. Run data validation
3. Run drift detection
4. Run model predictions
5. Compute performance metrics
6. Save all reports
7. Print summary with alerts

Usage:
    python main.py                              # Use default drifted data
    python main.py data/raw/churn_new_clean.csv # Use specific file
"""

import os
import sys
import json
import subprocess
import pandas as pd
from datetime import datetime

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from validation.data_validation import validate_data, print_report, save_report
from drift.drift_detection import detect_drift, print_drift_report, save_drift_report, generate_evidently_report
from model.train import train_model, load_and_preprocess, encode_features

DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODEL_DIR = os.path.join(BASE_DIR, "model")

PERFORMANCE_DROP_THRESHOLD = 0.05  # 5% accuracy drop triggers alert


def load_model_artifacts():
    """Load trained model and associated artifacts."""
    import pickle
    model_path = os.path.join(MODEL_DIR, "churn_model.pkl")
    encoder_path = os.path.join(MODEL_DIR, "encoders.pkl")
    feature_path = os.path.join(MODEL_DIR, "feature_columns.pkl")

    if not all(os.path.exists(p) for p in [model_path, encoder_path, feature_path]):
        return None, None, None

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        encoders = pickle.load(f)
    with open(feature_path, "rb") as f:
        feature_cols = pickle.load(f)

    return model, encoders, feature_cols


def run_predictions(df, model, encoders, feature_cols):
    """Run predictions and compute metrics."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score,
    )
    import numpy as np

    df_proc = df.copy()
    df_proc = df_proc.drop(columns=["customerID"], errors="ignore")
    df_proc["TotalCharges"] = pd.to_numeric(df_proc["TotalCharges"], errors="coerce")

    for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].fillna(df_proc[col].median())

    has_target = "Churn" in df_proc.columns
    if has_target:
        df_proc["Churn"] = df_proc["Churn"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    df_encoded, _ = encode_features(df_proc, encoders=encoders, fit=False)

    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    X = df_encoded[feature_cols]
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    metrics = None
    if has_target:
        y_true = df_proc["Churn"]
        metrics = {
            "accuracy": round(accuracy_score(y_true, predictions), 4),
            "precision": round(precision_score(y_true, predictions, zero_division=0), 4),
            "recall": round(recall_score(y_true, predictions, zero_division=0), 4),
            "f1_score": round(f1_score(y_true, predictions, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_true, probabilities), 4),
        }

    return predictions, probabilities, metrics


def check_performance_alerts(train_metrics, pred_metrics):
    """Check for performance degradation and generate alerts."""
    alerts = []

    if train_metrics is None or pred_metrics is None:
        return alerts

    for metric in ["accuracy", "f1_score", "roc_auc"]:
        train_val = train_metrics.get(metric, 0)
        pred_val = pred_metrics.get(metric, 0)
        drop = train_val - pred_val

        if drop > PERFORMANCE_DROP_THRESHOLD:
            alerts.append({
                "level": "CRITICAL",
                "metric": metric,
                "message": f"{metric} dropped by {drop:.1%} "
                           f"(train: {train_val:.4f} -> new: {pred_val:.4f}). "
                           f"Consider retraining the model.",
            })
        elif drop > 0.02:
            alerts.append({
                "level": "WARNING",
                "metric": metric,
                "message": f"{metric} decreased by {drop:.1%}. Monitor closely.",
            })

    return alerts


def run_pipeline(new_data_path=None):
    """Execute the full monitoring pipeline."""
    print("\n" + "=" * 70)
    print("  AUTOMATED ML PIPELINE MONITORING")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ---- Step 0: Check if model exists, train if needed ----
    model, encoders, feature_cols = load_model_artifacts()
    if model is None:
        print("\n[*] No trained model found. Training now...")
        train_data = os.path.join(DATA_DIR, "raw", "churn_train.csv")
        if not os.path.exists(train_data):
            print("[!] Training data not found. Generating synthetic data...")
            subprocess.run([sys.executable, os.path.join(DATA_DIR, "generate_data.py")], check=True)
        model, encoders, feature_cols, _ = train_model(train_data)

    # Load training metrics
    train_metrics_path = os.path.join(REPORTS_DIR, "training_metrics.json")
    train_metrics = None
    if os.path.exists(train_metrics_path):
        with open(train_metrics_path, "r") as f:
            train_metrics = json.load(f)

    # ---- Step 1: Load new data ----
    if new_data_path is None:
        new_data_path = os.path.join(DATA_DIR, "raw", "churn_new_drifted.csv")

    if not os.path.exists(new_data_path):
        print(f"[!] Data file not found: {new_data_path}")
        print("[!] Generating synthetic data...")
        subprocess.run([sys.executable, os.path.join(DATA_DIR, "generate_data.py")], check=True)

    print(f"\n{'-' * 70}")
    print("  STEP 1: Loading New Data")
    print(f"{'-' * 70}")
    new_df = pd.read_csv(new_data_path)
    dataset_name = os.path.splitext(os.path.basename(new_data_path))[0]
    print(f"[+] Loaded {len(new_df)} rows from {new_data_path}")

    # ---- Step 2: Data Validation ----
    print(f"\n{'-' * 70}")
    print("  STEP 2: Data Validation")
    print(f"{'-' * 70}")
    val_report = validate_data(new_df, dataset_name=dataset_name)
    print_report(val_report)
    save_report(val_report)

    # ---- Step 3: Drift Detection ----
    print(f"\n{'-' * 70}")
    print("  STEP 3: Drift Detection")
    print(f"{'-' * 70}")
    ref_path = os.path.join(DATA_DIR, "raw", "churn_train.csv")
    ref_df = pd.read_csv(ref_path)
    drift_report = detect_drift(ref_df, new_df, dataset_name=dataset_name)
    print_drift_report(drift_report)
    save_drift_report(drift_report)

    # Generate Evidently HTML report
    generate_evidently_report(ref_df, new_df, report_name=f"evidently_{dataset_name}")

    # ---- Step 4: Model Prediction ----
    print(f"\n{'-' * 70}")
    print("  STEP 4: Model Prediction")
    print(f"{'-' * 70}")
    predictions, probabilities, pred_metrics = run_predictions(
        new_df, model, encoders, feature_cols
    )
    print(f"[+] Predictions complete for {len(predictions)} rows")
    print(f"[+] Predicted churn rate: {predictions.mean():.2%}")

    if pred_metrics:
        print("\n[+] New Data Performance:")
        for k, v in pred_metrics.items():
            print(f"    {k:>12}: {v}")

        # Save prediction metrics
        pred_metrics_path = os.path.join(REPORTS_DIR, "prediction_metrics.json")
        with open(pred_metrics_path, "w") as f:
            json.dump(pred_metrics, f, indent=2)
        print(f"[+] Prediction metrics saved: {pred_metrics_path}")

    # ---- Step 5: Performance Monitoring & Alerts ----
    print(f"\n{'-' * 70}")
    print("  STEP 5: Performance Monitoring")
    print(f"{'-' * 70}")
    alerts = check_performance_alerts(train_metrics, pred_metrics)
    drift_alerts = []

    if drift_report["drift_detected"]:
        drift_alerts.append({
            "level": "WARNING" if drift_report["drift_severity"] in ("LOW", "MODERATE") else "CRITICAL",
            "metric": "data_drift",
            "message": f"Data drift detected in {drift_report['drifted_features_count']} features: "
                       f"{', '.join(drift_report['drifted_features'])}",
        })

    if val_report["overall_status"] == "FAIL":
        drift_alerts.append({
            "level": "CRITICAL",
            "metric": "data_quality",
            "message": f"Data validation failed. Issues: {'; '.join(val_report['issues'])}",
        })

    all_alerts = alerts + drift_alerts

    if all_alerts:
        print(f"\n[!] {len(all_alerts)} Alert(s) Generated:")
        for alert in all_alerts:
            level = alert["level"]
            print(f"    [{level}] {alert['message']}")
    else:
        print("\n[OK] No alerts. Pipeline is healthy.")

    # Save full pipeline report
    pipeline_report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_name,
        "data_path": new_data_path,
        "rows_processed": len(new_df),
        "validation_status": val_report["overall_status"],
        "validation_issues": len(val_report.get("issues", [])),
        "drift_detected": drift_report["drift_detected"],
        "drift_severity": drift_report["drift_severity"],
        "drifted_features": drift_report["drifted_features"],
        "training_metrics": train_metrics,
        "prediction_metrics": pred_metrics,
        "alerts": all_alerts,
        "overall_health": "HEALTHY" if not all_alerts else
                          "DEGRADED" if all(a["level"] == "WARNING" for a in all_alerts) else
                          "CRITICAL",
    }

    pipeline_path = os.path.join(REPORTS_DIR, "pipeline_report.json")
    with open(pipeline_path, "w") as f:
        json.dump(pipeline_report, f, indent=2, default=str)
    print(f"\n[+] Pipeline report saved: {pipeline_path}")

    # ---- Summary ----
    print(f"\n{'=' * 70}")
    print("  PIPELINE SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Dataset:          {dataset_name}")
    print(f"  Rows Processed:   {len(new_df)}")
    print(f"  Validation:       {val_report['overall_status']}")
    print(f"  Drift Severity:   {drift_report['drift_severity']}")
    if pred_metrics:
        print(f"  Accuracy (new):   {pred_metrics['accuracy']:.1%}")
        print(f"  F1 Score (new):   {pred_metrics['f1_score']:.3f}")
    print(f"  Alerts:           {len(all_alerts)}")
    print(f"  Overall Health:   {pipeline_report['overall_health']}")
    print(f"{'=' * 70}\n")

    print("[+] Dashboard: streamlit run monitoring/dashboard.py")

    return pipeline_report


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(data_path)
