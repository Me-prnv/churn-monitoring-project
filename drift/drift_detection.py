"""
Data Drift Detection Module.

Compares reference (training) data against new (current) data to detect
distribution shifts using statistical tests.

Uses:
- Kolmogorov-Smirnov test for numeric features
- Chi-squared test for categorical features
- Population Stability Index (PSI)

Also generates Evidently AI drift reports when available.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

KS_THRESHOLD = 0.05  # p-value threshold for KS test
CHI2_THRESHOLD = 0.05  # p-value threshold for chi-squared test
PSI_THRESHOLD = 0.2  # PSI > 0.2 indicates significant drift


def calculate_psi(reference, current, bins=10):
    """Calculate Population Stability Index between two distributions."""
    # Create bins based on reference data
    breakpoints = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        bins + 1,
    )

    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current, bins=breakpoints)[0]

    # Add small constant to avoid division by zero
    ref_pcts = (ref_counts + 1e-6) / (ref_counts.sum() + 1e-6 * len(ref_counts))
    cur_pcts = (cur_counts + 1e-6) / (cur_counts.sum() + 1e-6 * len(cur_counts))

    psi = np.sum((cur_pcts - ref_pcts) * np.log(cur_pcts / ref_pcts))
    return round(float(psi), 6)


def detect_numeric_drift(ref_df, cur_df):
    """Detect drift in numeric columns using KS test and PSI."""
    results = {}

    for col in NUMERIC_COLS:
        if col not in ref_df.columns or col not in cur_df.columns:
            continue

        ref_vals = pd.to_numeric(ref_df[col], errors="coerce").dropna()
        cur_vals = pd.to_numeric(cur_df[col], errors="coerce").dropna()

        if len(ref_vals) < 10 or len(cur_vals) < 10:
            continue

        # KS test
        ks_stat, ks_pvalue = stats.ks_2samp(ref_vals, cur_vals)

        # PSI
        psi_value = calculate_psi(ref_vals, cur_vals)

        # Statistics comparison
        drift_detected = ks_pvalue < KS_THRESHOLD or psi_value > PSI_THRESHOLD

        results[col] = {
            "type": "numeric",
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pvalue), 6),
            "psi": psi_value,
            "drift_detected": drift_detected,
            "reference_mean": round(float(ref_vals.mean()), 4),
            "current_mean": round(float(cur_vals.mean()), 4),
            "mean_shift": round(float(cur_vals.mean() - ref_vals.mean()), 4),
            "reference_std": round(float(ref_vals.std()), 4),
            "current_std": round(float(cur_vals.std()), 4),
        }

    return results


def detect_categorical_drift(ref_df, cur_df):
    """Detect drift in categorical columns using Chi-squared test."""
    results = {}

    for col in CATEGORICAL_COLS:
        if col not in ref_df.columns or col not in cur_df.columns:
            continue

        ref_vals = ref_df[col].dropna().astype(str)
        cur_vals = cur_df[col].dropna().astype(str)

        # Get all categories from both
        all_categories = sorted(set(ref_vals.unique()) | set(cur_vals.unique()))

        ref_counts = ref_vals.value_counts()
        cur_counts = cur_vals.value_counts()

        # Align categories
        ref_aligned = [ref_counts.get(c, 0) for c in all_categories]
        cur_aligned = [cur_counts.get(c, 0) for c in all_categories]

        # Normalize to same total
        ref_total = sum(ref_aligned)
        cur_total = sum(cur_aligned)
        if ref_total == 0 or cur_total == 0:
            continue

        ref_expected = [r * cur_total / ref_total for r in ref_aligned]

        # Chi-squared test
        try:
            chi2, chi2_pvalue = stats.chisquare(cur_aligned, f_exp=ref_expected)
        except Exception:
            chi2, chi2_pvalue = 0.0, 1.0

        drift_detected = chi2_pvalue < CHI2_THRESHOLD

        # Distribution comparison
        ref_dist = {c: round(v / ref_total, 4) for c, v in zip(all_categories, ref_aligned)}
        cur_dist = {c: round(v / cur_total, 4) for c, v in zip(all_categories, cur_aligned)}

        results[col] = {
            "type": "categorical",
            "chi2_statistic": round(float(chi2), 4),
            "chi2_pvalue": round(float(chi2_pvalue), 6),
            "drift_detected": drift_detected,
            "reference_distribution": ref_dist,
            "current_distribution": cur_dist,
        }

    return results


def detect_drift(ref_df, cur_df, dataset_name="unknown"):
    """Run full drift detection pipeline.

    Args:
        ref_df: Reference (training) DataFrame.
        cur_df: Current (new) DataFrame.
        dataset_name: Name for reporting.

    Returns:
        Drift detection report dict.
    """
    numeric_results = detect_numeric_drift(ref_df, cur_df)
    categorical_results = detect_categorical_drift(ref_df, cur_df)

    all_results = {**numeric_results, **categorical_results}
    drifted_features = [k for k, v in all_results.items() if v.get("drift_detected")]

    report = {
        "dataset_name": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "reference_rows": len(ref_df),
        "current_rows": len(cur_df),
        "total_features_checked": len(all_results),
        "drifted_features_count": len(drifted_features),
        "drifted_features": drifted_features,
        "drift_detected": len(drifted_features) > 0,
        "drift_severity": classify_drift_severity(len(drifted_features), len(all_results)),
        "feature_results": all_results,
    }

    return report


def classify_drift_severity(drifted_count, total_count):
    """Classify overall drift severity."""
    if total_count == 0:
        return "NO_DATA"
    ratio = drifted_count / total_count
    if ratio == 0:
        return "NONE"
    elif ratio < 0.2:
        return "LOW"
    elif ratio < 0.5:
        return "MODERATE"
    else:
        return "HIGH"


def print_drift_report(report):
    """Pretty-print a drift detection report."""
    severity_symbol = {
        "NONE": "[OK]",
        "LOW": "[!!]",
        "MODERATE": "[!!]",
        "HIGH": "[XX]",
        "NO_DATA": "[??]",
    }

    print("\n" + "=" * 60)
    print(f"  DRIFT DETECTION REPORT — {report['dataset_name']}")
    print(f"  Timestamp: {report['timestamp']}")
    print(f"  Reference: {report['reference_rows']} rows  |  Current: {report['current_rows']} rows")
    print("=" * 60)

    symbol = severity_symbol.get(report["drift_severity"], "[??]")
    print(f"\n  {symbol} Overall Drift: {report['drift_severity']}")
    print(f"  Drifted Features: {report['drifted_features_count']}/{report['total_features_checked']}")

    if report["drifted_features"]:
        print(f"\n  Drifted Features List:")
        for feat in report["drifted_features"]:
            result = report["feature_results"][feat]
            if result["type"] == "numeric":
                print(f"    - {feat}: KS p={result['ks_pvalue']:.4f}, "
                      f"PSI={result['psi']:.4f}, "
                      f"mean shift={result['mean_shift']:+.2f}")
            else:
                print(f"    - {feat}: Chi2 p={result['chi2_pvalue']:.4f}")

    print("\n  Feature Details:")
    for feat, result in report["feature_results"].items():
        drift_mark = " ** DRIFT **" if result["drift_detected"] else ""
        if result["type"] == "numeric":
            print(f"    {feat:>20}: KS={result['ks_statistic']:.4f} "
                  f"(p={result['ks_pvalue']:.4f}), PSI={result['psi']:.4f}{drift_mark}")
        else:
            print(f"    {feat:>20}: Chi2={result['chi2_statistic']:.4f} "
                  f"(p={result['chi2_pvalue']:.4f}){drift_mark}")

    print("=" * 60 + "\n")


def save_drift_report(report, filename=None):
    """Save drift report as JSON."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    if filename is None:
        filename = f"drift_report_{report['dataset_name']}.json"
    path = os.path.join(REPORTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[+] Drift report saved: {path}")
    return path


def generate_evidently_report(ref_df, cur_df, report_name="evidently_drift"):
    """Generate an Evidently AI HTML drift report (if installed)."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset, DataSummaryPreset

        drift_report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])

        # Drop non-feature columns
        drop = ["customerID"]
        ref_clean = ref_df.drop(columns=[c for c in drop if c in ref_df.columns])
        cur_clean = cur_df.drop(columns=[c for c in drop if c in cur_df.columns])

        snapshot = drift_report.run(reference_data=ref_clean, current_data=cur_clean)

        os.makedirs(REPORTS_DIR, exist_ok=True)
        html_path = os.path.join(REPORTS_DIR, f"{report_name}.html")
        snapshot.save_html(html_path)
        print(f"[+] Evidently HTML report saved: {html_path}")
        return html_path
    except ImportError:
        print("[!] Evidently not installed. Skipping HTML report generation.")
        print("    Install with: pip install evidently")
        return None
    except Exception as e:
        print(f"[!] Evidently report generation failed: {e}")
        return None


def run_drift_detection(ref_path, cur_path, dataset_name=None):
    """Full drift detection from file paths."""
    ref_df = pd.read_csv(ref_path)
    cur_df = pd.read_csv(cur_path)

    if dataset_name is None:
        dataset_name = os.path.splitext(os.path.basename(cur_path))[0]

    print(f"\n[*] Running drift detection...")
    print(f"    Reference: {ref_path} ({len(ref_df)} rows)")
    print(f"    Current:   {cur_path} ({len(cur_df)} rows)")

    report = detect_drift(ref_df, cur_df, dataset_name=dataset_name)
    print_drift_report(report)
    save_drift_report(report)

    # Try Evidently HTML report
    generate_evidently_report(ref_df, cur_df, report_name=f"evidently_{dataset_name}")

    return report


if __name__ == "__main__":
    import sys

    ref = os.path.join(DATA_DIR, "raw", "churn_train.csv")

    if len(sys.argv) > 1:
        cur = sys.argv[1]
    else:
        cur = os.path.join(DATA_DIR, "raw", "churn_new_drifted.csv")

    run_drift_detection(ref, cur)
