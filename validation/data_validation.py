"""
Data Validation Module.

Performs comprehensive data quality checks on incoming data:
- Schema validation (column existence, data types)
- Missing value detection
- Range checks for numeric columns
- Duplicate row detection
- Unexpected category detection

Returns a structured validation report.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# ----- Expected Schema Definition -----
EXPECTED_SCHEMA = {
    "customerID": "object",
    "gender": "object",
    "SeniorCitizen": "int64",
    "Partner": "object",
    "Dependents": "object",
    "tenure": "int64",
    "PhoneService": "object",
    "MultipleLines": "object",
    "InternetService": "object",
    "OnlineSecurity": "object",
    "OnlineBackup": "object",
    "DeviceProtection": "object",
    "TechSupport": "object",
    "StreamingTV": "object",
    "StreamingMovies": "object",
    "Contract": "object",
    "PaperlessBilling": "object",
    "PaymentMethod": "object",
    "MonthlyCharges": "float64",
    "TotalCharges": "float64",
    "Churn": "object",
}

EXPECTED_CATEGORIES = {
    "gender": ["Male", "Female"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    "Churn": ["Yes", "No"],
}

NUMERIC_RANGES = {
    "SeniorCitizen": {"min": 0, "max": 1},
    "tenure": {"min": 0, "max": 120},
    "MonthlyCharges": {"min": 0, "max": 500},
    "TotalCharges": {"min": 0, "max": 100000},
}

NULL_THRESHOLD = 0.05  # Alert if more than 5% nulls in any column


def validate_data(df, dataset_name="unknown"):
    """Run all validation checks and return a structured report.

    Args:
        df: pandas DataFrame to validate.
        dataset_name: Name for the dataset (used in reports).

    Returns:
        dict with validation results including pass/fail status,
        issues found, and summary statistics.
    """
    report = {
        "dataset_name": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "checks": {},
        "issues": [],
        "warnings": [],
        "overall_status": "PASS",
    }

    # ----- 1. Column Existence Check -----
    check_columns(df, report)

    # ----- 2. Data Type Check -----
    check_dtypes(df, report)

    # ----- 3. Missing Value Check -----
    check_missing_values(df, report)

    # ----- 4. Numeric Range Check -----
    check_numeric_ranges(df, report)

    # ----- 5. Duplicate Row Check -----
    check_duplicates(df, report)

    # ----- 6. Unexpected Category Check -----
    check_categories(df, report)

    # ----- Determine overall status -----
    if report["issues"]:
        report["overall_status"] = "FAIL"
    elif report["warnings"]:
        report["overall_status"] = "WARNING"

    return report


def check_columns(df, report):
    """Check if all expected columns exist."""
    expected = set(EXPECTED_SCHEMA.keys())
    actual = set(df.columns)
    missing = expected - actual
    extra = actual - expected

    result = {
        "name": "Column Existence",
        "status": "PASS",
        "missing_columns": list(missing),
        "extra_columns": list(extra),
    }

    if missing:
        result["status"] = "FAIL"
        report["issues"].append(f"Missing columns: {missing}")
    if extra:
        report["warnings"].append(f"Extra columns found: {extra}")

    report["checks"]["column_existence"] = result


def check_dtypes(df, report):
    """Check data types match expected schema."""
    mismatched = {}
    for col, expected_dtype in EXPECTED_SCHEMA.items():
        if col not in df.columns:
            continue
        actual_dtype = str(df[col].dtype)
        # Allow float for int columns (happens with NaN)
        if expected_dtype == "int64" and actual_dtype == "float64":
            continue
        if actual_dtype != expected_dtype:
            mismatched[col] = {"expected": expected_dtype, "actual": actual_dtype}

    result = {
        "name": "Data Types",
        "status": "PASS" if not mismatched else "FAIL",
        "mismatched": mismatched,
    }

    if mismatched:
        report["issues"].append(f"Data type mismatches: {list(mismatched.keys())}")

    report["checks"]["data_types"] = result


def check_missing_values(df, report):
    """Check for missing values exceeding threshold."""
    null_counts = df.isnull().sum()
    null_pcts = (null_counts / len(df)).round(4)

    columns_above_threshold = {}
    for col in df.columns:
        pct = float(null_pcts[col])
        if pct > 0:
            columns_above_threshold[col] = {
                "null_count": int(null_counts[col]),
                "null_percentage": round(pct * 100, 2),
                "exceeds_threshold": pct > NULL_THRESHOLD,
            }

    above = {k: v for k, v in columns_above_threshold.items() if v["exceeds_threshold"]}

    result = {
        "name": "Missing Values",
        "status": "PASS" if not above else "WARNING",
        "total_nulls": int(null_counts.sum()),
        "columns_with_nulls": columns_above_threshold,
    }

    if above:
        report["warnings"].append(
            f"Columns with >5% missing values: {list(above.keys())}"
        )

    report["checks"]["missing_values"] = result


def check_numeric_ranges(df, report):
    """Check numeric columns are within expected ranges."""
    out_of_range = {}

    for col, bounds in NUMERIC_RANGES.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        below = int((series < bounds["min"]).sum())
        above = int((series > bounds["max"]).sum())

        if below > 0 or above > 0:
            out_of_range[col] = {
                "expected_min": bounds["min"],
                "expected_max": bounds["max"],
                "actual_min": float(series.min()) if not series.isna().all() else None,
                "actual_max": float(series.max()) if not series.isna().all() else None,
                "below_min_count": below,
                "above_max_count": above,
            }

    result = {
        "name": "Numeric Ranges",
        "status": "PASS" if not out_of_range else "FAIL",
        "out_of_range_columns": out_of_range,
    }

    if out_of_range:
        report["issues"].append(
            f"Columns with out-of-range values: {list(out_of_range.keys())}"
        )

    report["checks"]["numeric_ranges"] = result


def check_duplicates(df, report):
    """Check for duplicate rows."""
    # Exclude customerID for duplicate check
    check_cols = [c for c in df.columns if c != "customerID"]
    dup_count = int(df.duplicated(subset=check_cols, keep="first").sum())

    result = {
        "name": "Duplicate Rows",
        "status": "PASS" if dup_count == 0 else "WARNING",
        "duplicate_count": dup_count,
        "duplicate_percentage": round(dup_count / len(df) * 100, 2) if len(df) > 0 else 0,
    }

    if dup_count > 0:
        report["warnings"].append(f"Found {dup_count} duplicate rows")

    report["checks"]["duplicates"] = result


def check_categories(df, report):
    """Check for unexpected categories in categorical columns."""
    unexpected = {}

    for col, allowed in EXPECTED_CATEGORIES.items():
        if col not in df.columns:
            continue
        actual_values = set(df[col].dropna().unique())
        bad_values = actual_values - set(allowed)
        if bad_values:
            unexpected[col] = {
                "unexpected_values": list(bad_values),
                "count": int(df[col].isin(bad_values).sum()),
            }

    result = {
        "name": "Category Values",
        "status": "PASS" if not unexpected else "FAIL",
        "unexpected_categories": unexpected,
    }

    if unexpected:
        report["issues"].append(
            f"Unexpected categories in: {list(unexpected.keys())}"
        )

    report["checks"]["categories"] = result


def print_report(report):
    """Pretty-print a validation report."""
    status_symbol = {"PASS": "[OK]", "WARNING": "[!!]", "FAIL": "[XX]"}
    overall = report["overall_status"]

    print("\n" + "=" * 60)
    print(f"  DATA VALIDATION REPORT — {report['dataset_name']}")
    print(f"  Timestamp: {report['timestamp']}")
    print(f"  Rows: {report['num_rows']}  |  Columns: {report['num_columns']}")
    print("=" * 60)

    for key, check in report["checks"].items():
        symbol = status_symbol.get(check["status"], "[??]")
        print(f"\n  {symbol} {check['name']}: {check['status']}")

        if check["status"] != "PASS":
            # Print details depending on check type
            if key == "column_existence" and check.get("missing_columns"):
                print(f"      Missing: {check['missing_columns']}")
            if key == "data_types" and check.get("mismatched"):
                for col, info in check["mismatched"].items():
                    print(f"      {col}: expected {info['expected']}, got {info['actual']}")
            if key == "missing_values":
                for col, info in check.get("columns_with_nulls", {}).items():
                    if info["exceeds_threshold"]:
                        print(f"      {col}: {info['null_percentage']}% missing ({info['null_count']} rows)")
            if key == "numeric_ranges":
                for col, info in check.get("out_of_range_columns", {}).items():
                    print(f"      {col}: {info['below_min_count']} below min, {info['above_max_count']} above max")
                    print(f"        Range: [{info['actual_min']}, {info['actual_max']}] vs expected [{info['expected_min']}, {info['expected_max']}]")
            if key == "duplicates" and check.get("duplicate_count", 0) > 0:
                print(f"      {check['duplicate_count']} duplicate rows ({check['duplicate_percentage']}%)")
            if key == "categories":
                for col, info in check.get("unexpected_categories", {}).items():
                    print(f"      {col}: unexpected values {info['unexpected_values']} ({info['count']} rows)")

    print("\n" + "-" * 60)
    symbol = status_symbol.get(overall, "[??]")
    print(f"  OVERALL: {symbol} {overall}")

    if report["issues"]:
        print(f"\n  Issues ({len(report['issues'])}):")
        for issue in report["issues"]:
            print(f"    - {issue}")
    if report["warnings"]:
        print(f"\n  Warnings ({len(report['warnings'])}):")
        for w in report["warnings"]:
            print(f"    - {w}")

    print("=" * 60 + "\n")


def save_report(report, filename=None):
    """Save validation report as JSON."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    if filename is None:
        filename = f"validation_report_{report['dataset_name']}.json"
    path = os.path.join(REPORTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[+] Validation report saved: {path}")
    return path


def validate_file(filepath, dataset_name=None):
    """Validate a CSV file end-to-end."""
    if dataset_name is None:
        dataset_name = os.path.splitext(os.path.basename(filepath))[0]

    print(f"\n[*] Validating: {filepath}")
    df = pd.read_csv(filepath)
    report = validate_data(df, dataset_name=dataset_name)
    print_report(report)
    save_report(report)
    return report


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        validate_file(sys.argv[1])
    else:
        # Validate all raw data files
        raw_dir = os.path.join(BASE_DIR, "data", "raw")
        if os.path.exists(raw_dir):
            for fname in sorted(os.listdir(raw_dir)):
                if fname.endswith(".csv"):
                    validate_file(os.path.join(raw_dir, fname))
