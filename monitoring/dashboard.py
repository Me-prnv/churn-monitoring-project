"""
Streamlit Monitoring Dashboard.

Displays:
- Data validation results
- Drift detection summary
- Model performance metrics
- Alerts and recommendations

Run: streamlit run monitoring/dashboard.py
"""

import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_json(filepath):
    """Load a JSON report file."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None


def get_report_files():
    """Find all report files in the reports directory."""
    reports = {
        "validation": [],
        "drift": [],
        "metrics": [],
    }
    if not os.path.exists(REPORTS_DIR):
        return reports

    for fname in os.listdir(REPORTS_DIR):
        fpath = os.path.join(REPORTS_DIR, fname)
        if fname.startswith("validation_report") and fname.endswith(".json"):
            reports["validation"].append(fpath)
        elif fname.startswith("drift_report") and fname.endswith(".json"):
            reports["drift"].append(fpath)
        elif fname.endswith("_metrics.json"):
            reports["metrics"].append(fpath)

    return reports


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="ML Pipeline Monitor",
    page_icon="📊",
    layout="wide",
)

st.title("ML Pipeline Monitoring Dashboard")
st.markdown("**Automated Data Quality & Drift Detection for Customer Churn Prediction**")
st.markdown("---")

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Data Validation", "Drift Detection", "Model Performance", "Reports Viewer", "Run Pipeline"],
)

report_files = get_report_files()


# ============================================================
# OVERVIEW PAGE
# ============================================================
if page == "Overview":
    st.header("System Overview")

    col1, col2, col3 = st.columns(3)

    # Validation status
    val_reports = report_files["validation"]
    if val_reports:
        latest_val = load_json(sorted(val_reports)[-1])
        status = latest_val.get("overall_status", "UNKNOWN") if latest_val else "NO DATA"
        col1.metric(
            "Data Validation",
            status,
            delta="Issues found" if status == "FAIL" else "Clean" if status == "PASS" else "Check warnings",
        )
    else:
        col1.metric("Data Validation", "NO DATA", delta="Run pipeline first")

    # Drift status
    drift_reports = report_files["drift"]
    if drift_reports:
        latest_drift = load_json(sorted(drift_reports)[-1])
        severity = latest_drift.get("drift_severity", "UNKNOWN") if latest_drift else "NO DATA"
        drifted = latest_drift.get("drifted_features_count", 0) if latest_drift else 0
        col2.metric(
            "Data Drift",
            severity,
            delta=f"{drifted} features drifted",
        )
    else:
        col2.metric("Data Drift", "NO DATA", delta="Run pipeline first")

    # Model performance
    train_metrics = load_json(os.path.join(REPORTS_DIR, "training_metrics.json"))
    pred_metrics = load_json(os.path.join(REPORTS_DIR, "prediction_metrics.json"))

    if train_metrics:
        col3.metric(
            "Model Accuracy",
            f"{train_metrics.get('accuracy', 0):.1%}",
            delta=f"F1: {train_metrics.get('f1_score', 0):.3f}",
        )
    else:
        col3.metric("Model Performance", "NO DATA", delta="Train model first")

    # Alerts section
    st.markdown("---")
    st.subheader("Alerts")

    alerts = []
    if val_reports:
        latest_val = load_json(sorted(val_reports)[-1])
        if latest_val and latest_val.get("overall_status") == "FAIL":
            for issue in latest_val.get("issues", []):
                alerts.append(("error", f"Validation: {issue}"))
        if latest_val:
            for warn in latest_val.get("warnings", []):
                alerts.append(("warning", f"Validation: {warn}"))

    if drift_reports:
        latest_drift = load_json(sorted(drift_reports)[-1])
        if latest_drift and latest_drift.get("drift_detected"):
            alerts.append(("error", f"Drift detected in {latest_drift['drifted_features_count']} features: {', '.join(latest_drift['drifted_features'])}"))

    if pred_metrics and train_metrics:
        acc_drop = train_metrics["accuracy"] - pred_metrics["accuracy"]
        if acc_drop > 0.05:
            alerts.append(("error", f"Model accuracy dropped by {acc_drop:.1%} on new data"))

    if alerts:
        for level, msg in alerts:
            if level == "error":
                st.error(msg)
            else:
                st.warning(msg)
    else:
        st.success("No alerts. System is healthy.")

    # Architecture diagram
    st.markdown("---")
    st.subheader("Pipeline Architecture")
    st.code("""
    Data Source (CSV)
         |
    Data Validation Layer  -->  Validation Report
         |
    Drift Detection  ---------->  Drift Report
         |
    Model Training / Prediction ->  Performance Metrics
         |
    Monitoring Dashboard  <----  All Reports
    """, language="text")


# ============================================================
# DATA VALIDATION PAGE
# ============================================================
elif page == "Data Validation":
    st.header("Data Validation Results")

    val_reports = report_files["validation"]
    if not val_reports:
        st.info("No validation reports found. Run the pipeline first.")
    else:
        selected = st.selectbox(
            "Select Report",
            sorted(val_reports),
            format_func=lambda x: os.path.basename(x),
        )
        report = load_json(selected)

        if report:
            # Status banner
            status = report["overall_status"]
            if status == "PASS":
                st.success(f"Overall Status: {status}")
            elif status == "WARNING":
                st.warning(f"Overall Status: {status}")
            else:
                st.error(f"Overall Status: {status}")

            st.markdown(f"**Dataset:** {report['dataset_name']}  |  "
                       f"**Rows:** {report['num_rows']}  |  "
                       f"**Columns:** {report['num_columns']}  |  "
                       f"**Time:** {report['timestamp']}")

            # Check details
            st.subheader("Validation Checks")
            for key, check in report.get("checks", {}).items():
                with st.expander(f"{check['name']} — {check['status']}", expanded=check["status"] != "PASS"):
                    if key == "column_existence":
                        if check.get("missing_columns"):
                            st.error(f"Missing columns: {check['missing_columns']}")
                        if check.get("extra_columns"):
                            st.warning(f"Extra columns: {check['extra_columns']}")
                        if check["status"] == "PASS":
                            st.success("All expected columns present.")

                    elif key == "data_types":
                        if check.get("mismatched"):
                            df_types = pd.DataFrame([
                                {"Column": col, "Expected": info["expected"], "Actual": info["actual"]}
                                for col, info in check["mismatched"].items()
                            ])
                            st.dataframe(df_types, use_container_width=True)
                        else:
                            st.success("All data types match expected schema.")

                    elif key == "missing_values":
                        cols_with_nulls = check.get("columns_with_nulls", {})
                        if cols_with_nulls:
                            df_nulls = pd.DataFrame([
                                {"Column": col, "Null Count": info["null_count"],
                                 "Null %": info["null_percentage"],
                                 "Exceeds Threshold": info["exceeds_threshold"]}
                                for col, info in cols_with_nulls.items()
                            ])
                            st.dataframe(df_nulls, use_container_width=True)

                            # Visualize
                            fig, ax = plt.subplots(figsize=(10, 4))
                            cols = list(cols_with_nulls.keys())
                            pcts = [cols_with_nulls[c]["null_percentage"] for c in cols]
                            colors = ["red" if cols_with_nulls[c]["exceeds_threshold"] else "orange" for c in cols]
                            ax.barh(cols, pcts, color=colors)
                            ax.set_xlabel("Missing %")
                            ax.set_title("Missing Values by Column")
                            ax.axvline(x=5.0, color="red", linestyle="--", label="Threshold (5%)")
                            ax.legend()
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.success("No missing values detected.")

                    elif key == "numeric_ranges":
                        oor = check.get("out_of_range_columns", {})
                        if oor:
                            df_range = pd.DataFrame([
                                {"Column": col, "Min": info["actual_min"], "Max": info["actual_max"],
                                 "Expected Min": info["expected_min"], "Expected Max": info["expected_max"],
                                 "Below Min": info["below_min_count"], "Above Max": info["above_max_count"]}
                                for col, info in oor.items()
                            ])
                            st.dataframe(df_range, use_container_width=True)
                        else:
                            st.success("All numeric values within expected ranges.")

                    elif key == "duplicates":
                        dup_count = check.get("duplicate_count", 0)
                        if dup_count > 0:
                            st.warning(f"Found {dup_count} duplicate rows ({check.get('duplicate_percentage', 0)}%)")
                        else:
                            st.success("No duplicate rows found.")

                    elif key == "categories":
                        unexpected = check.get("unexpected_categories", {})
                        if unexpected:
                            for col, info in unexpected.items():
                                st.error(f"**{col}**: unexpected values {info['unexpected_values']} ({info['count']} rows)")
                        else:
                            st.success("All categorical values are expected.")

            # Issues and Warnings summary
            if report.get("issues"):
                st.subheader("Issues")
                for issue in report["issues"]:
                    st.error(issue)
            if report.get("warnings"):
                st.subheader("Warnings")
                for warn in report["warnings"]:
                    st.warning(warn)


# ============================================================
# DRIFT DETECTION PAGE
# ============================================================
elif page == "Drift Detection":
    st.header("Data Drift Detection")

    drift_reports = report_files["drift"]
    if not drift_reports:
        st.info("No drift reports found. Run the pipeline first.")
    else:
        selected = st.selectbox(
            "Select Report",
            sorted(drift_reports),
            format_func=lambda x: os.path.basename(x),
        )
        report = load_json(selected)

        if report:
            # Status banner
            severity = report.get("drift_severity", "UNKNOWN")
            if severity == "NONE":
                st.success(f"Drift Severity: {severity}")
            elif severity in ("LOW", "MODERATE"):
                st.warning(f"Drift Severity: {severity}")
            else:
                st.error(f"Drift Severity: {severity}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Features Checked", report["total_features_checked"])
            col2.metric("Features Drifted", report["drifted_features_count"])
            col3.metric("Drift Detected", "Yes" if report["drift_detected"] else "No")

            # Drifted features list
            if report["drifted_features"]:
                st.subheader("Drifted Features")
                for feat in report["drifted_features"]:
                    result = report["feature_results"][feat]
                    if result["type"] == "numeric":
                        st.markdown(f"- **{feat}**: KS p-value={result['ks_pvalue']:.4f}, "
                                   f"PSI={result['psi']:.4f}, "
                                   f"Mean shift: {result['reference_mean']:.2f} -> {result['current_mean']:.2f}")
                    else:
                        st.markdown(f"- **{feat}**: Chi-squared p-value={result['chi2_pvalue']:.4f}")

            # Numeric features detail
            st.subheader("Numeric Feature Drift")
            numeric_data = []
            for feat, result in report["feature_results"].items():
                if result["type"] == "numeric":
                    numeric_data.append({
                        "Feature": feat,
                        "KS Statistic": result["ks_statistic"],
                        "KS p-value": result["ks_pvalue"],
                        "PSI": result["psi"],
                        "Ref Mean": result["reference_mean"],
                        "Cur Mean": result["current_mean"],
                        "Mean Shift": result["mean_shift"],
                        "Drift": "Yes" if result["drift_detected"] else "No",
                    })
            if numeric_data:
                st.dataframe(pd.DataFrame(numeric_data), use_container_width=True)

                # PSI bar chart
                PSI_THRESHOLD = 0.2
                fig, ax = plt.subplots(figsize=(10, 4))
                names = [d["Feature"] for d in numeric_data]
                psi_vals = [d["PSI"] for d in numeric_data]
                colors = ["red" if p > PSI_THRESHOLD else "green" for p in psi_vals]
                ax.bar(names, psi_vals, color=colors)
                ax.axhline(y=PSI_THRESHOLD, color="red", linestyle="--", label=f"Threshold ({PSI_THRESHOLD})")
                ax.set_ylabel("PSI Score")
                ax.set_title("Population Stability Index by Feature")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            # Categorical features detail
            st.subheader("Categorical Feature Drift")
            cat_data = []
            for feat, result in report["feature_results"].items():
                if result["type"] == "categorical":
                    cat_data.append({
                        "Feature": feat,
                        "Chi2 Statistic": result["chi2_statistic"],
                        "Chi2 p-value": result["chi2_pvalue"],
                        "Drift": "Yes" if result["drift_detected"] else "No",
                    })
            if cat_data:
                st.dataframe(pd.DataFrame(cat_data), use_container_width=True)

            # Evidently HTML report link
            evidently_path = os.path.join(REPORTS_DIR, f"evidently_{report['dataset_name']}.html")
            if os.path.exists(evidently_path):
                st.subheader("Evidently AI Report")
                st.markdown(f"Full interactive report available at: `{evidently_path}`")
                with open(evidently_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=600, scrolling=True)


# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================
elif page == "Model Performance":
    st.header("Model Performance Monitoring")

    train_metrics = load_json(os.path.join(REPORTS_DIR, "training_metrics.json"))
    pred_metrics = load_json(os.path.join(REPORTS_DIR, "prediction_metrics.json"))

    if not train_metrics:
        st.info("No training metrics found. Train the model first.")
    else:
        st.subheader("Training Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{train_metrics['accuracy']:.1%}")
        col2.metric("Precision", f"{train_metrics['precision']:.1%}")
        col3.metric("Recall", f"{train_metrics['recall']:.1%}")
        col4.metric("F1 Score", f"{train_metrics['f1_score']:.3f}")
        col5.metric("ROC AUC", f"{train_metrics['roc_auc']:.3f}")

    if pred_metrics:
        st.subheader("New Data Prediction Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)

        acc_delta = pred_metrics["accuracy"] - train_metrics["accuracy"] if train_metrics else 0
        f1_delta = pred_metrics["f1_score"] - train_metrics["f1_score"] if train_metrics else 0

        col1.metric("Accuracy", f"{pred_metrics['accuracy']:.1%}", delta=f"{acc_delta:+.1%}")
        col2.metric("Precision", f"{pred_metrics['precision']:.1%}")
        col3.metric("Recall", f"{pred_metrics['recall']:.1%}")
        col4.metric("F1 Score", f"{pred_metrics['f1_score']:.3f}", delta=f"{f1_delta:+.3f}")
        col5.metric("ROC AUC", f"{pred_metrics['roc_auc']:.3f}")

        # Performance comparison chart
        if train_metrics:
            st.subheader("Training vs New Data Performance")
            fig, ax = plt.subplots(figsize=(10, 5))
            metrics_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
            train_vals = [train_metrics[m] for m in metrics_names]
            pred_vals = [pred_metrics[m] for m in metrics_names]

            x = np.arange(len(metrics_names))
            width = 0.35
            ax.bar(x - width / 2, train_vals, width, label="Training", color="steelblue")
            ax.bar(x + width / 2, pred_vals, width, label="New Data", color="coral")
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace("_", " ").title() for m in metrics_names])
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("Score")
            ax.set_title("Model Performance Comparison")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Alerts
            if acc_delta < -0.05:
                st.error(f"Model accuracy dropped by {abs(acc_delta):.1%} on new data. "
                        f"Consider retraining the model.")
            elif acc_delta < -0.02:
                st.warning(f"Minor accuracy drop of {abs(acc_delta):.1%} detected. Monitor closely.")
            else:
                st.success("Model performance is stable on new data.")


# ============================================================
# RUN PIPELINE PAGE
# ============================================================
elif page == "Run Pipeline":
    st.header("Run Monitoring Pipeline")
    st.markdown("Execute the full pipeline from here.")

    st.subheader("1. Generate Data")
    if st.button("Generate Synthetic Data"):
        with st.spinner("Generating data..."):
            result = subprocess.run(
                [sys.executable, os.path.join(DATA_DIR, "generate_data.py")],
                capture_output=True, text=True
            )
        if result.returncode == 0:
            st.success("Data generated. Check data/raw/ directory.")
            if result.stdout:
                st.code(result.stdout)
        else:
            st.error(f"Data generation failed:\n{result.stderr}")

    st.subheader("2. Train Model")
    if st.button("Train Churn Model"):
        with st.spinner("Training model..."):
            result = subprocess.run(
                [sys.executable, os.path.join(BASE_DIR, "model", "train.py")],
                capture_output=True, text=True
            )
        if result.returncode == 0:
            st.success("Model training complete.")
            if result.stdout:
                st.code(result.stdout)
            st.rerun()
        else:
            st.error(f"Model training failed:\n{result.stderr}")

    st.subheader("3. Run Full Pipeline")
    data_option = st.selectbox(
        "Select data for validation & drift detection:",
        ["churn_new_clean.csv", "churn_new_drifted.csv", "churn_new_corrupted.csv"],
    )

    if st.button("Run Full Pipeline"):
        with st.spinner("Running pipeline..."):
            data_path = os.path.join(DATA_DIR, "raw", data_option)
            result = subprocess.run(
                [sys.executable, os.path.join(BASE_DIR, "main.py"), data_path],
                capture_output=True, text=True
            )
        if result.returncode == 0:
            st.success("Pipeline complete. Navigate to other pages to see results.")
            if result.stdout:
                st.code(result.stdout)
            st.rerun()
        else:
            st.error(f"Pipeline failed:\n{result.stderr}")
            if result.stdout:
                st.code(result.stdout)

    st.subheader("4. Report Files")
    if os.path.exists(REPORTS_DIR):
        files = os.listdir(REPORTS_DIR)
        if files:
            st.write("Available reports:")
            for f in sorted(files):
                st.text(f"  - {f}")
        else:
            st.info("No reports generated yet.")
    else:
        st.info("Reports directory does not exist yet.")


# ============================================================
# REPORTS VIEWER PAGE
# ============================================================
elif page == "Reports Viewer":
    st.header("📄 Reports Viewer")
    st.markdown("Browse and inspect all generated pipeline reports.")

    if not os.path.exists(REPORTS_DIR):
        st.info("Reports directory does not exist yet. Run the pipeline first.")
    else:
        all_files = sorted(os.listdir(REPORTS_DIR))
        json_files = [f for f in all_files if f.endswith(".json")]
        html_files = [f for f in all_files if f.endswith(".html")]

        if not all_files:
            st.info("No reports generated yet. Run the pipeline first.")
        else:
            # --- Summary ---
            st.subheader("Report Inventory")
            col1, col2, col3 = st.columns(3)
            col1.metric("JSON Reports", len(json_files))
            col2.metric("HTML Reports", len(html_files))
            col3.metric("Total Files", len(all_files))

            st.markdown("---")

            # --- JSON Report Viewer ---
            st.subheader("JSON Report Viewer")
            if json_files:
                selected_json = st.selectbox(
                    "Select a JSON report to inspect:",
                    json_files,
                    key="json_selector",
                )
                json_path = os.path.join(REPORTS_DIR, selected_json)
                report_data = load_json(json_path)

                if report_data:
                    # File metadata
                    file_stat = os.stat(json_path)
                    from datetime import datetime as dt
                    modified_time = dt.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    file_size_kb = round(file_stat.st_size / 1024, 2)
                    st.caption(f"Last modified: {modified_time}  |  Size: {file_size_kb} KB")

                    # Quick summary based on report type
                    if "overall_status" in report_data:
                        status = report_data["overall_status"]
                        if status == "PASS":
                            st.success(f"Validation Status: {status}")
                        elif status == "WARNING":
                            st.warning(f"Validation Status: {status}")
                        else:
                            st.error(f"Validation Status: {status}")

                    if "drift_severity" in report_data:
                        severity = report_data["drift_severity"]
                        if severity == "NONE":
                            st.success(f"Drift Severity: {severity}")
                        elif severity in ("LOW", "MODERATE"):
                            st.warning(f"Drift Severity: {severity}")
                        else:
                            st.error(f"Drift Severity: {severity}")

                    if "overall_health" in report_data:
                        health = report_data["overall_health"]
                        if health == "HEALTHY":
                            st.success(f"Pipeline Health: {health}")
                        elif health == "DEGRADED":
                            st.warning(f"Pipeline Health: {health}")
                        else:
                            st.error(f"Pipeline Health: {health}")

                    # Expandable full JSON
                    with st.expander("View Full JSON Content", expanded=False):
                        st.json(report_data)

                    # Download button
                    json_str = json.dumps(report_data, indent=2, default=str)
                    st.download_button(
                        label=f"⬇️ Download {selected_json}",
                        data=json_str,
                        file_name=selected_json,
                        mime="application/json",
                    )
            else:
                st.info("No JSON reports found.")

            st.markdown("---")

            # --- HTML Report Viewer ---
            st.subheader("Evidently HTML Reports")
            if html_files:
                selected_html = st.selectbox(
                    "Select an HTML report to view:",
                    html_files,
                    key="html_selector",
                )
                html_path = os.path.join(REPORTS_DIR, selected_html)

                file_stat = os.stat(html_path)
                from datetime import datetime as dt
                modified_time = dt.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                file_size_kb = round(file_stat.st_size / 1024, 2)
                st.caption(f"Last modified: {modified_time}  |  Size: {file_size_kb} KB")

                # Embed HTML report
                with open(html_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=800, scrolling=True)

                # Download button
                st.download_button(
                    label=f"⬇️ Download {selected_html}",
                    data=html_content,
                    file_name=selected_html,
                    mime="text/html",
                )
            else:
                st.info("No HTML reports found. Run drift detection to generate Evidently reports.")

            st.markdown("---")

            # --- All Files Table ---
            st.subheader("All Report Files")
            file_info = []
            for fname in all_files:
                fpath = os.path.join(REPORTS_DIR, fname)
                fstat = os.stat(fpath)
                from datetime import datetime as dt
                file_info.append({
                    "File Name": fname,
                    "Type": fname.split(".")[-1].upper(),
                    "Size (KB)": round(fstat.st_size / 1024, 2),
                    "Last Modified": dt.fromtimestamp(fstat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                })
            st.dataframe(pd.DataFrame(file_info), use_container_width=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "*Automated Data Validation & Drift Detection Framework for Customer Churn Prediction*  \n"
    "Major Project | MLOps Pipeline Monitoring"
)
