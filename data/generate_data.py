"""
Synthetic Telco Customer Churn Dataset Generator.

Generates data mimicking the IBM Telco Customer Churn dataset from Kaggle.
Replace this with the real dataset by downloading from:
https://www.kaggle.com/blastchar/telco-customer-churn

Usage:
    python data/generate_data.py
"""

import numpy as np
import pandas as pd
import os

SEED = 42


def generate_churn_data(n_samples=7043, seed=SEED, drift=False, drift_intensity=0.0):
    """Generate synthetic telco customer churn data.

    Args:
        n_samples: Number of rows to generate.
        seed: Random seed for reproducibility.
        drift: If True, introduce distribution shifts to simulate data drift.
        drift_intensity: Float 0-1 controlling how much drift to introduce.
    """
    rng = np.random.RandomState(seed)

    # --- Customer demographics ---
    gender = rng.choice(["Male", "Female"], n_samples)
    senior_citizen = rng.choice([0, 1], n_samples, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n_samples, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n_samples, p=[0.30, 0.70])

    # --- Account info ---
    tenure = rng.randint(0, 73, n_samples)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        n_samples,
        p=[0.55, 0.21, 0.24],
    )
    paperless_billing = rng.choice(["Yes", "No"], n_samples, p=[0.59, 0.41])
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        n_samples,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    # --- Services ---
    phone_service = rng.choice(["Yes", "No"], n_samples, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], n_samples, p=[0.42, 0.58]),
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], n_samples, p=[0.34, 0.44, 0.22]
    )
    online_security = np.where(
        internet_service == "No",
        "No internet service",
        rng.choice(["Yes", "No"], n_samples, p=[0.29, 0.71]),
    )
    online_backup = np.where(
        internet_service == "No",
        "No internet service",
        rng.choice(["Yes", "No"], n_samples, p=[0.34, 0.66]),
    )
    device_protection = np.where(
        internet_service == "No",
        "No internet service",
        rng.choice(["Yes", "No"], n_samples, p=[0.34, 0.66]),
    )
    tech_support = np.where(
        internet_service == "No",
        "No internet service",
        rng.choice(["Yes", "No"], n_samples, p=[0.29, 0.71]),
    )
    streaming_tv = np.where(
        internet_service == "No",
        "No internet service",
        rng.choice(["Yes", "No"], n_samples, p=[0.38, 0.62]),
    )
    streaming_movies = np.where(
        internet_service == "No",
        "No internet service",
        rng.choice(["Yes", "No"], n_samples, p=[0.39, 0.61]),
    )

    # --- Charges ---
    monthly_charges = np.round(rng.uniform(18, 118, n_samples), 2)
    total_charges = np.round(monthly_charges * tenure + rng.uniform(-50, 50, n_samples), 2)
    total_charges = np.clip(total_charges, 0, None)

    # --- Target: Churn ---
    # Base churn probability driven by contract type, tenure, and charges
    churn_prob = np.full(n_samples, 0.27)
    churn_prob[contract == "Month-to-month"] += 0.15
    churn_prob[contract == "Two year"] -= 0.15
    churn_prob[tenure < 12] += 0.10
    churn_prob[tenure > 48] -= 0.10
    churn_prob[monthly_charges > 80] += 0.08
    churn_prob = np.clip(churn_prob, 0.05, 0.85)
    churn = np.where(rng.rand(n_samples) < churn_prob, "Yes", "No")

    # --- Apply drift if requested ---
    if drift and drift_intensity > 0:
        n_drift = int(n_samples * drift_intensity)
        idx = rng.choice(n_samples, n_drift, replace=False)

        # Shift monthly charges upward
        monthly_charges[idx] += rng.uniform(20, 60, n_drift)
        monthly_charges = np.round(monthly_charges, 2)

        # Shift contract distribution toward month-to-month
        contract[idx] = rng.choice(
            ["Month-to-month", "One year", "Two year"],
            n_drift,
            p=[0.80, 0.12, 0.08],
        )

        # Increase fiber optic users
        internet_service[idx] = rng.choice(
            ["DSL", "Fiber optic", "No"], n_drift, p=[0.15, 0.70, 0.15]
        )

        # More senior citizens
        senior_citizen[idx] = rng.choice([0, 1], n_drift, p=[0.60, 0.40])

    # --- Build DataFrame ---
    df = pd.DataFrame(
        {
            "customerID": [f"CUST-{i:05d}" for i in range(n_samples)],
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn,
        }
    )
    return df


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Reference / training data ---
    df_train = generate_churn_data(n_samples=5000, seed=42, drift=False)
    train_path = os.path.join(base_dir, "raw", "churn_train.csv")
    df_train.to_csv(train_path, index=False)
    print(f"[+] Training data saved: {train_path}  ({len(df_train)} rows)")

    # --- Clean new data (no drift) ---
    df_new_clean = generate_churn_data(n_samples=1500, seed=99, drift=False)
    clean_path = os.path.join(base_dir, "raw", "churn_new_clean.csv")
    df_new_clean.to_csv(clean_path, index=False)
    print(f"[+] Clean new data saved: {clean_path}  ({len(df_new_clean)} rows)")

    # --- Drifted new data ---
    df_new_drift = generate_churn_data(
        n_samples=1500, seed=77, drift=True, drift_intensity=0.6
    )
    drift_path = os.path.join(base_dir, "raw", "churn_new_drifted.csv")
    df_new_drift.to_csv(drift_path, index=False)
    print(f"[+] Drifted new data saved: {drift_path}  ({len(df_new_drift)} rows)")

    # --- Corrupted data (for validation testing) ---
    df_corrupt = df_new_clean.copy()
    rng = np.random.RandomState(123)
    # Inject missing values
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        mask = rng.rand(len(df_corrupt)) < 0.08
        df_corrupt.loc[mask, col] = np.nan
    # Inject invalid tenure
    bad_idx = rng.choice(len(df_corrupt), 30, replace=False)
    df_corrupt.loc[bad_idx, "tenure"] = -rng.randint(1, 20, 30)
    # Inject unexpected categories
    bad_idx2 = rng.choice(len(df_corrupt), 20, replace=False)
    df_corrupt.loc[bad_idx2, "Contract"] = "Weekly"
    # Inject duplicates
    dup_rows = df_corrupt.sample(40, random_state=rng)
    df_corrupt = pd.concat([df_corrupt, dup_rows], ignore_index=True)
    corrupt_path = os.path.join(base_dir, "raw", "churn_new_corrupted.csv")
    df_corrupt.to_csv(corrupt_path, index=False)
    print(f"[+] Corrupted new data saved: {corrupt_path}  ({len(df_corrupt)} rows)")


if __name__ == "__main__":
    main()
