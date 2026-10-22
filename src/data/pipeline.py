"""Explicit prediction-time feature contract and untouched held-out splits."""

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
TARGET = "total_annual_cost"
NUMERIC_FEATURES = [
    "age",
    "bmi",
    "smoker",
    "chronic_conditions_count",
    "previous_hospitalizations",
    "previous_er_visits",
    "previous_office_visits",
    "medication_count",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "cholesterol_total",
    "glucose_fasting",
]
CATEGORICAL_FEATURES = ["gender", "state", "insurance_type", "vaccination_status"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
# Deny by default: adding a new column never silently changes the model contract.
EXCLUDED_FEATURES = {
    "patient_id": "Identifier; retained only for split integrity checks.",
    "data_date": "Dataset creation date, not a prospective observation date.",
    "zip_code": "Synthetic identifier with no justified predictive use.",
    "chronic_conditions": "Redundant unvalidated diagnostic text; use count only.",
    "total_annual_cost": "Target.",
    "inpatient_cost": "Same-year component of target.",
    "outpatient_cost": "Same-year component of target.",
    "pharmacy_cost": "Same-year component of target.",
    "emergency_cost": "Same-year component of target.",
    "insurance_paid": "Target times coverage rate.",
    "patient_responsibility": "Target minus insurance payment.",
    "payment_status": "Observed after billing; unavailable at prediction time.",
    "outstanding_balance": "Target-derived balance observed after billing.",
    "risk_category": "Thresholded target.",
    "previous_year_cost": "Generator shares current-year random base; chronology unverified.",
    "financial_risk_score": "Ratio derived from target and patient responsibility.",
    "cost_per_visit": "Uses excluded previous_year_cost.",
    "age_group": "Redundant; age is available directly.",
    "bmi_category": "Redundant; BMI is available directly.",
    "high_utilizer": "Old full-dataset quantile; replaced by raw prior counts.",
    "comorbidity_score": "Unvalidated heuristic; use its raw inputs.",
    "insurance_coverage_score": "Unvalidated heuristic; use insurance category.",
    "healthcare_intensity": "Unvalidated heuristic; use raw prior counts.",
}


def make_preprocessor():
    """Return an unfitted transformer; fit only inside a training Pipeline."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def load_dataset(path=ROOT / "data/raw/patient_data.csv"):
    data = pd.read_csv(path, keep_default_na=False, na_values=[""])
    required = ["patient_id", TARGET, *FEATURES]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")
    if (
        len(data) < 50
        or data.patient_id.isna().any()
        or data.patient_id.duplicated().any()
    ):
        raise ValueError("Need at least 50 unique patient records with nonempty IDs")
    numeric = data[[*NUMERIC_FEATURES, TARGET]].apply(pd.to_numeric, errors="raise")
    if numeric.isin([float("inf"), float("-inf")]).any().any():
        raise ValueError("Infinite numeric values are unsupported")
    if data[TARGET].isna().any() or (data[TARGET] < 0).any():
        raise ValueError("Target must be observed and nonnegative")
    # Row-local quality rule; no population statistics or target-dependent filtering.
    invalid_pressure = data.blood_pressure_diastolic >= data.blood_pressure_systolic
    data.loc[
        invalid_pressure, ["blood_pressure_systolic", "blood_pressure_diastolic"]
    ] = float("nan")
    return data


def split_dataset(data, seed=42):
    """60/20/20 patient split before any fitted preprocessing or selection."""
    train, test = train_test_split(data, test_size=0.2, random_state=seed)
    train, validation = train_test_split(train, test_size=0.25, random_state=seed)
    return {"train": train, "validation": validation, "test": test}


def dataset_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_pipeline(
    data_path=ROOT / "data/raw/patient_data.csv", output_dir=ROOT / "artifacts/splits"
):
    splits = split_dataset(load_dataset(data_path))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame[["patient_id", *FEATURES, TARGET]].to_csv(
            output_dir / f"{name}.csv", index=False
        )
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/patient_data.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/splits")
    args = parser.parse_args()
    splits = run_pipeline(args.data, args.output)
    print({name: len(frame) for name, frame in splits.items()})


if __name__ == "__main__":
    main()
