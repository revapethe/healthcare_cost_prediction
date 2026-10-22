"""Validated, deterministic inference shared by the API and dashboard."""

import os
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.data.pipeline import ROOT, FEATURES


class PatientInput(BaseModel):
    """Inputs assumed measured before the forecast year; no billed cost fields."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    age: int = Field(ge=18, le=100)
    gender: Literal["M", "F"]
    bmi: float = Field(ge=15, le=60)
    smoker: bool = False
    chronic_conditions_count: int = Field(default=0, ge=0, le=20)
    previous_office_visits: int = Field(default=0, ge=0, le=365)
    previous_er_visits: int = Field(default=0, ge=0, le=100)
    previous_hospitalizations: int = Field(default=0, ge=0, le=100)
    medication_count: int = Field(default=0, ge=0, le=100)
    blood_pressure_systolic: int | None = Field(default=120, ge=80, le=220)
    blood_pressure_diastolic: int | None = Field(default=80, ge=40, le=140)
    cholesterol_total: int = Field(default=200, ge=100, le=400)
    glucose_fasting: int = Field(default=100, ge=70, le=300)
    insurance_type: Literal[
        "Medicare", "Medicaid", "Private", "Self-Pay", "HMO", "PPO"
    ] = "Private"
    state: Literal["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI"] = "CA"
    vaccination_status: Literal["Complete", "Partial", "None"] = "Complete"

    @field_validator(
        "age",
        "chronic_conditions_count",
        "previous_office_visits",
        "previous_er_visits",
        "previous_hospitalizations",
        "medication_count",
        "blood_pressure_systolic",
        "blood_pressure_diastolic",
        "cholesterol_total",
        "glucose_fasting",
        mode="before",
    )
    @classmethod
    def reject_boolean_counts(cls, value):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Integer measurements cannot be booleans")
        return value

    @model_validator(mode="after")
    def check_pressure(self):
        if (
            self.blood_pressure_diastolic is not None
            and self.blood_pressure_systolic is not None
            and self.blood_pressure_diastolic >= self.blood_pressure_systolic
        ):
            raise ValueError("Diastolic pressure must be below systolic pressure")
        return self


def categorize_risk(cost):
    """Illustrative cost bands, not clinical risk classifications."""
    for threshold, label in [(5000, "Low"), (25000, "Medium"), (100000, "High")]:
        if cost < threshold:
            return label
    return "Catastrophic"


class ArtifactError(RuntimeError):
    """Missing, corrupt or incompatible trusted local model artifact."""


class CostPredictor:
    def __init__(self, model_path=None):
        configured_path = model_path or os.environ.get("MODEL_PATH")
        if configured_path:
            path = Path(configured_path)
        else:
            local_path = ROOT / "artifacts/model.joblib"
            path = (
                local_path if local_path.exists() else ROOT / "deployment/model.joblib"
            )
        try:
            bundle = joblib.load(path)
            if bundle["schema_version"] != 1 or bundle["features"] != FEATURES:
                raise ValueError("Incompatible artifact schema or feature contract")
            self.model = bundle["pipeline"]
            self.metadata = bundle["metadata"]
        except Exception as exc:
            raise ArtifactError(
                "Model artifact unavailable or incompatible. Run python -m src.models.train."
            ) from exc

    def predict(self, data):
        patient = (
            data
            if isinstance(data, PatientInput)
            else PatientInput.model_validate(data)
        )
        frame = pd.DataFrame([patient.model_dump()])[FEATURES]
        cost = float(self.model.predict(frame)[0])
        if not np.isfinite(cost):
            raise ArtifactError("Model generated a nonfinite prediction")
        cost = round(max(0.0, cost), 2)
        return {
            "predicted_cost": cost,
            "monthly_cost": round(cost / 12, 2),
            "risk_category": categorize_risk(cost),
            "model_name": self.metadata["selected_model"],
            "dataset_sha256": self.metadata["dataset_sha256"],
            "notice": "Synthetic educational estimate; no calibrated confidence interval or clinical validity.",
        }

    def predict_population(self, frame):
        # Validate every row with exactly the same contract as single prediction.
        inputs = frame[FEATURES].astype(object).where(pd.notna(frame[FEATURES]), None)
        records = [
            PatientInput.model_validate(row).model_dump()
            for row in inputs.to_dict("records")
        ]
        costs = np.asarray(self.model.predict(pd.DataFrame(records)[FEATURES]))
        if not np.isfinite(costs).all():
            raise ArtifactError("Model generated nonfinite predictions")
        return np.maximum(0, costs).round(2)
