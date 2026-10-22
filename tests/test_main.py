"""Regression tests for leakage, training boundaries and shared real inference."""

import json

from fastapi.testclient import TestClient
import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.api.app import create_app
from src.data.pipeline import (
    ROOT,
    FEATURES,
    EXCLUDED_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    load_dataset,
    make_preprocessor,
    run_pipeline,
    split_dataset,
)
from src.models.predict import (
    ArtifactError,
    CostPredictor,
    PatientInput,
    categorize_risk,
)
from src.models.train import ensemble_weights, regression_metrics, train


@pytest.fixture(scope="session")
def dataset():
    return load_dataset()


@pytest.fixture(scope="session")
def trained_path():
    path = ROOT / "artifacts/model.joblib"
    if not path.exists():
        train()
    return path


@pytest.fixture
def predictor(trained_path):
    return CostPredictor(trained_path)


@pytest.fixture
def patient():
    return PatientInput(
        age=55,
        gender="M",
        bmi=28.0,
        chronic_conditions_count=2,
        previous_office_visits=5,
        previous_er_visits=1,
        previous_hospitalizations=1,
        medication_count=3,
    )


def test_raw_and_engineered_features_are_all_audited(dataset):
    assert not (set(FEATURES) & set(EXCLUDED_FEATURES))
    assert set(dataset) == set(FEATURES) | (set(EXCLUDED_FEATURES) & set(dataset))
    assert set(PatientInput.model_fields) == set(FEATURES)
    assert len(FEATURES) == 16


def test_split_is_complete_disjoint_and_reproducible(dataset, tmp_path):
    splits = run_pipeline(output_dir=tmp_path)
    ids = [set(frame.patient_id) for frame in splits.values()]
    assert not ids[0] & ids[1] and not ids[1] & ids[2] and not ids[0] & ids[2]
    assert set.union(*ids) == set(dataset.patient_id)
    for name, frame in split_dataset(dataset).items():
        assert frame.patient_id.tolist() == splits[name].patient_id.tolist()
        saved = pd.read_csv(tmp_path / f"{name}.csv")
        assert set(saved) == {"patient_id", TARGET, *FEATURES}


def test_duplicate_patient_cannot_cross_splits(dataset, tmp_path):
    invalid = pd.concat([dataset, dataset.iloc[:1]])
    path = tmp_path / "duplicate.csv"
    invalid.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_dataset(path)


def test_preprocessing_never_reads_targets_or_unapproved_columns(dataset):
    preprocessor = make_preprocessor().fit(dataset.iloc[:100])
    original = preprocessor.transform(dataset.iloc[100:105])
    poisoned = dataset.iloc[100:105].copy()
    for name in EXCLUDED_FEATURES:
        poisoned[name] = 1e12
    poisoned["future_bill"] = -1e99
    np.testing.assert_allclose(original, preprocessor.transform(poisoned))


def test_imputation_and_scaling_fit_training_only(dataset, predictor):
    training = split_dataset(dataset)["train"]
    model = predictor.model
    # VotingRegressor contains independently fitted full pipelines.
    pipelines = model.estimators_ if hasattr(model, "estimators_") else [model]
    for pipeline in pipelines:
        numeric = pipeline.named_steps["preprocessor"].named_transformers_["numeric"]
        medians = training[NUMERIC_FEATURES].median().to_numpy()
        np.testing.assert_allclose(numeric.named_steps["imputer"].statistics_, medians)
        imputed = training[NUMERIC_FEATURES].fillna(training[NUMERIC_FEATURES].median())
        np.testing.assert_allclose(
            numeric.named_steps["scale"].mean_, imputed.mean().to_numpy()
        )


def test_literal_none_vaccination_is_not_missing(dataset):
    assert (dataset.vaccination_status == "None").any()
    assert not dataset.vaccination_status.isna().any()


def test_inconsistent_pressure_is_missing_before_training(dataset):
    assert not (
        dataset.blood_pressure_diastolic >= dataset.blood_pressure_systolic
    ).any()
    assert (
        dataset.blood_pressure_systolic.isna().sum()
        == dataset.blood_pressure_diastolic.isna().sum()
    )


@pytest.mark.parametrize(
    "errors", [[0, 0, 0], [0, 1, 2], [1e-300, 1e300, 5], [1, 2, 3]]
)
def test_ensemble_weights_are_finite_even_for_zero_rmse(errors):
    weights = ensemble_weights(errors)
    assert np.isfinite(weights).all()
    assert (weights >= 0).all()
    assert weights.sum() == pytest.approx(1)
    if 0 in errors:
        assert all(weight == 0 for error, weight in zip(errors, weights) if error > 0)


@pytest.mark.parametrize("errors", [[], [np.nan], [np.inf], [-1], [[1, 2]]])
def test_invalid_ensemble_errors_rejected(errors):
    with pytest.raises(ValueError):
        ensemble_weights(errors)


def test_metric_math():
    result = regression_metrics(np.array([1, 3]), np.array([2, 3]))
    assert result["MAE"] == 0.5
    assert result["RMSE"] == pytest.approx(np.sqrt(0.5))
    assert result["R2"] == 0.5


@pytest.mark.parametrize(
    "cost,expected",
    [(4999.99, "Low"), (5000, "Medium"), (25000, "High"), (100000, "Catastrophic")],
)
def test_cost_bands(cost, expected):
    assert categorize_risk(cost) == expected


def test_shared_inference_deterministic_and_pipeline_exact(predictor, patient):
    first = predictor.predict(patient)
    assert first == predictor.predict(patient.model_dump())
    expected = max(
        0, predictor.model.predict(pd.DataFrame([patient.model_dump()])[FEATURES])[0]
    )
    assert first["predicted_cost"] == round(expected, 2)
    assert first["monthly_cost"] == round(first["predicted_cost"] / 12, 2)
    assert "confidence_interval" not in first


def test_population_matches_single_inference(predictor, patient):
    frame = pd.DataFrame([patient.model_dump(), patient.model_dump()])
    costs = predictor.predict_population(frame)
    np.testing.assert_array_equal(
        costs, [predictor.predict(patient)["predicted_cost"]] * 2
    )


@pytest.mark.parametrize(
    "override",
    [
        {"age": 17},
        {"age": True},
        {"bmi": float("nan")},
        {"bmi": float("inf")},
        {"previous_er_visits": -1},
        {"gender": "unknown"},
        {"state": "unknown"},
        {"total_annual_cost": 100},
        {"previous_year_cost": 5000},
        {"blood_pressure_systolic": 90, "blood_pressure_diastolic": 100},
    ],
)
def test_invalid_prediction_inputs_rejected(patient, override):
    with pytest.raises(ValidationError):
        PatientInput.model_validate({**patient.model_dump(), **override})


def test_missing_corrupt_and_legacy_artifacts_fail_honestly(tmp_path):
    for path in [
        tmp_path / "missing.joblib",
        tmp_path / "bad.joblib",
        tmp_path / "legacy.joblib",
    ]:
        if path.name == "bad.joblib":
            path.write_bytes(b"corrupt")
        elif path.name == "legacy.joblib":
            joblib.dump({"schema_version": 0}, path)
        with pytest.raises(ArtifactError, match="unavailable"):
            CostPredictor(path)
        with TestClient(create_app(path)) as client:
            assert client.get("/health").status_code == 503
            assert (
                client.post(
                    "/api/v1/predict", json={"age": 55, "gender": "M", "bmi": 28}
                ).status_code
                == 503
            )


def test_api_single_batch_and_dashboard_service_match(trained_path, predictor, patient):
    with TestClient(create_app(trained_path)) as client:
        assert client.get("/health").json()["model_loaded"]
        result = client.post("/api/v1/predict", json=patient.model_dump())
        assert result.status_code == 200
        assert result.json() == predictor.predict(patient)
        batch = client.post(
            "/api/v1/predict/batch", json={"patients": [patient.model_dump()] * 2}
        )
        assert batch.status_code == 200
        assert batch.json()["predictions"] == [result.json()] * 2
        assert (
            client.post("/api/v1/predict/batch", json={"patients": []}).status_code
            == 422
        )
        assert client.post("/api/v1/predict", json={"age": -1}).status_code == 422
        assert (
            client.get("/api/v1/stats").json()["selected_model"]
            == predictor.metadata["selected_model"]
        )
        assert client.get("/api/v1/risk-categories").status_code == 200


def test_report_selection_and_test_metrics_are_reproducible(dataset, predictor):
    report = predictor.metadata
    assert report["selected_model"] == min(
        report["validation"], key=lambda name: report["validation"][name]["RMSE"]
    )
    test = split_dataset(dataset)["test"]
    measured = regression_metrics(
        test[TARGET], np.maximum(0, predictor.model.predict(test[FEATURES]))
    )
    for metric, value in measured.items():
        assert report["test"][report["selected_model"]][metric] == pytest.approx(value)
    assert report["test"]["Mean baseline"]["RMSE"] > 0
    assert report["test"][report["selected_model"]]["R2"] < 0.99
    assert len(json.dumps(report, allow_nan=False)) > 0


def test_test_targets_cannot_change_model_selection_or_fit(
    dataset, tmp_path, monkeypatch
):
    """Poison held-out outcomes and prove selection/preprocessing remain identical."""
    import importlib

    module = importlib.import_module("src.models.train")
    splits = split_dataset(dataset.iloc[:300])
    monkeypatch.setattr(module, "run_pipeline", lambda *args: splits)
    first_path = tmp_path / "first"
    first = train(
        output_dir=first_path,
        report_path=tmp_path / "first.json",
        include_boosters=False,
    )
    first_model = CostPredictor(first_path / "model.joblib")
    splits["test"] = splits["test"].copy()
    splits["test"][TARGET] += 100000
    second_path = tmp_path / "second"
    second = train(
        output_dir=second_path,
        report_path=tmp_path / "second.json",
        include_boosters=False,
    )
    second_model = CostPredictor(second_path / "model.joblib")
    assert first["selected_model"] == second["selected_model"]
    # Parallel reductions can differ by a few floating-point bits across fits.
    # Keep model/member selection exact and numerical comparisons tight enough
    # to detect any material influence from the poisoned held-out outcomes.
    assert first["validation"].keys() == second["validation"].keys()
    for name, metrics in first["validation"].items():
        assert metrics == pytest.approx(
            second["validation"][name], rel=1e-10, abs=1e-10
        )
    assert first["ensemble"]["members"] == second["ensemble"]["members"]
    np.testing.assert_allclose(
        first["ensemble"]["weights"],
        second["ensemble"]["weights"],
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        first_model.model.predict(splits["test"][FEATURES]),
        second_model.model.predict(splits["test"][FEATURES]),
        rtol=1e-10,
        atol=1e-7,
    )
    assert first["test"] != second["test"]
