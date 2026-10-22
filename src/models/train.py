"""Train supported regressors; select on validation, report untouched test results."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.data.pipeline import (
    ROOT,
    FEATURES,
    TARGET,
    dataset_digest,
    make_preprocessor,
    run_pipeline,
)


def regression_metrics(actual, predicted):
    if not np.isfinite(predicted).all():
        raise ValueError("Model generated nonfinite predictions")
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def ensemble_weights(errors):
    """Finite normalized weights, including exact zero; use unrounded errors."""
    errors = np.asarray(errors, dtype=float)
    if (
        errors.ndim != 1
        or not len(errors)
        or not np.isfinite(errors).all()
        or (errors < 0).any()
    ):
        raise ValueError("Ensemble errors must be a nonempty finite nonnegative vector")
    zeros = errors == 0
    if zeros.any():
        return zeros.astype(float) / zeros.sum()
    # Scale before inversion to avoid overflow for tiny nonzero errors.
    inverse = errors.min() / errors
    return inverse / inverse.sum()


def supported_models(include_boosters=True):
    models = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, min_samples_leaf=5, random_state=42, n_jobs=2
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, max_depth=2, loss="squared_error", random_state=42
        ),
    }
    if include_boosters:
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor

        models.update(
            {
                "XGBoost": XGBRegressor(
                    n_estimators=150,
                    max_depth=3,
                    learning_rate=0.05,
                    random_state=42,
                    n_jobs=2,
                ),
                "LightGBM": LGBMRegressor(
                    n_estimators=150,
                    max_depth=4,
                    num_leaves=15,
                    learning_rate=0.05,
                    random_state=42,
                    n_jobs=2,
                    verbosity=-1,
                ),
            }
        )
    return models


def train(
    data_path=ROOT / "data/raw/patient_data.csv",
    output_dir=ROOT / "artifacts",
    report_path=ROOT / "reports/evaluation.json",
    include_boosters=True,
):
    splits = run_pipeline(data_path, Path(output_dir) / "splits")
    training, validation, test = (
        splits[key] for key in ["train", "validation", "test"]
    )
    models, validation_results = {}, {}
    for name, estimator in supported_models(include_boosters).items():
        pipeline = Pipeline(
            [("preprocessor", make_preprocessor()), ("regressor", estimator)]
        )
        pipeline.fit(training[FEATURES], training[TARGET])
        models[name] = pipeline
        validation_results[name] = regression_metrics(
            validation[TARGET], np.maximum(0, pipeline.predict(validation[FEATURES]))
        )
    members = sorted(
        (n for n in models if n != "Mean baseline"),
        key=lambda n: validation_results[n]["RMSE"],
    )[:3]
    weights = ensemble_weights([validation_results[n]["RMSE"] for n in members])
    ensemble = VotingRegressor(
        [(f"model{i}", models[name]) for i, name in enumerate(members)], weights=weights
    )
    ensemble.fit(training[FEATURES], training[TARGET])
    models["Validation ensemble"] = ensemble
    validation_results["Validation ensemble"] = regression_metrics(
        validation[TARGET], np.maximum(0, ensemble.predict(validation[FEATURES]))
    )
    selected = min(validation_results, key=lambda n: validation_results[n]["RMSE"])
    # The selection decision is final before accessing any test targets.
    test_results = {
        name: regression_metrics(
            test[TARGET], np.maximum(0, model.predict(test[FEATURES]))
        )
        for name, model in models.items()
    }
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": dataset_digest(data_path),
        "synthetic": True,
        "rows": sum(len(s) for s in splits.values()),
        "split_seed": 42,
        "cleaning": {
            "invalid_pressure_pairs_imputed": int(
                sum(s.blood_pressure_systolic.isna().sum() for s in splits.values())
            ),
            "rule": "Inconsistent BP pairs set to missing; numeric imputation fitted on train only. Literal vaccination category None preserved.",
        },
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "features": FEATURES,
        "selected_model": selected,
        "selection_rule": "Minimum validation RMSE, including baseline; no test-based tuning or refit",
        "validation": validation_results,
        "test": test_results,
        "ensemble": {"members": members, "weights": weights.tolist()},
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "limitations": [
            "Synthetic data only; no clinical validation",
            "Random patient split, not temporal or external validation",
            "Historical inputs assumed known before forecast; source contains no observation timestamps",
            "No calibrated uncertainty or causal intervention estimates",
        ],
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": 1,
            "pipeline": models[selected],
            "features": FEATURES,
            "metadata": report,
        },
        Path(output_dir) / "model.joblib",
    )
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"selected_model": selected, "test": test_results}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/patient_data.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/evaluation.json")
    parser.add_argument(
        "--simple", action="store_true", help="Train scikit-learn models only"
    )
    args = parser.parse_args()
    train(args.data, args.output, args.report, not args.simple)


if __name__ == "__main__":
    main()
