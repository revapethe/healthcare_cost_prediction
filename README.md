# Healthcare Cost Prediction & Risk Analytics

A Python application for exploring **synthetic** healthcare costs, forecasting annual cost with a fitted scikit-learn pipeline, and comparing model estimates. Built with pandas, NumPy, scikit-learn, Streamlit, Plotly and FastAPI. XGBoost and LightGBM are supported training candidates; Docker and GitHub Actions provide reproducible validation workflows.

**Educational portfolio project, not a validated clinical decision tool.** No clinical efficacy, deployment, financial savings, patient outcomes or throughput claims are made. Do not enter real patient information into this demonstration.

## Run locally

Requires Python **3.12**. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.lock
python -m src.data.pipeline
python -m src.models.train
python -m scripts.test_system
pytest -v
black --check src dashboards scripts tests
ruff check src dashboards scripts tests
```

Start these in separate terminals with the virtual environment activated:

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
streamlit run dashboards/main_dashboard.py
```

The dashboard runs on port 8501; the API serves its OpenAPI documentation at `/docs` on port 8000. No database, cloud account or credentials are needed. `bash setup.sh` performs installation, training and checks without hiding failures. Runtime-only installation: `pip install -r requirements.lock`.

Example API request:

```bash
curl --fail -X POST http://127.0.0.1:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":55,"gender":"M","bmi":28,"chronic_conditions_count":2,"previous_office_visits":5,"previous_er_visits":1,"previous_hospitalizations":1,"medication_count":3}'
```

Omitted optional measurements use the defaults in `PatientInput`; the dashboard exposes every input explicitly. Missing or incompatible artifacts produce an explanatory dashboard error and HTTP 503, including on `/health`. Invalid API inputs produce HTTP 422. The batch endpoint accepts `{"patients":[...]}`, limited to 1,000 records. There is no authentication or clinical safety validation; do not expose this local demo as a public service.

## Dataset and evaluation

The committed sample contains **10,000 synthetic patients** and 31 raw columns. Its SHA-256 is `d1acf0cb98bd7955eca81abf8709142689849992c1754fd758ee699159158372`. The source generator is illustrative, with assumed prior utilization but no patient observation timestamps. The previous-year cost field shares the current-year random base in the generator and is therefore excluded conservatively. This is a random patient holdout, **not** a prospective or external clinical validation.

A deterministic split (seed 42) uses 6,000 training, 2,000 validation and 2,000 untouched test rows. Numeric median imputation, scaling and categorical imputation/one-hot encoding fit training rows only. Unknown categories are supported by the transformer; public input categories are constrained to the synthetic dataset's supported schema. No target-based outlier removal occurs. The loader preserves the literal vaccination category `None` and treats 87 inconsistent blood-pressure pairs as missing, then imputes them using training data only.

Selection uses **validation RMSE**, including a mean baseline. A validation-weighted ensemble combines the three strongest individual candidates, with explicit handling of zero errors. No rounded errors enter weight calculations. Models are not refitted on validation or test rows. Negative predictions are clipped to zero consistently in evaluation and serving. Test results from the actual run in [reports/evaluation.json](reports/evaluation.json):

| Model | MAE ($) | RMSE ($) | R² |
|---|---:|---:|---:|
| Mean baseline | 4,246.86 | 5,205.25 | -0.0001 |
| Linear Regression | 2,128.67 | 2,804.21 | 0.7098 |
| Ridge Regression | 2,128.71 | 2,804.05 | 0.7098 |
| Random Forest | 1,501.60 | 2,049.27 | 0.8450 |
| Gradient Boosting | 1,457.91 | 1,967.33 | 0.8571 |
| XGBoost | 1,452.36 | 1,962.90 | 0.8578 |
| LightGBM | 1,449.63 | 1,963.43 | 0.8577 |
| Validation ensemble | 1,447.35 | 1,956.80 | 0.8587 |

The selected model is **Validation ensemble**. Full validation metrics, weights, feature names, software versions and data fingerprint are recorded with the artifact. Test scores are reported for transparency, not used to tune the model. Future repeated development against this public holdout can overfit it; obtain a new temporal or external evaluation before making stronger claims.

## Features and leakage controls

Only 16 allowlisted inputs reach training and inference: age, BMI, smoking, chronic condition count, prior hospitalizations/ER/office visits, medication count, four measurements, gender, state, insurance type and vaccination status. They are assumed known at the start of the forecast period. This assumption is not independently verified by the synthetic source.

Excluded: same-year billed components, insurance payments, patient responsibility, outstanding balance, payment status, target-derived risk labels and financial score, identifiers, unvalidated diagnosis text and previous-year cost. All old engineered features were removed rather than carrying full-dataset quantiles or unvalidated heuristics into inference. The detailed [feature audit](reports/FEATURE_AUDIT.md) covers every raw and former engineered column. Regression tests poison excluded columns and held-out targets to detect leakage.

## Application

- **Executive Overview:** filtered cohort size, observed costs and cost distributions.
- **Patient Cost Predictor:** deterministic trained-model estimate with explicit inputs and export.
- **Risk Stratification:** model-based population cost bands, separately labeled from observed outcomes.
- **Cost Analysis:** descriptive relationships and observed billed components, never model predictors.
- **Model Performance:** measured holdout comparison, provenance and limitations.
- **What-If Scenarios:** two estimates from the same fitted pipeline; differences are sensitivity analysis, not causal savings.

Population pages provide state, insurance and age filters, interactive Plotly charts and CSV exports. No fabricated confidence intervals, cost allocations, ROI, clinical recommendations or random prediction multipliers are displayed. Cost bands (<$5k, <$25k, <$100k, ≥$100k) are illustrative financial thresholds, not clinical risk classes. Population predictions include training records and must not be mistaken for held-out performance.

## Architecture and artifacts

```text
synthetic CSV → explicit feature contract → 60/20/20 patient split
  → per-model fitted imputation / scaling / encoding → supported regressors
  → validation selection and weights → untouched test metrics
  → artifacts/model.joblib (pipeline + provenance)
       ↳ CostPredictor + PatientInput ↳ FastAPI / Streamlit / scenario comparison
```

`python -m src.data.pipeline` writes raw allowlisted splits under ignored `artifacts/splits/`. Training repeats the same split and writes `artifacts/model.joblib` plus the reviewable `reports/evaluation.json`. Fitted artifacts and old leaked feature matrices are not versioned. Train after cloning; never load an untrusted joblib/pickle file. `MODEL_PATH` optionally selects a trusted artifact. Restart services after replacing it. The dashboard rejects dataset/artifact fingerprint mismatches.

The seeded generator can be exercised without overwriting the committed sample:

```bash
python -m scripts.generate_data --n_patients 10000 --output_dir /tmp/healthcare-synthetic
python -m src.models.train --data /tmp/healthcare-synthetic/patient_data.csv \
  --output /tmp/healthcare-model --report /tmp/healthcare-evaluation.json
```

The generator's date fields are creation metadata; seed reproducibility applies to values for the same generator/runtime, not calendar timestamps. `python -m src.models.train --simple` trains the scikit-learn candidates and ensemble without booster imports.

## Docker and validation

```bash
docker compose config --quiet
docker compose build
docker compose up
```

The image trains its own artifact on the committed sample, runs as a non-root user, and serves the API by default. Compose also starts Streamlit using the same image definition. Health checks fail on non-200 responses. No image publishing or deployment action is configured.

[EXECUTION_REPORT.md](EXECUTION_REPORT.md) records the actual local validation results and remaining limitations. GitHub Actions installs the lockfile, checks formatting/lint, runs the pipeline, trains and tests, and builds Docker in a separate job. Local checks are not evidence that hosted GitHub Actions has run. No merge or deployment is part of this change.
