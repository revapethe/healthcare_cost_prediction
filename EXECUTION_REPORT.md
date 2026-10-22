# Validation evidence

Validation performed on 22 October 2026 with Python 3.12.14 on branch `improve/reliable-risk-analytics`. This report describes local execution, not hosted CI, deployment or clinical validation.

## Reproduced before changes

- Original pytest run: **14 passed, 2 failed**. The derived-feature fixture omitted required visit/cost fields; the MAE test referenced an undefined `rmse`.
- Original simplified training completed individual models but failed in ensemble evaluation with NaN predictions. Rounded RMSE zero caused division by zero followed by infinite-weight normalization.
- API inference used 17 manually encoded inputs while the trained model expected 34, producing HTTP 500. The UI separately used a formula and random multiplier.
- Four observed billed cost components reconstructed the target within $0.02. Insurance payments and patient responsibility reconstructed it within $0.01. A deliberately leaky regression achieved near-perfect test scores; these were invalid forecasting evidence.
- Documentation claimed perfect accuracy, clinical/business impact, deployment and coverage without supporting results. The old system-test script printed seven passes regardless of exceptions.

## Current evaluation

The unchanged committed sample contains 10,000 synthetic patients. See `reports/evaluation.json` for its SHA-256, exact numeric metrics, software versions, split sizes, validation results and ensemble weights. Preprocessing fits 6,000 training rows only; model selection uses 2,000 validation rows; 2,000 test rows remain separate. No target-based trimming occurs.

| Candidate | Test MAE ($) | Test RMSE ($) | Test R² |
|---|---:|---:|---:|
| Mean baseline | 4,246.86 | 5,205.25 | -0.0001 |
| Linear Regression | 2,128.67 | 2,804.21 | 0.7098 |
| Ridge Regression | 2,128.71 | 2,804.05 | 0.7098 |
| Random Forest | 1,501.60 | 2,049.27 | 0.8450 |
| Gradient Boosting | 1,457.91 | 1,967.33 | 0.8571 |
| XGBoost | 1,452.36 | 1,962.90 | 0.8578 |
| LightGBM | 1,449.63 | 1,963.43 | 0.8577 |
| Validation ensemble (selected) | 1,447.35 | 1,956.80 | 0.8587 |

Results reflect the synthetic generator, not expected performance in a real population. No uncertainty calibration, temporal validation, causal treatment evaluation or clinical validation was performed. See `reports/FEATURE_AUDIT.md` for all predictor decisions, including the conservative exclusion of previous-year cost.

## Executed checks

- Installed the resolved `requirements-dev.lock` into the environment with `uv pip sync`, removing undeclared ambient dependencies; `uv pip check` passed for all 74 installed packages.
- `python -m src.data.pipeline`: passed, emitted 6,000 / 2,000 / 2,000 allowlisted splits.
- `python -m src.models.train`: all six supported regressors, baseline and ensemble completed; finite metrics and loadable fitted artifact saved.
- `python -m scripts.generate_data --n_patients 10000 --output_dir /tmp/healthcare-generated-validation` followed by training against that separate generated CSV: passed. This secondary run validates generation-to-training compatibility; headline metrics above use the committed sample, not the newly generated data.
- Clean locked-environment `pytest -q --cov=src --cov-report=term-missing`: **47 passed, 1 warning in 16.60s**, 89% source coverage. No failures, skips or expected failures. One upstream Starlette deprecation warns about its httpx test adapter; requests pass. Raw output is preserved in `reports/validation.txt`.
- Regression coverage includes poisoned excluded columns, poisoned test outcomes, train-only transformer statistics, disjoint patient IDs, zero/extreme ensemble errors, deterministic repeated inference, single/population/API parity, invalid inputs and missing/corrupt/legacy artifacts.
- Streamlit AppTest executed all six pages and submitted both forms. Identical what-if profiles produced a zero difference; empty population filters and missing artifacts were checked. These are functional app tests, not a visual screenshot review or a cross-browser responsive-layout audit.
- `python -m scripts.test_system`: passed for 2,000 held-out predictions and API/shared-service parity.
- Live Uvicorn: `/health` and `/api/v1/predict` returned HTTP 200. The example profile in README returned $13,599.18 annually and exactly matched direct `CostPredictor` output.
- Live Streamlit on test port 8502: `/_stcore/health` returned HTTP 200 `ok`. Default port 8501 was occupied by an earlier process; this did not replace the AppTest functional checks.
- `black --check src dashboards scripts tests`, `ruff check src dashboards scripts tests`, and `git diff --check`: passed after formatting.
- `docker compose config --quiet`: passed. Docker build validation is recorded below separately.

## Docker and hosted CI

Initial local Docker builds could not resolve Debian package hosts from the build container. Host-network and proxy-forwarding diagnoses were attempted without disabling TLS or package signature checks. See the final Docker outcome below.

The replacement GitHub Actions workflow installs locked dependencies, checks formatting/lint, prepares data, trains and runs pytest; a separate job validates Compose and builds Docker. Placeholder Docker Hub login/push and deployment steps were removed. **Hosted GitHub Actions has not run as part of this local change.** No remote CI success, merge, publication or deployment is claimed.

**Final Docker outcome: passed.** After resolving the cloud proxy hostname and forwarding the existing trusted CA bundle as an ephemeral BuildKit secret, the image built successfully (`healthcare-analytics:review`, image SHA-256 `6d48fc54375a655815c0dc6d3b375321ce97b95bd90e1f09b4811fc26f5856f0`). TLS and Debian package signature verification remained enabled. The Dockerfile supports an optional `build_ca` secret without copying the cloud CA into the image.

Training in the image reproduced the local selected model's metrics exactly. Ephemeral containers passed API readiness and real prediction parity ($13,599.18 for the README example), dashboard readiness, and AppTest execution of all six pages. They exposed no host ports and were stopped after validation. This is local container validation, not deployment. The image is local only; it was not pushed to a registry.

## Remaining limitations

- Hosted GitHub Actions has not been observed; local success is not hosted CI success.
- No screenshot review, mobile browser matrix, clinical validation, external dataset, calibrated uncertainty, authentication or public deployment was performed.
- One upstream test-client deprecation warning remains; it does not fail requests.
- Artifact loading is restricted by convention to trusted local files. Do not load untrusted pickle/joblib files.
- No merge into main or deployment is included.
