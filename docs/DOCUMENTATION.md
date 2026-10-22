# Technical documentation

See [README](../README.md) for architecture, measured results and local commands, [feature audit](../reports/FEATURE_AUDIT.md) for predictor availability, and [evaluation JSON](../reports/evaluation.json) for machine-readable evidence.

## Data and preprocessing

The committed CSV has 10,000 synthetic patients and 31 raw columns. All raw and former engineered features have an explicit decision in `src/data/pipeline.py`. Split by unique patient before fitting. Missing numeric values are median-imputed and scaled using training rows; categorical values are imputed and one-hot encoded with unknown-category support. Invalid blood-pressure pairs are replaced with missing measurements using a row-local rule. Literal vaccination status `None` is a category, not a missing value.

Training compares a constant mean baseline and six supported regressors. The three lowest validation RMSE candidates form a weighted ensemble; exact zero errors distribute weight across exact-zero members, and positive errors are scaled before inversion. Full precision is retained. Validation chooses the final model including the baseline. Test outcomes never inform fitting, weights or selection. The fitted preprocessing remains inside every model pipeline.

## API contract

`PatientInput` in `src/models/predict.py` is the authoritative schema. Age, gender and BMI are required; remaining forecast-start measurements have explicit defaults. Unknown fields and target-derived cost inputs are rejected. The schema checks finite bounds, integer measurements, categorical values and blood-pressure ordering. Missing BP can be represented as JSON null and uses the training-fitted median.

- `GET /health`: HTTP 200 only when a compatible model is loaded; otherwise 503.
- `POST /api/v1/predict`: one patient; returns annual and monthly estimated costs, illustrative cost band, model identity and dataset hash.
- `POST /api/v1/predict/batch`: `{ "patients": [...] }`, one to 1,000 records.
- `GET /api/v1/stats`: metadata from the loaded artifact, including actual evaluation.
- `GET /api/v1/risk-categories`: definitions of illustrative financial bands.
- `GET /docs`: generated OpenAPI documentation.

Invalid requests return 422; missing/corrupt/incompatible artifacts return 503. Load only trusted local joblib files. `MODEL_PATH` can override the artifact location. Both interfaces load on startup/rerun, with no implicit training or replacement by a formula. Restart the API after changing artifacts. No persistence of user-entered profiles, authentication, deployment or clinical advice is provided.

## Metrics and scenarios

MAE and RMSE are dollar errors; R² is a regression statistic, not classification accuracy. Cost bands are deterministic thresholds, not a separately validated classifier. Uncertainty intervals and bill component forecasts are intentionally omitted because they have not been estimated. What-if differences describe model sensitivity, not causal effects, achievable savings or recommendations. Population scoring includes training rows and is labeled accordingly.

## Extension requirements

For real data, establish a forecast index date and verify every input's timestamp and provenance. Use grouped temporal/external validation; audit protected attribute performance; evaluate uncertainty and distribution shift; implement consent, privacy, authentication, observability and clinical governance before considering real use. These are future requirements, not implemented capabilities.
