# Feature and prediction-time availability audit

Audit applies to the 31 raw columns and all eight former engineered columns. Deny-by-default selection is enforced in `FEATURES`; categorical encoders and numeric imputers/scalers fit training data only. No model consumes raw CSV columns implicitly.

## Retained inputs

| Input | Availability assumption |
|---|---|
| `age` | Measured or recorded at forecast start; synthetic source has no timestamps to prove chronology |
| `bmi` | Measured or recorded at forecast start; synthetic source has no timestamps to prove chronology |
| `smoker` | Measured or recorded at forecast start; synthetic source has no timestamps to prove chronology |
| `chronic_conditions_count` | Measured or recorded at forecast start; synthetic source has no timestamps to prove chronology |
| `previous_hospitalizations` | Recorded before the forecast year; historical utilization count |
| `previous_er_visits` | Recorded before the forecast year; historical utilization count |
| `previous_office_visits` | Recorded before the forecast year; historical utilization count |
| `medication_count` | Measured or recorded at forecast start; synthetic source has no timestamps to prove chronology |
| `blood_pressure_systolic` | Measured or recorded at forecast start; synthetic source has no timestamps to prove chronology |
| `blood_pressure_diastolic` | Measured or recorded at forecast start; synthetic source has no timestamps to prove chronology |
| `cholesterol_total` | Measured or recorded at forecast start; synthetic source has no timestamps to prove chronology |
| `glucose_fasting` | Measured or recorded at forecast start; synthetic source has no timestamps to prove chronology |
| `gender` | Measured or recorded at forecast start; synthetic source has no timestamps to prove chronology |
| `state` | Measured or recorded at forecast start; synthetic source has no timestamps to prove chronology |
| `insurance_type` | Measured or recorded at forecast start; synthetic source has no timestamps to prove chronology |
| `vaccination_status` | Measured or recorded at forecast start; synthetic source has no timestamps to prove chronology |

These are forecasting assumptions, not proof of clinical availability. Medication count and measurements must be collected before the target period in any replacement dataset. Gender and insurance can encode inequity; the demo has no validated fairness claim.

## Excluded raw and former engineered inputs

| Input | Reason |
|---|---|
| `patient_id` | Identifier; retained only for split integrity checks. |
| `data_date` | Dataset creation date, not a prospective observation date. |
| `zip_code` | Synthetic identifier with no justified predictive use. |
| `chronic_conditions` | Redundant unvalidated diagnostic text; use count only. |
| `total_annual_cost` | Target. |
| `inpatient_cost` | Same-year component of target. |
| `outpatient_cost` | Same-year component of target. |
| `pharmacy_cost` | Same-year component of target. |
| `emergency_cost` | Same-year component of target. |
| `insurance_paid` | Target times coverage rate. |
| `patient_responsibility` | Target minus insurance payment. |
| `payment_status` | Observed after billing; unavailable at prediction time. |
| `outstanding_balance` | Target-derived balance observed after billing. |
| `risk_category` | Thresholded target. |
| `previous_year_cost` | Generator shares current-year random base; chronology unverified. |
| `financial_risk_score` | Ratio derived from target and patient responsibility. |
| `cost_per_visit` | Uses excluded previous_year_cost. |
| `age_group` | Redundant; age is available directly. |
| `bmi_category` | Redundant; BMI is available directly. |
| `high_utilizer` | Old full-dataset quantile; replaced by raw prior counts. |
| `comorbidity_score` | Unvalidated heuristic; use its raw inputs. |
| `insurance_coverage_score` | Unvalidated heuristic; use insurance category. |
| `healthcare_intensity` | Unvalidated heuristic; use raw prior counts. |

## Reproduced leakage

On the committed 10,000-row sample, the absolute difference between the four rounded billed components and the target is at most $0.02; insurance payment plus patient responsibility reconstructs it within $0.01. A linear model on the four billed components, split with seed 42 and 25% test, measured MAE $0.002772 and R² 0.9999999999989798. This is outcome reconstruction, not prospective prediction. These are deliberately leaky diagnostic results and are not application performance.

The old flow also imputed and encoded full data, derived a full-data quantile, removed target-based outliers and selected models/ensemble weights using test outcomes. All were removed. Age/BMI grouping and heuristic scores are redundant rather than inherently target-derived, but are removed to keep a small auditable raw input contract. The ratio financial_risk_score may reduce to insurance coverage in this generator, but still depends on the observed target and is not an acceptable predictor construction.

## Data quality

87 contradictory BP pairs are converted to missing with a row-local rule. Imputation values come only from training. No patients or high-cost outcomes are discarded. The literal vaccination label `None` is preserved rather than treated as a CSV missing marker. IDs must be unique and nonempty so repeated patients cannot cross splits.

See `tests/test_main.py` for exclusion, poisoned-column, held-out-target, split and train-only preprocessing regression tests.
