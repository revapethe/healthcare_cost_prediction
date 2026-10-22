<div align="center">

# Healthcare Cost Prediction & Risk Analytics

### From patient profiles to transparent cost estimates

Explore healthcare spending, estimate annual patient costs, and compare financial risk through an interactive machine learning application.

**Python · pandas · NumPy · scikit-learn · Streamlit · Plotly · FastAPI · Docker**

</div>

---

> **Educational application using synthetic data.** This project is not a validated clinical decision tool. Its estimates should not guide patient care or financial decisions.

## Overview

Healthcare spending varies with patient characteristics, chronic conditions, and prior service utilization. This project explores those relationships through an integrated analytics dashboard and prediction API.

The application connects three capabilities:

- **Understand spending:** Explore observed costs across a synthetic patient population.
- **Estimate future costs:** Generate annual cost estimates from information assumed available before the forecast year.
- **Explore scenarios:** Compare model estimates for different patient profiles.

A shared prediction service keeps preprocessing and inference consistent across the dashboard, API, and scenario comparisons.

## Application Features

| Workspace | What it provides |
|---|---|
| **Executive Overview** | Population summaries, observed spending, average costs, and cost distributions |
| **Patient Cost Predictor** | Deterministic annual cost estimates from a patient profile |
| **Risk Stratification** | Population grouping using predicted financial cost bands |
| **Cost Analysis** | Exploration of observed costs and synthetic billing components |
| **Model Performance** | Measured model comparisons, evaluation details, and provenance |
| **What-If Scenarios** | Side-by-side estimates for original and alternative patient profiles |

Population views include interactive filters for age, state, and insurance type. Plotly charts support exploration, and CSV exports make results available for further analysis.

## Dataset

The committed dataset contains **10,000 synthetic patients** and **31 raw columns**.

It includes:

- Demographic characteristics
- Chronic condition counts
- Prior hospitalizations and emergency and office visits
- Medication counts and selected measurements
- Insurance categories
- Observed annual costs and billing components

The dataset is generated from predefined relationships and random variation. It does not contain real patient records or establish how the models would perform in a clinical population.

## Machine Learning Approach

### An Explicit Feature Contract

Training and inference use the same **16 allowlisted inputs**:

| Category | Inputs |
|---|---|
| Demographics | Age, gender, state |
| Patient characteristics | BMI, smoking status, chronic condition count |
| Prior utilization | Hospitalizations, emergency visits, office visits |
| Medication and prevention | Medication count, vaccination status |
| Measurements | Systolic and diastolic blood pressure, total cholesterol, fasting glucose |
| Coverage | Insurance type |

These inputs are assumed to be known before the forecast year. The synthetic dataset does not contain observation timestamps that independently verify this assumption.

### Preventing Target Leakage

Some billing fields already contain information about the outcome being predicted. Including them would produce misleadingly strong results.

The prediction pipeline excludes:

- Same-year inpatient, outpatient, pharmacy, and emergency costs
- Insurance payments and patient responsibility
- Outstanding balances and payment status
- Target-derived risk labels and financial scores
- Identifiers and unvalidated diagnostic text
- Previous-year cost, which shares the current-year random base in the generator

The four billed cost components reconstruct the annual target within rounding error. Their exclusion is essential to evaluating forecasting rather than reconstructing an observed bill.

The complete column-by-column assessment is documented in the [Feature Audit](reports/FEATURE_AUDIT.md).

### Training and Evaluation

The dataset is divided into three disjoint patient groups:

| Dataset split | Patients | Purpose |
|---|---:|---|
| Training | 6,000 | Fit preprocessing and model parameters |
| Validation | 2,000 | Compare candidates, calculate ensemble weights, and select a model |
| Test | 2,000 | Measure held-out performance after selection |

Numeric imputation, scaling, and categorical encoding are fitted on training rows only.

Model selection uses validation RMSE. Test outcomes do not influence preprocessing, model fitting, ensemble weights, or selection. No target-based outlier removal is applied.

## Measured Model Performance

Results below were measured on **2,000 held-out synthetic patients**.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean baseline | $4,246.86 | $5,205.25 | −0.0001 |
| Linear Regression | $2,128.67 | $2,804.21 | 0.7098 |
| Ridge Regression | $2,128.71 | $2,804.05 | 0.7098 |
| Random Forest | $1,501.60 | $2,049.27 | 0.8450 |
| Gradient Boosting | $1,457.91 | $1,967.33 | 0.8571 |
| XGBoost | $1,452.36 | $1,962.90 | 0.8578 |
| LightGBM | $1,449.63 | $1,963.43 | 0.8577 |
| **Validation ensemble** | **$1,447.35** | **$1,956.80** | **0.8587** |

The **validation-weighted ensemble** was selected using validation performance.

- **MAE** measures the average absolute prediction error in dollars.
- **RMSE** gives greater weight to larger prediction errors.
- **R²** describes explained variation; it is not classification accuracy.

These results measure performance within the synthetic dataset. They do not establish prospective accuracy or generalization to real healthcare populations.

Full-precision metrics, ensemble weights, software versions, and the dataset fingerprint are recorded in [Evaluation Results](reports/evaluation.json).

## Shared Prediction Architecture

```text
Synthetic patient dataset
           |
           v
Explicit feature contract
           |
           v
Disjoint training / validation / test splits
           |
           v
Training-fitted preprocessing and candidate models
           |
           v
Validation-based selection
           |
           v
Saved pipeline and evaluation metadata
           |
           v
Shared input validation and prediction service
           |
           +-------------------+-------------------+
           |                   |                   |
           v                   v                   v
     Streamlit app        FastAPI service    What-if comparison
```

The fitted preprocessing pipeline remains attached to the model during inference.

Identical inputs produce identical predictions from the same loaded artifact. Missing or incompatible artifacts produce explicit errors rather than replacement estimates.

Prediction outputs include the model identity and dataset fingerprint. The application does not fabricate confidence intervals, billing allocations, clinical recommendations, or financial savings.

## Financial Cost Bands

Predicted annual costs are grouped into illustrative financial bands:

| Band | Estimated annual cost |
|---|---:|
| Low | Below $5,000 |
| Medium | $5,000 to below $25,000 |
| High | $25,000 to below $100,000 |
| Catastrophic | $100,000 or more |

These thresholds describe estimated spending, not validated clinical risk.

Population predictions include training records. Held-out model performance is reported separately on the Model Performance page.

## What-If Analysis

The scenario workspace compares two estimates generated by the same fitted pipeline.

For example, a user can compare an original profile with an alternative BMI or smoking status while keeping the other entered characteristics unchanged.

The resulting difference describes **model sensitivity**. It does not establish:

- A causal treatment effect
- Achievable financial savings
- A recommended intervention
- A clinically realistic change in patient circumstances

## Technology Stack

| Layer | Technologies |
|---|---|
| Data processing | Python, pandas, NumPy |
| Machine learning | scikit-learn, XGBoost, LightGBM |
| Interactive analytics | Streamlit, Plotly |
| Prediction service | FastAPI, Pydantic, Uvicorn |
| Testing and code quality | pytest, coverage, Black, Ruff |
| Containers | Docker, Docker Compose |
| Automation | GitHub Actions |

## Testing and Validation

Cloud-environment validation completed with **47 passing tests** and **89% source coverage**.

Regression tests cover:

- Exclusion of target-derived predictors
- Training-only preprocessing
- Disjoint patient splits
- Independence from held-out test outcomes
- Finite ensemble weights, including zero-error cases
- Deterministic repeated predictions
- Consistency across individual, population, and API inference
- Invalid input and unavailable-artifact handling
- All six dashboard pages and prediction forms

Local formatting, lint, Docker build, and container smoke checks also passed.

Hosted GitHub Actions and Windows validation are separate from these cloud results. A numerical-tolerance update was added after a Windows regression-test failure; Windows confirmation remains pending.

Detailed evidence is available in the [Execution Report](EXECUTION_REPORT.md).

## Repository Structure

```text
healthcare_cost_prediction/
├── dashboards/          Streamlit analytics application
├── data/
│   └── raw/             Committed synthetic patient sample
├── docs/                Technical documentation
├── notebooks/           Exploratory analysis
├── reports/             Feature audit, metrics, and validation evidence
├── scripts/             Synthetic generation and system checks
├── src/
│   ├── api/             FastAPI endpoints
│   ├── data/            Feature contract and dataset splitting
│   └── models/          Training and shared inference
├── tests/               Modeling, API, and dashboard regressions
└── .github/
    └── workflows/       Automated validation definition
```

## Limitations

This application has not undergone clinical, prospective, or external validation.

The synthetic generator defines the relationships the models learn. A random patient split does not assess performance across future time periods or different healthcare systems.

The project does not establish:

- Fairness across real patient populations
- Calibrated prediction uncertainty
- Clinical safety or effectiveness
- Causal intervention benefits
- Production security or operational reliability
- Financial impact or deployment success

Real-world use would require verified prediction-time data, temporal and external evaluation, subgroup analysis, privacy controls, security review, and appropriate clinical governance.

## Project Documentation

- [Feature Audit](reports/FEATURE_AUDIT.md)
- [Evaluation Results](reports/evaluation.json)
- [Execution Report](EXECUTION_REPORT.md)
- [Technical Documentation](docs/DOCUMENTATION.md)

## License

Distributed under the [MIT License](LICENSE).
