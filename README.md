Healthcare Cost Prediction & Risk Analytics
From patient profiles to transparent cost estimates
An interactive machine learning application that explores healthcare spending, estimates annual patient costs, and compares financial risk across a synthetic population.

The project brings together an auditable modeling workflow, a Streamlit analytics dashboard, and a FastAPI prediction service. Every prediction uses the same fitted preprocessing pipeline, keeping results consistent across interfaces.
Educational application using synthetic data. This project is not a validated clinical decision tool, and its estimates should not guide patient care or financial decisions.

Project Overview
Healthcare costs vary with patient characteristics, chronic conditions, and prior service utilization. This application examines those relationships through three connected capabilities:
- Explore observed costs across patient groups.
- Estimate annual spending from information assumed available before the forecast year.
- Compare alternative profiles to understand how model estimates respond to different inputs.
The emphasis is on transparent evaluation and consistent inference. Observed spending, predicted spending, and scenario comparisons are clearly distinguished throughout the application.
Application Experience
Workspace	Purpose
Executive Overview	Explore population size, total spending, average costs, and cost distributions.
Patient Cost Predictor	Generate a deterministic annual cost estimate from a patient profile.
Risk Stratification	Group patients into illustrative financial cost bands using model predictions.
Cost Analysis	Examine observed spending patterns and the composition of synthetic healthcare bills.
Model Performance	Compare measured model results and inspect evaluation provenance.
What-If Scenarios	Compare estimates for an original profile and an alternative profile.


Population views include age, state, and insurance filters. Interactive Plotly charts support exploration, while CSV exports make results available for further analysis.
Dataset
The committed dataset contains 10,000 synthetic patients and 31 raw columns, covering:
- Demographic characteristics
- Chronic condition counts
- Prior hospitalizations and emergency and office visits
- Medication counts and selected measurements
- Insurance categories
- Observed annual costs and billing components
The data is generated from predefined relationships and random variation. It does not represent real patients or establish how the application would perform in a clinical population.
Modeling Approach
Preventing target leakage
A model can appear exceptionally accurate when its predictors already contain the answer. In this dataset, inpatient, outpatient, pharmacy, and emergency costs reconstruct the annual cost target.

Those billing components are excluded from prediction, together with insurance payments, patient responsibility, outstanding balances, and target-derived risk labels.

The model uses an explicit 16-feature allowlist. Inputs are assumed to be available at the start of the forecast period. The synthetic source does not contain observation timestamps that independently verify that assumption.

previous_year_cost is also excluded because the generator creates it from the same random base as current-year costs.

The complete reasoning is documented in the [Feature Audit](reports/FEATURE_AUDIT.md).
Training and evaluation
The dataset is split into three disjoint patient groups:
Split	Patients	Role
Training	6,000	Fit preprocessing and model parameters
Validation	2,000	Compare candidates, calculate ensemble weights, and select a model
Test	2,000	Measure held-out performance after selection


Numeric imputation, scaling, and categorical encoding are fitted on training rows only. Test outcomes do not influence model fitting, ensemble weights, or selection.

Supported candidates include a mean baseline, Linear Regression, Ridge Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, and a validation-weighted ensemble.
Measured Results
The validation-selected ensemble achieved the following results on 2,000 held-out synthetic patients:
Model	MAE	RMSE	R²
Mean baseline	$4,246.86	$5,205.25	−0.0001
Linear Regression	$2,128.67	$2,804.21	0.7098
Ridge Regression	$2,128.71	$2,804.05	0.7098
Random Forest	$1,501.60	$2,049.27	0.8450
Gradient Boosting	$1,457.91	$1,967.33	0.8571
XGBoost	$1,452.36	$1,962.90	0.8578
LightGBM	$1,449.63	$1,963.43	0.8577
Validation ensemble	$1,447.35	$1,956.80	0.8587


MAE and RMSE measure prediction error in dollars. R² describes explained variation; it is not classification accuracy.

These results measure performance within the synthetic dataset. They do not establish prospective accuracy or generalization to real healthcare data.

Full metrics, ensemble weights, software versions, and the dataset fingerprint are recorded in [Evaluation Results](reports/evaluation.json).
Consistent Predictions Across Interfaces
The dashboard and API share one prediction implementation:
Patient profile
      │
      ▼
Shared input validation
      │
      ▼
Saved preprocessing pipeline
      │
      ▼
Trained prediction model
      │
      ▼
Annual cost estimate and illustrative cost band
      │
      ├── Streamlit dashboard
      ├── FastAPI service
      └── What-if comparison
Identical inputs produce identical predictions from the same loaded artifact. Missing or incompatible artifacts produce explicit errors.

Predictions include the model identity and dataset fingerprint. The application does not fabricate confidence intervals, bill allocations, clinical recommendations, or financial savings.
Financial Cost Bands
Predicted annual spending is grouped into illustrative bands:
Band	Estimated annual cost
Low	Below $5,000
Medium	$5,000 to below $25,000
High	$25,000 to below $100,000
Catastrophic	$100,000 or more


These are financial thresholds, not validated clinical risk classifications.

What-if comparisons describe model sensitivity. A difference between two estimates does not establish a treatment effect or achievable savings.
Technology Stack
Layer	Technologies
Data processing	Python, pandas, NumPy
Machine learning	scikit-learn, XGBoost, LightGBM
Dashboard	Streamlit, Plotly
Prediction API	FastAPI, Pydantic, Uvicorn
Validation	pytest, coverage, Black, Ruff
Containers and automation	Docker, Docker Compose, GitHub Actions


Quality and Validation
Local validation completed with 47 passing tests and 89% source coverage.

Regression checks cover:
- Exclusion of target-derived predictors
- Training-only preprocessing and disjoint patient splits
- Independence from held-out test outcomes
- Finite ensemble weights, including zero-error cases
- Deterministic and consistent inference
- API validation and unavailable-artifact handling
- All six dashboard pages and prediction forms
Formatting, lint, Docker build, and local container smoke checks also passed. Hosted GitHub Actions status is separate from these local results.

Detailed evidence is available in the [Execution Report](EXECUTION_REPORT.md).
Project Structure
healthcare_cost_prediction/
├── dashboards/          Streamlit analytics application
├── data/raw/            Committed synthetic patient sample
├── docs/                Technical documentation
├── notebooks/           Exploratory analysis
├── reports/             Feature audit, metrics, and validation evidence
├── scripts/             Synthetic generation and system checks
├── src/
│   ├── api/             FastAPI endpoints
│   ├── data/            Feature contract and dataset splitting
│   └── models/          Training and shared inference
├── tests/               Modeling, API, and dashboard regressions
└── .github/workflows/   Automated validation definition
Limitations
The application has not undergone clinical, prospective, or external validation. Its random patient split does not assess performance across future time periods or different healthcare systems.

The synthetic generator defines the relationships the models learn. Fairness, calibrated uncertainty, privacy controls for real patient data, and public-service security have not been established.

Population analytics include training records; held-out performance should be assessed through the evaluation report.
