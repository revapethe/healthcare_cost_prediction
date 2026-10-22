# Project summary

A synthetic healthcare cost and risk analytics application built with Python, pandas, NumPy, scikit-learn, Streamlit, Plotly and FastAPI. Docker builds an artifact from the sample, and GitHub Actions defines repeatable validation without publishing images.

The sample has 10,000 patients; the forecast contract has 16 inputs. Supported models are a mean baseline, linear regression, ridge, random forest, gradient boosting, XGBoost, LightGBM and a validation-weighted ensemble. Model selection uses validation RMSE, followed by untouched test evaluation.

All user-facing predictions use the fitted preprocessing pipeline through `CostPredictor`. Six dashboard pages distinguish observed data, forecasts and scenario sensitivity. Generated artifacts are ignored; measured metrics and provenance are in `reports/evaluation.json`.

See [README](README.md) for current commands and limitations and [EXECUTION_REPORT](EXECUTION_REPORT.md) for verification. This project is educational and has not been clinically validated or deployed by this change.
