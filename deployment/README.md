# Streamlit demonstration model

`model.joblib` is a trusted, repository-supplied fitted pipeline for the committed synthetic patient dataset. It contains the same model and evaluation metadata as the measured run in `reports/evaluation.json`. It is included so Streamlit Community Cloud can serve predictions without executing training during app startup.

Artifact generation used Python 3.12 and the pinned runtime dependencies. Configure the hosted app for Python 3.12. `packages.txt` supplies the OpenMP library used by the supported boosters. Do not replace pinned dependencies without rebuilding and validating the artifact.

Loading priority is an explicitly supplied path, then `MODEL_PATH`, then a local `artifacts/model.joblib` if present, then this bundled demonstration artifact. Explicitly configured missing paths and corrupt local artifacts produce errors; they do not silently fall back. No external model download or startup training occurs.

For an intentional dataset/model update, retrain with `python -m src.models.train`, validate, then copy `artifacts/model.joblib` here and update the manifest. Commit the model, evaluation report and manifest together. Never load artifacts from untrusted sources. The dashboard checks the dataset fingerprint before serving model pages.

The artifact is approximately 1 MB and contains a model trained only on synthetic data. It is an educational demonstration, not a validated clinical model. Publishing this branch or changing an existing app's deployed branch is a separate user-controlled action.

Validation for this deployment fix: 49 tests passed with one upstream test-client deprecation warning. The prediction-dependent Streamlit pages also passed AppTest using this bundled artifact explicitly. A regression test simulates a fresh checkout with no local training outputs and verifies API readiness/inference through the fallback. Hosted deployment remains user-controlled and has not been verified by these local checks.
