# Installation and troubleshooting

Use Python 3.12 and the commands in [README](README.md#run-locally). Install `requirements-dev.lock` for tests and formatting, or `requirements.lock` for runtime. `requirements*.txt` contains pinned direct dependencies; `requirements*.lock` includes the resolved transitive set. Update locks deliberately after changing direct dependencies.

```bash
python -m pip install uv
uv pip compile requirements.txt -o requirements.lock
uv pip compile requirements-dev.txt -o requirements-dev.lock
```

Run every module command from the repository root. Virtual environments, model artifacts and splits are ignored by Git. The committed sample is sufficient; data generation is optional and should target another directory unless replacing the sample is intentional.

If a model is unavailable, run `python -m src.models.train` and restart the API/dashboard. An old `models/saved/best_model.pkl` is deliberately unsupported: it uses leaked features and incompatible preprocessing. If dataset fingerprints differ, retrain against the current CSV. The saved bundle contains its fitted preprocessor; do not fit a separate transformer at inference time.

If a port is occupied, select another Uvicorn/Streamlit port. Docker requires a running daemon and access to its socket. The image needs Python package indexes and Debian packages for OpenMP (`libgomp1`); no database is needed. Build validation is separate from deployment.

A prediction is an educational synthetic estimate. Enter only synthetic profiles. Missing required fields, unsupported categories, impossible numeric bounds and contradictory entered blood pressures are rejected. Measurement defaults are explicit in `PatientInput`; they are not estimates of a particular patient's unobserved measurements.

For a build behind a TLS-inspecting corporate/cloud proxy, provide its trusted CA bundle without turning off verification:

```bash
docker build --secret id=build_ca,src=/path/to/trusted-ca-bundle.pem -t healthcare-analytics .
```

The optional BuildKit secret is mounted only during package installation and is not stored in the image. Forward the proxy build arguments and configure a resolvable proxy hostname when your environment requires them. Ordinary direct Internet builds need no custom CA secret.
