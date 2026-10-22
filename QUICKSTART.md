# Quick start

See [README](README.md#run-locally) for the tested commands. From the repository root with Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.lock
python -m src.data.pipeline
python -m src.models.train
pytest -q
```

Then run `uvicorn src.api.app:app --port 8000` and `streamlit run dashboards/main_dashboard.py` in separate terminals. The API and dashboard use the same fitted pipeline. No database or credentials are required. Use synthetic inputs only; this is not a validated clinical tool.
