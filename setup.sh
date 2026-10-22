#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.lock
python -m src.data.pipeline
python -m src.models.train
pytest -v
black --check src dashboards scripts tests
ruff check src dashboards scripts tests
