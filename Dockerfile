FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.lock .
# Optional custom CA bundle for trusted corporate/cloud build proxies.
# BuildKit secrets are ephemeral and are not copied into the image.
RUN --mount=type=secret,id=build_ca \
    if [ -f /run/secrets/build_ca ]; then \
      PIP_CERT=/run/secrets/build_ca pip install --no-cache-dir -r requirements.lock; \
    else pip install --no-cache-dir -r requirements.lock; fi
COPY src src
COPY dashboards dashboards
COPY .streamlit .streamlit
COPY data/raw/patient_data.csv data/raw/patient_data.csv
# Fit artifact inside image; no host artifact or untrusted pickle is loaded.
RUN python -m src.models.train
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
