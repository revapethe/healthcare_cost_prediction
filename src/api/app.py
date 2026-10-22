"""FastAPI adapter for the same inference service used by Streamlit."""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.models.predict import ArtifactError, CostPredictor, PatientInput

logger = logging.getLogger(__name__)


def create_app(model_path=None):
    @asynccontextmanager
    async def lifespan(app):
        try:
            app.state.predictor = CostPredictor(model_path)
        except ArtifactError:
            logger.warning("Model unavailable; prediction endpoints return 503")
            app.state.predictor = None
        yield

    app = FastAPI(
        title="Healthcare Cost & Risk Analytics",
        version="2.0.0",
        lifespan=lifespan,
        description="Synthetic educational estimates. Not a clinical decision tool.",
    )

    def get_predictor(request):
        predictor = getattr(request.app.state, "predictor", None)
        if predictor is None:
            raise HTTPException(
                503, "Model unavailable. Train the model and restart the service."
            )
        return predictor

    @app.get("/")
    def root():
        return {"application": "Healthcare Cost & Risk Analytics", "educational": True}

    @app.get("/health")
    def health(request: Request):
        get_predictor(request)
        return {"status": "ready", "model_loaded": True, "version": "2.0.0"}

    @app.post("/api/v1/predict")
    def predict(patient: PatientInput, request: Request):
        try:
            return get_predictor(request).predict(patient)
        except ArtifactError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post("/api/v1/predict/batch")
    def batch(payload: BatchInput, request: Request):
        predictor = get_predictor(request)
        try:
            return {"predictions": [predictor.predict(p) for p in payload.patients]}
        except ArtifactError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/api/v1/stats")
    def stats(request: Request):
        return get_predictor(request).metadata

    @app.get("/api/v1/risk-categories")
    def categories():
        return {
            "notice": "Illustrative annual cost bands, not clinical risk",
            "bands": {
                "Low": "<5000",
                "Medium": "5000–24999.99",
                "High": "25000–99999.99",
                "Catastrophic": ">=100000",
            },
        }

    return app


class BatchInput(BaseModel):
    patients: list[PatientInput] = Field(min_length=1, max_length=1000)


app = create_app()
