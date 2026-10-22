"""System check: real artifact, held-out inference and API prediction parity."""

from fastapi.testclient import TestClient
import numpy as np

from src.api.app import create_app
from src.data.pipeline import load_dataset, split_dataset
from src.models.predict import CostPredictor, PatientInput


def main():
    predictor = CostPredictor()
    test = split_dataset(load_dataset())["test"]
    predictions = predictor.predict_population(test)
    assert len(predictions) == len(test) and np.isfinite(predictions).all()
    patient = PatientInput(age=55, gender="M", bmi=28)
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/api/v1/predict", json=patient.model_dump())
        assert response.status_code == 200
        assert response.json() == predictor.predict(patient)
    print(f"PASS: {len(test)} held-out predictions and API/shared inference parity")


if __name__ == "__main__":
    main()
