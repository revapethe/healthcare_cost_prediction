"""Execute real Streamlit pages, forms, filters and missing-artifact paths."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.models.predict import CostPredictor, PatientInput

APP = str(Path(__file__).resolve().parents[1] / "dashboards/main_dashboard.py")
PAGES = [
    "Executive Overview",
    "Patient Cost Predictor",
    "Risk Stratification",
    "Cost Analysis",
    "Model Performance",
    "What-If Scenarios",
]


def open_page(page):
    app = AppTest.from_file(APP, default_timeout=45).run()
    app.radio[0].set_value(page).run()
    assert not app.exception
    assert not app.error
    return app


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_errors(page):
    open_page(page)


def test_patient_form_runs_real_shared_inference():
    app = open_page("Patient Cost Predictor")
    app.button[0].click().run()
    assert not app.exception and not app.error
    patient = PatientInput(
        age=55,
        gender="M",
        bmi=28,
        chronic_conditions_count=2,
        previous_office_visits=5,
        previous_er_visits=1,
        previous_hospitalizations=1,
        medication_count=3,
    )
    expected = CostPredictor().predict(patient)
    assert app.metric[0].value == f"${expected['predicted_cost']:,.0f}"
    first = [m.value for m in app.metric]
    app.button[0].click().run()
    assert [m.value for m in app.metric] == first


def test_what_if_identical_profiles_have_zero_delta():
    app = open_page("What-If Scenarios")
    [x for x in app.number_input if x.label == "Scenario BMI"][0].set_value(28.0)
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert app.metric[0].value == "$0.00"


def test_filters_handle_empty_cohort():
    app = open_page("Executive Overview")
    app.slider[0].set_value((100, 100)).run()
    assert not app.exception and not app.error
    assert any("No patients" in w.value for w in app.warning)


def test_dashboard_reports_missing_model(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    app = AppTest.from_file(APP, default_timeout=45).run()
    app.radio[0].set_value("Patient Cost Predictor").run()
    assert not app.exception
    assert any("requires a compatible" in e.value for e in app.error)
