"""Healthcare cost analytics on synthetic data with shared model inference."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st
from pydantic import ValidationError

from src.data.pipeline import ROOT, dataset_digest, load_dataset
from src.models.predict import ArtifactError, CostPredictor, categorize_risk

PAGES = [
    "Executive Overview",
    "Patient Cost Predictor",
    "Risk Stratification",
    "Cost Analysis",
    "Model Performance",
    "What-If Scenarios",
]
COLORS = ["#007F82", "#173A60", "#3BA9B2", "#E2A34B"]
st.set_page_config(
    page_title="Healthcare Cost & Risk Analytics", page_icon="⚕", layout="wide"
)
st.markdown(
    """<style>
.block-container {max-width: 1450px; padding-top: 2rem;}
h1,h2,h3 {letter-spacing: -.025em; color: #142D4E;}
[data-testid="stMetric"] {background: white; border: 1px solid #DFE8F0; border-radius: 12px; padding: 1rem;}
[data-testid="stSidebar"] {border-right: 1px solid #DFE8F0;}
.stButton>button {border-radius: 8px;}
@media(max-width: 700px) {.block-container {padding: 1rem;}}
</style>""",
    unsafe_allow_html=True,
)


def chart(fig):
    fig.update_layout(
        template="plotly_white",
        font_color="#142D4E",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=35, b=15),
        legend_title_text="",
    )
    st.plotly_chart(fig, width="stretch")


def export(frame, label="Export filtered data", key="export"):
    st.download_button(
        label,
        frame.to_csv(index=False).encode(),
        file_name=f"{key}.csv",
        mime="text/csv",
        key=key,
    )


def patient_form(prefix):
    """All model inputs are collected; no hidden random or cost heuristics."""
    a, b, c = st.columns(3)
    with a:
        age = st.number_input(
            "Age", 18, 100, 55, key=f"{prefix}_age", help="Age at forecast start"
        )
        gender = st.selectbox(
            "Gender",
            ["M", "F"],
            key=f"{prefix}_gender",
            help="Synthetic dataset uses two categories; limits generalizability",
        )
        bmi = st.number_input("BMI", 15.0, 60.0, 28.0, step=0.1, key=f"{prefix}_bmi")
        smoker = st.checkbox("Smoker", key=f"{prefix}_smoker")
    with b:
        chronic = st.number_input(
            "Chronic conditions", 0, 20, 2, key=f"{prefix}_chronic"
        )
        office = st.number_input(
            "Prior office visits",
            0,
            365,
            5,
            key=f"{prefix}_office",
            help="Visits before the forecast year",
        )
        er = st.number_input("Prior ER visits", 0, 100, 1, key=f"{prefix}_er")
        hospital = st.number_input(
            "Prior hospitalizations", 0, 100, 1, key=f"{prefix}_hospital"
        )
    with c:
        meds = st.number_input("Medication count", 0, 100, 3, key=f"{prefix}_meds")
        insurance = st.selectbox(
            "Insurance",
            ["Private", "Medicare", "Medicaid", "Self-Pay", "HMO", "PPO"],
            key=f"{prefix}_insurance",
        )
        state = st.selectbox(
            "State",
            ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI"],
            key=f"{prefix}_state",
        )
        vaccination = st.selectbox(
            "Vaccination status",
            ["Complete", "Partial", "None"],
            key=f"{prefix}_vaccination",
        )
    with st.expander("Measurements at forecast start", expanded=False):
        left, right = st.columns(2)
        with left:
            systolic = st.number_input(
                "Systolic BP", 80, 220, 120, key=f"{prefix}_systolic"
            )
            diastolic = st.number_input(
                "Diastolic BP", 40, 140, 80, key=f"{prefix}_diastolic"
            )
        with right:
            cholesterol = st.number_input(
                "Total cholesterol", 100, 400, 200, key=f"{prefix}_cholesterol"
            )
            glucose = st.number_input(
                "Fasting glucose", 70, 300, 100, key=f"{prefix}_glucose"
            )
    return dict(
        age=age,
        gender=gender,
        bmi=bmi,
        smoker=smoker,
        chronic_conditions_count=chronic,
        previous_office_visits=office,
        previous_er_visits=er,
        previous_hospitalizations=hospital,
        medication_count=meds,
        insurance_type=insurance,
        state=state,
        vaccination_status=vaccination,
        blood_pressure_systolic=systolic,
        blood_pressure_diastolic=diastolic,
        cholesterol_total=cholesterol,
        glucose_fasting=glucose,
    )


def display_prediction(result):
    a, b, c = st.columns(3)
    a.metric("Estimated annual cost", f"${result['predicted_cost']:,.0f}")
    b.metric("Annual estimate ÷ 12", f"${result['monthly_cost']:,.0f}")
    c.metric("Illustrative cost band", result["risk_category"])
    st.caption(
        f"Model: {result['model_name']}. No calibrated uncertainty interval is available."
    )
    export(pd.DataFrame([result]), "Export prediction", "prediction")


def main():
    st.sidebar.title("HEALTHCARE")
    st.sidebar.caption("Cost & Risk Analytics · Synthetic dataset")
    page = st.sidebar.radio("Navigation", PAGES)
    st.title(page)
    st.caption("Explore costs. Inspect model evidence. Compare patient scenarios.")
    st.info(
        "Educational application using synthetic patients. Estimates and cost bands are not validated for clinical or financial decisions."
    )
    try:
        data = load_dataset()
    except (OSError, ValueError) as exc:
        st.error(f"Dataset unavailable: {exc}")
        return
    try:
        predictor = CostPredictor()
    except ArtifactError as exc:
        predictor = None
        st.warning(str(exc))
    if predictor is not None and predictor.metadata["dataset_sha256"] != dataset_digest(
        ROOT / "data/raw/patient_data.csv"
    ):
        st.error(
            "Dataset differs from the trained artifact. Retrain before comparing model performance or population predictions."
        )
        predictor = None
    model_pages = [
        "Patient Cost Predictor",
        "Risk Stratification",
        "Model Performance",
        "What-If Scenarios",
    ]
    if page in model_pages and predictor is None:
        st.error(
            "This page requires a compatible trained model. Run python -m src.models.train."
        )
        return
    if page == "Patient Cost Predictor":
        st.subheader("Patient profile")
        st.caption(
            "Enter information known before the forecast year. All inputs use the saved preprocessing pipeline."
        )
        with st.form("patient_form"):
            inputs = patient_form("patient")
            submitted = st.form_submit_button("Estimate annual cost", type="primary")
        if submitted:
            try:
                display_prediction(predictor.predict(inputs))
            except (ValidationError, ArtifactError, ValueError) as exc:
                st.error(f"Unable to predict: {exc}")
        return
    if page == "What-If Scenarios":
        st.subheader("Compare two model estimates")
        st.caption(
            "Sensitivity analysis, not a causal treatment or savings estimate. Changing one input may create an unrealistic patient profile."
        )
        with st.form("scenario_form"):
            inputs = patient_form("scenario")
            st.markdown("#### Alternative profile")
            a, b = st.columns(2)
            bmi = a.number_input("Scenario BMI", 15.0, 60.0, 25.0, step=0.1)
            smoker = b.checkbox("Scenario smoker")
            submitted = st.form_submit_button("Compare scenarios", type="primary")
        if submitted:
            try:
                baseline = predictor.predict(inputs)
                alternative = predictor.predict(
                    {**inputs, "bmi": bmi, "smoker": smoker}
                )
                frame = pd.DataFrame(
                    [
                        {"Profile": "Original", **baseline},
                        {"Profile": "Alternative", **alternative},
                    ]
                )
                chart(
                    px.bar(
                        frame,
                        x="Profile",
                        y="predicted_cost",
                        color="Profile",
                        color_discrete_sequence=COLORS,
                        labels={"predicted_cost": "Estimated annual cost ($)"},
                    )
                )
                st.metric(
                    "Alternative minus original",
                    f"${alternative['predicted_cost'] - baseline['predicted_cost']:,.2f}",
                )
                export(frame, "Export scenario comparison", "scenarios")
            except (ValidationError, ArtifactError, ValueError) as exc:
                st.error(f"Unable to compare: {exc}")
        return
    if page == "Model Performance":
        report = predictor.metadata
        selected = report["selected_model"]
        scores = report["test"][selected]
        a, b, c = st.columns(3)
        a.metric("Held-out MAE", f"${scores['MAE']:,.2f}")
        b.metric("Held-out RMSE", f"${scores['RMSE']:,.2f}")
        c.metric("Held-out R²", f"{scores['R2']:.4f}")
        st.caption(
            f"Selected on validation: {selected}. Train/validation/test sizes: {report['split_sizes']}. All preprocessing fits training rows only."
        )
        comparison = pd.DataFrame(report["test"]).T.rename_axis("Model").reset_index()
        chart(
            px.bar(
                comparison,
                x="RMSE",
                y="Model",
                orientation="h",
                color_discrete_sequence=COLORS,
                labels={"RMSE": "Held-out RMSE ($)"},
            )
        )
        st.dataframe(comparison, hide_index=True, width="stretch")
        export(comparison, "Export evaluation metrics", "evaluation")
        st.subheader("Provenance & limitations")
        st.code(
            f"Dataset SHA-256: {report['dataset_sha256']}\nTrained: {report['created_at']}"
        )
        st.write("Predictors: " + ", ".join(report["features"]))
        for limitation in report["limitations"]:
            st.caption("• " + limitation)
        return
    st.sidebar.subheader("Population filters")
    states = st.sidebar.multiselect("States", sorted(data.state.unique()))
    insurance = st.sidebar.multiselect(
        "Insurance types", sorted(data.insurance_type.unique())
    )
    ages = st.sidebar.slider("Age range", 18, 100, (18, 100))
    filtered = data[data.age.between(*ages)].copy()
    if states:
        filtered = filtered[filtered.state.isin(states)]
    if insurance:
        filtered = filtered[filtered.insurance_type.isin(insurance)]
    st.caption(
        f"{len(filtered):,} of {len(data):,} synthetic records match current filters. Dollar values are synthetic annual totals."
    )
    if filtered.empty:
        st.warning("No patients match these filters. Widen the selected population.")
        return
    if page == "Executive Overview":
        observed_high = filtered.total_annual_cost >= 25000
        a, b, c, d = st.columns(4)
        a.metric("Synthetic patients", f"{len(filtered):,}")
        b.metric("Observed annual cost", f"${filtered.total_annual_cost.sum():,.0f}")
        c.metric("Mean observed cost", f"${filtered.total_annual_cost.mean():,.0f}")
        d.metric("Observed cost ≥ $25k", f"{observed_high.mean():.1%}")
        left, right = st.columns(2)
        with left:
            chart(
                px.histogram(
                    filtered,
                    x="total_annual_cost",
                    nbins=35,
                    color_discrete_sequence=COLORS,
                    labels={"total_annual_cost": "Observed annual cost ($)"},
                    title="Observed cost distribution",
                )
            )
        with right:
            grouped = filtered.groupby(
                "insurance_type", as_index=False
            ).total_annual_cost.mean()
            chart(
                px.bar(
                    grouped,
                    x="insurance_type",
                    y="total_annual_cost",
                    color_discrete_sequence=COLORS,
                    labels={
                        "insurance_type": "Insurance",
                        "total_annual_cost": "Mean observed cost ($)",
                    },
                    title="Observed costs by insurance",
                )
            )
    elif page == "Risk Stratification":
        try:
            filtered["predicted_cost"] = predictor.predict_population(filtered)
        except (ValidationError, ArtifactError, ValueError) as exc:
            st.error(f"Population inference failed: {exc}")
            return
        filtered["predicted_band"] = filtered.predicted_cost.map(categorize_risk)
        st.caption(
            "Bands use model estimates: Low < $5k; Medium < $25k; High < $100k; Catastrophic ≥ $100k. This population includes training records; use Model Performance for held-out evaluation."
        )
        distribution = filtered.groupby("predicted_band", as_index=False).agg(
            patients=("patient_id", "count"),
            mean_predicted_cost=("predicted_cost", "mean"),
        )
        chart(
            px.bar(
                distribution,
                x="predicted_band",
                y="patients",
                color="predicted_band",
                color_discrete_sequence=COLORS,
            )
        )
        st.dataframe(distribution, hide_index=True, width="stretch")
        st.dataframe(
            filtered[["patient_id", "age", "predicted_cost", "predicted_band"]]
            .sort_values("predicted_cost", ascending=False)
            .head(100),
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption(
            "Descriptive analysis of observed synthetic bills. Billed components below are excluded from all model inputs."
        )
        axis = st.selectbox(
            "Compare costs against",
            ["age", "bmi", "chronic_conditions_count", "previous_hospitalizations"],
        )
        chart(
            px.scatter(
                filtered,
                x=axis,
                y="total_annual_cost",
                color="insurance_type",
                opacity=0.45,
                color_discrete_sequence=COLORS,
                labels={"total_annual_cost": "Observed annual cost ($)"},
            )
        )
        components = [
            "inpatient_cost",
            "outpatient_cost",
            "pharmacy_cost",
            "emergency_cost",
        ]
        totals = (
            filtered[components]
            .sum()
            .rename_axis("Component")
            .reset_index(name="Observed cost ($)")
        )
        chart(
            px.bar(
                totals,
                x="Component",
                y="Observed cost ($)",
                color_discrete_sequence=COLORS,
            )
        )
    export(filtered, key="population")


if __name__ == "__main__":
    main()
