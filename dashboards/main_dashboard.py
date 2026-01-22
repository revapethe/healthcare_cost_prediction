"""
Healthcare Cost Prediction Dashboard
Interactive Streamlit application for exploring predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="Healthcare Cost Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
    }
    .stMetric label {
        color: #000000 !important;
    }
    .stMetric .metric-value {
        color: #000000 !important;
    }
    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #262730 !important;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load processed data"""
    try:
        df = pd.read_csv('data/processed/processed_data.csv')
        return df
    except FileNotFoundError:
        st.error("Data not found. Please run the data generation script first.")
        return None


def main():
    """Main dashboard function"""
    
    # Title and description
    st.title("🏥 Healthcare Cost Prediction & Risk Assessment")
    st.markdown("### Comprehensive analytics for healthcare cost management")
    
    # Sidebar

    with st.sidebar:
        st.markdown("### 🏥 Healthcare Analytics Platform")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["Executive Dashboard", "Patient Predictor", "Risk Stratification", 
             "Cost Analysis", "Model Performance"]
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info("This dashboard provides healthcare cost predictions and risk assessments using machine learning.")
    
    # Load data
    df = load_data()
    
    if df is None:
        st.stop()
    
    # Route to selected page
    if page == "Executive Dashboard":
        executive_dashboard(df)
    elif page == "Patient Predictor":
        patient_predictor()
    elif page == "Risk Stratification":
        risk_stratification(df)
    elif page == "Cost Analysis":
        cost_analysis(df)
    elif page == "Model Performance":
        model_performance()


def executive_dashboard(df):
    """Executive summary dashboard"""
    
    st.header("Executive Dashboard")
    st.markdown("### Key Performance Indicators")
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_patients = len(df)
        st.metric("Total Patients", f"{total_patients:,}")
    
    with col2:
        avg_cost = df['total_annual_cost'].mean()
        st.metric("Average Annual Cost", f"${avg_cost:,.0f}")
    
    with col3:
        total_cost = df['total_annual_cost'].sum()
        st.metric("Total Annual Costs", f"${total_cost/1e6:.1f}M")
    
    with col4:
        high_risk = len(df[df['total_annual_cost'] > 25000])
        st.metric("High-Risk Patients", f"{high_risk:,}", 
                 delta=f"{high_risk/total_patients*100:.1f}%")
    
    st.markdown("---")
    
    # Charts row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cost Distribution by Risk Category")
        risk_dist = df.groupby('risk_category')['total_annual_cost'].agg(['count', 'mean'])
        risk_dist = risk_dist.reset_index()
        
        fig = px.bar(risk_dist, x='risk_category', y='count',
                    color='mean', color_continuous_scale='Reds',
                    labels={'count': 'Number of Patients', 'risk_category': 'Risk Category'},
                    text='count')
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Average Cost by Age Group")
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 65, 100],
                                 labels=['<30', '30-50', '50-65', '65+'])
        age_costs = df.groupby('age_group')['total_annual_cost'].mean().reset_index()
        
        fig = px.line(age_costs, x='age_group', y='total_annual_cost',
                     markers=True, labels={'total_annual_cost': 'Average Cost'})
        fig.update_traces(line_color='#FF6B6B', marker=dict(size=12))
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Cost Drivers")
        cost_drivers = {
            'Chronic Conditions': df['chronic_conditions_count'].corr(df['total_annual_cost']),
            'Age': df['age'].corr(df['total_annual_cost']),
            'BMI': df['bmi'].corr(df['total_annual_cost']),
            'Hospitalizations': df['previous_hospitalizations'].corr(df['total_annual_cost']),
            'ER Visits': df['previous_er_visits'].corr(df['total_annual_cost'])
        }
        
        drivers_df = pd.DataFrame(list(cost_drivers.items()), 
                                 columns=['Factor', 'Correlation'])
        drivers_df = drivers_df.sort_values('Correlation', ascending=True)
        
        fig = px.bar(drivers_df, y='Factor', x='Correlation', orientation='h',
                    color='Correlation', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Cost Distribution")
        fig = px.histogram(df, x='total_annual_cost', nbins=50,
                          labels={'total_annual_cost': 'Annual Cost'},
                          color_discrete_sequence=['#4CAF50'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def patient_predictor():
    """Interactive patient cost predictor"""
    
    st.header("Patient Cost Predictor")
    st.markdown("### Enter patient information to predict healthcare costs")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        age = st.slider("Age", 18, 100, 45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        bmi = st.number_input("BMI", 15.0, 60.0, 28.0, 0.1)
        smoker = st.checkbox("Smoker")
        insurance = st.selectbox("Insurance Type", 
                                ["Medicare", "Medicaid", "Private", "HMO", "PPO", "Self-Pay"])
    
    with col2:
        st.subheader("Clinical History")
        chronic_conditions = st.number_input("Chronic Conditions", 0, 10, 2)
        office_visits = st.number_input("Office Visits (Last Year)", 0, 50, 5)
        er_visits = st.number_input("ER Visits (Last Year)", 0, 20, 1)
        hospitalizations = st.number_input("Hospitalizations (Last Year)", 0, 10, 0)
        medications = st.number_input("Current Medications", 0, 20, 3)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Vital Signs")
        bp_systolic = st.number_input("Blood Pressure (Systolic)", 80, 200, 120)
        bp_diastolic = st.number_input("Blood Pressure (Diastolic)", 50, 120, 80)
    
    with col2:
        st.subheader("Lab Values")
        cholesterol = st.number_input("Total Cholesterol", 100, 400, 200)
        glucose = st.number_input("Fasting Glucose", 70, 300, 100)
    
    if st.button("Predict Cost", type="primary"):
        # Simple prediction logic (in production, call your model)
        base_cost = 5000
        base_cost += age * 50
        base_cost += (bmi - 25) * 100 if bmi > 25 else 0
        base_cost += smoker * 3000
        base_cost += chronic_conditions * 2500
        base_cost += office_visits * 150
        base_cost += er_visits * 1500
        base_cost += hospitalizations * 8000
        base_cost += medications * 150
        
        predicted_cost = base_cost * np.random.uniform(0.8, 1.2)
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Annual Cost", f"${predicted_cost:,.0f}")
        
        with col2:
            risk = "Low" if predicted_cost < 5000 else \
                   "Medium" if predicted_cost < 25000 else \
                   "High" if predicted_cost < 100000 else "Catastrophic"
            st.metric("Risk Category", risk)
        
        with col3:
            monthly = predicted_cost / 12
            st.metric("Monthly Cost", f"${monthly:,.0f}")
        
        # Cost breakdown
        st.subheader("Estimated Cost Breakdown")
        breakdown = pd.DataFrame({
            'Category': ['Inpatient', 'Outpatient', 'Pharmacy', 'Emergency', 'Other'],
            'Cost': [predicted_cost * 0.35, predicted_cost * 0.25, 
                    predicted_cost * 0.20, predicted_cost * 0.15, predicted_cost * 0.05]
        })
        
        fig = px.pie(breakdown, values='Cost', names='Category', 
                    color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("Recommendations")
        recommendations = []
        
        if bmi > 30:
            recommendations.append("✓ Consider weight management program")
        if smoker:
            recommendations.append("✓ Smoking cessation program recommended")
        if chronic_conditions > 2:
            recommendations.append("✓ Enroll in chronic disease management")
        if er_visits > 2:
            recommendations.append("✓ Increase preventive care visits")
        if predicted_cost > 25000:
            recommendations.append("✓ Consult financial counselor for payment plans")
        
        for rec in recommendations:
            st.info(rec)


def risk_stratification(df):
    """Risk stratification analysis"""
    
    st.header("Risk Stratification Analysis")
    
    # Risk distribution
    risk_counts = df['risk_category'].value_counts()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Risk Distribution")
        for category in ['Low', 'Medium', 'High', 'Catastrophic']:
            if category in risk_counts.index:
                count = risk_counts[category]
                pct = count / len(df) * 100
                st.metric(f"{category} Risk", f"{count:,}", f"{pct:.1f}%")
    
    with col2:
        st.subheader("Risk Category Breakdown")
        fig = px.sunburst(df, path=['risk_category'], values='total_annual_cost',
                         color='total_annual_cost', color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    
    # High-risk patients analysis
    st.subheader("High-Risk Patient Analysis")
    high_risk_df = df[df['risk_category'].isin(['High', 'Catastrophic'])]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Characteristics of High-Risk Patients**")
        characteristics = pd.DataFrame({
            'Metric': ['Average Age', 'Average BMI', '% Smokers', 'Avg Chronic Conditions'],
            'Value': [
                f"{high_risk_df['age'].mean():.1f}",
                f"{high_risk_df['bmi'].mean():.1f}",
                f"{high_risk_df['smoker'].mean()*100:.1f}%",
                f"{high_risk_df['chronic_conditions_count'].mean():.1f}"
            ]
        })
        st.dataframe(characteristics, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Cost Impact**")
        st.info(f"High-risk patients ({len(high_risk_df):,}) represent "
               f"{len(high_risk_df)/len(df)*100:.1f}% of population but "
               f"{high_risk_df['total_annual_cost'].sum()/df['total_annual_cost'].sum()*100:.1f}% of total costs")


def cost_analysis(df):
    """Detailed cost analysis"""
    
    st.header("Cost Analysis")
    
    # Filters
    st.sidebar.subheader("Filters")
    age_range = st.sidebar.slider("Age Range", 18, 100, (18, 100))
    selected_risk = st.sidebar.multiselect("Risk Categories", 
                                          df['risk_category'].unique().tolist(),
                                          default=df['risk_category'].unique().tolist())
    
    # Filter data
    filtered_df = df[(df['age'] >= age_range[0]) & 
                    (df['age'] <= age_range[1]) &
                    (df['risk_category'].isin(selected_risk))]
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Median Cost", f"${filtered_df['total_annual_cost'].median():,.0f}")
    with col2:
        st.metric("Mean Cost", f"${filtered_df['total_annual_cost'].mean():,.0f}")
    with col3:
        st.metric("90th Percentile", f"${filtered_df['total_annual_cost'].quantile(0.9):,.0f}")
    with col4:
        st.metric("Max Cost", f"${filtered_df['total_annual_cost'].max():,.0f}")
    
    # Cost trends
    st.subheader("Cost Analysis by Demographics")
    
    analysis_type = st.selectbox("Analysis Type", 
                                ["By Age", "By BMI", "By Chronic Conditions", "By Insurance"])
    
    if analysis_type == "By Age":
        fig = px.scatter(filtered_df, x='age', y='total_annual_cost', 
                        color='risk_category', size='bmi',
                        labels={'total_annual_cost': 'Annual Cost'},
                        trendline="lowess")
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "By BMI":
        fig = px.scatter(filtered_df, x='bmi', y='total_annual_cost',
                        color='risk_category',
                        labels={'total_annual_cost': 'Annual Cost'},
                        trendline="ols")
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "By Chronic Conditions":
        chronic_costs = filtered_df.groupby('chronic_conditions_count')['total_annual_cost'].mean().reset_index()
        fig = px.bar(chronic_costs, x='chronic_conditions_count', y='total_annual_cost',
                    labels={'chronic_conditions_count': 'Number of Chronic Conditions',
                           'total_annual_cost': 'Average Annual Cost'})
        st.plotly_chart(fig, use_container_width=True)
    
    else:  # By Insurance
        insurance_costs = filtered_df.groupby('insurance_type')['total_annual_cost'].agg(['mean', 'count']).reset_index()
        fig = px.bar(insurance_costs, x='insurance_type', y='mean',
                    text='count', labels={'mean': 'Average Cost', 'count': 'Patients'})
        st.plotly_chart(fig, use_container_width=True)


def model_performance():
    """Display model performance metrics"""
    
    st.header("Model Performance")
    
    st.subheader("Performance Metrics")
    
    # Mock performance data (in production, load from actual metrics)
    metrics_data = {
        'Model': ['XGBoost', 'Random Forest', 'LightGBM', 'Gradient Boosting'],
        'RMSE': [3245, 3567, 3421, 3789],
        'MAE': [2156, 2389, 2267, 2501],
        'R²': [0.87, 0.84, 0.86, 0.82],
        'Training Time (s)': [45, 32, 38, 56]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Display table
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)
    
    # Visualize comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(metrics_df, x='Model', y='RMSE', 
                    color='RMSE', color_continuous_scale='Reds_r',
                    title="Model Comparison - RMSE")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(metrics_df, x='Model', y='R²',
                    color='R²', color_continuous_scale='Greens',
                    title="Model Comparison - R² Score")
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (mock data)
    st.subheader("Feature Importance")
    
    feature_importance = pd.DataFrame({
        'Feature': ['Previous Year Cost', 'Age', 'Chronic Conditions', 'BMI', 
                   'Hospitalizations', 'Medications', 'ER Visits', 'Office Visits'],
        'Importance': [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05]
    })
    
    fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                color='Importance', color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
