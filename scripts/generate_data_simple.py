"""
Simplified Data Generator - No External Dependencies
Generates synthetic healthcare data using only numpy and pandas
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

print("Starting data generation...")

# Constants
ICD10_CODES = {
    'E11': 'Type 2 Diabetes',
    'I10': 'Hypertension',
    'J44': 'COPD',
    'I25': 'Chronic Ischemic Heart Disease',
    'M17': 'Osteoarthritis of Knee',
    'F32': 'Major Depressive Disorder',
    'N18': 'Chronic Kidney Disease',
    'E78': 'Hyperlipidemia',
    'J45': 'Asthma',
    'K21': 'GERD'
}

INSURANCE_TYPES = ['Medicare', 'Medicaid', 'Private', 'Self-Pay', 'HMO', 'PPO']
STATES = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']

# Number of patients
n_patients = 10000
print(f"Generating data for {n_patients} patients...")

# Generate patient IDs
patient_ids = [f'P{str(i).zfill(7)}' for i in range(1, n_patients + 1)]

# Demographics
print("Generating demographics...")
ages = np.random.normal(55, 18, n_patients).clip(18, 95).astype(int)
genders = np.random.choice(['M', 'F'], n_patients, p=[0.48, 0.52])
states = np.random.choice(STATES, n_patients)
zip_codes = [f'{random.randint(10000, 99999)}' for _ in range(n_patients)]
insurance_types = np.random.choice(INSURANCE_TYPES, n_patients, p=[0.25, 0.15, 0.35, 0.05, 0.10, 0.10])

# BMI and smoking
bmis = np.random.normal(28, 6, n_patients).clip(15, 55).round(1)
smoking_prob = np.where(ages < 40, 0.15, 0.20)
smokers = np.random.binomial(1, smoking_prob)

# Clinical data
print("Generating clinical data...")
age_factor = (ages - 18) / 77
bmi_factor = (bmis - 15) / 40
chronic_conditions_prob = 0.3 * age_factor + 0.2 * bmi_factor + 0.1 * smokers
chronic_conditions_prob = np.clip(chronic_conditions_prob, 0, 0.8)
chronic_conditions_count = np.random.binomial(5, chronic_conditions_prob)

# Specific conditions
chronic_conditions = []
for n_cond in chronic_conditions_count:
    if n_cond > 0:
        selected = np.random.choice(list(ICD10_CODES.keys()), 
                                   size=min(n_cond, len(ICD10_CODES)), 
                                   replace=False)
        chronic_conditions.append(';'.join(selected))
    else:
        chronic_conditions.append('')

# Healthcare utilization
previous_hospitalizations = np.random.poisson(age_factor * 2, n_patients)
previous_er_visits = np.random.poisson(chronic_conditions_prob * 3, n_patients)
previous_office_visits = np.random.poisson(5 + chronic_conditions_prob * 10, n_patients)
medication_count = np.random.poisson(chronic_conditions_count * 1.5, n_patients)
vaccination_status = np.random.choice(['Complete', 'Partial', 'None'], n_patients, p=[0.6, 0.25, 0.15])

# Lab values
blood_pressure_systolic = np.random.normal(130, 18, n_patients).clip(90, 200).astype(int)
blood_pressure_diastolic = np.random.normal(80, 12, n_patients).clip(60, 120).astype(int)
cholesterol_total = np.random.normal(200, 40, n_patients).clip(100, 400).astype(int)
glucose_fasting = np.random.normal(100, 30, n_patients).clip(70, 300).astype(int)

# Financial data
print("Generating financial data...")
base_cost = 5000 + (age_factor * 8000) + (chronic_conditions_count / 5 * 15000) + (bmi_factor * 5000)
base_cost = base_cost + (smokers * 3000)
base_cost = base_cost * np.random.uniform(0.7, 1.3, n_patients)

previous_year_cost = base_cost * np.random.uniform(0.6, 1.2, n_patients)

inpatient_cost = base_cost * np.random.uniform(0.3, 0.5, n_patients) * (previous_hospitalizations > 0)
outpatient_cost = base_cost * np.random.uniform(0.2, 0.4, n_patients)
pharmacy_cost = medication_count * 150 * np.random.uniform(0.8, 1.5, n_patients)
emergency_cost = previous_er_visits * 1500 * np.random.uniform(0.8, 1.2, n_patients)

total_annual_cost = inpatient_cost + outpatient_cost + pharmacy_cost + emergency_cost

# Insurance coverage
insurance_coverage_map = {
    'Medicare': 0.80, 'Medicaid': 0.85, 'Private': 0.75,
    'Self-Pay': 0.0, 'HMO': 0.78, 'PPO': 0.72
}
coverage_rates = np.array([insurance_coverage_map[ins] for ins in insurance_types])
insurance_paid = total_annual_cost * coverage_rates
patient_responsibility = total_annual_cost - insurance_paid

payment_status = np.random.choice(['Paid', 'Partial', 'Outstanding'], n_patients, p=[0.7, 0.2, 0.1])
outstanding_balance = patient_responsibility * (payment_status == 'Outstanding')

# Risk categories
def assign_risk(cost):
    if cost < 5000:
        return 'Low'
    elif cost < 25000:
        return 'Medium'
    elif cost < 100000:
        return 'High'
    else:
        return 'Catastrophic'

risk_categories = [assign_risk(cost) for cost in total_annual_cost]

# Create DataFrame
print("Creating DataFrame...")
df = pd.DataFrame({
    'patient_id': patient_ids,
    'age': ages,
    'gender': genders,
    'state': states,
    'zip_code': zip_codes,
    'insurance_type': insurance_types,
    'bmi': bmis,
    'smoker': smokers,
    'chronic_conditions_count': chronic_conditions_count,
    'chronic_conditions': chronic_conditions,
    'previous_hospitalizations': previous_hospitalizations,
    'previous_er_visits': previous_er_visits,
    'previous_office_visits': previous_office_visits,
    'medication_count': medication_count,
    'vaccination_status': vaccination_status,
    'blood_pressure_systolic': blood_pressure_systolic,
    'blood_pressure_diastolic': blood_pressure_diastolic,
    'cholesterol_total': cholesterol_total,
    'glucose_fasting': glucose_fasting,
    'total_annual_cost': total_annual_cost.round(2),
    'inpatient_cost': inpatient_cost.round(2),
    'outpatient_cost': outpatient_cost.round(2),
    'pharmacy_cost': pharmacy_cost.round(2),
    'emergency_cost': emergency_cost.round(2),
    'insurance_paid': insurance_paid.round(2),
    'patient_responsibility': patient_responsibility.round(2),
    'previous_year_cost': previous_year_cost.round(2),
    'payment_status': payment_status,
    'outstanding_balance': outstanding_balance.round(2),
    'risk_category': risk_categories,
    'data_date': datetime.now().strftime('%Y-%m-%d')
})

# Save data
print("Saving data...")
df.to_csv('data/raw/patient_data.csv', index=False)

# Generate summary
print("\n" + "="*60)
print("DATA GENERATION SUMMARY")
print("="*60)
print(f"Total patients: {len(df)}")
print(f"\nRisk Category Distribution:")
print(df['risk_category'].value_counts())
print(f"\nCost Statistics:")
print(df['total_annual_cost'].describe())
print(f"\nAverage costs by risk category:")
print(df.groupby('risk_category')['total_annual_cost'].mean())
print("="*60)
print("\n✅ Data generation complete!")
print(f"File saved to: data/raw/patient_data.csv")
