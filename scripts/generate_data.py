"""
Generate Synthetic Healthcare and Financial Data
This script creates realistic patient data for training and testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
import click
from pathlib import Path

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

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

CPT_CODES = {
    '99213': 'Office Visit - Moderate Complexity',
    '99214': 'Office Visit - High Complexity',
    '80053': 'Comprehensive Metabolic Panel',
    '93000': 'Electrocardiogram',
    '71046': 'Chest X-Ray',
    '36415': 'Blood Draw',
    '99385': 'Preventive Visit',
    '90471': 'Immunization Administration'
}

INSURANCE_TYPES = ['Medicare', 'Medicaid', 'Private', 'Self-Pay', 'HMO', 'PPO']
STATES = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']


def generate_patient_demographics(n_patients):
    """Generate patient demographic information"""
    
    data = {
        'patient_id': [f'P{str(i).zfill(7)}' for i in range(1, n_patients + 1)],
        'age': np.random.normal(55, 18, n_patients).clip(18, 95).astype(int),
        'gender': np.random.choice(['M', 'F'], n_patients, p=[0.48, 0.52]),
        'state': np.random.choice(STATES, n_patients),
        'zip_code': [fake.zipcode()[:5] for _ in range(n_patients)],
        'insurance_type': np.random.choice(INSURANCE_TYPES, n_patients, 
                                          p=[0.25, 0.15, 0.35, 0.05, 0.10, 0.10])
    }
    
    # Generate BMI with realistic distribution
    data['bmi'] = np.random.normal(28, 6, n_patients).clip(15, 55).round(1)
    
    # Smoking status (age-dependent)
    smoking_prob = np.where(data['age'] < 40, 0.15, 0.20)
    data['smoker'] = np.random.binomial(1, smoking_prob)
    
    return pd.DataFrame(data)


def generate_clinical_data(demographics_df):
    """Generate clinical conditions and history"""
    
    n_patients = len(demographics_df)
    
    # Number of chronic conditions (age and BMI dependent)
    age_factor = (demographics_df['age'] - 18) / 77  # Normalize to 0-1
    bmi_factor = (demographics_df['bmi'] - 15) / 40
    
    chronic_conditions_prob = 0.3 * age_factor + 0.2 * bmi_factor + 0.1 * demographics_df['smoker']
    chronic_conditions_prob = chronic_conditions_prob.clip(0, 0.8)
    
    n_conditions = np.random.binomial(5, chronic_conditions_prob)
    
    # Generate specific conditions
    conditions = []
    for i, n_cond in enumerate(n_conditions):
        patient_conditions = []
        if n_cond > 0:
            selected_conditions = np.random.choice(
                list(ICD10_CODES.keys()), 
                size=min(n_cond, len(ICD10_CODES)), 
                replace=False
            )
            patient_conditions = ';'.join(selected_conditions)
        conditions.append(patient_conditions)
    
    clinical_data = {
        'patient_id': demographics_df['patient_id'],
        'chronic_conditions_count': n_conditions,
        'chronic_conditions': conditions,
        'previous_hospitalizations': np.random.poisson(age_factor * 2, n_patients),
        'previous_er_visits': np.random.poisson(chronic_conditions_prob * 3, n_patients),
        'previous_office_visits': np.random.poisson(5 + chronic_conditions_prob * 10, n_patients),
        'medication_count': np.random.poisson(n_conditions * 1.5, n_patients),
        'vaccination_status': np.random.choice(['Complete', 'Partial', 'None'], 
                                               n_patients, p=[0.6, 0.25, 0.15])
    }
    
    # Generate lab values
    clinical_data['blood_pressure_systolic'] = np.random.normal(130, 18, n_patients).clip(90, 200).astype(int)
    clinical_data['blood_pressure_diastolic'] = np.random.normal(80, 12, n_patients).clip(60, 120).astype(int)
    clinical_data['cholesterol_total'] = np.random.normal(200, 40, n_patients).clip(100, 400).astype(int)
    clinical_data['glucose_fasting'] = np.random.normal(100, 30, n_patients).clip(70, 300).astype(int)
    
    return pd.DataFrame(clinical_data)


def generate_financial_data(demographics_df, clinical_df):
    """Generate financial and cost data"""
    
    n_patients = len(demographics_df)
    
    # Base cost calculation with multiple factors
    age_factor = demographics_df['age'] / 100
    bmi_factor = (demographics_df['bmi'] - 25).clip(0, 20) / 20
    chronic_factor = clinical_df['chronic_conditions_count'] / 5
    
    # Base annual cost
    base_cost = 5000 + (age_factor * 8000) + (chronic_factor * 15000) + (bmi_factor * 5000)
    
    # Add smoker premium
    base_cost = base_cost + (demographics_df['smoker'] * 3000)
    
    # Add random variation
    base_cost = base_cost * np.random.uniform(0.7, 1.3, n_patients)
    
    # Previous year costs
    previous_year_cost = base_cost * np.random.uniform(0.6, 1.2, n_patients)
    
    # Calculate components
    inpatient_cost = base_cost * np.random.uniform(0.3, 0.5, n_patients) * (clinical_df['previous_hospitalizations'] > 0)
    outpatient_cost = base_cost * np.random.uniform(0.2, 0.4, n_patients)
    pharmacy_cost = clinical_df['medication_count'] * 150 * np.random.uniform(0.8, 1.5, n_patients)
    emergency_cost = clinical_df['previous_er_visits'] * 1500 * np.random.uniform(0.8, 1.2, n_patients)
    
    total_cost = inpatient_cost + outpatient_cost + pharmacy_cost + emergency_cost
    
    # Insurance coverage
    insurance_coverage_rate = {
        'Medicare': 0.80,
        'Medicaid': 0.85,
        'Private': 0.75,
        'Self-Pay': 0.0,
        'HMO': 0.78,
        'PPO': 0.72
    }
    
    coverage_rates = demographics_df['insurance_type'].map(insurance_coverage_rate)
    insurance_paid = total_cost * coverage_rates
    patient_paid = total_cost - insurance_paid
    
    # Payment status
    payment_prob = np.where(patient_paid < 1000, 0.95, 
                   np.where(patient_paid < 5000, 0.75, 0.50))
    payment_status = np.random.choice(['Paid', 'Partial', 'Outstanding'], 
                                     n_patients, 
                                     p=[0.7, 0.2, 0.1])
    
    financial_data = {
        'patient_id': demographics_df['patient_id'],
        'total_annual_cost': total_cost.round(2),
        'inpatient_cost': inpatient_cost.round(2),
        'outpatient_cost': outpatient_cost.round(2),
        'pharmacy_cost': pharmacy_cost.round(2),
        'emergency_cost': emergency_cost.round(2),
        'insurance_paid': insurance_paid.round(2),
        'patient_responsibility': patient_paid.round(2),
        'previous_year_cost': previous_year_cost.round(2),
        'payment_status': payment_status,
        'outstanding_balance': (patient_paid * (payment_status == 'Outstanding')).round(2)
    }
    
    return pd.DataFrame(financial_data)


def calculate_risk_category(financial_df):
    """Calculate risk categories based on predicted costs"""
    
    def assign_risk(cost):
        if cost < 5000:
            return 'Low'
        elif cost < 25000:
            return 'Medium'
        elif cost < 100000:
            return 'High'
        else:
            return 'Catastrophic'
    
    return financial_df['total_annual_cost'].apply(assign_risk)


def generate_time_series_data(patient_ids, n_months=12):
    """Generate monthly utilization data"""
    
    time_series_data = []
    
    for patient_id in patient_ids[:1000]:  # Generate for subset to keep file size manageable
        start_date = datetime.now() - timedelta(days=365)
        
        for month in range(n_months):
            date = start_date + timedelta(days=30 * month)
            
            record = {
                'patient_id': patient_id,
                'month': date.strftime('%Y-%m'),
                'office_visits': np.random.poisson(0.5),
                'er_visits': np.random.poisson(0.1),
                'hospitalizations': np.random.poisson(0.05),
                'pharmacy_fills': np.random.poisson(2),
                'monthly_cost': np.random.uniform(200, 2000)
            }
            time_series_data.append(record)
    
    return pd.DataFrame(time_series_data)


@click.command()
@click.option('--n_patients', default=50000, help='Number of patients to generate')
@click.option('--output_dir', default='data/raw', help='Output directory for data files')
def main(n_patients, output_dir):
    """Generate synthetic healthcare and financial data"""
    
    click.echo(f"Generating data for {n_patients} patients...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    click.echo("Generating demographics...")
    demographics_df = generate_patient_demographics(n_patients)
    
    click.echo("Generating clinical data...")
    clinical_df = generate_clinical_data(demographics_df)
    
    click.echo("Generating financial data...")
    financial_df = generate_financial_data(demographics_df, clinical_df)
    
    # Merge all data
    click.echo("Merging datasets...")
    full_data = demographics_df.merge(clinical_df, on='patient_id')
    full_data = full_data.merge(financial_df, on='patient_id')
    
    # Add risk category
    full_data['risk_category'] = calculate_risk_category(financial_df)
    
    # Add timestamp
    full_data['data_date'] = datetime.now().strftime('%Y-%m-%d')
    
    # Save main dataset
    output_file = output_path / 'patient_data.csv'
    full_data.to_csv(output_file, index=False)
    click.echo(f"Saved main dataset to {output_file}")
    
    # Generate and save time series data
    click.echo("Generating time series data...")
    time_series_df = generate_time_series_data(demographics_df['patient_id'])
    ts_output_file = output_path / 'patient_time_series.csv'
    time_series_df.to_csv(ts_output_file, index=False)
    click.echo(f"Saved time series data to {ts_output_file}")
    
    # Generate data summary
    click.echo("\n" + "="*60)
    click.echo("DATA GENERATION SUMMARY")
    click.echo("="*60)
    click.echo(f"Total patients: {len(full_data)}")
    click.echo(f"Date range: {full_data['data_date'].min()} to {full_data['data_date'].max()}")
    click.echo(f"\nRisk Category Distribution:")
    click.echo(full_data['risk_category'].value_counts())
    click.echo(f"\nCost Statistics:")
    click.echo(full_data['total_annual_cost'].describe())
    click.echo(f"\nAverage costs by risk category:")
    click.echo(full_data.groupby('risk_category')['total_annual_cost'].mean())
    click.echo("="*60)
    
    click.echo(f"\n✅ Data generation complete!")


if __name__ == '__main__':
    main()
