"""
Prediction Module
Load trained models and make predictions
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CostPredictor:
    """Make cost predictions using trained models"""
    
    def __init__(self, model_path='models/saved/best_model.pkl',
                 preprocessor_path='data/processed/preprocessor.pkl'):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.model = None
        self.preprocessor = None
        
        self.load_model()
        self.load_preprocessor()
    
    def load_model(self):
        """Load trained model"""
        if self.model_path.exists():
            logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")
    
    def load_preprocessor(self):
        """Load preprocessor"""
        if self.preprocessor_path.exists():
            logger.info(f"Loading preprocessor from {self.preprocessor_path}")
            self.preprocessor = joblib.load(self.preprocessor_path)
        else:
            logger.warning(f"Preprocessor not found at {self.preprocessor_path}")
    
    def preprocess_input(self, data):
        """Preprocess input data for prediction"""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Apply scaling if preprocessor is available
        if self.preprocessor and 'scalers' in self.preprocessor:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if 'standard' in self.preprocessor['scalers']:
                df[numeric_cols] = self.preprocessor['scalers']['standard'].transform(df[numeric_cols])
        
        return df
    
    def predict(self, data, return_details=True):
        """Make prediction for input data"""
        # Preprocess
        processed_data = self.preprocess_input(data)
        
        # Predict
        prediction = self.model.predict(processed_data)[0]
        
        if return_details:
            risk_category = self.categorize_risk(prediction)
            confidence_interval = self.calculate_confidence_interval(prediction)
            
            return {
                'predicted_cost': round(prediction, 2),
                'risk_category': risk_category,
                'confidence_interval': confidence_interval,
                'monthly_cost': round(prediction / 12, 2),
                'cost_breakdown_estimate': self.estimate_cost_breakdown(prediction)
            }
        
        return prediction
    
    def categorize_risk(self, cost):
        """Categorize patient into risk category"""
        if cost < 5000:
            return 'Low'
        elif cost < 25000:
            return 'Medium'
        elif cost < 100000:
            return 'High'
        else:
            return 'Catastrophic'
    
    def calculate_confidence_interval(self, prediction, confidence=0.95):
        """Calculate confidence interval for prediction"""
        # Simplified CI calculation (in production, use model-specific methods)
        std_error = prediction * 0.15  # Assuming 15% standard error
        margin = 1.96 * std_error  # 95% CI
        
        return {
            'lower': round(max(0, prediction - margin), 2),
            'upper': round(prediction + margin, 2),
            'confidence': confidence
        }
    
    def estimate_cost_breakdown(self, total_cost):
        """Estimate breakdown of costs"""
        # Simplified breakdown (adjust based on your data analysis)
        return {
            'inpatient': round(total_cost * 0.35, 2),
            'outpatient': round(total_cost * 0.25, 2),
            'pharmacy': round(total_cost * 0.20, 2),
            'emergency': round(total_cost * 0.15, 2),
            'other': round(total_cost * 0.05, 2)
        }
    
    def predict_batch(self, data_path, output_path=None):
        """Make predictions for batch of patients"""
        logger.info(f"Loading batch data from {data_path}")
        df = pd.read_csv(data_path)
        
        logger.info(f"Making predictions for {len(df)} patients...")
        
        # Make predictions
        predictions = []
        for idx, row in df.iterrows():
            pred = self.predict(row, return_details=True)
            pred['patient_id'] = row.get('patient_id', f'P{idx}')
            predictions.append(pred)
        
        results_df = pd.DataFrame(predictions)
        
        if output_path:
            results_df.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")
        
        return results_df


class RiskStratifier:
    """Stratify patients by risk level"""
    
    def __init__(self, predictor):
        self.predictor = predictor
    
    def stratify_population(self, patient_data):
        """Stratify entire patient population"""
        logger.info(f"Stratifying {len(patient_data)} patients...")
        
        predictions = self.predictor.predict_batch(patient_data)
        
        # Group by risk category
        risk_distribution = predictions['risk_category'].value_counts()
        
        # Calculate statistics by risk group
        risk_stats = predictions.groupby('risk_category').agg({
            'predicted_cost': ['mean', 'median', 'min', 'max', 'count']
        }).round(2)
        
        return {
            'distribution': risk_distribution.to_dict(),
            'statistics': risk_stats,
            'predictions': predictions
        }
    
    def identify_high_risk_patients(self, patient_data, threshold=25000):
        """Identify patients above cost threshold"""
        predictions = self.predictor.predict_batch(patient_data)
        high_risk = predictions[predictions['predicted_cost'] > threshold]
        
        logger.info(f"Identified {len(high_risk)} high-risk patients (>{threshold})")
        
        return high_risk.sort_values('predicted_cost', ascending=False)


def demo_prediction():
    """Demonstrate prediction functionality"""
    
    print("\n" + "="*60)
    print("HEALTHCARE COST PREDICTION - DEMO")
    print("="*60)
    
    # Initialize predictor
    predictor = CostPredictor()
    
    # Example patient data
    patient_data = {
        'age': 55,
        'gender': 1,  # Encoded: 1 = Male
        'bmi': 32.5,
        'smoker': 1,
        'chronic_conditions_count': 3,
        'previous_office_visits': 8,
        'previous_er_visits': 2,
        'previous_hospitalizations': 1,
        'medication_count': 5,
        'blood_pressure_systolic': 145,
        'blood_pressure_diastolic': 92,
        'cholesterol_total': 240,
        'glucose_fasting': 135,
        'previous_year_cost': 18500,
        'insurance_type': 2,  # Encoded
        'state': 5,  # Encoded
        'vaccination_status': 1  # Encoded
    }
    
    print("\nPatient Profile:")
    print(f"  Age: {patient_data['age']}")
    print(f"  BMI: {patient_data['bmi']}")
    print(f"  Chronic Conditions: {patient_data['chronic_conditions_count']}")
    print(f"  Previous Year Cost: ${patient_data['previous_year_cost']:,.2f}")
    
    # Make prediction
    prediction = predictor.predict(patient_data)
    
    print("\n" + "-"*60)
    print("PREDICTION RESULTS")
    print("-"*60)
    print(f"Predicted Annual Cost: ${prediction['predicted_cost']:,.2f}")
    print(f"Risk Category: {prediction['risk_category']}")
    print(f"Monthly Cost: ${prediction['monthly_cost']:,.2f}")
    print(f"\nConfidence Interval (95%):")
    print(f"  Lower: ${prediction['confidence_interval']['lower']:,.2f}")
    print(f"  Upper: ${prediction['confidence_interval']['upper']:,.2f}")
    
    print(f"\nCost Breakdown Estimate:")
    for component, cost in prediction['cost_breakdown_estimate'].items():
        print(f"  {component.capitalize()}: ${cost:,.2f}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    demo_prediction()
