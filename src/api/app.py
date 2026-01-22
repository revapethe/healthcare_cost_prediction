"""
FastAPI Application for Healthcare Cost Prediction
RESTful API for making predictions
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Healthcare Cost Prediction API",
    description="API for predicting patient healthcare costs and risk stratification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessor
model = None
preprocessor = None


# Pydantic models for request/response
class PatientInput(BaseModel):
    """Input schema for patient data"""
    age: int = Field(..., ge=18, le=100, description="Patient age")
    gender: str = Field(..., description="Gender (M/F)")
    bmi: float = Field(..., ge=15, le=60, description="Body Mass Index")
    smoker: bool = Field(default=False, description="Smoking status")
    chronic_conditions_count: int = Field(default=0, ge=0, description="Number of chronic conditions")
    previous_office_visits: int = Field(default=0, ge=0, description="Previous office visits")
    previous_er_visits: int = Field(default=0, ge=0, description="Previous ER visits")
    previous_hospitalizations: int = Field(default=0, ge=0, description="Previous hospitalizations")
    medication_count: int = Field(default=0, ge=0, description="Number of medications")
    blood_pressure_systolic: Optional[int] = Field(120, ge=80, le=200)
    blood_pressure_diastolic: Optional[int] = Field(80, ge=50, le=120)
    cholesterol_total: Optional[int] = Field(200, ge=100, le=400)
    glucose_fasting: Optional[int] = Field(100, ge=70, le=300)
    previous_year_cost: Optional[float] = Field(5000, ge=0)
    insurance_type: Optional[str] = Field("Private", description="Insurance type")
    state: Optional[str] = Field("CA", description="State code")
    
    class Config:
        schema_extra = {
            "example": {
                "age": 55,
                "gender": "M",
                "bmi": 28.5,
                "smoker": False,
                "chronic_conditions_count": 2,
                "previous_office_visits": 5,
                "previous_er_visits": 1,
                "previous_hospitalizations": 0,
                "medication_count": 3,
                "blood_pressure_systolic": 130,
                "blood_pressure_diastolic": 85,
                "cholesterol_total": 210,
                "glucose_fasting": 105,
                "previous_year_cost": 8500,
                "insurance_type": "Private",
                "state": "CA"
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    predicted_cost: float
    risk_category: str
    monthly_cost: float
    confidence_interval: Dict[str, float]
    cost_breakdown: Dict[str, float]
    recommendations: List[str]


class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions"""
    patients: List[PatientInput]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str


def load_model_and_preprocessor():
    """Load model and preprocessor at startup"""
    global model, preprocessor
    
    try:
        model_path = Path('models/saved/best_model.pkl')
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}")
            
        preprocessor_path = Path('data/processed/preprocessor.pkl')
        if preprocessor_path.exists():
            preprocessor = joblib.load(preprocessor_path)
            logger.info(f"Preprocessor loaded from {preprocessor_path}")
        else:
            logger.warning(f"Preprocessor not found at {preprocessor_path}")
            
    except Exception as e:
        logger.error(f"Error loading model/preprocessor: {e}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model_and_preprocessor()


def encode_patient_data(patient: PatientInput) -> pd.DataFrame:
    """Encode patient data for model input"""
    
    # Simple encoding mappings
    gender_map = {'M': 1, 'F': 0}
    insurance_map = {
        'Medicare': 0, 'Medicaid': 1, 'Private': 2, 
        'Self-Pay': 3, 'HMO': 4, 'PPO': 5
    }
    state_map = {
        'CA': 0, 'TX': 1, 'FL': 2, 'NY': 3, 'PA': 4,
        'IL': 5, 'OH': 6, 'GA': 7, 'NC': 8, 'MI': 9
    }
    
    data = {
        'age': patient.age,
        'gender': gender_map.get(patient.gender, 1),
        'bmi': patient.bmi,
        'smoker': int(patient.smoker),
        'chronic_conditions_count': patient.chronic_conditions_count,
        'previous_office_visits': patient.previous_office_visits,
        'previous_er_visits': patient.previous_er_visits,
        'previous_hospitalizations': patient.previous_hospitalizations,
        'medication_count': patient.medication_count,
        'blood_pressure_systolic': patient.blood_pressure_systolic,
        'blood_pressure_diastolic': patient.blood_pressure_diastolic,
        'cholesterol_total': patient.cholesterol_total,
        'glucose_fasting': patient.glucose_fasting,
        'previous_year_cost': patient.previous_year_cost,
        'insurance_type': insurance_map.get(patient.insurance_type, 2),
        'state': state_map.get(patient.state, 0),
        'vaccination_status': 0  # Default
    }
    
    return pd.DataFrame([data])


def categorize_risk(cost: float) -> str:
    """Categorize patient risk"""
    if cost < 5000:
        return 'Low'
    elif cost < 25000:
        return 'Medium'
    elif cost < 100000:
        return 'High'
    else:
        return 'Catastrophic'


def generate_recommendations(patient: PatientInput, predicted_cost: float) -> List[str]:
    """Generate personalized recommendations"""
    recommendations = []
    
    if patient.bmi > 30:
        recommendations.append("Consider weight management program to reduce obesity-related risks")
    
    if patient.smoker:
        recommendations.append("Smoking cessation program recommended - can reduce costs by up to 30%")
    
    if patient.chronic_conditions_count > 2:
        recommendations.append("Enroll in chronic disease management program")
    
    if patient.previous_er_visits > 2:
        recommendations.append("Preventive care visits may help reduce emergency room usage")
    
    if patient.blood_pressure_systolic and patient.blood_pressure_systolic > 140:
        recommendations.append("Blood pressure management needed - consult with physician")
    
    if predicted_cost > 25000:
        recommendations.append("Consider meeting with financial counselor for payment plan options")
        recommendations.append("Explore patient assistance programs for medication costs")
    
    if not recommendations:
        recommendations.append("Continue current preventive care routine")
    
    return recommendations


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_cost(patient: PatientInput):
    """
    Predict healthcare costs for a single patient
    
    Returns predicted annual cost, risk category, and recommendations
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Encode patient data
        patient_df = encode_patient_data(patient)
        
        # Make prediction
        predicted_cost = float(model.predict(patient_df)[0])
        
        # Calculate additional metrics
        risk_category = categorize_risk(predicted_cost)
        monthly_cost = predicted_cost / 12
        
        # Confidence interval (simplified)
        std_error = predicted_cost * 0.15
        margin = 1.96 * std_error
        confidence_interval = {
            'lower': max(0, predicted_cost - margin),
            'upper': predicted_cost + margin,
            'confidence': 0.95
        }
        
        # Cost breakdown estimate
        cost_breakdown = {
            'inpatient': predicted_cost * 0.35,
            'outpatient': predicted_cost * 0.25,
            'pharmacy': predicted_cost * 0.20,
            'emergency': predicted_cost * 0.15,
            'other': predicted_cost * 0.05
        }
        
        # Generate recommendations
        recommendations = generate_recommendations(patient, predicted_cost)
        
        return {
            'predicted_cost': round(predicted_cost, 2),
            'risk_category': risk_category,
            'monthly_cost': round(monthly_cost, 2),
            'confidence_interval': {k: round(v, 2) for k, v in confidence_interval.items()},
            'cost_breakdown': {k: round(v, 2) for k, v in cost_breakdown.items()},
            'recommendations': recommendations
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/predict/batch")
async def predict_batch(batch_input: BatchPredictionInput):
    """
    Predict healthcare costs for multiple patients
    
    Returns predictions for all patients in the batch
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        
        for patient in batch_input.patients:
            # Make individual prediction
            result = await predict_cost(patient)
            predictions.append(result)
        
        return {
            'count': len(predictions),
            'predictions': predictions
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/risk-categories")
async def get_risk_categories():
    """Get information about risk categories"""
    return {
        'categories': [
            {
                'name': 'Low',
                'range': '$0 - $5,000',
                'description': 'Minimal healthcare utilization',
                'percentage': '~60%'
            },
            {
                'name': 'Medium',
                'range': '$5,000 - $25,000',
                'description': 'Moderate healthcare needs',
                'percentage': '~25%'
            },
            {
                'name': 'High',
                'range': '$25,000 - $100,000',
                'description': 'Significant healthcare needs',
                'percentage': '~13%'
            },
            {
                'name': 'Catastrophic',
                'range': '$100,000+',
                'description': 'Complex medical conditions',
                'percentage': '~2%'
            }
        ]
    }


@app.get("/api/v1/stats")
async def get_model_stats():
    """Get model performance statistics"""
    try:
        # Load model metrics if available
        metrics_path = Path('models/saved/xgboost_metrics.json')
        if metrics_path.exists():
            import json
            with open(metrics_path) as f:
                metrics = json.load(f)
            return metrics
        else:
            return {
                'message': 'Model metrics not available',
                'model_loaded': model is not None
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
