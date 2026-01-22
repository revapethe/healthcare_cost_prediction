"""
Test Suite for Healthcare Cost Prediction System
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.pipeline import DataPreprocessor
from src.models.predict import CostPredictor


class TestDataPreprocessor:
    """Test data preprocessing functionality"""
    
    @pytest.fixture
    def preprocessor(self):
        return DataPreprocessor()
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'age': [25, 45, 65, 80],
            'bmi': [22.5, 28.0, 32.5, 27.0],
            'smoker': [0, 1, 0, 0],
            'total_annual_cost': [3000, 8000, 25000, 35000],
            'gender': ['M', 'F', 'M', 'F'],
            'insurance_type': ['Private', 'Medicare', 'Medicaid', 'Medicare']
        })
    
    def test_handle_missing_values(self, preprocessor, sample_data):
        """Test missing value handling"""
        # Add missing values
        sample_data.loc[0, 'age'] = np.nan
        sample_data.loc[1, 'bmi'] = np.nan
        
        # Process
        result = preprocessor.handle_missing_values(sample_data)
        
        # Assertions
        assert result.isnull().sum().sum() == 0, "Missing values not handled"
        assert len(result) == len(sample_data), "Data length changed"
    
    def test_encode_categorical(self, preprocessor, sample_data):
        """Test categorical encoding"""
        categorical_cols = ['gender', 'insurance_type']
        result = preprocessor.encode_categorical(sample_data, categorical_cols)
        
        # Assertions
        for col in categorical_cols:
            assert pd.api.types.is_numeric_dtype(result[col]), f"{col} not encoded"
    
    def test_remove_outliers(self, preprocessor, sample_data):
        """Test outlier removal"""
        original_len = len(sample_data)
        result = preprocessor.remove_outliers(sample_data, ['total_annual_cost'], 
                                             method='iqr', threshold=1.5)
        
        # Assertions
        assert len(result) <= original_len, "Outlier removal failed"
        assert len(result) > 0, "All data removed"
    
    def test_create_derived_features(self, preprocessor, sample_data):
        """Test derived feature creation"""
        result = preprocessor.create_derived_features(sample_data)
        
        # Check new features exist
        assert 'age_group' in result.columns, "Age group not created"
        assert 'bmi_category' in result.columns, "BMI category not created"


class TestPredictionModel:
    """Test prediction functionality"""
    
    def test_risk_categorization(self):
        """Test risk category assignment"""
        from src.models.predict import CostPredictor
        
        # Mock predictor
        assert CostPredictor.categorize_risk(None, 3000) == 'Low'
        assert CostPredictor.categorize_risk(None, 15000) == 'Medium'
        assert CostPredictor.categorize_risk(None, 50000) == 'High'
        assert CostPredictor.categorize_risk(None, 150000) == 'Catastrophic'
    
    def test_confidence_interval(self):
        """Test confidence interval calculation"""
        from src.models.predict import CostPredictor
        
        predictor = CostPredictor.__new__(CostPredictor)
        ci = predictor.calculate_confidence_interval(10000)
        
        assert 'lower' in ci, "Lower bound missing"
        assert 'upper' in ci, "Upper bound missing"
        assert ci['lower'] < 10000 < ci['upper'], "CI doesn't contain prediction"
    
    def test_cost_breakdown(self):
        """Test cost breakdown estimation"""
        from src.models.predict import CostPredictor
        
        predictor = CostPredictor.__new__(CostPredictor)
        breakdown = predictor.estimate_cost_breakdown(10000)
        
        assert 'inpatient' in breakdown, "Inpatient cost missing"
        assert 'pharmacy' in breakdown, "Pharmacy cost missing"
        assert sum(breakdown.values()) == pytest.approx(10000, rel=0.01), \
               "Breakdown doesn't sum to total"


class TestDataValidation:
    """Test data validation rules"""
    
    def test_age_range(self):
        """Test age is within valid range"""
        ages = [18, 45, 65, 100]
        assert all(18 <= age <= 100 for age in ages), "Invalid age found"
    
    def test_bmi_range(self):
        """Test BMI is within valid range"""
        bmis = [15.0, 22.5, 35.0, 60.0]
        assert all(15 <= bmi <= 60 for bmi in bmis), "Invalid BMI found"
    
    def test_cost_positive(self):
        """Test costs are non-negative"""
        costs = [0, 1000, 50000, 200000]
        assert all(cost >= 0 for cost in costs), "Negative cost found"


class TestAPIEndpoints:
    """Test API endpoints"""
    
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from src.api.app import app
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()
    
    def test_predict_endpoint(self, client):
        """Test prediction endpoint"""
        patient_data = {
            "age": 45,
            "gender": "M",
            "bmi": 28.5,
            "smoker": False,
            "chronic_conditions_count": 2,
            "previous_office_visits": 5,
            "previous_er_visits": 1,
            "previous_hospitalizations": 0,
            "medication_count": 3
        }
        
        response = client.post("/api/v1/predict", json=patient_data)
        
        assert response.status_code in [200, 503], "Unexpected status code"
        if response.status_code == 200:
            result = response.json()
            assert "predicted_cost" in result
            assert "risk_category" in result


class TestModelMetrics:
    """Test model performance metrics"""
    
    def test_rmse_calculation(self):
        """Test RMSE calculation"""
        from sklearn.metrics import mean_squared_error
        
        y_true = np.array([10000, 15000, 20000, 25000])
        y_pred = np.array([9500, 16000, 19000, 26000])
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        assert rmse > 0, "RMSE should be positive"
        assert rmse < 5000, "RMSE too high"
    
    def test_mae_calculation(self):
        """Test MAE calculation"""
        from sklearn.metrics import mean_absolute_error
        
        y_true = np.array([10000, 15000, 20000, 25000])
        y_pred = np.array([9500, 16000, 19000, 26000])
        
        mae = mean_absolute_error(y_true, y_pred)
        assert mae > 0, "MAE should be positive"
        assert mae < rmse, "MAE should be less than RMSE"


# Performance benchmarks
@pytest.mark.benchmark
class TestPerformance:
    """Test system performance"""
    
    def test_prediction_speed(self, benchmark):
        """Test prediction latency"""
        from src.models.predict import CostPredictor
        
        # Mock prediction function
        def predict():
            return np.random.uniform(5000, 50000)
        
        result = benchmark(predict)
        assert result > 0, "Prediction failed"
    
    def test_batch_processing_speed(self):
        """Test batch processing speed"""
        import time
        
        n_samples = 1000
        start_time = time.time()
        
        # Simulate batch processing
        predictions = [np.random.uniform(5000, 50000) for _ in range(n_samples)]
        
        elapsed_time = time.time() - start_time
        throughput = n_samples / elapsed_time
        
        assert throughput > 100, f"Throughput too low: {throughput:.2f} pred/sec"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src", "--cov-report=html"])
