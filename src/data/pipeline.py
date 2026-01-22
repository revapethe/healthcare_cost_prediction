"""
Data Preprocessing Pipeline
Handles data cleaning, validation, and transformation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handle all data preprocessing tasks"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.scalers = {}
        self.encoders = {}
        
    def load_data(self, filepath):
        """Load data from CSV file"""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        logger.info("Handling missing values...")
        
        # Numeric columns - fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Categorical columns - fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        logger.info(f"Missing values handled. Remaining nulls: {df.isnull().sum().sum()}")
        return df
    
    def remove_outliers(self, df, columns, method='iqr', threshold=3):
        """Remove outliers using IQR or Z-score method"""
        logger.info(f"Removing outliers using {method} method...")
        
        original_len = len(df)
        
        if method == 'iqr':
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        elif method == 'zscore':
            for col in columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < threshold]
        
        removed = original_len - len(df)
        logger.info(f"Removed {removed} outliers ({removed/original_len*100:.2f}%)")
        
        return df
    
    def encode_categorical(self, df, categorical_columns):
        """Encode categorical variables"""
        logger.info("Encoding categorical variables...")
        
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df_encoded[col] = self.encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    df_encoded[col] = self.encoders[col].transform(df_encoded[col].astype(str))
        
        logger.info(f"Encoded {len(categorical_columns)} categorical columns")
        return df_encoded
    
    def create_derived_features(self, df):
        """Create derived features for better predictions"""
        logger.info("Creating derived features...")
        
        df_features = df.copy()
        
        # Age groups
        df_features['age_group'] = pd.cut(df['age'], 
                                          bins=[0, 30, 50, 65, 100],
                                          labels=['Young', 'Middle', 'Senior', 'Elderly'])
        
        # BMI categories
        df_features['bmi_category'] = pd.cut(df['bmi'],
                                            bins=[0, 18.5, 25, 30, 100],
                                            labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        
        # Cost per visit
        total_visits = (df_features['previous_office_visits'] + 
                       df_features['previous_er_visits'] + 
                       df_features['previous_hospitalizations'])
        df_features['cost_per_visit'] = np.where(
            total_visits > 0,
            df_features['previous_year_cost'] / total_visits,
            0
        )
        
        # High utilizer flag
        df_features['high_utilizer'] = (
            (df_features['previous_office_visits'] > df_features['previous_office_visits'].quantile(0.75)) |
            (df_features['previous_er_visits'] > 2) |
            (df_features['previous_hospitalizations'] > 1)
        ).astype(int)
        
        # Comorbidity score (simplified)
        df_features['comorbidity_score'] = (
            df_features['chronic_conditions_count'] * 2 +
            df_features['medication_count'] * 0.5 +
            df_features['smoker'] * 3
        )
        
        # Insurance adequacy score
        insurance_coverage_map = {
            'Medicare': 0.80,
            'Medicaid': 0.85,
            'Private': 0.75,
            'Self-Pay': 0.0,
            'HMO': 0.78,
            'PPO': 0.72
        }
        df_features['insurance_coverage_score'] = df_features['insurance_type'].map(insurance_coverage_map)
        
        # Financial risk score
        df_features['financial_risk_score'] = (
            (df_features['patient_responsibility'] / df_features['total_annual_cost']).fillna(0) * 100
        )
        
        # Healthcare intensity
        df_features['healthcare_intensity'] = (
            df_features['previous_office_visits'] * 1 +
            df_features['previous_er_visits'] * 5 +
            df_features['previous_hospitalizations'] * 10
        )
        
        logger.info(f"Created {len(df_features.columns) - len(df.columns)} new features")
        
        return df_features
    
    def scale_features(self, df, numeric_columns, fit=True):
        """Scale numeric features"""
        logger.info("Scaling numeric features...")
        
        df_scaled = df.copy()
        
        if fit:
            self.scalers['standard'] = StandardScaler()
            df_scaled[numeric_columns] = self.scalers['standard'].fit_transform(df[numeric_columns])
        else:
            df_scaled[numeric_columns] = self.scalers['standard'].transform(df[numeric_columns])
        
        return df_scaled
    
    def prepare_model_data(self, df, target_column, test_size=0.2, random_state=42):
        """Prepare data for model training"""
        logger.info("Preparing data for modeling...")
        
        # Separate features and target
        X = df.drop(columns=[target_column, 'patient_id', 'data_date'], errors='ignore')
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self, filepath):
        """Save preprocessor objects"""
        logger.info(f"Saving preprocessor to {filepath}")
        joblib.dump({
            'scalers': self.scalers,
            'encoders': self.encoders
        }, filepath)
    
    def load_preprocessor(self, filepath):
        """Load preprocessor objects"""
        logger.info(f"Loading preprocessor from {filepath}")
        objects = joblib.load(filepath)
        self.scalers = objects['scalers']
        self.encoders = objects['encoders']


def run_pipeline():
    """Run the complete preprocessing pipeline"""
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    df = preprocessor.load_data('data/raw/patient_data.csv')
    
    # Handle missing values
    df = preprocessor.handle_missing_values(df)
    
    # Create derived features
    df = preprocessor.create_derived_features(df)
    
    # Encode categorical variables
    categorical_cols = ['gender', 'state', 'insurance_type', 'vaccination_status', 
                       'payment_status', 'age_group', 'bmi_category']
    df = preprocessor.encode_categorical(df, categorical_cols)
    
    # Remove outliers from cost columns
    cost_columns = ['total_annual_cost', 'previous_year_cost']
    df = preprocessor.remove_outliers(df, cost_columns, method='iqr', threshold=3)
    
    # Save processed data
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / 'processed_data.csv', index=False)
    logger.info(f"Saved processed data to {output_dir / 'processed_data.csv'}")
    
    # Prepare data for modeling
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [col for col in numeric_features if col not in ['patient_id', 'total_annual_cost']]
    
    # Split for cost prediction
    X_train, X_test, y_train, y_test = preprocessor.prepare_model_data(
        df, 'total_annual_cost', test_size=0.2
    )
    
    # Scale features
    X_train_scaled = preprocessor.scale_features(X_train[numeric_features], numeric_features, fit=True)
    X_test_scaled = preprocessor.scale_features(X_test[numeric_features], numeric_features, fit=False)
    
    # Save train/test splits
    X_train_scaled.to_csv(output_dir / 'X_train.csv', index=False)
    X_test_scaled.to_csv(output_dir / 'X_test.csv', index=False)
    y_train.to_csv(output_dir / 'y_train.csv', index=False, header=['total_annual_cost'])
    y_test.to_csv(output_dir / 'y_test.csv', index=False, header=['total_annual_cost'])
    
    # Save preprocessor
    preprocessor.save_preprocessor(output_dir / 'preprocessor.pkl')
    
    logger.info("✅ Pipeline complete!")
    logger.info(f"Final dataset shape: {df.shape}")
    logger.info(f"Features: {len(X_train.columns)}")
    
    return df, X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == '__main__':
    run_pipeline()
