"""
Model Training Script
Trains multiple models and selects the best performer
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
from pathlib import Path
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and evaluate multiple models"""
    
    def __init__(self, model_dir='models/saved'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.results = {}
        
    def initialize_models(self):
        """Initialize all models to be trained"""
        logger.info("Initializing models...")
        
        self.models = {
            'Linear Regression': LinearRegression(),
            
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            
            'XGBoost': XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            
            'LightGBM': LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=7,
                num_leaves=31,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
        }
        
        logger.info(f"Initialized {len(self.models)} models")
    
    def train_model(self, name, model, X_train, y_train):
        """Train a single model"""
        logger.info(f"Training {name}...")
        
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"{name} trained in {training_time:.2f} seconds")
        
        return model, training_time
    
    def evaluate_model(self, name, model, X_test, y_test):
        """Evaluate model performance"""
        logger.info(f"Evaluating {name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Calculate percentage within threshold
        threshold_10 = np.mean(np.abs((y_test - y_pred) / y_test) < 0.10) * 100
        threshold_20 = np.mean(np.abs((y_test - y_pred) / y_test) < 0.20) * 100
        
        metrics = {
            'RMSE': round(rmse, 2),
            'MAE': round(mae, 2),
            'R2': round(r2, 4),
            'MAPE': round(mape, 2),
            'Within_10%': round(threshold_10, 2),
            'Within_20%': round(threshold_20, 2)
        }
        
        logger.info(f"{name} - RMSE: ${rmse:.2f}, MAE: ${mae:.2f}, R²: {r2:.4f}")
        
        return metrics, y_pred
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train and evaluate all models"""
        logger.info("="*60)
        logger.info("TRAINING ALL MODELS")
        logger.info("="*60)
        
        self.initialize_models()
        
        for name, model in self.models.items():
            # Train
            trained_model, training_time = self.train_model(name, model, X_train, y_train)
            
            # Evaluate
            metrics, predictions = self.evaluate_model(name, trained_model, X_test, y_test)
            
            # Store results
            self.results[name] = {
                'model': trained_model,
                'metrics': metrics,
                'training_time': training_time,
                'predictions': predictions
            }
            
            logger.info("-"*60)
        
        # Display comparison
        self.display_results()
        
        # Save best model
        best_model_name = self.get_best_model()
        self.save_model(best_model_name)
        
        return self.results
    
    def display_results(self):
        """Display comparison of all models"""
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON")
        logger.info("="*60)
        
        comparison_df = pd.DataFrame({
            name: {
                'RMSE': results['metrics']['RMSE'],
                'MAE': results['metrics']['MAE'],
                'R²': results['metrics']['R2'],
                'MAPE': results['metrics']['MAPE'],
                'Training Time (s)': results['training_time']
            }
            for name, results in self.results.items()
        }).T
        
        comparison_df = comparison_df.sort_values('RMSE')
        print("\n", comparison_df.to_string())
        
        # Save comparison
        comparison_df.to_csv(self.model_dir / 'model_comparison.csv')
        logger.info(f"\nComparison saved to {self.model_dir / 'model_comparison.csv'}")
    
    def get_best_model(self):
        """Get the name of the best performing model"""
        best_model = min(self.results.items(), 
                        key=lambda x: x[1]['metrics']['RMSE'])
        return best_model[0]
    
    def save_model(self, model_name):
        """Save the specified model"""
        logger.info(f"\nSaving best model: {model_name}")
        
        model_info = self.results[model_name]
        
        # Save model
        model_path = self.model_dir / f'{model_name.replace(" ", "_").lower()}_model.pkl'
        joblib.dump(model_info['model'], model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metrics
        metrics_path = self.model_dir / f'{model_name.replace(" ", "_").lower()}_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({
                'model_name': model_name,
                'metrics': model_info['metrics'],
                'training_time': model_info['training_time'],
                'trained_date': datetime.now().isoformat()
            }, f, indent=4)
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Save best model as default
        default_path = self.model_dir / 'best_model.pkl'
        joblib.dump(model_info['model'], default_path)
        logger.info(f"Best model also saved to {default_path}")
    
    def create_ensemble(self, X_train, y_train, X_test, y_test):
        """Create an ensemble of top models"""
        logger.info("\n" + "="*60)
        logger.info("CREATING ENSEMBLE MODEL")
        logger.info("="*60)
        
        # Get top 3 models by RMSE
        sorted_models = sorted(self.results.items(), 
                              key=lambda x: x[1]['metrics']['RMSE'])[:3]
        
        logger.info("Top 3 models for ensemble:")
        for name, _ in sorted_models:
            logger.info(f"  - {name}")
        
        # Make predictions with each model
        predictions = []
        weights = []
        
        for name, results in sorted_models:
            model = results['model']
            pred = model.predict(X_test)
            predictions.append(pred)
            
            # Weight by inverse RMSE (better models get higher weight)
            weight = 1 / results['metrics']['RMSE']
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Calculate weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        # Evaluate ensemble
        rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        mae = mean_absolute_error(y_test, ensemble_pred)
        r2 = r2_score(y_test, ensemble_pred)
        mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
        
        logger.info(f"\nEnsemble Performance:")
        logger.info(f"  RMSE: ${rmse:.2f}")
        logger.info(f"  MAE: ${mae:.2f}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        
        # Save ensemble info
        ensemble_info = {
            'models': [name for name, _ in sorted_models],
            'weights': weights.tolist(),
            'metrics': {
                'RMSE': round(rmse, 2),
                'MAE': round(mae, 2),
                'R2': round(r2, 4),
                'MAPE': round(mape, 2)
            }
        }
        
        with open(self.model_dir / 'ensemble_info.json', 'w') as f:
            json.dump(ensemble_info, f, indent=4)
        
        logger.info(f"Ensemble info saved to {self.model_dir / 'ensemble_info.json'}")
        
        return ensemble_pred, ensemble_info


def main():
    """Main training function"""
    
    logger.info("="*60)
    logger.info("HEALTHCARE COST PREDICTION - MODEL TRAINING")
    logger.info("="*60)
    
    # Load processed data
    logger.info("\nLoading processed data...")
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train all models
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Create ensemble
    ensemble_pred, ensemble_info = trainer.create_ensemble(X_train, y_train, X_test, y_test)
    
    logger.info("\n" + "="*60)
    logger.info("✅ MODEL TRAINING COMPLETE!")
    logger.info("="*60)
    
    best_model = trainer.get_best_model()
    logger.info(f"\nBest Model: {best_model}")
    logger.info(f"Best RMSE: ${trainer.results[best_model]['metrics']['RMSE']:.2f}")
    logger.info(f"\nAll models saved to: {trainer.model_dir}")


if __name__ == '__main__':
    main()
