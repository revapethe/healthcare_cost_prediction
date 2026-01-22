"""
Simplified Model Training - Using only scikit-learn
Trains multiple models and evaluates performance
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
from pathlib import Path
import json
from datetime import datetime

print("="*60)
print("HEALTHCARE COST PREDICTION - MODEL TRAINING")
print("="*60)

# Load data
print("\nLoading processed data...")
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Initialize models
models = {
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
    )
}

print(f"\nInitialized {len(models)} models")

# Train and evaluate models
results = {}

print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

for name, model in models.items():
    print(f"\nTraining {name}...")
    start_time = datetime.now()
    
    # Train
    model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Store results
    results[name] = {
        'model': model,
        'metrics': {
            'RMSE': round(rmse, 2),
            'MAE': round(mae, 2),
            'R2': round(r2, 4),
            'MAPE': round(mape, 2)
        },
        'training_time': round(training_time, 2),
        'predictions': y_pred
    }
    
    print(f"{name} - RMSE: ${rmse:.2f}, MAE: ${mae:.2f}, R²: {r2:.4f}, Time: {training_time:.2f}s")
    print("-"*60)

# Display comparison
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

comparison_df = pd.DataFrame({
    name: {
        'RMSE': results[name]['metrics']['RMSE'],
        'MAE': results[name]['metrics']['MAE'],
        'R²': results[name]['metrics']['R2'],
        'MAPE': results[name]['metrics']['MAPE'],
        'Training Time (s)': results[name]['training_time']
    }
    for name in results.keys()
}).T

comparison_df = comparison_df.sort_values('RMSE')
print("\n", comparison_df.to_string())

# Save best model
best_model_name = comparison_df.index[0]
best_model = results[best_model_name]['model']

print(f"\n\nBest Model: {best_model_name}")
print(f"Best RMSE: ${results[best_model_name]['metrics']['RMSE']:.2f}")

# Create models directory
model_dir = Path('models/saved')
model_dir.mkdir(parents=True, exist_ok=True)

# Save best model
model_path = model_dir / 'best_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"\nBest model saved to {model_path}")

# Save specific model
specific_path = model_dir / f'{best_model_name.replace(" ", "_").lower()}_model.pkl'
with open(specific_path, 'wb') as f:
    pickle.dump(best_model, f)

# Save metrics
metrics_path = model_dir / f'{best_model_name.replace(" ", "_").lower()}_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump({
        'model_name': best_model_name,
        'metrics': results[best_model_name]['metrics'],
        'training_time': results[best_model_name]['training_time'],
        'trained_date': datetime.now().isoformat()
    }, f, indent=4)
print(f"Metrics saved to {metrics_path}")

# Save comparison
comparison_path = model_dir / 'model_comparison.csv'
comparison_df.to_csv(comparison_path)
print(f"Comparison saved to {comparison_path}")

# Create ensemble
print("\n" + "="*60)
print("CREATING ENSEMBLE MODEL")
print("="*60)

# Get top 3 models
top_models = comparison_df.head(3)
print("\nTop 3 models for ensemble:")
for name in top_models.index:
    print(f"  - {name}")

# Calculate ensemble predictions (simple average)
ensemble_predictions = []
weights = []

for name in top_models.index:
    pred = results[name]['predictions']
    ensemble_predictions.append(pred)
    weight = 1 / results[name]['metrics']['RMSE']
    weights.append(weight)

# Normalize weights
weights = np.array(weights) / sum(weights)

# Weighted average
ensemble_pred = np.average(ensemble_predictions, axis=0, weights=weights)

# Evaluate ensemble
rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
mae = mean_absolute_error(y_test, ensemble_pred)
r2 = r2_score(y_test, ensemble_pred)
mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100

print(f"\nEnsemble Performance:")
print(f"  RMSE: ${rmse:.2f}")
print(f"  MAE: ${mae:.2f}")
print(f"  R²: {r2:.4f}")
print(f"  MAPE: {mape:.2f}%")

# Save ensemble info
ensemble_info = {
    'models': [name for name in top_models.index],
    'weights': weights.tolist(),
    'metrics': {
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2),
        'R2': round(r2, 4),
        'MAPE': round(mape, 2)
    }
}

with open(model_dir / 'ensemble_info.json', 'w') as f:
    json.dump(ensemble_info, f, indent=4)

print("\n" + "="*60)
print("✅ MODEL TRAINING COMPLETE!")
print("="*60)
print(f"\nAll models saved to: {model_dir}")
