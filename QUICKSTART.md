# Quick Start Guide

Get up and running with Healthcare Cost Prediction in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- 4GB RAM minimum
- 2GB free disk space

## Installation

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/healthcare-cost-prediction.git
cd healthcare-cost-prediction

# Run automated setup (handles everything)
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate data
python scripts/generate_data.py --n_patients 10000

# 4. Preprocess data
python src/data/pipeline.py

# 5. Train models
python src/models/train.py
```

## Quick Test

### Test the API

```bash
# Terminal 1: Start API server
uvicorn src.api.app:app --reload --port 8000

# Terminal 2: Test prediction
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "gender": "M",
    "bmi": 28.5,
    "smoker": false,
    "chronic_conditions_count": 2,
    "previous_office_visits": 5,
    "previous_er_visits": 1,
    "previous_hospitalizations": 0,
    "medication_count": 3
  }'
```

### Launch Dashboard

```bash
streamlit run dashboards/main_dashboard.py
```

Then open: http://localhost:8501

## Docker Quick Start

```bash
# Build and run with Docker
docker-compose up -d

# Access services
# API: http://localhost:8000
# Dashboard: http://localhost:8501
# API Docs: http://localhost:8000/docs
```

## Project Structure

```
healthcare-cost-prediction/
├── data/                  # Data files
│   ├── raw/              # Generated synthetic data
│   └── processed/        # Preprocessed data
├── src/                  # Source code
│   ├── api/             # FastAPI application
│   ├── models/          # Model training & prediction
│   └── data/            # Data processing
├── dashboards/          # Streamlit dashboards
├── notebooks/           # Jupyter notebooks
├── tests/              # Test suite
└── models/             # Saved models
    └── saved/
```

## Common Commands

```bash
# Generate more data
python scripts/generate_data.py --n_patients 50000

# Retrain models
python src/models/train.py

# Run tests
pytest tests/ -v

# Check code coverage
pytest tests/ --cov=src --cov-report=html

# Format code
black src/

# Run linter
flake8 src/

# Start Jupyter
jupyter notebook notebooks/
```

## Example Usage

### Python SDK

```python
from src.models.predict import CostPredictor

# Initialize predictor
predictor = CostPredictor()

# Make prediction
patient = {
    'age': 55,
    'gender': 1,
    'bmi': 32.5,
    'smoker': 1,
    'chronic_conditions_count': 3,
    # ... other features
}

result = predictor.predict(patient)
print(f"Predicted Cost: ${result['predicted_cost']:,.2f}")
print(f"Risk Category: {result['risk_category']}")
```

### API Usage

```python
import requests

url = "http://localhost:8000/api/v1/predict"
data = {
    "age": 55,
    "gender": "M",
    "bmi": 32.5,
    # ... other fields
}

response = requests.post(url, json=data)
result = response.json()

print(f"Predicted Cost: ${result['predicted_cost']:,.2f}")
```

## Troubleshooting

### Issue: Module not found
```bash
# Solution: Activate virtual environment
source venv/bin/activate
```

### Issue: Data not found
```bash
# Solution: Generate data first
python scripts/generate_data.py
```

### Issue: Model not loaded (503 error)
```bash
# Solution: Train models first
python src/models/train.py
```

### Issue: Port already in use
```bash
# Solution: Change port or kill existing process
# Change port:
uvicorn src.api.app:app --port 8001

# Or kill process:
lsof -ti:8000 | xargs kill -9
```

## Next Steps

1. **Explore the Dashboard**: Check out all the interactive visualizations
2. **Read the Docs**: Full documentation in `docs/DOCUMENTATION.md`
3. **Try the Notebooks**: Jupyter notebooks in `notebooks/`
4. **Customize Models**: Modify hyperparameters in `src/models/train.py`
5. **Add Features**: Extend the feature engineering pipeline
6. **Deploy**: Follow deployment guide for production

## Resources

- **API Documentation**: http://localhost:8000/docs (when API is running)
- **Full Documentation**: [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)
- **Sample Notebooks**: [notebooks/](notebooks/)
- **Test Coverage Report**: `htmlcov/index.html` (after running tests)

## Support

- **GitHub Issues**: Report bugs or request features
- **Email**: your.email@example.com
- **Documentation**: Comprehensive guides in `docs/`

---

**You're all set! Happy predicting! 🏥📊**
