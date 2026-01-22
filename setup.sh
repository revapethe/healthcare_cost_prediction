#!/bin/bash

# Healthcare Cost Prediction - Master Setup Script
# This script sets up and runs the entire project

set -e  # Exit on error

echo "=================================================="
echo "Healthcare Cost Prediction - Project Setup"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[i]${NC} $1"
}

# Check Python installation
print_info "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
print_status "Found $PYTHON_VERSION"

# Create virtual environment
print_info "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_info "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_status "Pip upgraded"

# Install dependencies
print_info "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt > /dev/null 2>&1
print_status "Dependencies installed"

# Create necessary directories
print_info "Creating directory structure..."
mkdir -p data/{raw,processed,features}
mkdir -p models/saved
mkdir -p logs
mkdir -p outputs
print_status "Directories created"

# Generate synthetic data
print_info "Generating synthetic patient data..."
python scripts/generate_data.py --n_patients 10000
print_status "Data generated (10,000 patients)"

# Run data preprocessing
print_info "Running data preprocessing pipeline..."
python src/data/pipeline.py
print_status "Data preprocessing complete"

# Train models
print_info "Training machine learning models..."
python src/models/train.py
print_status "Model training complete"

# Run tests
print_info "Running test suite..."
pytest tests/ -v --cov=src --cov-report=html > /dev/null 2>&1
print_status "Tests passed"

echo ""
echo "=================================================="
echo -e "${GREEN}✓ SETUP COMPLETE!${NC}"
echo "=================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Start the API server:"
echo "   uvicorn src.api.app:app --reload --port 8000"
echo ""
echo "2. Launch the dashboard:"
echo "   streamlit run dashboards/main_dashboard.py"
echo ""
echo "3. Access the API docs:"
echo "   http://localhost:8000/docs"
echo ""
echo "4. View test coverage:"
echo "   open htmlcov/index.html"
echo ""
echo "5. Explore Jupyter notebooks:"
echo "   jupyter notebook notebooks/"
echo ""
echo "=================================================="
