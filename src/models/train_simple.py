"""Compatibility entry point for the supported scikit-learn training workflow.

Run from repository root: python -m src.models.train_simple
"""

from src.models.train import train

if __name__ == "__main__":
    train(include_boosters=False)
