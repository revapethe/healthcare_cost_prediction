"""Compatibility entry point; use the single seeded synthetic data generator.

Run: python -m scripts.generate_data_simple --n_patients 10000
"""

from scripts.generate_data import main

if __name__ == "__main__":
    main()
