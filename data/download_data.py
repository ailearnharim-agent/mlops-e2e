#!/usr/bin/env python
"""CLI entry point to download the UCI Heart Disease dataset.

Usage:
    python data/download_data.py

Writes the combined raw dataset to data/raw/heart_disease_raw.csv.
Run this before src/data/preprocess.py or src/models/train.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.download import main  # noqa: E402

if __name__ == "__main__":
    main()
