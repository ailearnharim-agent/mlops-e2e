"""Load the packaged model and run inference on new patient records."""

import functools
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd

from src.config import ALL_FEATURES, MODEL_PATH


@functools.lru_cache(maxsize=1)
def load_model(model_path: Path = MODEL_PATH):
    """Load and cache the packaged sklearn Pipeline (preprocessing + classifier)."""
    return joblib.load(model_path)


def predict_one(features: Dict, model_path: Path = MODEL_PATH) -> Tuple[int, float]:
    """Predict heart disease presence for a single patient record.

    Args:
        features: dict with keys matching src.config.ALL_FEATURES.

    Returns:
        (prediction, confidence) where prediction is 0/1 and confidence is the
        model's predicted probability of the predicted class.
    """
    model = load_model(model_path)
    row = pd.DataFrame([{col: features.get(col) for col in ALL_FEATURES}])
    proba = model.predict_proba(row)[0]
    prediction = int(proba.argmax())
    confidence = float(proba[prediction])
    return prediction, confidence
