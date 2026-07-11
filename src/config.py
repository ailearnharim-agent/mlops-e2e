"""Central paths and constants shared across the pipeline."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "heart_disease_raw.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "heart_disease_processed.csv"

MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"

MLRUNS_DIR = ROOT_DIR / "mlruns"
MLFLOW_TRACKING_URI = f"file:{MLRUNS_DIR}"
MLFLOW_EXPERIMENT_NAME = "heart-disease-classification"

REPORTS_DIR = ROOT_DIR / "reports"

# UCI ML Repository dataset id for "Heart Disease" (Cleveland + Hungary + Switzerland + VA Long Beach)
UCI_DATASET_ID = 45

# Original UCI target is 0 (no disease) .. 4 (varying severity of disease).
# We binarize: 0 stays 0, anything >0 becomes 1 ("presence of disease").
TARGET_COL = "target"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "ca", "thal"]
BINARY_FEATURES = ["sex", "fbs", "exang"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.2
