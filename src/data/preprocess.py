"""Cleaning and feature-preprocessing for the Heart Disease dataset.

Two distinct concerns are kept separate on purpose:

1. ``clean_dataframe`` — light, leakage-free cleaning (dedup, dtype fixes,
   target binarization) that is safe to apply once and persist as
   ``data/processed/heart_disease_processed.csv``.
2. ``build_preprocessor`` — a scikit-learn ``ColumnTransformer`` (imputation,
   scaling, one-hot encoding) that must be *fit only on the training split*
   to avoid leaking test-set statistics. It is composed into the model
   ``Pipeline`` in ``src/models/train.py`` rather than baked into the CSV.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    TARGET_COL,
)

logger = logging.getLogger(__name__)


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate, fix dtypes, and binarize the target.

    The original UCI target ("num") ranges 0-4 (0 = no disease, 1-4 = disease
    present with increasing severity/number of vessels). We keep it as
    ``target_multiclass`` for reference/EDA and binarize the modelling target
    to 0/1, which matches the assignment's "presence/absence" framing.
    """
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    if len(df) != before:
        logger.info("Dropped %d duplicate rows", before - len(df))

    # target -> target_multiclass (kept for EDA), target -> binarized 0/1
    df = df.rename(columns={TARGET_COL: "target_multiclass"})
    df["target_multiclass"] = df["target_multiclass"].astype(int)
    df[TARGET_COL] = (df["target_multiclass"] > 0).astype(int)

    # Categorical/binary columns arrive as floats (due to NaNs); keep them as
    # nullable floats here, the ColumnTransformer's imputers handle NaNs
    # downstream. Reorder columns for readability.
    ordered_cols = (
        ["source"]
        + NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + BINARY_FEATURES
        + ["target_multiclass", TARGET_COL]
    )
    df = df[ordered_cols]
    return df


def build_preprocessor() -> ColumnTransformer:
    """Build the reusable, leakage-safe feature preprocessing pipeline.

    - Numeric features: median imputation + standard scaling.
    - Categorical features (cp, restecg, slope, ca, thal): missingness here
      is substantial (up to ~66% for `ca`) because 3 of the 4 source sites
      rarely recorded them, so a constant sentinel category (-1, outside the
      valid value range) is imputed instead of the mode — this explicitly
      models "not recorded" as its own one-hot category rather than papering
      over it.
    - Binary features (sex, fbs, exang): mode imputation, passed through
      as-is (already 0/1).
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("binary", binary_pipeline, BINARY_FEATURES),
        ]
    )
    return preprocessor


def save_processed(df: pd.DataFrame, output_path: Path = PROCESSED_DATA_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved processed dataset to %s", output_path)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw_df = load_raw()
    clean_df = clean_dataframe(raw_df)
    save_processed(clean_df)
    logger.info(
        "Processed dataset: %d rows, target balance:\n%s",
        len(clean_df),
        clean_df["target"].value_counts(normalize=True).round(3).to_dict(),
    )


if __name__ == "__main__":
    main()
