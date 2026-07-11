import numpy as np
import pandas as pd

from src.config import ALL_FEATURES, TARGET_COL
from src.data.preprocess import build_preprocessor, clean_dataframe


def test_clean_dataframe_binarizes_target(raw_sample_df):
    clean = clean_dataframe(raw_sample_df)
    assert set(clean[TARGET_COL].unique()) <= {0, 1}
    # any row with original multiclass target > 0 must map to binary target 1
    assert (clean.loc[clean["target_multiclass"] > 0, TARGET_COL] == 1).all()
    assert (clean.loc[clean["target_multiclass"] == 0, TARGET_COL] == 0).all()


def test_clean_dataframe_drops_exact_duplicates():
    df = pd.DataFrame(
        {
            "age": [50, 50],
            "sex": [1, 1],
            "cp": [1, 1],
            "trestbps": [120, 120],
            "chol": [200, 200],
            "fbs": [0, 0],
            "restecg": [0, 0],
            "thalach": [150, 150],
            "exang": [0, 0],
            "oldpeak": [0.0, 0.0],
            "slope": [1, 1],
            "ca": [0, 0],
            "thal": [3, 3],
            "target": [0, 0],
            "source": ["cleveland", "cleveland"],
        }
    )
    clean = clean_dataframe(df)
    assert len(clean) == 1


def test_clean_dataframe_preserves_row_count_minus_duplicates(raw_sample_df):
    clean = clean_dataframe(raw_sample_df)
    assert len(clean) <= len(raw_sample_df)
    assert len(clean) > 0


def test_preprocessor_output_has_no_nans(raw_sample_df):
    clean = clean_dataframe(raw_sample_df)
    X = clean[ALL_FEATURES]

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    assert not np.isnan(X_transformed).any()


def test_preprocessor_output_row_count_matches_input(raw_sample_df):
    clean = clean_dataframe(raw_sample_df)
    X = clean[ALL_FEATURES]

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    assert X_transformed.shape[0] == len(X)
    assert X_transformed.shape[1] > len(ALL_FEATURES)  # one-hot encoding expands columns


def test_preprocessor_handles_column_with_only_missing_values():
    # Simulate a batch where a normally-present categorical is entirely missing,
    # which happens for non-Cleveland sites in production.
    X = pd.DataFrame(
        {
            "age": [40, 55, 60],
            "trestbps": [120, 130, 140],
            "chol": [200, 210, 220],
            "thalach": [150, 140, 130],
            "oldpeak": [0.0, 1.0, 2.0],
            "cp": [1, 2, 3],
            "restecg": [0, 1, 2],
            "slope": [np.nan, np.nan, np.nan],
            "ca": [np.nan, np.nan, np.nan],
            "thal": [np.nan, np.nan, np.nan],
            "sex": [1, 0, 1],
            "fbs": [0, 1, 0],
            "exang": [0, 0, 1],
        }
    )
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    assert not np.isnan(X_transformed).any()
