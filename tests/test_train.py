import numpy as np
from sklearn.linear_model import LogisticRegression

from src.config import ALL_FEATURES, TARGET_COL
from src.data.preprocess import clean_dataframe
from src.models.evaluate import compute_metrics
from src.models.train import CANDIDATES, build_pipeline


def test_candidates_have_estimator_and_param_grid():
    assert len(CANDIDATES) >= 2, "assignment requires at least two classification models"
    for name, cfg in CANDIDATES.items():
        assert "estimator" in cfg
        assert "param_grid" in cfg
        assert all(k.startswith("clf__") for k in cfg["param_grid"])


def test_build_pipeline_fits_and_predicts(raw_sample_df):
    clean = clean_dataframe(raw_sample_df)
    X = clean[ALL_FEATURES]
    y = clean[TARGET_COL]

    pipeline = build_pipeline(LogisticRegression(max_iter=1000, random_state=42))
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    probas = pipeline.predict_proba(X)

    assert preds.shape == (len(X),)
    assert probas.shape == (len(X), 2)
    assert set(np.unique(preds)) <= {0, 1}
    # probabilities for each row must sum to 1
    assert np.allclose(probas.sum(axis=1), 1.0)


def test_compute_metrics_returns_expected_keys():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    y_proba = [0.1, 0.9, 0.4, 0.2, 0.8]

    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_compute_metrics_perfect_predictions():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    y_proba = [0.01, 0.99, 0.02, 0.98]

    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["roc_auc"] == 1.0
