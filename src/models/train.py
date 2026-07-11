"""Train, tune, and compare classification models with MLflow experiment tracking.

Usage:
    python -m src.models.train

For each candidate model (Logistic Regression, Random Forest,
HistGradientBoosting) this:
  1. Runs GridSearchCV with repeated stratified CV on the training split.
  2. Logs params/CV results/test metrics/diagnostic plots to MLflow.
  3. Evaluates the tuned model on a held-out test split (never used in CV).

The model with the best held-out ROC-AUC is then refit on the FULL dataset
(train+test) with its selected hyperparameters — a separate, clearly-tagged
"final_model" MLflow run — and saved to models/model.pkl for serving. Held-out
metrics reported in the report/README come from the earlier CV-selection run,
not the refit-on-full-data run (which has no honest held-out set).
"""

import logging
from pathlib import Path
from typing import Dict

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    ALL_FEATURES,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COL,
    TEST_SIZE,
)
from src.data.preprocess import build_preprocessor
from src.models.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_precision_recall_curve,
    plot_roc_curve,
)

logger = logging.getLogger(__name__)

CANDIDATES = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "param_grid": {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__class_weight": [None, "balanced"],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
        },
    },
    "hist_gradient_boosting": {
        "estimator": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "clf__max_iter": [100, 200],
            "clf__max_depth": [None, 3, 5],
            "clf__learning_rate": [0.05, 0.1],
        },
    },
}


def load_dataset(path: Path = PROCESSED_DATA_PATH):
    df = pd.read_csv(path)
    X = df[ALL_FEATURES]
    y = df[TARGET_COL]
    return X, y


def build_pipeline(estimator) -> Pipeline:
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("clf", estimator)])


def train_and_tune_one(
    name: str,
    estimator,
    param_grid: Dict,
    X_train,
    y_train,
    X_test,
    y_test,
    plots_dir: Path,
) -> Dict:
    """Run GridSearchCV for one model type, log everything to MLflow, return summary."""
    pipeline = build_pipeline(estimator)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)

    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    with mlflow.start_run(run_name=name):
        logger.info(
            "Fitting GridSearchCV for %s (%d candidates x %d CV splits)",
            name,
            len(list(_expand(param_grid))),
            cv.get_n_splits(),
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_pred, y_proba)

        mlflow.log_param("model_type", name)
        mlflow.log_params({k.replace("clf__", ""): v for k, v in search.best_params_.items()})
        mlflow.log_metric("cv_best_roc_auc", search.best_score_)
        for metric_name, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        cm_path = plot_confusion_matrix(
            y_test, y_pred, plots_dir / f"{name}_confusion_matrix.png", title=f"{name}: Confusion Matrix"
        )
        roc_path = plot_roc_curve(
            y_test, y_proba, plots_dir / f"{name}_roc_curve.png", title=f"{name}: ROC Curve"
        )
        pr_path = plot_precision_recall_curve(
            y_test, y_proba, plots_dir / f"{name}_pr_curve.png", title=f"{name}: Precision-Recall Curve"
        )
        for p in (cm_path, roc_path, pr_path):
            mlflow.log_artifact(str(p))

        mlflow.sklearn.log_model(best_model, artifact_path="model")

        logger.info(
            "%s: cv_roc_auc=%.4f test_roc_auc=%.4f test_accuracy=%.4f",
            name,
            search.best_score_,
            test_metrics["roc_auc"],
            test_metrics["accuracy"],
        )

        return {
            "name": name,
            "best_params": search.best_params_,
            "cv_best_roc_auc": search.best_score_,
            "test_metrics": test_metrics,
            "estimator": estimator,
        }


def _expand(param_grid: Dict):
    from itertools import product

    keys = list(param_grid.keys())
    for combo in product(*param_grid.values()):
        yield dict(zip(keys, combo))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    Path(MLFLOW_TRACKING_URI.replace("file:", "")).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    plots_dir = REPORTS_DIR / "screenshots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train: %d rows, Test: %d rows", len(X_train), len(X_test))

    results = {}
    for name, cfg in CANDIDATES.items():
        summary = train_and_tune_one(
            name, cfg["estimator"], cfg["param_grid"], X_train, y_train, X_test, y_test, plots_dir
        )
        results[name] = summary

    comparison_metrics = {name: r["test_metrics"] for name, r in results.items()}
    plot_model_comparison(comparison_metrics, plots_dir / "model_comparison_roc_auc.png")

    best_name = max(results, key=lambda n: results[n]["test_metrics"]["roc_auc"])
    best = results[best_name]
    logger.info("Best model: %s (test ROC-AUC=%.4f)", best_name, best["test_metrics"]["roc_auc"])

    # Refit the winning model type + hyperparameters on the FULL dataset for deployment.
    final_estimator = CANDIDATES[best_name]["estimator"].__class__(
        **{
            **CANDIDATES[best_name]["estimator"].get_params(),
        }
    )
    final_pipeline = build_pipeline(final_estimator)
    final_pipeline.set_params(**best["best_params"])

    with mlflow.start_run(run_name=f"{best_name}_final_full_data"):
        final_pipeline.fit(X, y)
        mlflow.log_param("model_type", best_name)
        mlflow.log_params({k.replace("clf__", ""): v for k, v in best["best_params"].items()})
        mlflow.log_param("trained_on", "full_dataset")
        mlflow.log_metric("held_out_cv_roc_auc", best["cv_best_roc_auc"])
        mlflow.log_metric("held_out_test_roc_auc", best["test_metrics"]["roc_auc"])
        mlflow.sklearn.log_model(final_pipeline, artifact_path="model")
        mlflow.set_tag("selected_as_production_model", "true")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODEL_PATH)
    logger.info("Saved final production model to %s", MODEL_PATH)

    summary_df = pd.DataFrame(
        {name: {**r["test_metrics"], "cv_best_roc_auc": r["cv_best_roc_auc"]} for name, r in results.items()}
    ).T
    summary_df.index.name = "model"
    summary_df["selected"] = summary_df.index == best_name
    summary_path = REPORTS_DIR / "model_comparison.csv"
    summary_df.to_csv(summary_path)
    logger.info("Wrote model comparison summary to %s\n%s", summary_path, summary_df.round(4))


if __name__ == "__main__":
    main()
