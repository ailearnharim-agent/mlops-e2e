"""Evaluation metrics and diagnostic plots, reused by training and tests."""

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def plot_confusion_matrix(y_true, y_pred, output_path: Path, title: str = "Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["No disease", "Disease"], cmap="Blues", ax=ax
    )
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_roc_curve(y_true, y_proba, output_path: Path, title: str = "ROC Curve"):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_precision_recall_curve(y_true, y_proba, output_path: Path, title: str = "Precision-Recall Curve"):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_model_comparison(results: Dict[str, Dict[str, float]], output_path: Path, metric: str = "roc_auc"):
    names = list(results.keys())
    values = [results[n][metric] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(names, values, color=["#2e86ab", "#c0392b", "#27ae60"][: len(names)])
    ax.set_ylabel(metric.upper())
    ax.set_ylim(0, 1)
    ax.set_title(f"Model comparison ({metric.upper()}, held-out test set)")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
