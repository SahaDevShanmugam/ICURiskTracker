"""Metrics and figures for deterioration risk evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def classification_report_binary(y_true: np.ndarray, y_score: np.ndarray, name: str) -> dict:
    y_hat = (y_score >= 0.5).astype(int)
    out = {
        "name": name,
        "roc_auc": float(roc_auc_score(y_true, y_score))
        if len(np.unique(y_true)) > 1
        else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_score))
        if y_true.sum() > 0
        else float("nan"),
        "brier": float(brier_score_loss(y_true, y_score)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
    }
    return out


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_calibration_curve(
    y_true: np.ndarray, y_score: np.ndarray, title: str, out_path: Path, n_bins: int = 10
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(np.unique(y_true)) < 2:
        return
    prob_true, prob_pred = calibration_curve(
        y_true, y_score, n_bins=n_bins, strategy="uniform"
    )
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
    plt.xlabel("Mean predicted risk")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
