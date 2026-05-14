"""Deterioration risk estimators with optional probability calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_estimator(
    model_type: str,
    random_state: int,
    *,
    scale_pos_weight: float | None = None,
) -> Any:
    t = model_type.lower()
    if t == "logistic":
        return LogisticRegression(
            max_iter=5000,
            tol=1e-3,
            class_weight="balanced",
            random_state=random_state,
        )
    if t == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        )
    if t == "gradient_boosting":
        return HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.06,
            max_iter=200,
            random_state=random_state,
            class_weight="balanced",
            n_iter_no_change=15,
        )
    if t == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            random_state=random_state,
        )
    if t == "xgboost":
        from xgboost import XGBClassifier

        spw = float(scale_pos_weight) if scale_pos_weight and scale_pos_weight > 0 else 1.0
        return XGBClassifier(
            max_depth=6,
            learning_rate=0.06,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=1,
            scale_pos_weight=spw,
            eval_metric="logloss",
        )
    raise ValueError(f"Unknown model type: {model_type}")


def imbalance_scale_pos_weight(y: np.ndarray) -> float:
    """n_negative / n_positive for XGBoost scale_pos_weight (safe if no positives)."""
    y = np.asarray(y).astype(int).ravel()
    n_pos = max(int(y.sum()), 1)
    n_neg = max(int(len(y) - y.sum()), 1)
    return float(n_neg) / float(n_pos)


def make_pipeline(
    model_type: str,
    random_state: int,
    *,
    scale_pos_weight: float | None = None,
) -> Pipeline:
    t = model_type.lower()
    est = build_estimator(
        model_type,
        random_state,
        scale_pos_weight=scale_pos_weight if t == "xgboost" else None,
    )
    if t == "mlp":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", est),
            ]
        )
    if t == "logistic":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", est),
            ]
        )
    return Pipeline([("clf", est)])


def fit_calibrated(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    calibration: str,
    random_state: int,
) -> Any:
    cal = calibration.lower()
    if cal == "none":
        pipeline.fit(X, y)
        return pipeline
    method = "isotonic" if cal == "isotonic" else "sigmoid"
    wrapped = CalibratedClassifierCV(
        pipeline, method=method, cv=3, n_jobs=1
    )
    wrapped.fit(X, y)
    return wrapped


def save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Any:
    return joblib.load(path)
