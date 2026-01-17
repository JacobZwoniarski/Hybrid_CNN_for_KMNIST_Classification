"""Metrics utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    f1_macro: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
    """
    y_true, y_pred: shape (N,), integer class ids
    """
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    return ClassificationMetrics(accuracy=acc, f1_macro=f1)
