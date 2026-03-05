from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def compute_classification_metrics(
    y_true, y_pred, y_proba=None
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    if y_proba is not None:
        # Assume binary classification with positive class probability at index 1.
        pos_proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, pos_proba))
        except ValueError:
            # Fallback if only one class present.
            metrics["roc_auc"] = float("nan")
    return metrics

