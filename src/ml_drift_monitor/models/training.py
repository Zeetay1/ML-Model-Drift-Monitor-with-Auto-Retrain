from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(
    X, y, random_state: int = 42
) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X, y)
    return model


def predict_with_proba(model, X) -> Tuple[np.ndarray, np.ndarray]:
    proba = model.predict_proba(X)
    preds = (proba[:, 1] >= 0.5).astype(int)
    return preds, proba

