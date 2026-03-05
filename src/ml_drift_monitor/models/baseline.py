from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_drift_monitor.config import ProjectConfig, get_default_config
from ml_drift_monitor.data.generator import generate_all_months
from ml_drift_monitor.data.storage import ensure_directories, save_month_batch, save_feature_schema
from ml_drift_monitor.data.generator import generate_feature_schema
from ml_drift_monitor.logging_utils.logger import get_logger
from ml_drift_monitor.models.evaluation import compute_classification_metrics
from ml_drift_monitor.models.training import predict_with_proba, train_logistic_regression

import joblib


logger = get_logger()


def _prepare_training_data(cfg: ProjectConfig) -> pd.DataFrame:
    """
    Generate months 1–6 if needed and persist them, returning a concatenated
    DataFrame for months 1–2 to use as training data.
    """
    ensure_directories(cfg)
    # Generate all months; this is deterministic due to the fixed seed.
    months = generate_all_months(cfg.data_generation)
    schema = generate_feature_schema(cfg.data_generation)
    save_feature_schema(schema, cfg)

    for month, df in months.items():
        save_month_batch(df, month, cfg)

    train_df = pd.concat([months[1], months[2]], ignore_index=True)
    return train_df


def _train_initial_model(cfg: ProjectConfig) -> Tuple[object, Dict]:
    train_df = _prepare_training_data(cfg)
    X = train_df.drop(columns=["label"])
    y = train_df["label"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=cfg.data_generation.random_seed
    )

    model = train_logistic_regression(X_train, y_train, random_state=cfg.data_generation.random_seed)
    y_val_pred, y_val_proba = predict_with_proba(model, X_val)
    metrics = compute_classification_metrics(y_val, y_val_pred, y_val_proba)

    meta = {
        "training_window": [1, 2],
        "metrics": metrics,
        "promotion_decision": "initial_champion",
    }
    return model, meta


def ensure_initial_champion(cfg: ProjectConfig | None = None) -> Tuple[object, Dict]:
    """
    Ensure that an initial champion model exists on disk.
    If not present, train it on months 1–2 and save as champion_v1.
    """
    from ml_drift_monitor.models.registry import (
        get_current_champion,
        save_new_champion_version,
    )

    project_cfg = cfg or get_default_config()
    model, meta = get_current_champion(project_cfg)
    if model is not None:
        return model, meta

    logger.info("No champion model found. Training initial champion on months 1–2.")
    model, meta = _train_initial_model(project_cfg)
    save_new_champion_version(model, meta, project_cfg)
    logger.info("Initial champion trained and saved.")
    return model, meta

