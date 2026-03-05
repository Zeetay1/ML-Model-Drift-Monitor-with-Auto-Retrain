from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from ml_drift_monitor.config import ProjectConfig, get_default_config
from ml_drift_monitor.logging_utils.logger import get_logger


logger = get_logger()


def ensure_directories(paths: ProjectConfig | None = None) -> None:
    cfg = paths or get_default_config()
    cfg.paths.raw_data_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.metadata_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.drift_reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.mlflow_root.mkdir(parents=True, exist_ok=True)
    logger.info("Ensured artifacts directories exist under %s", cfg.paths.artifacts_root)


def save_month_batch(df: pd.DataFrame, month: int, config: ProjectConfig | None = None) -> Path:
    cfg = config or get_default_config()
    ensure_directories(cfg)
    path = cfg.paths.raw_data_dir / f"month_{month}.csv"
    df.to_csv(path, index=False)
    logger.info("Saved month %d batch to %s", month, path)
    return path


def load_month_batch(month: int, config: ProjectConfig | None = None) -> pd.DataFrame:
    cfg = config or get_default_config()
    path = cfg.paths.raw_data_dir / f"month_{month}.csv"
    df = pd.read_csv(path)
    logger.info("Loaded month %d batch from %s", month, path)
    return df


def save_feature_schema(schema: pd.DataFrame, config: ProjectConfig | None = None) -> Path:
    cfg = config or get_default_config()
    ensure_directories(cfg)
    path = cfg.paths.metadata_dir / "feature_schema.json"
    schema.to_json(path, orient="records", indent=2)
    logger.info("Saved feature schema to %s", path)
    return path

