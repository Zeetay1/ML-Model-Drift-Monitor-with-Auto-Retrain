from __future__ import annotations

from dataclasses import asdict
from typing import List

import pandas as pd

from ml_drift_monitor.config import DataGenerationConfig, FeatureDriftSpec, ProjectConfig, get_default_config
from ml_drift_monitor.logging_utils.logger import get_logger


logger = get_logger()


def build_feature_ground_truth(config: DataGenerationConfig) -> pd.DataFrame:
    """
    Build a feature-level ground truth table mirroring the drift specs.
    """
    records = [asdict(spec) for spec in config.drift_specs]
    df = pd.DataFrame.from_records(records)
    logger.info("Built feature-level ground truth with %d records", len(df))
    return df


def build_monthly_ground_truth(config: DataGenerationConfig) -> pd.DataFrame:
    """
    Build a per-feature, per-month ground truth table indicating whether
    each feature is in a drifted state for a given month.
    """
    rows = []
    for spec in config.drift_specs:
        for month in config.months:
            is_drifted = spec.drift_start_month is not None and month >= spec.drift_start_month
            rows.append(
                {
                    "month": month,
                    "feature_name": spec.feature_name,
                    "type": spec.feature_type,
                    "drift_type": spec.drift_type,
                    "drift_start_month": spec.drift_start_month,
                    "is_drifted": bool(is_drifted and spec.drift_type != "none"),
                    "drift_magnitude": spec.drift_magnitude,
                }
            )
    df = pd.DataFrame(rows)
    logger.info("Built monthly ground truth with %d rows", len(df))
    return df


def save_ground_truth(config: ProjectConfig | None = None) -> None:
    """
    Build and persist ground truth logs to disk.

    This should be run after the generator configuration is finalized,
    but before any drift detection is implemented, so it acts as the
    authoritative reference for tests and monitoring.
    """
    project_cfg = config or get_default_config()
    feature_df = build_feature_ground_truth(project_cfg.data_generation)
    monthly_df = build_monthly_ground_truth(project_cfg.data_generation)

    project_cfg.paths.metadata_dir.mkdir(parents=True, exist_ok=True)
    feature_path = project_cfg.paths.metadata_dir / "drift_ground_truth.json"
    monthly_path = project_cfg.paths.metadata_dir / "drift_by_month.json"

    feature_df.to_json(feature_path, orient="records", indent=2)
    monthly_df.to_json(monthly_path, orient="records", indent=2)
    logger.info(
        "Saved ground truth: feature-level to %s, monthly breakdown to %s",
        feature_path,
        monthly_path,
    )

