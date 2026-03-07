from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PathConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    artifacts_root: Path = field(init=False)
    raw_data_dir: Path = field(init=False)
    metadata_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    drift_reports_dir: Path = field(init=False)
    mlflow_root: Path = field(init=False)
    job_state_db_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.artifacts_root = self.project_root / "artifacts"
        self.raw_data_dir = self.artifacts_root / "data" / "raw"
        self.metadata_dir = self.artifacts_root / "data" / "metadata"
        self.models_dir = self.artifacts_root / "models"
        self.logs_dir = self.artifacts_root / "logs"
        self.drift_reports_dir = self.artifacts_root / "drift_reports"
        self.mlflow_root = self.artifacts_root / "mlflow"
        self.job_state_db_path = self.artifacts_root / "job_state.db"


@dataclass
class FeatureDriftSpec:
    feature_name: str
    feature_type: str  # "numerical" or "categorical"
    drift_type: str  # "none", "mean_shift", "variance_shift", "category_shift"
    drift_start_month: Optional[int]
    drift_magnitude: float  # abstract magnitude; interpreted per type


@dataclass
class DataGenerationConfig:
    months: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    rows_per_month: int = 5000
    random_seed: int = 42
    drift_specs: List[FeatureDriftSpec] = field(default_factory=list)


@dataclass
class DriftThresholdConfig:
    feature_drift_score_threshold: float = 0.2
    min_drifted_features_for_flag: int = 2
    prediction_drift_threshold: float = 0.1


@dataclass
class PromotionConfig:
    metric_name: str = "roc_auc"
    min_relative_improvement: float = 0.01


@dataclass
class CostTrackingConfig:
    """Cost tracking for retrain runs (no LLM; use inference time + estimated compute as proxy)."""
    estimated_compute_cost_per_retrain: float = 0.01  # e.g. USD per retrain run
    cost_per_cpu_hour: float = 0.05  # for deriving cost from inference_seconds if needed


@dataclass
class ProjectConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    data_generation: DataGenerationConfig = field(default_factory=DataGenerationConfig)
    drift_thresholds: DriftThresholdConfig = field(default_factory=DriftThresholdConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    cost_tracking: CostTrackingConfig = field(default_factory=CostTrackingConfig)


def get_default_feature_drift_specs() -> List[FeatureDriftSpec]:
    """
    Default feature set:
    - Some non-drift control features.
    - Numerical and categorical features with gradual drift starting at month 3.
    """
    return [
        FeatureDriftSpec(
            feature_name="age",
            feature_type="numerical",
            drift_type="none",
            drift_start_month=None,
            drift_magnitude=0.0,
        ),
        FeatureDriftSpec(
            feature_name="income",
            feature_type="numerical",
            drift_type="mean_shift",
            drift_start_month=3,
            drift_magnitude=0.5,
        ),
        FeatureDriftSpec(
            feature_name="tenure",
            feature_type="numerical",
            drift_type="variance_shift",
            drift_start_month=3,
            drift_magnitude=0.5,
        ),
        FeatureDriftSpec(
            feature_name="transactions_last_month",
            feature_type="numerical",
            drift_type="mean_shift",
            drift_start_month=3,
            drift_magnitude=0.7,
        ),
        FeatureDriftSpec(
            feature_name="score",
            feature_type="numerical",
            drift_type="mean_shift",
            drift_start_month=4,
            drift_magnitude=0.3,
        ),
        FeatureDriftSpec(
            feature_name="region",
            feature_type="categorical",
            drift_type="category_shift",
            drift_start_month=3,
            drift_magnitude=0.3,
        ),
        FeatureDriftSpec(
            feature_name="channel",
            feature_type="categorical",
            drift_type="category_shift",
            drift_start_month=3,
            drift_magnitude=0.2,
        ),
    ]


def get_default_config() -> ProjectConfig:
    cfg = ProjectConfig()
    if not cfg.data_generation.drift_specs:
        cfg.data_generation.drift_specs = get_default_feature_drift_specs()
    return cfg

