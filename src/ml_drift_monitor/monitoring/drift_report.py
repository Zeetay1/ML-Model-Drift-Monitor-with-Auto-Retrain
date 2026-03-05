from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ml_drift_monitor.config import DriftThresholdConfig


@dataclass
class FeatureDriftResult:
    feature_name: str
    drift_score: float
    threshold: float
    drift_detected: bool
    direction: Optional[str] = None
    p_value: Optional[float] = None
    stat: Optional[float] = None
    verdict: str = ""


@dataclass
class PredictionDriftResult:
    drift_score: float
    threshold: float
    drift_detected: bool
    verdict: str = ""


@dataclass
class DriftReport:
    month: int
    overall_drift_flag: bool
    feature_results: List[FeatureDriftResult] = field(default_factory=list)
    prediction_result: Optional[PredictionDriftResult] = None
    top_contributing_features: List[str] = field(default_factory=list)
    config_used: Dict[str, float] = field(default_factory=dict)


def compute_overall_flag(
    feature_results: List[FeatureDriftResult],
    prediction_result: Optional[PredictionDriftResult],
    thresholds: DriftThresholdConfig,
) -> bool:
    drifted_features = [f for f in feature_results if f.drift_detected]
    if len(drifted_features) >= thresholds.min_drifted_features_for_flag:
        return True
    if prediction_result and prediction_result.drift_detected:
        return True
    return False


def rank_top_features(feature_results: List[FeatureDriftResult], top_k: int = 5) -> List[str]:
    sorted_feats = sorted(feature_results, key=lambda f: abs(f.drift_score), reverse=True)
    return [f.feature_name for f in sorted_feats[:top_k]]

