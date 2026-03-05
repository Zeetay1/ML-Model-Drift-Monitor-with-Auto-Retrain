from __future__ import annotations

from typing import Dict, List

import pandas as pd
from evidently.metrics import PredictionDriftMetric
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from evidently import ColumnMapping

from ml_drift_monitor.config import DriftThresholdConfig, ProjectConfig, get_default_config
from ml_drift_monitor.monitoring.drift_report import (
    DriftReport,
    FeatureDriftResult,
    PredictionDriftResult,
    compute_overall_flag,
    rank_top_features,
)
from ml_drift_monitor.logging_utils.logger import get_logger


logger = get_logger()


def _build_column_mapping(feature_schema: pd.DataFrame) -> ColumnMapping:
    num_features = feature_schema.loc[feature_schema["feature_type"] == "numerical", "feature_name"].tolist()
    cat_features = feature_schema.loc[feature_schema["feature_type"] == "categorical", "feature_name"].tolist()
    mapping = ColumnMapping()
    mapping.numerical_features = num_features
    mapping.categorical_features = cat_features
    mapping.target = "label"
    mapping.prediction = "prediction"
    return mapping


def _extract_feature_results(
    metric_result: Dict, thresholds: DriftThresholdConfig
) -> List[FeatureDriftResult]:
    by_col = metric_result.get("drift_by_columns", {})
    feature_results: List[FeatureDriftResult] = []
    for feature_name, info in by_col.items():
        drift_score = float(info.get("drift_score", 0.0))
        # Use our own threshold for decision, even though Evidently may also provide one.
        drift_detected = drift_score >= thresholds.feature_drift_score_threshold
        direction = info.get("drift_detected")  # placeholder; Evidently may not expose direction directly
        p_value = info.get("p_value")
        stat = info.get("stat")
        if drift_detected:
            verdict = f"Drift detected (score={drift_score:.3f} ≥ {thresholds.feature_drift_score_threshold})"
        else:
            verdict = f"No drift (score={drift_score:.3f} < {thresholds.feature_drift_score_threshold})"
        feature_results.append(
            FeatureDriftResult(
                feature_name=feature_name,
                drift_score=drift_score,
                threshold=thresholds.feature_drift_score_threshold,
                drift_detected=drift_detected,
                direction=str(direction) if direction is not None else None,
                p_value=p_value,
                stat=stat,
                verdict=verdict,
            )
        )
    return feature_results


def _extract_prediction_result(result: Dict, thresholds: DriftThresholdConfig) -> PredictionDriftResult:
    drift_score = float(result.get("drift_score", 0.0))
    drift_detected = drift_score >= thresholds.prediction_drift_threshold
    if drift_detected:
        verdict = f"Prediction drift detected (score={drift_score:.3f} ≥ {thresholds.prediction_drift_threshold})"
    else:
        verdict = f"No prediction drift (score={drift_score:.3f} < {thresholds.prediction_drift_threshold})"
    return PredictionDriftResult(
        drift_score=drift_score,
        threshold=thresholds.prediction_drift_threshold,
        drift_detected=drift_detected,
        verdict=verdict,
    )


def run_evidently_drift(
    month: int,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_schema: pd.DataFrame,
    config: ProjectConfig | None = None,
) -> DriftReport:
    """
    Run Evidently-based feature and prediction drift detection and return a structured DriftReport.
    """
    project_cfg = config or get_default_config()
    thresholds = project_cfg.drift_thresholds

    mapping = _build_column_mapping(feature_schema)
    report = Report(metrics=[DataDriftPreset(), PredictionDriftMetric()])
    report.run(reference_data=reference, current_data=current, column_mapping=mapping)
    report_dict = report.as_dict()

    # Expect first metric to be data drift, second to be prediction drift.
    metrics = report_dict.get("metrics", [])
    data_drift_result = {}
    prediction_drift_result = {}
    for metric in metrics:
        metric_name = metric.get("metric", "")
        if "DataDrift" in metric_name:
            data_drift_result = metric.get("result", {})
        if "PredictionDrift" in metric_name:
            prediction_drift_result = metric.get("result", {})

    feature_results = _extract_feature_results(data_drift_result, thresholds)
    prediction_result = _extract_prediction_result(prediction_drift_result, thresholds)

    overall_flag = compute_overall_flag(feature_results, prediction_result, thresholds)
    top_features = rank_top_features(feature_results)

    drift_report = DriftReport(
        month=month,
        overall_drift_flag=overall_flag,
        feature_results=feature_results,
        prediction_result=prediction_result,
        top_contributing_features=top_features,
        config_used={
            "feature_drift_score_threshold": thresholds.feature_drift_score_threshold,
            "min_drifted_features_for_flag": thresholds.min_drifted_features_for_flag,
            "prediction_drift_threshold": thresholds.prediction_drift_threshold,
        },
    )

    # Persist for later inspection and dashboard use.
    project_cfg.paths.drift_reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = project_cfg.paths.drift_reports_dir / f"month_{month}_report.json"
    pd.Series(report_dict).to_json(out_path)
    logger.info("Saved Evidently drift report for month %d to %s", month, out_path)

    return drift_report

