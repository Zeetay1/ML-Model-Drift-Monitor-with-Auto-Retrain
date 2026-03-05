from __future__ import annotations

import pandas as pd

from ml_drift_monitor.config import get_default_config
from ml_drift_monitor.data.generator import generate_all_months, generate_feature_schema
from ml_drift_monitor.data.ground_truth import build_monthly_ground_truth
from ml_drift_monitor.monitoring.evidently_drift import run_evidently_drift
from ml_drift_monitor.monitoring.validators import (
    get_drifted_features_from_report,
    get_ground_truth_drifted_features,
)


def _build_reference_and_schema():
    cfg = get_default_config()
    months = generate_all_months(cfg.data_generation)
    reference = pd.concat([months[1], months[2]], ignore_index=True)
    schema = generate_feature_schema(cfg.data_generation)
    ground_truth_monthly = build_monthly_ground_truth(cfg.data_generation)
    return cfg, months, reference, schema, ground_truth_monthly


def test_zero_false_positives_on_months_1_and_2():
    cfg, months, reference, schema, ground_truth_monthly = _build_reference_and_schema()

    for month in [1, 2]:
        current = months[month]
        report = run_evidently_drift(month, reference, current, schema, cfg)
        assert not report.overall_drift_flag, f"Expected no overall drift for month {month}"
        drifted_features = get_drifted_features_from_report(report)
        assert not drifted_features, f"Expected no drifted features for month {month}"


def test_drift_flagged_by_month_4_and_matches_ground_truth():
    cfg, months, reference, schema, ground_truth_monthly = _build_reference_and_schema()

    # Use month 4 as a representative drifted month.
    current = months[4]
    report = run_evidently_drift(4, reference, current, schema, cfg)
    assert report.overall_drift_flag, "Expected overall drift flag by month 4"

    detected = get_drifted_features_from_report(report)
    truth = get_ground_truth_drifted_features(ground_truth_monthly, 4)

    # At least one ground-truth drifted feature should be detected.
    assert detected & truth, "Expected at least one ground-truth drifted feature to be detected by Evidently"

