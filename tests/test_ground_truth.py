from __future__ import annotations

from ml_drift_monitor.config import get_default_config
from ml_drift_monitor.data.ground_truth import build_feature_ground_truth, build_monthly_ground_truth


def test_ground_truth_feature_specs_match_config():
    cfg = get_default_config()
    feature_df = build_feature_ground_truth(cfg.data_generation)

    assert len(feature_df) == len(cfg.data_generation.drift_specs)
    for spec in cfg.data_generation.drift_specs:
        row = feature_df.loc[feature_df["feature_name"] == spec.feature_name].iloc[0]
        assert row["drift_type"] == spec.drift_type
        assert row["drift_start_month"] == spec.drift_start_month


def test_ground_truth_monthly_no_drift_for_months_1_and_2():
    cfg = get_default_config()
    monthly_df = build_monthly_ground_truth(cfg.data_generation)

    for month in [1, 2]:
        month_df = monthly_df[monthly_df["month"] == month]
        assert not month_df["is_drifted"].any(), f"Expected no drift in month {month}"

