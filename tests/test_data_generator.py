from __future__ import annotations

import numpy as np
import pandas as pd

from ml_drift_monitor.config import get_default_config
from ml_drift_monitor.data.generator import generate_all_months


def test_months_1_and_2_are_similar_for_non_drift_features():
    cfg = get_default_config()
    months = generate_all_months(cfg.data_generation)
    df1 = months[1]
    df2 = months[2]

    non_drift_features = [
        spec.feature_name for spec in cfg.data_generation.drift_specs if spec.drift_type == "none"
    ]
    assert non_drift_features, "There should be at least one non-drift feature for control."

    for feature in non_drift_features:
        diff_mean = abs(df1[feature].mean() - df2[feature].mean())
        diff_std = abs(df1[feature].std() - df2[feature].std())
        # For normally distributed controls, mean and std differences should be small.
        assert diff_mean < 0.5, f"{feature} mean differs too much between months 1 and 2"
        assert diff_std < 0.5, f"{feature} std differs too much between months 1 and 2"


def test_drift_is_detectable_from_month_3_onward_for_drift_features():
    cfg = get_default_config()
    months = generate_all_months(cfg.data_generation)
    df1 = months[1]
    df3 = months[3]

    drift_features = [
        spec.feature_name
        for spec in cfg.data_generation.drift_specs
        if spec.drift_type != "none" and (spec.drift_start_month or 99) <= 3
    ]
    assert drift_features, "There should be at least one feature starting drift by month 3."

    for feature in drift_features:
        # Use a simple effect size heuristic to assert meaningful drift.
        m1 = df1[feature]
        m3 = df3[feature]
        if pd.api.types.is_numeric_dtype(m1):
            pooled_std = np.sqrt((m1.var() + m3.var()) / 2.0)
            effect_size = abs(m1.mean() - m3.mean()) / (pooled_std + 1e-8)
            assert effect_size > 0.3, f"{feature} drift effect size too small between months 1 and 3"

