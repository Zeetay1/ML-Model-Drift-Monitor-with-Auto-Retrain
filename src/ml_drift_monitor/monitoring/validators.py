from __future__ import annotations

from typing import Iterable, Set

import pandas as pd

from ml_drift_monitor.monitoring.drift_report import DriftReport


def get_drifted_features_from_report(report: DriftReport) -> Set[str]:
    return {f.feature_name for f in report.feature_results if f.drift_detected}


def get_ground_truth_drifted_features(ground_truth_monthly: pd.DataFrame, month: int) -> Set[str]:
    month_df = ground_truth_monthly[ground_truth_monthly["month"] == month]
    return set(month_df.loc[month_df["is_drifted"], "feature_name"].tolist())


