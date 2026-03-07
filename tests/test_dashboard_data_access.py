"""
Dashboard data access: load drift reports and retrain events from artifacts.
"""

from __future__ import annotations

from ml_drift_monitor.config import get_default_config
from ml_drift_monitor.dashboard.data_access import load_drift_reports, load_retrain_events


def test_load_drift_reports_returns_dict():
    cfg = get_default_config()
    reports = load_drift_reports(cfg)
    assert isinstance(reports, dict)


def test_load_retrain_events_returns_dataframe():
    cfg = get_default_config()
    events = load_retrain_events(cfg)
    assert hasattr(events, "columns")
    assert hasattr(events, "empty")
