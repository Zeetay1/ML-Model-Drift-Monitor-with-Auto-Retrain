from __future__ import annotations

from ml_drift_monitor.config import get_default_config
from ml_drift_monitor.orchestration.prefect_flows import monitor_and_retrain_flow
from ml_drift_monitor.tracking.event_log import get_all_events


def test_pipeline_idempotency_for_same_window():
    cfg = get_default_config()

    # Run the flow twice for a drifted month (e.g., month 4).
    decision_first = monitor_and_retrain_flow(4)
    decision_second = monitor_and_retrain_flow(4)

    assert decision_first in {"promoted", "rejected"}
    assert decision_second == "skipped_already_processed"

    events = get_all_events(cfg)
    triggered_events = [e for e in events if e.window_id == "month_4" and e.retrain_triggered]
    assert len(triggered_events) == 1, "Retrain should only be triggered once for the same window"

