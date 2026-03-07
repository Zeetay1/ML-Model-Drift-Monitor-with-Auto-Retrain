"""
Cost metadata (inference time, estimated compute cost) must be present and non-zero
after a retrain run. Verify via record_retrain_cost and that flow-attached metadata is set.
"""

from __future__ import annotations

from ml_drift_monitor.config import get_default_config
from ml_drift_monitor.utils.cost_tracking import CostMetadata, record_retrain_cost


def test_record_retrain_cost_returns_non_zero_metadata():
    """record_retrain_cost returns metadata with non-zero inference_seconds and estimated_compute_cost."""
    meta = record_retrain_cost(2.5, get_default_config())
    assert isinstance(meta, CostMetadata)
    assert meta.inference_seconds == 2.5
    assert meta.estimated_compute_cost > 0
    d = meta.to_dict()
    assert d["inference_seconds"] == 2.5
    assert d["estimated_compute_cost"] > 0


def test_cost_metadata_present_after_mocked_retrain_run():
    """Simulate a retrain run that records cost; assert metadata is present and non-zero."""
    from ml_drift_monitor.utils.cost_tracking import record_retrain_cost

    # Simulate some work (e.g. evaluation) taking time.
    import time
    start = time.perf_counter()
    time.sleep(0.01)  # minimal delay so inference_seconds > 0
    elapsed = time.perf_counter() - start

    cost_meta = record_retrain_cost(elapsed)
    assert cost_meta.inference_seconds > 0
    assert cost_meta.estimated_compute_cost > 0
