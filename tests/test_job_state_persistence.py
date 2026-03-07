"""
Job state must be persisted to a database and survive process restart.
Verified by: write state, re-open DB (simulate new process), read and assert.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ml_drift_monitor.config import PathConfig, ProjectConfig, get_default_config
from ml_drift_monitor.db.job_state import (
    get_job_state,
    has_retrain_run_for_window,
    set_job_completed,
)


def test_job_state_survives_new_connection():
    """Write job state to SQLite, close, open new connection, assert state is present."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "job_state.db"
        # Build a config that points to this temp DB.
        cfg = get_default_config()
        cfg.paths.job_state_db_path = db_path
        cfg.paths.artifacts_root = Path(tmp)
        cfg.paths.job_state_db_path = db_path

        # Simulate completing a retrain for window month_3.
        set_job_completed(
            "month_3",
            status="completed",
            promotion_decision="promoted",
            champion_version=2,
            challenger_version=1,
            drift_flag=True,
            retrain_triggered=True,
            extra={"cost_metadata": {"inference_seconds": 1.5, "estimated_compute_cost": 0.01}},
            cfg=cfg,
        )

        # "Restart": new connection (job_state uses new _conn each time).
        state = get_job_state("month_3", cfg)
        assert state is not None
        assert state["window_id"] == "month_3"
        assert state["status"] == "completed"
        assert state["promotion_decision"] == "promoted"
        assert state["retrain_triggered"] == 1

        assert has_retrain_run_for_window("month_3", cfg) is True
        assert has_retrain_run_for_window("month_99", cfg) is False
