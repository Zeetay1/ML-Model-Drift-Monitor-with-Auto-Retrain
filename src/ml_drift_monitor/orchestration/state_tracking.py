"""
Idempotency and job state: use DB-backed state so that 'already processed' survives restart.
"""

from __future__ import annotations

from typing import Optional

from ml_drift_monitor.config import ProjectConfig, get_default_config
from ml_drift_monitor.db.job_state import get_job_state, has_retrain_run_for_window as db_has_retrain_run
from ml_drift_monitor.tracking.event_log import RetrainEvent, get_all_events


def has_retrain_run_for_window(window_id: str, cfg: ProjectConfig | None = None) -> bool:
    """True if a retrain has already been run for this window (from DB)."""
    return db_has_retrain_run(window_id, cfg)


def get_last_event_for_window(window_id: str, cfg: ProjectConfig | None = None) -> Optional[RetrainEvent]:
    """Last retrain event for window from event log (audit trail)."""
    events = [e for e in get_all_events(cfg) if e.window_id == window_id]
    if not events:
        return None
    return events[-1]


def get_window_state_from_db(window_id: str, cfg: ProjectConfig | None = None) -> Optional[dict]:
    """Raw job state from DB."""
    return get_job_state(window_id, cfg)
