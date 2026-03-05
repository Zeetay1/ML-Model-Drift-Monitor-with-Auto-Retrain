from __future__ import annotations

from typing import Optional

from ml_drift_monitor.config import ProjectConfig, get_default_config
from ml_drift_monitor.tracking.event_log import RetrainEvent, get_all_events


def has_retrain_run_for_window(window_id: str, cfg: ProjectConfig | None = None) -> bool:
    events = get_all_events(cfg)
    return any(e.window_id == window_id and e.retrain_triggered for e in events)


def get_last_event_for_window(window_id: str, cfg: ProjectConfig | None = None) -> Optional[RetrainEvent]:
    events = [e for e in get_all_events(cfg) if e.window_id == window_id]
    if not events:
        return None
    return events[-1]

