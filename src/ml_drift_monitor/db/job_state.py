"""
SQLite-backed job state. Persists window_id, status, timestamps, decision so that
idempotency can be enforced and state survives process restart.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional

from ml_drift_monitor.config import ProjectConfig, get_default_config


_TABLE = """
CREATE TABLE IF NOT EXISTS job_runs (
    window_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    promotion_decision TEXT,
    champion_version INTEGER,
    challenger_version INTEGER,
    drift_flag INTEGER,
    retrain_triggered INTEGER,
    extra_json TEXT
)
"""


@contextmanager
def _conn(cfg: ProjectConfig | None = None) -> Iterator[sqlite3.Connection]:
    project_cfg = cfg or get_default_config()
    db_path = project_cfg.paths.job_state_db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_TABLE)
        conn.commit()
        yield conn
    finally:
        conn.close()


def set_job_started(
    window_id: str,
    cfg: ProjectConfig | None = None,
) -> None:
    with _conn(cfg) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO job_runs
            (window_id, status, started_at, completed_at, promotion_decision,
             champion_version, challenger_version, drift_flag, retrain_triggered, extra_json)
            VALUES (?, 'started', datetime('now'), NULL, NULL, NULL, NULL, NULL, NULL, NULL)
            """,
            (window_id,),
        )
        conn.commit()


def set_job_completed(
    window_id: str,
    status: str,
    promotion_decision: str,
    champion_version: Optional[int] = None,
    challenger_version: Optional[int] = None,
    drift_flag: bool = False,
    retrain_triggered: bool = False,
    extra: Optional[Dict[str, Any]] = None,
    cfg: ProjectConfig | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _conn(cfg) as conn:
        existing = conn.execute(
            "SELECT started_at FROM job_runs WHERE window_id = ?", (window_id,)
        ).fetchone()
        started = existing[0] if existing else now
        conn.execute(
            """
            INSERT OR REPLACE INTO job_runs
            (window_id, status, started_at, completed_at, promotion_decision,
             champion_version, challenger_version, drift_flag, retrain_triggered, extra_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                window_id,
                status,
                started,
                now,
                promotion_decision,
                champion_version,
                challenger_version,
                1 if drift_flag else 0,
                1 if retrain_triggered else 0,
                json.dumps(extra) if extra else None,
            ),
        )
        conn.commit()


def get_job_state(window_id: str, cfg: ProjectConfig | None = None) -> Optional[Dict[str, Any]]:
    with _conn(cfg) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM job_runs WHERE window_id = ?", (window_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)


def has_retrain_run_for_window(window_id: str, cfg: ProjectConfig | None = None) -> bool:
    """True if this window has already had a retrain triggered (completed)."""
    state = get_job_state(window_id, cfg)
    if state is None:
        return False
    return bool(state.get("retrain_triggered"))
