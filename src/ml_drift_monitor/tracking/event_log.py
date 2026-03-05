from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ml_drift_monitor.config import ProjectConfig, get_default_config
from ml_drift_monitor.logging_utils.logger import get_logger


logger = get_logger()


@dataclass
class RetrainEvent:
    window_id: str
    drift_flag: bool
    retrain_triggered: bool
    promotion_decision: str
    champion_metrics: Dict[str, Any]
    challenger_metrics: Optional[Dict[str, Any]]
    model_versions: Dict[str, Any]
    timestamp: str


def _events_path(cfg: ProjectConfig) -> Path:
    cfg.paths.logs_dir.mkdir(parents=True, exist_ok=True)
    return cfg.paths.logs_dir / "retrain_events.jsonl"


def log_retrain_event(event: RetrainEvent, cfg: ProjectConfig | None = None) -> None:
    project_cfg = cfg or get_default_config()
    path = _events_path(project_cfg)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(event)) + "\n")
    logger.info("Logged retrain event for window %s", event.window_id)


def get_all_events(cfg: ProjectConfig | None = None) -> List[RetrainEvent]:
    project_cfg = cfg or get_default_config()
    path = _events_path(project_cfg)
    events: List[RetrainEvent] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            events.append(RetrainEvent(**data))
    return events


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

