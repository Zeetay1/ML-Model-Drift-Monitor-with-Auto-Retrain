from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ml_drift_monitor.config import ProjectConfig, get_default_config


def load_drift_reports(cfg: ProjectConfig | None = None) -> Dict[int, Dict]:
    project_cfg = cfg or get_default_config()
    reports: Dict[int, Dict] = {}
    if not project_cfg.paths.drift_reports_dir.exists():
        return reports
    for path in sorted(project_cfg.paths.drift_reports_dir.glob("month_*_report.json")):
        month_str = path.stem.split("_")[1]
        month = int(month_str)
        data = json.loads(path.read_text(encoding="utf-8"))
        # pd.Series(report_dict).to_json() saves as {"0": report_dict}; use report dict.
        if isinstance(data, dict) and "0" in data and "metrics" in data.get("0", {}):
            data = data["0"]
        reports[month] = data
    return reports


def load_retrain_events(cfg: ProjectConfig | None = None) -> pd.DataFrame:
    from ml_drift_monitor.tracking.event_log import get_all_events

    events = get_all_events(cfg)
    if not events:
        return pd.DataFrame()
    return pd.DataFrame([e.__dict__ for e in events])


def load_model_metadata(cfg: ProjectConfig | None = None) -> pd.DataFrame:
    project_cfg = cfg or get_default_config()
    rows: List[Dict] = []
    for path in project_cfg.paths.models_dir.glob("*_meta.json"):
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["file"] = path.name
        rows.append(meta)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

