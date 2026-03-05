from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib

from ml_drift_monitor.config import ProjectConfig, get_default_config
from ml_drift_monitor.logging_utils.logger import get_logger


logger = get_logger()


def _model_files(cfg: ProjectConfig, prefix: str) -> Dict[int, Path]:
    cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"{prefix}_v(\d+)\.pkl$")
    versions: Dict[int, Path] = {}
    for path in cfg.paths.models_dir.glob(f"{prefix}_v*.pkl"):
        m = pattern.match(path.name)
        if m:
            versions[int(m.group(1))] = path
    return versions


def _latest_version(versions: Dict[int, Path]) -> Optional[Tuple[int, Path]]:
    if not versions:
        return None
    v = max(versions.keys())
    return v, versions[v]


def _meta_path_for(prefix: str, version: int, cfg: ProjectConfig) -> Path:
    return cfg.paths.models_dir / f"{prefix}_v{version}_meta.json"


def get_current_champion(
    cfg: ProjectConfig | None = None,
) -> Tuple[Optional[object], Optional[Dict[str, Any]]]:
    project_cfg = cfg or get_default_config()
    versions = _model_files(project_cfg, "champion")
    latest = _latest_version(versions)
    if latest is None:
        return None, None
    version, path = latest
    model = joblib.load(path)
    meta_path = _meta_path_for("champion", version, project_cfg)
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["version"] = version
    return model, meta


def save_new_champion_version(model: object, meta: Dict[str, Any], cfg: ProjectConfig | None = None) -> int:
    project_cfg = cfg or get_default_config()
    versions = _model_files(project_cfg, "champion")
    latest = _latest_version(versions)
    next_version = (latest[0] + 1) if latest is not None else 1
    project_cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = project_cfg.paths.models_dir / f"champion_v{next_version}.pkl"
    joblib.dump(model, model_path)
    meta_path = _meta_path_for("champion", next_version, project_cfg)
    meta_with_version = dict(meta)
    meta_with_version["version"] = next_version
    meta_path.write_text(json.dumps(meta_with_version, indent=2), encoding="utf-8")
    logger.info("Saved champion model version %d to %s", next_version, model_path)
    return next_version


def save_challenger(model: object, meta: Dict[str, Any], cfg: ProjectConfig | None = None) -> int:
    project_cfg = cfg or get_default_config()
    versions = _model_files(project_cfg, "challenger")
    latest = _latest_version(versions)
    next_version = (latest[0] + 1) if latest is not None else 1
    project_cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = project_cfg.paths.models_dir / f"challenger_v{next_version}.pkl"
    joblib.dump(model, model_path)
    meta_path = _meta_path_for("challenger", next_version, project_cfg)
    meta_with_version = dict(meta)
    meta_with_version["version"] = next_version
    meta_path.write_text(json.dumps(meta_with_version, indent=2), encoding="utf-8")
    logger.info("Saved challenger model version %d to %s", next_version, model_path)
    return next_version

