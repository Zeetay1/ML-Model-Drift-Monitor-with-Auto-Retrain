from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional

import mlflow

from ml_drift_monitor.config import ProjectConfig, get_default_config


def configure_mlflow(cfg: ProjectConfig | None = None) -> None:
    project_cfg = cfg or get_default_config()
    tracking_dir = project_cfg.paths.mlflow_root
    tracking_dir.mkdir(parents=True, exist_ok=True)
    uri = f"file://{tracking_dir.resolve()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("drift_monitoring")


@contextmanager
def start_run(
    run_name: str,
    cfg: ProjectConfig | None = None,
    tags: Optional[Dict[str, str]] = None,
) -> Iterator[mlflow.ActiveRun]:
    configure_mlflow(cfg)
    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        yield run

