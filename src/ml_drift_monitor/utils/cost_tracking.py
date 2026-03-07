"""
Cost tracking for retrain runs. No LLM in this project; we record inference time
and estimated compute cost per retrain (from config constants). Metadata is attached
to result objects and persisted with job state / event log.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict

from ml_drift_monitor.config import CostTrackingConfig, ProjectConfig, get_default_config


@dataclass
class CostMetadata:
    inference_seconds: float
    estimated_compute_cost: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inference_seconds": self.inference_seconds,
            "estimated_compute_cost": self.estimated_compute_cost,
        }


def record_retrain_cost(
    inference_seconds: float,
    cfg: ProjectConfig | None = None,
) -> CostMetadata:
    """
    Build cost metadata for a retrain run using config pricing.
    Caller should pass measured inference_seconds (e.g. from timing the evaluation step).
    """
    project_cfg = cfg or get_default_config()
    cost_cfg: CostTrackingConfig = project_cfg.cost_tracking
    # Use fixed cost per retrain as specified in config.
    estimated = cost_cfg.estimated_compute_cost_per_retrain
    return CostMetadata(inference_seconds=inference_seconds, estimated_compute_cost=estimated)


def timed_retrain_section(fn: Callable[[], Any], cfg: ProjectConfig | None = None) -> tuple[Any, CostMetadata]:
    """
    Run fn(), measure elapsed time, return (result, cost_metadata).
    Use for the retrain/evaluation section so we get non-zero inference_seconds.
    """
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    meta = record_retrain_cost(elapsed, cfg)
    return result, meta
