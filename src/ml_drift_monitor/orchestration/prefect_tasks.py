from __future__ import annotations

import time
from typing import Dict, Tuple

import pandas as pd
from prefect import task

from ml_drift_monitor.config import ProjectConfig, get_default_config
from ml_drift_monitor.data.generator import generate_feature_schema
from ml_drift_monitor.data.storage import load_month_batch
from ml_drift_monitor.db.job_state import set_job_completed
from ml_drift_monitor.logging_utils.logger import get_logger
from ml_drift_monitor.models.baseline import ensure_initial_champion
from ml_drift_monitor.models.evaluation import compute_classification_metrics
from ml_drift_monitor.models.registry import save_challenger, save_new_champion_version
from ml_drift_monitor.models.training import predict_with_proba, train_logistic_regression
from ml_drift_monitor.monitoring.evidently_drift import run_evidently_drift
from ml_drift_monitor.orchestration.state_tracking import has_retrain_run_for_window
from ml_drift_monitor.tracking.event_log import RetrainEvent, log_retrain_event, utc_now_iso
from ml_drift_monitor.tracking.mlflow_utils import start_run
from ml_drift_monitor.utils.cost_tracking import record_retrain_cost


logger = get_logger()


@task
def load_config_task() -> ProjectConfig:
    return get_default_config()


@task
def ensure_champion_task(cfg: ProjectConfig):
    return ensure_initial_champion(cfg)


@task
def load_reference_data_task(cfg: ProjectConfig) -> pd.DataFrame:
    # Reference is months 1–2 concatenated from stored CSVs.
    df1 = load_month_batch(1, cfg)
    df2 = load_month_batch(2, cfg)
    return pd.concat([df1, df2], ignore_index=True)


@task
def load_current_batch_task(month: int, cfg: ProjectConfig) -> pd.DataFrame:
    return load_month_batch(month, cfg)


@task
def add_predictions_to_batch_task(df: pd.DataFrame, model) -> pd.DataFrame:
    """Add 'prediction' column (positive-class probability) for Evidently prediction drift."""
    X = df.drop(columns=["label"], errors="ignore")
    _, proba = predict_with_proba(model, X)
    out = df.copy()
    out["prediction"] = proba[:, 1] if proba.ndim == 2 else proba
    return out


@task
def run_drift_detection_task(
    month: int,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    cfg: ProjectConfig,
):
    schema = generate_feature_schema(cfg.data_generation)
    return run_evidently_drift(month, reference, current, schema, cfg)


@task
def check_retrain_already_run_task(window_id: str, cfg: ProjectConfig) -> bool:
    return has_retrain_run_for_window(window_id, cfg)


@task
def train_challenger_task(cfg: ProjectConfig, up_to_month: int):
    # Train challenger on all data up to and including the current month.
    dfs = [load_month_batch(m, cfg) for m in cfg.data_generation.months if m <= up_to_month]
    full_df = pd.concat(dfs, ignore_index=True)
    X = full_df.drop(columns=["label"])
    y = full_df["label"].values
    model = train_logistic_regression(X, y, random_state=cfg.data_generation.random_seed + up_to_month)
    return model


@task
def evaluate_models_task(
    month: int,
    cfg: ProjectConfig,
    champion_model,
    challenger_model,
) -> Tuple[Dict, Dict]:
    df = load_month_batch(month, cfg)
    X = df.drop(columns=["label"])
    y = df["label"].values

    champ_preds, champ_proba = predict_with_proba(champion_model, X)
    chall_preds, chall_proba = predict_with_proba(challenger_model, X)

    champ_metrics = compute_classification_metrics(y, champ_preds, champ_proba)
    chall_metrics = compute_classification_metrics(y, chall_preds, chall_proba)
    return champ_metrics, chall_metrics


@task
def decide_and_persist_task(
    window_id: str,
    month: int,
    cfg: ProjectConfig,
    drift_report,
    champion_model,
    champion_meta,
    challenger_model,
    champion_metrics: Dict,
    challenger_metrics: Dict,
):
    from ml_drift_monitor.config import PromotionConfig

    start_time = time.perf_counter()
    prom_cfg: PromotionConfig = cfg.promotion
    metric_name = prom_cfg.metric_name
    champ_score = champion_metrics.get(metric_name, float("nan"))
    chall_score = challenger_metrics.get(metric_name, float("nan"))

    retrain_triggered = True
    decision: str

    if (
        chall_score is not None
        and champ_score is not None
        and chall_score >= champ_score + prom_cfg.min_relative_improvement
    ):
        decision = "promoted"
        challenger_meta = {
            "training_window": champion_meta.get("training_window", [1, month]),
            "metrics": challenger_metrics,
            "promotion_decision": "promoted",
            "trained_on_window": window_id,
        }
        challenger_version = save_challenger(challenger_model, challenger_meta, cfg)
        new_champion_version = save_new_champion_version(challenger_model, challenger_meta, cfg)
        model_versions = {
            "champion_before": champion_meta.get("version"),
            "champion_after": new_champion_version,
            "challenger": challenger_version,
        }
    else:
        decision = "rejected"
        challenger_meta = {
            "training_window": [1, month],
            "metrics": challenger_metrics,
            "promotion_decision": "rejected",
            "trained_on_window": window_id,
        }
        challenger_version = save_challenger(challenger_model, challenger_meta, cfg)
        model_versions = {
            "champion_before": champion_meta.get("version"),
            "champion_after": champion_meta.get("version"),
            "challenger": challenger_version,
        }

    inference_seconds = time.perf_counter() - start_time
    cost_meta = record_retrain_cost(inference_seconds, cfg)
    cost_metadata = cost_meta.to_dict()

    event = RetrainEvent(
        window_id=window_id,
        drift_flag=drift_report.overall_drift_flag,
        retrain_triggered=retrain_triggered,
        promotion_decision=decision,
        champion_metrics=champion_metrics,
        challenger_metrics=challenger_metrics,
        model_versions=model_versions,
        timestamp=utc_now_iso(),
        cost_metadata=cost_metadata,
    )
    log_retrain_event(event, cfg)

    set_job_completed(
        window_id,
        status="completed",
        promotion_decision=decision,
        champion_version=model_versions.get("champion_after") or champion_meta.get("version"),
        challenger_version=model_versions.get("challenger"),
        drift_flag=drift_report.overall_drift_flag,
        retrain_triggered=True,
        extra={"cost_metadata": cost_metadata},
        cfg=cfg,
    )

    # Log to MLflow as well.
    with start_run(run_name=f"retrain_{window_id}", cfg=cfg, tags={"window_id": window_id}) as run:
        import mlflow

        mlflow.log_metric(f"champion_{metric_name}", champ_score)
        mlflow.log_metric(f"challenger_{metric_name}", chall_score)
        mlflow.log_params(
            {
                "promotion_decision": decision,
                "window_id": window_id,
            }
        )

    return decision

