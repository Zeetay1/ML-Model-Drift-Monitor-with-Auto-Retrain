from __future__ import annotations

from prefect import flow

from ml_drift_monitor.logging_utils.logger import get_logger
from ml_drift_monitor.orchestration.prefect_tasks import (
    check_retrain_already_run_task,
    decide_and_persist_task,
    ensure_champion_task,
    evaluate_models_task,
    load_config_task,
    load_current_batch_task,
    load_reference_data_task,
    run_drift_detection_task,
    train_challenger_task,
)
from ml_drift_monitor.tracking.event_log import RetrainEvent, log_retrain_event, utc_now_iso


logger = get_logger()


@flow(name="monitor_and_retrain_flow")
def monitor_and_retrain_flow(month: int) -> str:
    """
    Orchestrated flow:
    - Detect drift for the given month.
    - If no drift: log and return.
    - If drift and retrain not yet run for this window: train challenger, evaluate, decide, and promote/reject.
    - If retrain already run for this window: skip retrain and log the reason.
    """
    cfg = load_config_task()
    window_id = f"month_{month}"

    # Ensure a champion exists and training data / artifacts are generated.
    champion_model, champion_meta = ensure_champion_task(cfg)

    reference = load_reference_data_task(cfg)
    current = load_current_batch_task(month, cfg)

    drift_report = run_drift_detection_task(month, reference, current, cfg)
    if not drift_report.overall_drift_flag:
        event = RetrainEvent(
            window_id=window_id,
            drift_flag=False,
            retrain_triggered=False,
            promotion_decision="no_drift",
            champion_metrics={},
            challenger_metrics=None,
            model_versions={},
            timestamp=utc_now_iso(),
        )
        log_retrain_event(event, cfg)
        logger.info("No drift detected for %s. Skipping retrain.", window_id)
        return "no_drift"

    already_run = check_retrain_already_run_task(window_id, cfg)
    if already_run:
        event = RetrainEvent(
            window_id=window_id,
            drift_flag=True,
            retrain_triggered=False,
            promotion_decision="skipped_already_processed",
            champion_metrics={},
            challenger_metrics=None,
            model_versions={},
            timestamp=utc_now_iso(),
        )
        log_retrain_event(event, cfg)
        logger.info("Retrain already processed for %s. Skipping.", window_id)
        return "skipped_already_processed"

    challenger_model = train_challenger_task(cfg, up_to_month=month)
    champion_metrics, challenger_metrics = evaluate_models_task(
        month, cfg, champion_model, challenger_model
    )
    decision = decide_and_persist_task(
        window_id,
        month,
        cfg,
        drift_report,
        champion_model,
        champion_meta or {},
        challenger_model,
        champion_metrics,
        challenger_metrics,
    )
    logger.info("Completed monitor_and_retrain_flow for %s with decision %s", window_id, decision)
    return decision

