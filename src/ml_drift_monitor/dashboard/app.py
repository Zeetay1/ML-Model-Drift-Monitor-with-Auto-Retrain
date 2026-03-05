from __future__ import annotations

from pathlib import Path

import plotly.io as pio

from ml_drift_monitor.config import get_default_config
from ml_drift_monitor.dashboard.data_access import (
    load_drift_reports,
    load_model_metadata,
    load_retrain_events,
)
from ml_drift_monitor.dashboard.plots import (
    drift_trend_figure,
    model_version_timeline_figure,
    retrain_events_overlay_figure,
)


def main() -> None:
    """
    Simple CLI entrypoint to render dashboard plots to HTML files using Plotly.

    This keeps dependencies local and avoids running a web server while still
    providing a visual overview driven entirely from logged history.
    """
    cfg = get_default_config()
    drift_reports = load_drift_reports(cfg)
    events = load_retrain_events(cfg)

    if not drift_reports:
        print("No drift reports found under artifacts/drift_reports/. Run the pipeline first.")
        return

    # Use the default feature drift threshold from config.
    threshold = cfg.drift_thresholds.feature_drift_score_threshold
    # Render trend for a couple of key features if present.
    example_features = ["income", "tenure", "transactions_last_month"]

    output_dir = cfg.paths.artifacts_root / "dashboard"
    output_dir.mkdir(parents=True, exist_ok=True)

    for feature in example_features:
        fig = drift_trend_figure(drift_reports, feature, threshold)
        out_path = output_dir / f"drift_trend_{feature}.html"
        pio.write_html(fig, file=str(out_path), auto_open=False)

    version_fig = model_version_timeline_figure(events)
    pio.write_html(version_fig, file=str(output_dir / "model_versions.html"), auto_open=False)

    events_fig = retrain_events_overlay_figure(events)
    pio.write_html(events_fig, file=str(output_dir / "retrain_events.html"), auto_open=False)

    print(f"Dashboard HTML files written to {output_dir}")


if __name__ == "__main__":
    main()

