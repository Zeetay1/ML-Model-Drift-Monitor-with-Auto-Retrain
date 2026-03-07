"""
FastAPI app serving minimal HTML/JS dashboard. No Streamlit.
Plotly figures embedded as HTML fragments. Single command: uvicorn ml_drift_monitor.dashboard.server:app
"""

from __future__ import annotations

import json
from typing import List

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from ml_drift_monitor.config import get_default_config
from ml_drift_monitor.dashboard.data_access import (
    load_drift_reports,
    load_retrain_events,
)
from ml_drift_monitor.dashboard.plots import (
    drift_trend_figure,
    model_version_timeline_figure,
    retrain_events_overlay_figure,
)

app = FastAPI(title="ML Drift Monitor Dashboard")


def _fig_to_script(div_id: str, fig) -> str:
    j = fig.to_plotly_json()
    data_js = json.dumps(j["data"])
    layout_js = json.dumps(j["layout"])
    return f'<div id="{div_id}"></div><script>Plotly.newPlot("{div_id}", {data_js}, {layout_js});</script>'


def _layout_html(title: str, body_fragments: List[str]) -> str:
    fragments = "\n".join(body_fragments)
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
  <h1>{title}</h1>
  {fragments}
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def dashboard_root() -> HTMLResponse:
    """Serve dashboard index: drift trends, model versions, retrain events."""
    cfg = get_default_config()
    drift_reports = load_drift_reports(cfg)
    events = load_retrain_events(cfg)

    fragments: List[str] = []
    threshold = cfg.drift_thresholds.feature_drift_score_threshold
    for feature in ["income", "tenure", "transactions_last_month"]:
        fig = drift_trend_figure(drift_reports, feature, threshold)
        fragments.append(_fig_to_script(f"drift-{feature}", fig))

    if not events.empty:
        fig_v = model_version_timeline_figure(events)
        fragments.append(_fig_to_script("model-versions", fig_v))
        fig_e = retrain_events_overlay_figure(events)
        fragments.append(_fig_to_script("retrain-events", fig_e))

    if not fragments:
        fragments.append("<p>No drift reports or events yet. Run the pipeline first.</p>")

    html = _layout_html("ML Drift Monitor Dashboard", fragments)
    return HTMLResponse(html)


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)
