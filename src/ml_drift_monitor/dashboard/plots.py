from __future__ import annotations

from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go


def drift_trend_figure(
    drift_reports: Dict[int, Dict],
    feature_name: str,
    threshold: float,
) -> go.Figure:
    months = sorted(drift_reports.keys())
    scores: List[float] = []
    for m in months:
        metric_result = None
        for metric in drift_reports[m].get("metrics", []):
            if "DataDrift" in metric.get("metric", ""):
                metric_result = metric.get("result", {})
                break
        if metric_result is None:
            scores.append(0.0)
            continue
        col_info = metric_result.get("drift_by_columns", {}).get(feature_name, {})
        scores.append(float(col_info.get("drift_score", 0.0)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=scores, mode="lines+markers", name=feature_name))
    fig.add_hline(y=threshold, line_dash="dash", annotation_text="Threshold", annotation_position="top left")
    fig.update_layout(
        title=f"Drift score over time for {feature_name}",
        xaxis_title="Month",
        yaxis_title="Drift score",
    )
    return fig


def model_version_timeline_figure(events: pd.DataFrame) -> go.Figure:
    if events.empty:
        return go.Figure()

    df = events.copy()
    df["month"] = df["window_id"].str.extract(r"month_(\d+)").astype(int)
    df = df.sort_values("month")
    versions = df["model_versions"].apply(lambda mv: mv.get("champion_after") or mv.get("champion_before"))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["month"],
            y=versions,
            mode="lines+markers",
            name="Champion version",
        )
    )
    fig.update_layout(
        title="Champion model version over time",
        xaxis_title="Month",
        yaxis_title="Champion version",
    )
    return fig


def retrain_events_overlay_figure(events: pd.DataFrame) -> go.Figure:
    if events.empty:
        return go.Figure()

    df = events.copy()
    df["month"] = df["window_id"].str.extract(r"month_(\d+)").astype(int)

    fig = go.Figure()
    for decision, color in [("promoted", "green"), ("rejected", "red")]:
        sub = df[df["promotion_decision"] == decision]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["month"],
                y=[1] * len(sub),
                mode="markers",
                name=decision,
                marker=dict(color=color, size=10),
            )
        )
    fig.update_layout(
        title="Retrain events",
        xaxis_title="Month",
        yaxis_visible=False,
        yaxis_showticklabels=False,
    )
    return fig

