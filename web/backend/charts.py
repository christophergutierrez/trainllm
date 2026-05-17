"""Plotly figure construction — backend builds complete specs, frontend just renders."""

from typing import Any

import plotly.graph_objects as go


def loss_curve(
    steps: list[int],
    train_loss: list[float],
    eval_loss: list[float] | None = None,
    eval_steps: list[int] | None = None,
) -> dict[str, Any]:
    """Build a loss curve figure spec."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=train_loss,
        mode="lines", name="Train Loss",
        line=dict(color="#3b82f6", width=2),
    ))
    if eval_loss and eval_steps:
        fig.add_trace(go.Scatter(
            x=eval_steps, y=eval_loss,
            mode="lines+markers", name="Eval Loss",
            line=dict(color="#ef4444", width=2, dash="dot"),
            marker=dict(size=6),
        ))
    fig.update_layout(
        title=None,
        xaxis_title="Step",
        yaxis_title="Loss",
        template="plotly_dark",
        margin=dict(l=50, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=300,
    )
    return fig.to_dict()


def lr_schedule(steps: list[int], lrs: list[float]) -> dict[str, Any]:
    """Build a learning rate schedule figure spec."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=lrs,
        mode="lines", name="Learning Rate",
        line=dict(color="#10b981", width=2),
        fill="tozeroy",
        fillcolor="rgba(16, 185, 129, 0.1)",
    ))
    fig.update_layout(
        title=None,
        xaxis_title="Step",
        yaxis_title="LR",
        template="plotly_dark",
        margin=dict(l=50, r=20, t=20, b=40),
        height=300,
    )
    return fig.to_dict()


def score_distribution(scores: list[float], title: str = "Score Distribution") -> dict[str, Any]:
    """Build a histogram of scores."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=20,
        marker_color="#8b5cf6",
        opacity=0.8,
    ))
    fig.update_layout(
        title=None,
        xaxis_title="Score",
        yaxis_title="Count",
        template="plotly_dark",
        margin=dict(l=50, r=20, t=20, b=40),
        height=300,
    )
    return fig.to_dict()


def band_donut(band_counts: dict[str, int]) -> dict[str, Any]:
    """Build a donut chart of eval bands."""
    colors = {
        "EXCELLENT": "#10b981",
        "GOOD": "#3b82f6",
        "PARTIAL": "#f59e0b",
        "POOR": "#ef4444",
        "ERROR": "#6b7280",
    }
    labels = list(band_counts.keys())
    values = list(band_counts.values())
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker=dict(colors=[colors.get(l, "#6b7280") for l in labels]),
        textinfo="label+value",
    )])
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20),
        height=250,
        showlegend=False,
    )
    return fig.to_dict()


def timing_bars(step_names: list[str], durations_sec: list[float]) -> dict[str, Any]:
    """Build a horizontal bar chart of step durations."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=step_names,
        x=durations_sec,
        orientation="h",
        marker_color="#6366f1",
    ))
    fig.update_layout(
        title=None,
        xaxis_title="Duration (seconds)",
        template="plotly_dark",
        margin=dict(l=100, r=20, t=20, b=40),
        height=max(200, len(step_names) * 40),
    )
    return fig.to_dict()


def composite_trend(
    run_dates: list[str],
    composite_scores: list[float],
    similarity_scores: list[float] | None = None,
) -> dict[str, Any]:
    """Build a trend line of composite scores across runs."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=run_dates, y=composite_scores,
        mode="lines+markers", name="Composite",
        line=dict(color="#8b5cf6", width=3),
        marker=dict(size=8),
    ))
    if similarity_scores:
        fig.add_trace(go.Scatter(
            x=run_dates, y=similarity_scores,
            mode="lines+markers", name="Similarity",
            line=dict(color="#6b7280", width=2, dash="dot"),
            marker=dict(size=5),
        ))
    fig.update_layout(
        title=None,
        xaxis_title="Run",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        template="plotly_dark",
        margin=dict(l=50, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=300,
    )
    return fig.to_dict()
