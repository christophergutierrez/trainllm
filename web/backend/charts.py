"""Plotly figure construction — backend builds complete specs, frontend just renders."""

from typing import Any

import plotly.graph_objects as go


def loss_curve(
    steps: list[int],
    train_loss: list[float],
    eval_loss: list[float] | None = None,
    eval_steps: list[int] | None = None,
) -> dict[str, Any]:
    """Build a loss curve figure spec (linear by default; frontend toggles log)."""
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


COMPARISON_COLORS = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]


def loss_comparison(models_data: list[dict]) -> dict[str, Any]:
    """Overlaid loss curves for multiple models (log-scale y-axis)."""
    fig = go.Figure()
    for i, m in enumerate(models_data):
        history = m.get("loss_history")
        if not history:
            continue
        steps = [p[0] for p in history]
        losses = [p[1] for p in history]
        fig.add_trace(go.Scatter(
            x=steps, y=losses,
            mode="lines", name=m["model_name"],
            line=dict(color=COMPARISON_COLORS[i % len(COMPARISON_COLORS)], width=2),
        ))
    fig.update_layout(
        xaxis_title="Step",
        yaxis_title="Loss",
        yaxis_type="log",
        template="plotly_dark",
        margin=dict(l=50, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=320,
    )
    return fig.to_dict()


def score_comparison(models_data: list[dict]) -> dict[str, Any]:
    """Bar chart comparing composite scores across models."""
    names = [m["model_name"] for m in models_data]
    scores = [m.get("score") or 0 for m in models_data]
    colors = [COMPARISON_COLORS[i % len(COMPARISON_COLORS)] for i in range(len(names))]

    fig = go.Figure(data=[go.Bar(
        x=names, y=scores,
        marker_color=colors,
        text=[f"{s:.4f}" for s in scores],
        textposition="outside",
    )])
    fig.update_layout(
        yaxis_title="Composite Score",
        yaxis_range=[0, 1.05],
        template="plotly_dark",
        margin=dict(l=50, r=20, t=20, b=40),
        height=280,
    )
    return fig.to_dict()


def band_comparison(models_data: list[dict]) -> dict[str, Any]:
    """Stacked bar chart of band distributions per model."""
    band_order = ["EXCELLENT", "GOOD", "PARTIAL", "POOR", "ERROR"]
    band_colors = {
        "EXCELLENT": "#10b981",
        "GOOD": "#3b82f6",
        "PARTIAL": "#f59e0b",
        "POOR": "#ef4444",
        "ERROR": "#6b7280",
    }
    names = [m["model_name"] for m in models_data]

    fig = go.Figure()
    for band in band_order:
        counts = [m.get("band_counts", {}).get(band, 0) for m in models_data]
        if sum(counts) == 0:
            continue
        fig.add_trace(go.Bar(
            x=names, y=counts,
            name=band,
            marker_color=band_colors.get(band, "#6b7280"),
        ))
    fig.update_layout(
        barmode="stack",
        yaxis_title="Records",
        template="plotly_dark",
        margin=dict(l=50, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=280,
    )
    return fig.to_dict()


def convention_comparison(models_data: list[dict]) -> dict[str, Any]:
    """Grouped bar chart of per-convention scores across models."""
    all_conventions = {}
    for m in models_data:
        for c in m.get("convention_breakdown", []):
            all_conventions[c["convention"]] = True

    conv_names = sorted(all_conventions.keys())
    if not conv_names:
        return go.Figure().to_dict()

    fig = go.Figure()
    for i, m in enumerate(models_data):
        conv_map = {c["convention"]: c["avg"] for c in m.get("convention_breakdown", [])}
        scores = [conv_map.get(c, 0) for c in conv_names]
        fig.add_trace(go.Bar(
            x=conv_names, y=scores,
            name=m["model_name"],
            marker_color=COMPARISON_COLORS[i % len(COMPARISON_COLORS)],
        ))
    fig.update_layout(
        barmode="group",
        yaxis_title="Avg Score",
        yaxis_range=[0, 1.05],
        xaxis_tickangle=-30,
        template="plotly_dark",
        margin=dict(l=50, r=20, t=20, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=320,
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
