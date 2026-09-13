"""Interactive Plotly figures for the Streamlit front end."""

from __future__ import annotations

from typing import Sequence

import plotly.graph_objects as go

from .ga import Generation
from .maze import Coord, Maze

BG = "#0e1117"
GRID_FREE = "#eef2f7"
GRID_WALL = "#252d3a"
GA_PATH = "#ff5d5d"
OPTIMAL = "#3ba7ff"
START = "#1ecb7b"
GOAL = "#ffb020"
MUTED = "#8b949e"
ACCENT = "#c792ea"

_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(color="#e6edf3", size=12),
)
_MAZE_LAYOUT = dict(
    **_BASE,
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.04, x=0.5, xanchor="center"),
)
_CHART_LAYOUT = dict(
    **_BASE,
    margin=dict(l=64, r=64, t=64, b=52),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.02, yanchor="bottom", x=1,
                xanchor="right"),
)


def _axis(**extra) -> dict:
    return dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, **extra)


def maze_figure(
    maze: Maze,
    path: Sequence[Coord] | None = None,
    *,
    optimal: Sequence[Coord] | None = None,
    title: str = "",
    height: int = 520,
    clickable: bool = False,
) -> go.Figure:
    """Render the maze, optionally with the GA path and the exact optimum."""
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=maze.grid,
            colorscale=[[0.0, GRID_FREE], [1.0, GRID_WALL]],
            zmin=0,
            zmax=1,
            showscale=False,
            xgap=1,
            ygap=1,
            hoverinfo="skip",
        )
    )

    if optimal:
        fig.add_trace(
            go.Scatter(
                x=[c for _, c in optimal],
                y=[r for r, _ in optimal],
                mode="lines",
                line=dict(color=OPTIMAL, width=9),
                opacity=0.35,
                name=f"optimal · {len(optimal) - 1} steps",
                hoverinfo="skip",
            )
        )
    if path:
        fig.add_trace(
            go.Scatter(
                x=[c for _, c in path],
                y=[r for r, _ in path],
                mode="lines",
                line=dict(color=GA_PATH, width=4, shape="linear"),
                name=f"GA best · {len(path) - 1} steps",
                hovertemplate="row %{y}, col %{x}<extra></extra>",
            )
        )

    for cell, colour, symbol, label in (
        (maze.start, START, "circle", "start"),
        (maze.goal, GOAL, "star", "goal"),
    ):
        fig.add_trace(
            go.Scatter(
                x=[cell[1]],
                y=[cell[0]],
                mode="markers",
                marker=dict(color=colour, size=16, symbol=symbol, line=dict(color=BG, width=1.5)),
                name=label,
                hoverinfo="skip",
            )
        )

    if clickable:
        xs = [c for r in range(maze.rows) for c in range(maze.cols)]
        ys = [r for r in range(maze.rows) for _ in range(maze.cols)]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(size=max(6, int(360 / max(maze.rows, maze.cols))), opacity=0),
                showlegend=False,
                name="cells",
                hovertemplate="row %{y}, col %{x}<extra>click to edit</extra>",
            )
        )

    fig.update_layout(
        **_MAZE_LAYOUT,
        height=height,
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=14)),
        xaxis=_axis(range=[-0.5, maze.cols - 0.5]),
        yaxis=_axis(range=[maze.rows - 0.5, -0.5], scaleanchor="x", scaleratio=1),
        dragmode=False,
        clickmode="event+select" if clickable else "event",
    )
    return fig


def fitness_figure(history: Sequence[Generation], height: int = 300) -> go.Figure:
    gens = [g.index for g in history]
    upper = [g.mean + g.std for g in history]
    lower = [g.mean - g.std for g in history]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=gens + gens[::-1], y=upper + lower[::-1], fill="toself",
                   fillcolor="rgba(59,167,255,0.15)", line=dict(width=0),
                   hoverinfo="skip", showlegend=False)
    )
    fig.add_trace(go.Scatter(x=gens, y=[g.mean for g in history], name="population mean",
                             line=dict(color=OPTIMAL, width=2)))
    fig.add_trace(go.Scatter(x=gens, y=[g.best for g in history], name="best",
                             line=dict(color=GA_PATH, width=2.5)))

    solved = next((g.index for g in history if g.solved), None)
    if solved is not None:
        fig.add_vline(x=solved, line=dict(color=START, dash="dash", width=1.5),
                      annotation_text=f"solved @ {solved}", annotation_font_color=START)
    for gen in (g.index for g in history if g.event == "catastrophe restart"):
        fig.add_vline(x=gen, line=dict(color=GOAL, dash="dot", width=1))

    fig.update_layout(**_CHART_LAYOUT, height=height,
                      title=dict(text="Fitness convergence", x=0.01, y=0.97, font=dict(size=14)),
                      hovermode="x unified")
    fig.update_xaxes(title="generation", showgrid=True, gridcolor="#21262d")
    fig.update_yaxes(title="fitness", showgrid=True, gridcolor="#21262d")
    return fig


def diversity_figure(history: Sequence[Generation], height: int = 300) -> go.Figure:
    gens = [g.index for g in history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=[g.diversity for g in history], name="gene entropy",
                             line=dict(color=ACCENT, width=2)))
    fig.add_trace(go.Scatter(x=gens, y=[g.success_rate for g in history], name="share reaching goal",
                             line=dict(color=START, width=2)))
    fig.add_trace(go.Scatter(x=gens, y=[g.mutation_rate for g in history], name="mutation rate",
                             line=dict(color=GOAL, width=1.6, dash="dash"), yaxis="y2"))
    fig.update_layout(
        **_CHART_LAYOUT,
        height=height,
        title=dict(text="Population health", x=0.01, y=0.97, font=dict(size=14)),
        hovermode="x unified",
        yaxis2=dict(overlaying="y", side="right", showgrid=False, title="mutation rate",
                    tickfont=dict(color=MUTED)),
    )
    fig.update_xaxes(title="generation", showgrid=True, gridcolor="#21262d")
    fig.update_yaxes(title="normalised", range=[-0.02, 1.02], showgrid=True, gridcolor="#21262d")
    return fig


def benchmark_figure(rows: Sequence[dict], height: int = 340) -> go.Figure:
    names = [r["variant"] for r in rows]
    fig = go.Figure(
        go.Bar(
            x=[r["solve_rate"] for r in rows],
            y=names,
            orientation="h",
            marker=dict(color=[r["solve_rate"] for r in rows], colorscale="Teal", cmin=0, cmax=1),
            text=[f"{r['solve_rate']:.0%}" for r in rows],
            textposition="outside",
            hovertemplate="%{y}<br>solve rate %{x:.0%}<extra></extra>",
        )
    )
    fig.update_layout(**_CHART_LAYOUT, height=height, showlegend=False,
                      title=dict(text="Ablation — solve rate across seeds", x=0.01, y=0.97,
                                 font=dict(size=14)))
    fig.update_xaxes(range=[0, 1.12], tickformat=".0%", showgrid=True, gridcolor="#21262d")
    fig.update_yaxes(autorange="reversed")
    return fig


def contribution_figure(report, height: int = 240) -> go.Figure:
    """Waterfall of the fitness terms for the best individual."""
    terms = report.terms
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["relative"] * len(terms) + ["total"],
            x=list(terms) + ["score"],
            y=list(terms.values()) + [0],
            connector=dict(line=dict(color=MUTED, width=1)),
            increasing=dict(marker=dict(color=START)),
            decreasing=dict(marker=dict(color=GA_PATH)),
            totals=dict(marker=dict(color=OPTIMAL)),
        )
    )
    fig.update_layout(**_CHART_LAYOUT, height=height, showlegend=False,
                      title=dict(text="Where the best score comes from", x=0.01, y=0.97,
                                 font=dict(size=14)))
    fig.update_yaxes(showgrid=True, gridcolor="#21262d")
    return fig
