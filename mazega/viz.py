"""Matplotlib rendering for the CLI, notebooks and the README figures."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

import numpy as np

from .ga import GAResult
from .genome import prune_loops
from .maze import Coord, Maze
from .search import astar

WALL_COLOUR = "#1f2733"
FREE_COLOUR = "#f2f5f9"
PATH_COLOUR = "#ff5d5d"
OPTIMAL_COLOUR = "#3ba7ff"
START_COLOUR = "#1ecb7b"
GOAL_COLOUR = "#ffb020"


def use_dark_theme() -> None:
    matplotlib.rcParams.update(
        {
            "figure.facecolor": "#0e1117",
            "axes.facecolor": "#0e1117",
            "savefig.facecolor": "#0e1117",
            "text.color": "#e6edf3",
            "axes.labelcolor": "#e6edf3",
            "axes.edgecolor": "#30363d",
            "xtick.color": "#8b949e",
            "ytick.color": "#8b949e",
            "grid.color": "#21262d",
            "legend.facecolor": "#161b22",
            "legend.edgecolor": "#30363d",
            "font.size": 10,
        }
    )


def _grid_axes(ax, maze: Maze) -> None:
    from matplotlib.colors import ListedColormap

    ax.imshow(maze.grid, cmap=ListedColormap([FREE_COLOUR, WALL_COLOUR]), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_maze(
    maze: Maze,
    path: Sequence[Coord] | None = None,
    *,
    optimal: Sequence[Coord] | None = None,
    ax=None,
    title: str = "",
):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    _grid_axes(ax, maze)

    if optimal:
        ax.plot(
            [c for _, c in optimal],
            [r for r, _ in optimal],
            color=OPTIMAL_COLOUR,
            lw=4,
            alpha=0.45,
            solid_capstyle="round",
            label="A* optimal",
        )
    if path:
        ax.plot(
            [c for _, c in path],
            [r for r, _ in path],
            color=PATH_COLOUR,
            lw=2.2,
            solid_capstyle="round",
            label="GA best",
        )
    ax.scatter(*maze.start[::-1], s=110, marker="o", color=START_COLOUR, zorder=5, label="start")
    ax.scatter(*maze.goal[::-1], s=150, marker="*", color=GOAL_COLOUR, zorder=5, label="goal")
    if title:
        ax.set_title(title)
    return ax


def plot_fitness(result: GAResult, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.4))
    gens = [g.index for g in result.history]
    ax.plot(gens, [g.best for g in result.history], color=PATH_COLOUR, lw=2, label="best")
    ax.plot(gens, [g.mean for g in result.history], color=OPTIMAL_COLOUR, lw=1.6, label="mean")
    ax.fill_between(
        gens,
        [g.mean - g.std for g in result.history],
        [g.mean + g.std for g in result.history],
        color=OPTIMAL_COLOUR,
        alpha=0.15,
        linewidth=0,
    )
    solved = result.generation_solved
    if solved is not None:
        ax.axvline(solved, color=START_COLOUR, ls="--", lw=1.2, label=f"solved @ {solved}")
    for gen in (g.index for g in result.history if g.event == "catastrophe restart"):
        ax.axvline(gen, color=GOAL_COLOUR, ls=":", lw=0.9, alpha=0.7)
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness")
    ax.set_title("Fitness convergence")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="lower right", fontsize=8)
    return ax


def plot_diversity(result: GAResult, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.4))
    gens = [g.index for g in result.history]
    ax.plot(gens, [g.diversity for g in result.history], color="#c792ea", lw=2, label="gene entropy")
    ax.plot(gens, [g.success_rate for g in result.history], color=START_COLOUR, lw=1.6, label="share solving")
    twin = ax.twinx()
    twin.plot(gens, [g.mutation_rate for g in result.history], color=GOAL_COLOUR, lw=1.2, ls="--", label="mutation rate")
    twin.set_ylabel("mutation rate")
    twin.tick_params(colors="#8b949e")
    ax.set_xlabel("generation")
    ax.set_ylabel("normalised value")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Diversity, success rate and adaptive mutation")
    ax.grid(alpha=0.25, linewidth=0.6)
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="center right", fontsize=8)
    return ax


def overview_figure(result: GAResult):
    """Three-panel summary: solved maze, convergence, population health."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12.5, 4.6), constrained_layout=True)
    spec = fig.add_gridspec(2, 2, width_ratios=[1, 1.45])
    ax_maze = fig.add_subplot(spec[:, 0])
    plot_maze(
        result.maze,
        result.simple_path,
        optimal=astar(result.maze).path,
        ax=ax_maze,
        title=f"{result.maze.rows}x{result.maze.cols} maze — {result.summary()['path_steps']} steps",
    )
    ax_maze.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=8, frameon=False)
    plot_fitness(result, fig.add_subplot(spec[0, 1]))
    plot_diversity(result, fig.add_subplot(spec[1, 1]))
    return fig


def save_animation(result: GAResult, path: str | Path, *, fps: int = 8, max_frames: int = 120) -> Path:
    """Write a GIF of the best path improving generation by generation."""
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    history = result.history
    step = max(1, len(history) // max_frames)
    frames = list(range(0, len(history), step))
    if frames[-1] != len(history) - 1:
        frames.append(len(history) - 1)

    fig, ax = plt.subplots(figsize=(4.6, 4.9))

    def draw(i: int) -> None:
        ax.clear()
        record = history[i]
        plot_maze(result.maze, prune_loops(record.best_path), ax=ax)
        ax.set_title(
            f"gen {record.index:>3} · fitness {record.best:.3f} · "
            f"{'solved' if record.solved else 'searching'}",
            fontsize=10,
        )

    anim = animation.FuncAnimation(fig, draw, frames=frames, interval=1000 // fps)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out


def plot_benchmark(rows: Sequence[dict], ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 3.6))
    names = [r["variant"] for r in rows]
    y = np.arange(len(names))
    ax.barh(y, [r["solve_rate"] for r in rows], color=OPTIMAL_COLOUR, height=0.55)
    for i, row in enumerate(rows):
        label = f"{row['solve_rate']:.0%}"
        if row["mean_optimality"] is not None:
            label += f"  ·  optimality {row['mean_optimality']:.2f}"
        inside = row["solve_rate"] > 0.25
        ax.text(
            row["solve_rate"] - 0.015 if inside else row["solve_rate"] + 0.015,
            i,
            label,
            va="center",
            ha="right" if inside else "left",
            fontsize=9,
            color="#0e1117" if inside else "#e6edf3",
            fontweight="bold",
        )
    ax.set_yticks(y, names, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("solve rate across seeds")
    ax.set_title("Ablation: what each mechanism is worth")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.invert_yaxis()
    return ax
