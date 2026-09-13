"""Regenerate every figure used in the README.

    python scripts/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from mazega import GAConfig, MazeGA, astar, generate, viz  # noqa: E402
from mazega.benchmark import DEFAULT_VARIANTS, compare  # noqa: E402

OUT = ROOT / "docs" / "images"
MAZE_KIND, MAZE_SIZE, MAZE_SEED, GA_SEED = "braided", 31, 1, 7


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    viz.use_dark_theme()
    import matplotlib.pyplot as plt

    from mazega import prune_loops

    maze = generate(MAZE_KIND, MAZE_SIZE, MAZE_SIZE, seed=MAZE_SEED, braid=0.4)
    exact = astar(maze)
    print(f"maze {maze.rows}x{maze.cols} · optimal {exact.steps} steps")

    result = MazeGA(maze, GAConfig(generations=300, seed=GA_SEED)).run()
    print("solve:", result.summary())

    figure = viz.overview_figure(result)
    figure.savefig(OUT / "overview.png", dpi=150)
    plt.close(figure)

    solved_at = result.generation_solved or result.generations_run - 1
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), constrained_layout=True)
    for ax, gen in zip(axes, (0, solved_at // 2, solved_at)):
        record = result.history[gen]
        viz.plot_maze(maze, prune_loops(record.best_path), ax=ax,
                      title=f"generation {record.index} · fitness {record.best:.2f}")
    fig.suptitle("The best individual, three snapshots apart", fontsize=13)
    fig.savefig(OUT / "evolution-stages.png", dpi=150)
    plt.close(fig)

    bench_maze = generate("recursive_backtracker", 25, 25, seed=2)
    _, rows = compare(bench_maze, DEFAULT_VARIANTS,
                      base=GAConfig(generations=150), seeds=range(6))
    for row in rows:
        print(row)
    ax = viz.plot_benchmark(rows)
    ax.figure.savefig(OUT / "ablation.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)

    gif_result = MazeGA(maze, GAConfig(generations=300, seed=GA_SEED, patience=15)).run()
    viz.save_animation(gif_result, OUT / "evolution.gif", fps=6, max_frames=60)
    print("wrote figures to", OUT)


if __name__ == "__main__":
    main()
