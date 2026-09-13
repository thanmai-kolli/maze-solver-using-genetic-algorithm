"""Command-line front end: solve, benchmark or export a maze run.

    python cli.py solve --size 21 --seed 7 --save out/run.png
    python cli.py bench --seeds 5
    python cli.py gif --out out/evolution.gif
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mazega import (
    DEFAULT_VARIANTS,
    FitnessWeights,
    GAConfig,
    Maze,
    MazeGA,
    astar,
    compare,
    generate,
    preset,
)
from mazega.maze import GENERATORS, PRESETS


def build_maze(args: argparse.Namespace) -> Maze:
    if args.preset:
        return preset(args.preset)
    if args.file:
        return Maze.from_text(Path(args.file).read_text(encoding="utf-8"))
    return generate(args.kind, args.size, args.size, seed=args.maze_seed, density=args.density)


def build_config(args: argparse.Namespace) -> GAConfig:
    return GAConfig(
        population=args.population,
        generations=args.generations,
        chromosome=args.chromosome,
        selection=args.selection,
        crossover=args.crossover,
        mutation=args.mutation,
        mutation_rate=args.mutation_rate,
        decode=args.decode,
        lamarckian=not args.no_lamarckian,
        adaptive_mutation=not args.no_adaptive,
        catastrophe=not args.no_catastrophe,
        weights=FitnessWeights(heuristic=args.heuristic),
        seed=args.seed,
    )


def cmd_solve(args: argparse.Namespace) -> int:
    maze = build_maze(args)
    exact = astar(maze)
    engine = MazeGA(maze, build_config(args))

    print(f"maze {maze.rows}x{maze.cols} · open {maze.open_cells} · optimal {exact.steps} steps")
    print(f"chromosome {engine.length} moves · search space 4^{engine.length}")
    if not engine.encodable:
        print("warning: chromosome is shorter than the optimal path — no solution can be encoded",
              file=sys.stderr)

    def tick(record) -> None:
        if args.quiet:
            return
        marker = "*" if record.event else " "
        print(f"{marker} gen {record.index:>4} | best {record.best:7.4f} | mean {record.mean:7.4f} "
              f"| entropy {record.diversity:.2f} | steps {record.best_steps:>4} "
              f"| {record.event}".rstrip())

    result = engine.run(tick)
    print("\n" + json.dumps(result.summary(), indent=2))
    if not args.quiet and result.solved:
        print("\n" + render(maze, result.simple_path))

    if args.save:
        from mazega import viz

        viz.use_dark_theme()
        figure = viz.overview_figure(result)
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save, dpi=160)
        print(f"saved {args.save}")
    return 0 if result.solved else 1


def cmd_bench(args: argparse.Namespace) -> int:
    maze = build_maze(args)
    _, rows = compare(maze, DEFAULT_VARIANTS, base=build_config(args), seeds=range(args.seeds))
    width = max(len(r["variant"]) for r in rows)
    cell = lambda v: "-" if v is None else str(v)  # noqa: E731 - 0.0 is a real value here
    print(f"{'variant'.ljust(width)}  solve  gen*  steps  optimality")
    for row in rows:
        print(
            f"{row['variant'].ljust(width)}  {row['solve_rate']:>5.0%}  "
            f"{cell(row['median_gen_to_solve']):>4}  "
            f"{cell(row['mean_path_steps']):>5}  "
            f"{cell(row['mean_optimality']):>10}"
        )
    if args.save:
        from mazega import viz

        viz.use_dark_theme()
        ax = viz.plot_benchmark(rows)
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(args.save, dpi=160, bbox_inches="tight")
        print(f"saved {args.save}")
    return 0


def cmd_gif(args: argparse.Namespace) -> int:
    from mazega import viz

    maze = build_maze(args)
    result = MazeGA(maze, build_config(args)).run()
    viz.use_dark_theme()
    print("saved", viz.save_animation(result, args.out))
    return 0


def render(maze: Maze, path) -> str:
    cells = set(path)
    rows = []
    for r in range(maze.rows):
        line = ""
        for c in range(maze.cols):
            if (r, c) == maze.start:
                line += "S"
            elif (r, c) == maze.goal:
                line += "G"
            elif maze.grid[r, c]:
                line += "█"
            elif (r, c) in cells:
                line += "·"
            else:
                line += " "
        rows.append(line)
    return "\n".join(rows)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = root.add_subparsers(dest="command", required=True)

    def shared(p: argparse.ArgumentParser) -> None:
        p.add_argument("--kind", choices=GENERATORS, default="recursive_backtracker")
        p.add_argument("--size", type=int, default=21)
        p.add_argument("--maze-seed", type=int, default=7)
        p.add_argument("--density", type=float, default=0.28)
        p.add_argument("--preset", choices=list(PRESETS))
        p.add_argument("--file", help="load an ASCII maze instead of generating one")
        p.add_argument("--population", type=int, default=180)
        p.add_argument("--generations", type=int, default=250)
        p.add_argument("--chromosome", type=int)
        p.add_argument("--selection", default="tournament")
        p.add_argument("--crossover", default="two_point")
        p.add_argument("--mutation", default="point")
        p.add_argument("--mutation-rate", type=float, default=0.04)
        p.add_argument("--decode", choices=["strict", "skip", "repair", "guided"], default="repair")
        p.add_argument("--heuristic", choices=["geodesic", "manhattan"], default="geodesic")
        p.add_argument("--no-lamarckian", action="store_true")
        p.add_argument("--no-adaptive", action="store_true")
        p.add_argument("--no-catastrophe", action="store_true")
        p.add_argument("--seed", type=int, default=42)

    solve_p = sub.add_parser("solve", help="evolve a path and print the result")
    shared(solve_p)
    solve_p.add_argument("--save", help="write a summary PNG here")
    solve_p.add_argument("--quiet", action="store_true")
    solve_p.set_defaults(func=cmd_solve)

    bench_p = sub.add_parser("bench", help="ablate the mechanisms over several seeds")
    shared(bench_p)
    bench_p.add_argument("--seeds", type=int, default=5)
    bench_p.add_argument("--save", help="write a bar chart PNG here")
    bench_p.set_defaults(func=cmd_bench)

    gif_p = sub.add_parser("gif", help="render the evolution as an animated GIF")
    shared(gif_p)
    gif_p.add_argument("--out", default="docs/images/evolution.gif")
    gif_p.set_defaults(func=cmd_gif)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
