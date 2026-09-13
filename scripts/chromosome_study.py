"""Measure the chromosome-length effect quoted in the README."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mazega import GAConfig, MazeGA, astar, generate

SIZES = (21, 25)
SEEDS = range(10)


def main() -> None:
    for slack in (1.0, 1.5, 2.0, 2.5, 3.0):
        wins = total = 0
        for size in SIZES:
            for seed in SEEDS:
                maze = generate("recursive_backtracker", size, size, seed=seed)
                length = round(astar(maze).steps * slack) + 8
                result = MazeGA(maze, GAConfig(generations=250, seed=seed, chromosome=length)).run()
                wins += result.solved
                total += 1
        print(f"slack {slack:>4}  solve rate {wins / total:.0%}  ({wins}/{total})")

    print()
    maze = generate("recursive_backtracker", 25, 25, seed=2)
    for population in (60, 180, 400, 800):
        result = MazeGA(maze, GAConfig(generations=250, seed=3, population=population)).run()
        print(f"population {population:>4}  solved={result.solved!s:5s} "
              f"gen={result.generation_solved}")


if __name__ == "__main__":
    main()
