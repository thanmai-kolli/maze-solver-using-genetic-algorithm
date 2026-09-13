"""mazega — solving mazes with a genetic algorithm, measurably.

Quick start::

    from mazega import generate, solve

    maze = generate("recursive_backtracker", 21, 21, seed=7)
    result = solve(maze)
    print(result.summary())
"""

from __future__ import annotations

from .benchmark import DEFAULT_VARIANTS, aggregate, compare, run_trials
from .fitness import FitnessEvaluator, FitnessReport, FitnessWeights
from .ga import GAConfig, GAResult, Generation, MazeGA, solve
from .genome import Walk, decode, prune_loops, suggested_length, to_moves
from .maze import FREE, MOVES, PRESETS, WALL, Maze, from_rows, generate, preset
from .search import astar, bfs, distance_field, is_solvable

__version__ = "2.0.0"

__all__ = [
    "DEFAULT_VARIANTS",
    "FREE",
    "FitnessEvaluator",
    "FitnessReport",
    "FitnessWeights",
    "GAConfig",
    "GAResult",
    "Generation",
    "MOVES",
    "MazeGA",
    "Maze",
    "PRESETS",
    "WALL",
    "Walk",
    "aggregate",
    "astar",
    "bfs",
    "compare",
    "decode",
    "distance_field",
    "from_rows",
    "generate",
    "is_solvable",
    "preset",
    "prune_loops",
    "run_trials",
    "solve",
    "suggested_length",
    "to_moves",
]
