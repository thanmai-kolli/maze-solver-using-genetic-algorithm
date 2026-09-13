"""A decomposed, explainable fitness function."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .genome import DecodeMode, Walk, decode
from .maze import Coord, Maze
from .search import UNREACHABLE, bfs, distance_field

Heuristic = Literal["manhattan", "geodesic"]


@dataclass(frozen=True)
class FitnessWeights:
    """Relative importance of each fitness term.

    The score is a weighted sum of four normalised signals, so changing a weight
    changes behaviour predictably instead of rescaling the whole landscape.
    """

    goal: float = 2.0
    progress: float = 1.0
    efficiency: float = 1.0
    collision: float = 0.35
    revisit: float = 0.35
    heuristic: Heuristic = "geodesic"

    def as_dict(self) -> dict[str, float | str]:
        return {
            "goal": self.goal,
            "progress": self.progress,
            "efficiency": self.efficiency,
            "collision": self.collision,
            "revisit": self.revisit,
            "heuristic": self.heuristic,
        }


@dataclass(frozen=True)
class FitnessReport:
    """Per-individual score plus the breakdown that produced it."""

    score: float
    reached: bool
    progress: float
    efficiency: float
    collision_rate: float
    revisit_rate: float
    path_steps: int
    terms: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | bool | int]:
        return {
            "score": round(self.score, 5),
            "reached": self.reached,
            "progress": round(self.progress, 4),
            "efficiency": round(self.efficiency, 4),
            "collision_rate": round(self.collision_rate, 4),
            "revisit_rate": round(self.revisit_rate, 4),
            "path_steps": self.path_steps,
        }


class FitnessEvaluator:
    """Scores chromosomes against one maze, caching everything maze-dependent."""

    def __init__(
        self,
        maze: Maze,
        weights: FitnessWeights | None = None,
        *,
        decode_mode: DecodeMode = "repair",
        lamarckian: bool = False,
    ) -> None:
        self.maze = maze
        self.weights = weights or FitnessWeights()
        self.decode_mode = decode_mode
        self.lamarckian = lamarckian and decode_mode in ("repair", "guided")
        self.evaluations = 0

        self._field = distance_field(maze)
        optimal = bfs(maze)
        self.optimal_steps: int | None = optimal.steps if optimal.solved else None
        self.solvable = optimal.solved
        self._penalty = float(maze.open_cells)
        self._flat_distance = [
            float(v) if v != UNREACHABLE else float(d + self._penalty)
            for v, d in zip(self._field.ravel().tolist(), maze.tables.goal_distance)
        ]
        self._flat_manhattan = [float(d) for d in maze.tables.goal_distance]
        self._active_field = (
            self._flat_distance if self.weights.heuristic == "geodesic" else self._flat_manhattan
        )
        self._start_distance = max(self.distance(maze.start), 1.0)

    # ------------------------------------------------------------------ signals
    def distance(self, cell: Coord) -> float:
        """Distance from ``cell`` to the goal under the configured heuristic."""
        return self._distance_at(cell[0] * self.maze.cols + cell[1])

    def _distance_at(self, index: int) -> float:
        if self.weights.heuristic == "geodesic":
            # Walled-off cells fall back to an inflated Manhattan value so they
            # can never out-score a genuinely reachable cell.
            return self._flat_distance[index]
        return self._flat_manhattan[index]

    @property
    def lower_bound_steps(self) -> int:
        if self.optimal_steps is not None:
            return self.optimal_steps
        return self.maze.manhattan(self.maze.start, self.maze.goal)

    # ---------------------------------------------------------------- evaluation
    def evaluate(self, genome: np.ndarray) -> tuple[FitnessReport, Walk]:
        self.evaluations += 1
        walk = decode(
            genome,
            self.maze,
            self.decode_mode,
            writeback=self.lamarckian,
            distance=self._active_field,
        )
        if self.lamarckian and walk.genome is not None:
            genome[:] = walk.genome
        return self.score(walk), walk

    def score(self, walk: Walk) -> FitnessReport:
        w = self.weights
        # Credit the closest the walk ever came, not where it happened to stop.
        progress = float(min(max(1.0 - walk.closest / self._start_distance, 0.0), 1.0))
        efficiency = (
            min(max(self.lower_bound_steps / max(walk.steps, 1), 0.0), 1.0) if walk.reached else 0.0
        )
        collision_rate = walk.collisions / max(walk.genes_used, 1)
        revisit_rate = walk.revisits / max(walk.steps, 1)

        terms = {
            "goal": w.goal * float(walk.reached),
            "progress": w.progress * progress,
            "efficiency": w.efficiency * efficiency,
            "collision": -w.collision * collision_rate,
            "revisit": -w.revisit * revisit_rate,
        }
        return FitnessReport(
            score=float(sum(terms.values())),
            reached=walk.reached,
            progress=progress,
            efficiency=efficiency,
            collision_rate=collision_rate,
            revisit_rate=revisit_rate,
            path_steps=walk.steps,
            terms=terms,
        )

    def evaluate_population(
        self, population: np.ndarray
    ) -> tuple[np.ndarray, list[FitnessReport], list[Walk]]:
        reports: list[FitnessReport] = []
        walks: list[Walk] = []
        for genome in population:
            report, walk = self.evaluate(genome)
            reports.append(report)
            walks.append(walk)
        return np.fromiter((r.score for r in reports), dtype=float, count=len(reports)), reports, walks
