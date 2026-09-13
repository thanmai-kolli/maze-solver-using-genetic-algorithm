"""Repeated-trial benchmarking so claims about operators are backed by numbers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .ga import GAConfig, GAResult, MazeGA
from .fitness import FitnessWeights
from .maze import Maze


@dataclass(frozen=True)
class TrialSummary:
    variant: str
    seed: int
    solved: bool
    generations: int
    generation_solved: int | None
    best_fitness: float
    path_steps: int
    optimal_steps: int | None
    evaluations: int
    seconds: float

    @property
    def optimality(self) -> float | None:
        if not self.solved or not self.optimal_steps:
            return None
        return self.optimal_steps / max(self.path_steps, 1)


def _summarise(name: str, seed: int, result: GAResult) -> TrialSummary:
    info = result.summary()
    return TrialSummary(
        variant=name,
        seed=seed,
        solved=bool(info["solved"]),
        generations=int(info["generations"]),
        generation_solved=info["generation_solved"],
        best_fitness=float(info["best_fitness"]),
        path_steps=int(info["path_steps"]),
        optimal_steps=info["optimal_steps"],
        evaluations=int(info["evaluations"]),
        seconds=float(info["seconds"]),
    )


def run_trials(
    maze: Maze,
    config: GAConfig,
    seeds: Sequence[int],
    *,
    name: str = "default",
) -> list[TrialSummary]:
    """Run the same configuration under several seeds."""
    return [_summarise(name, seed, MazeGA(maze, replace(config, seed=seed)).run()) for seed in seeds]


def compare(
    maze: Maze,
    variants: Mapping[str, Mapping[str, Any]],
    *,
    base: GAConfig | None = None,
    seeds: Iterable[int] = range(5),
    progress: Any = None,
) -> tuple[list[TrialSummary], list[dict]]:
    """Benchmark several config overrides on one maze.

    Returns the raw trials plus an aggregated table (solve rate, median
    generations-to-solve, mean optimality, mean runtime).
    """
    base = (base or GAConfig()).validate()
    seeds = list(seeds)
    trials: list[TrialSummary] = []
    total = len(variants) * len(seeds)
    done = 0

    for name, overrides in variants.items():
        cfg = replace(base, **dict(overrides))
        for seed in seeds:
            trials.append(_summarise(name, seed, MazeGA(maze, replace(cfg, seed=seed)).run()))
            done += 1
            if progress is not None:
                progress(done / total, f"{name} · seed {seed}")

    return trials, aggregate(trials)


def aggregate(trials: Sequence[TrialSummary]) -> list[dict]:
    rows: list[dict] = []
    for name in dict.fromkeys(t.variant for t in trials):
        group = [t for t in trials if t.variant == name]
        wins = [t for t in group if t.solved]
        gens = [t.generation_solved for t in wins if t.generation_solved is not None]
        ratios = [t.optimality for t in wins if t.optimality is not None]
        rows.append(
            {
                "variant": name,
                "trials": len(group),
                "solve_rate": round(len(wins) / len(group), 3),
                "median_gen_to_solve": None if not gens else float(np.median(gens)),
                "mean_path_steps": None if not wins else round(float(np.mean([t.path_steps for t in wins])), 2),
                "mean_optimality": None if not ratios else round(float(np.mean(ratios)), 3),
                "mean_evaluations": round(float(np.mean([t.evaluations for t in group]))),
                "mean_seconds": round(float(np.mean([t.seconds for t in group])), 3),
            }
        )
    return rows


DEFAULT_VARIANTS: dict[str, dict[str, Any]] = {
    "naive baseline": {
        "decode": "strict",
        "lamarckian": False,
        "adaptive_mutation": False,
        "catastrophe": False,
        "immigrant_rate": 0.0,
        "selection": "truncation",
        "weights": FitnessWeights(heuristic="manhattan"),
    },
    "+ geodesic fitness": {
        "decode": "strict",
        "lamarckian": False,
        "adaptive_mutation": False,
        "catastrophe": False,
        "immigrant_rate": 0.0,
        "selection": "truncation",
    },
    "+ tournament & restarts": {"decode": "strict", "lamarckian": False},
    "+ skip decoding": {"decode": "skip", "lamarckian": False},
    "+ repair decoding": {"decode": "repair", "lamarckian": False},
    "+ Lamarckian writeback": {"decode": "repair", "lamarckian": True},
    "guided (memetic)": {"decode": "guided", "lamarckian": True},
}
