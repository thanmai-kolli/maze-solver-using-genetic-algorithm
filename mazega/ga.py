"""The evolutionary engine: a steady stream of measurable generations."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterator, Literal

import numpy as np

from . import operators as ops
from .fitness import FitnessEvaluator, FitnessReport, FitnessWeights
from .genome import (
    DecodeMode,
    Walk,
    prune_loops,
    random_genome,
    random_population,
    suggested_length,
    to_moves,
)
from .maze import Coord, Maze

StopRule = Literal["first_solution", "optimal", "never"]
STOP_RULES: tuple[str, ...] = ("first_solution", "optimal", "never")


@dataclass
class GAConfig:
    """Every knob of the search, in one reproducible object."""

    population: int = 180
    generations: int = 200
    chromosome: int | None = None  # None -> derived from the maze
    elite: int = 6

    selection: str = "tournament"
    tournament_size: int = 5
    rank_pressure: float = 1.8
    truncation_fraction: float = 0.5

    crossover: str = "two_point"
    crossover_rate: float = 0.9
    uniform_swap_rate: float = 0.5

    mutation: str = "point"
    mutation_rate: float = 0.04
    adaptive_mutation: bool = True
    max_mutation_rate: float = 0.30

    stagnation_limit: int = 18
    catastrophe: bool = True
    immigrant_rate: float = 0.04

    decode: DecodeMode = "repair"
    lamarckian: bool = True
    weights: FitnessWeights = field(default_factory=FitnessWeights)

    stop_rule: StopRule = "optimal"
    patience: int = 40
    seed: int | None = 42

    def validate(self) -> "GAConfig":
        if self.population < 4:
            raise ValueError("population must be >= 4")
        if not 0 <= self.elite < self.population:
            raise ValueError("elite must be in [0, population)")
        if self.selection not in ops.SELECTION_METHODS:
            raise ValueError(f"selection must be one of {ops.SELECTION_METHODS}")
        if self.crossover not in ops.CROSSOVER_METHODS:
            raise ValueError(f"crossover must be one of {ops.CROSSOVER_METHODS}")
        if self.mutation not in ops.MUTATION_METHODS:
            raise ValueError(f"mutation must be one of {ops.MUTATION_METHODS}")
        if self.stop_rule not in STOP_RULES:
            raise ValueError(f"stop_rule must be one of {STOP_RULES}")
        return self

    def as_dict(self) -> dict:
        data = asdict(self)
        data["weights"] = self.weights.as_dict()
        return data


@dataclass(frozen=True)
class Generation:
    """One row of the evolution log — everything the UI plots comes from here."""

    index: int
    best: float
    mean: float
    median: float
    worst: float
    std: float
    diversity: float
    success_rate: float
    mutation_rate: float
    best_steps: int
    solved: bool
    stagnation: int
    evaluations: int
    elapsed: float
    best_path: list[Coord] = field(repr=False, default_factory=list)
    report: FitnessReport | None = field(repr=False, default=None)
    event: str = ""

    def as_row(self) -> dict:
        return {
            "generation": self.index,
            "best": round(self.best, 5),
            "mean": round(self.mean, 5),
            "median": round(self.median, 5),
            "worst": round(self.worst, 5),
            "std": round(self.std, 5),
            "diversity": round(self.diversity, 4),
            "success_rate": round(self.success_rate, 4),
            "mutation_rate": round(self.mutation_rate, 4),
            "best_steps": self.best_steps,
            "solved": self.solved,
            "event": self.event,
        }


@dataclass
class GAResult:
    """Final outcome plus the full generation-by-generation history."""

    maze: Maze
    config: GAConfig
    history: list[Generation]
    best_genome: np.ndarray
    best_walk: Walk
    best_report: FitnessReport
    evaluations: int
    elapsed: float
    optimal_steps: int | None

    @property
    def solved(self) -> bool:
        return self.best_walk.reached

    @property
    def generations_run(self) -> int:
        return len(self.history)

    @property
    def best_path(self) -> list[Coord]:
        return self.best_walk.path

    @property
    def simple_path(self) -> list[Coord]:
        """Best path with cycles removed — what you would actually walk."""
        return prune_loops(self.best_walk.path)

    @property
    def generation_solved(self) -> int | None:
        return next((g.index for g in self.history if g.solved), None)

    @property
    def optimality_ratio(self) -> float | None:
        """``optimal / found`` — 1.0 means the GA matched the exact solver."""
        if not self.solved or not self.optimal_steps:
            return None
        return self.optimal_steps / max(len(self.simple_path) - 1, 1)

    def moves(self) -> str:
        return to_moves(self.best_genome)

    def summary(self) -> dict:
        ratio = self.optimality_ratio
        return {
            "solved": self.solved,
            "generations": self.generations_run,
            "generation_solved": self.generation_solved,
            "best_fitness": round(self.best_report.score, 5),
            "path_steps": max(len(self.simple_path) - 1, 0),
            "optimal_steps": self.optimal_steps,
            "optimality": None if ratio is None else round(ratio, 4),
            "evaluations": self.evaluations,
            "seconds": round(self.elapsed, 3),
        }

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame([g.as_row() for g in self.history])


class MazeGA:
    """Genetic algorithm over fixed-length ``UDLR`` chromosomes."""

    def __init__(self, maze: Maze, config: GAConfig | None = None) -> None:
        self.maze = maze
        self.config = (config or GAConfig()).validate()
        self.rng = np.random.default_rng(self.config.seed)
        self.evaluator = FitnessEvaluator(
            maze,
            self.config.weights,
            decode_mode=self.config.decode,
            lamarckian=self.config.lamarckian,
        )
        self.length = int(self.config.chromosome or suggested_length(maze))
        self._select = ops.get_selection(self.config.selection)
        self._cross = ops.get_crossover(self.config.crossover)
        self._mutate = ops.get_mutation(self.config.mutation)
        self.history: list[Generation] = []
        self._result: GAResult | None = None

    @property
    def result(self) -> GAResult:
        """The finished run; only valid once :meth:`evolve` has been exhausted."""
        if self._result is None:
            raise RuntimeError("the run has not finished yet")
        return self._result

    @property
    def encodable(self) -> bool:
        """False when the chromosome is too short to represent any solution."""
        optimal = self.evaluator.optimal_steps
        return optimal is None or self.length >= optimal

    # ------------------------------------------------------------------ helpers
    def _selection_kwargs(self) -> dict:
        cfg = self.config
        return {
            "k": cfg.tournament_size,
            "pressure": cfg.rank_pressure,
            "fraction": cfg.truncation_fraction,
        }

    def _current_mutation_rate(self, stagnation: int) -> float:
        cfg = self.config
        if not cfg.adaptive_mutation or cfg.stagnation_limit <= 0:
            return cfg.mutation_rate
        ramp = min(stagnation / cfg.stagnation_limit, 1.0)
        return float(min(cfg.mutation_rate * (1.0 + 6.0 * ramp), cfg.max_mutation_rate))

    def _breed(self, population: np.ndarray, scores: np.ndarray, rate: float) -> np.ndarray:
        cfg = self.config
        order = np.argsort(scores)[::-1]
        nxt = [population[i].copy() for i in order[: cfg.elite]]
        needed = cfg.population - len(nxt)
        parents = self._select(scores, needed + needed % 2, self.rng, **self._selection_kwargs())

        for i in range(0, len(parents) - 1, 2):
            a, b = population[parents[i]].copy(), population[parents[i + 1]].copy()
            if self.rng.random() < cfg.crossover_rate:
                if cfg.crossover == "uniform":
                    a, b = self._cross(a, b, self.rng, cfg.uniform_swap_rate)
                else:
                    a, b = self._cross(a, b, self.rng)
            nxt.append(self._mutate(a, rate, self.rng))
            if len(nxt) < cfg.population:
                nxt.append(self._mutate(b, rate, self.rng))

        immigrants = int(cfg.population * max(cfg.immigrant_rate, 0.0))
        for i in range(immigrants):
            slot = cfg.population - 1 - i
            if slot > cfg.elite:
                nxt[slot] = random_genome(self.length, self.rng)
        return np.asarray(nxt[: cfg.population], dtype=np.int8)

    def _restart(self, population: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Hypermutation catastrophe: keep the elites, rebuild everyone else."""
        order = np.argsort(scores)[::-1]
        fresh = random_population(self.config.population, self.length, self.rng)
        for slot, idx in enumerate(order[: max(self.config.elite, 1)]):
            fresh[slot] = population[idx]
        return fresh

    # ------------------------------------------------------------------- driver
    def evolve(self) -> Iterator[Generation]:
        """Run the search, yielding one :class:`Generation` at a time."""
        cfg = self.config
        started = time.perf_counter()
        population = random_population(cfg.population, self.length, self.rng)

        best_score = -np.inf
        best_genome = population[0].copy()
        best_walk: Walk | None = None
        best_report: FitnessReport | None = None
        stagnation = 0
        solved_at: int | None = None

        for gen in range(cfg.generations):
            scores, reports, walks = self.evaluator.evaluate_population(population)
            champion = int(np.argmax(scores))
            event = ""

            if scores[champion] > best_score + 1e-12:
                best_score = float(scores[champion])
                best_genome = population[champion].copy()
                best_walk, best_report = walks[champion], reports[champion]
                stagnation = 0
            else:
                stagnation += 1

            assert best_walk is not None and best_report is not None
            if best_walk.reached and solved_at is None:
                solved_at = gen
                event = "solution found"

            record = Generation(
                index=gen,
                best=float(scores[champion]),
                mean=float(scores.mean()),
                median=float(np.median(scores)),
                worst=float(scores.min()),
                std=float(scores.std()),
                diversity=ops.locus_entropy(population),
                success_rate=float(np.mean([r.reached for r in reports])),
                mutation_rate=self._current_mutation_rate(stagnation),
                best_steps=max(len(prune_loops(best_walk.path)) - 1, 0),
                solved=best_walk.reached,
                stagnation=stagnation,
                evaluations=self.evaluator.evaluations,
                elapsed=time.perf_counter() - started,
                best_path=list(best_walk.path),
                report=best_report,
                event=event,
            )

            if self._should_stop(record, solved_at, gen):
                record = _with_event(record, record.event or "stop rule met")
                self.history.append(record)
                yield record
                break

            if cfg.catastrophe and stagnation >= cfg.stagnation_limit:
                population = self._restart(population, scores)
                stagnation = 0
                record = _with_event(record, "catastrophe restart")
                self.history.append(record)
                yield record
                continue

            self.history.append(record)
            yield record
            population = self._breed(population, scores, record.mutation_rate)

        self._result = GAResult(
            maze=self.maze,
            config=cfg,
            history=self.history,
            best_genome=best_genome,
            best_walk=best_walk,  # type: ignore[arg-type]
            best_report=best_report,  # type: ignore[arg-type]
            evaluations=self.evaluator.evaluations,
            elapsed=time.perf_counter() - started,
            optimal_steps=self.evaluator.optimal_steps,
        )

    def _should_stop(self, record: Generation, solved_at: int | None, gen: int) -> bool:
        cfg = self.config
        if not record.solved:
            return False
        if cfg.stop_rule == "first_solution":
            return True
        if cfg.stop_rule == "optimal":
            if self.evaluator.optimal_steps is not None and record.best_steps <= self.evaluator.optimal_steps:
                return True
            return solved_at is not None and (gen - solved_at) >= cfg.patience
        return False

    def run(self, on_generation: Callable[[Generation], None] | None = None) -> GAResult:
        for record in self.evolve():
            if on_generation is not None:
                on_generation(record)
        return self.result


def _with_event(record: Generation, event: str) -> Generation:
    data = {f: getattr(record, f) for f in record.__dataclass_fields__}
    data["event"] = event
    return Generation(**data)


def solve(maze: Maze, config: GAConfig | None = None, **overrides) -> GAResult:
    """One-liner entry point: ``solve(maze, generations=300, seed=7)``."""
    cfg = config or GAConfig()
    if overrides:
        cfg = GAConfig(**{**cfg.as_dict(), **overrides, "weights": cfg.weights})
    return MazeGA(maze, cfg).run()
