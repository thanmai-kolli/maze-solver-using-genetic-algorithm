"""Selection, crossover, mutation and diversity measurement."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .maze import DIRECTIONS

N_GENES = len(DIRECTIONS)

SELECTION_METHODS = ("tournament", "roulette", "rank", "sus", "truncation")
CROSSOVER_METHODS = ("one_point", "two_point", "uniform")
MUTATION_METHODS = ("point", "swap", "scramble", "mixed")


# ------------------------------------------------------------------- selection
def _probabilities(weights: np.ndarray) -> np.ndarray:
    weights = np.maximum(weights, 0.0)
    total = weights.sum()
    if total <= 0:
        return np.full(weights.size, 1.0 / weights.size)
    return weights / total


def tournament(
    fitness: np.ndarray, n: int, rng: np.random.Generator, k: int = 5, **_: object
) -> np.ndarray:
    k = int(np.clip(k, 2, max(2, fitness.size)))
    contenders = rng.integers(0, fitness.size, size=(n, k))
    return contenders[np.arange(n), np.argmax(fitness[contenders], axis=1)]


def roulette(fitness: np.ndarray, n: int, rng: np.random.Generator, **_: object) -> np.ndarray:
    shifted = fitness - fitness.min() + 1e-9
    return rng.choice(fitness.size, size=n, p=_probabilities(shifted))


def rank(
    fitness: np.ndarray, n: int, rng: np.random.Generator, pressure: float = 1.8, **_: object
) -> np.ndarray:
    """Linear ranking: selection odds depend on order, not on raw score gaps."""
    size = fitness.size
    order = np.argsort(fitness)
    ranks = np.empty(size, dtype=float)
    ranks[order] = np.arange(size)
    pressure = float(np.clip(pressure, 1.0, 2.0))
    weights = (2 - pressure) + 2 * (pressure - 1) * ranks / max(size - 1, 1)
    return rng.choice(size, size=n, p=_probabilities(weights))


def sus(fitness: np.ndarray, n: int, rng: np.random.Generator, **_: object) -> np.ndarray:
    """Stochastic universal sampling — one spin, evenly spaced pointers."""
    probs = _probabilities(fitness - fitness.min() + 1e-9)
    pointers = (rng.random() + np.arange(n)) / n
    return np.searchsorted(np.cumsum(probs), pointers, side="left").clip(0, fitness.size - 1)


def truncation(
    fitness: np.ndarray, n: int, rng: np.random.Generator, fraction: float = 0.5, **_: object
) -> np.ndarray:
    cut = max(2, int(fitness.size * float(np.clip(fraction, 0.05, 1.0))))
    elite_idx = np.argsort(fitness)[::-1][:cut]
    return rng.choice(elite_idx, size=n)


def get_selection(name: str) -> Callable[..., np.ndarray]:
    table = {
        "tournament": tournament,
        "roulette": roulette,
        "rank": rank,
        "sus": sus,
        "truncation": truncation,
    }
    try:
        return table[name]
    except KeyError as exc:
        raise ValueError(f"unknown selection {name!r}; expected one of {SELECTION_METHODS}") from exc


# ------------------------------------------------------------------- crossover
def one_point(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    point = int(rng.integers(1, a.size))
    return np.concatenate((a[:point], b[point:])), np.concatenate((b[:point], a[point:]))


def two_point(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    i, j = sorted(rng.choice(np.arange(1, a.size), size=2, replace=False)) if a.size > 2 else (1, a.size)
    c1, c2 = a.copy(), b.copy()
    c1[i:j], c2[i:j] = b[i:j], a[i:j]
    return c1, c2


def uniform(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator, swap_rate: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    mask = rng.random(a.size) < swap_rate
    c1, c2 = a.copy(), b.copy()
    c1[mask], c2[mask] = b[mask], a[mask]
    return c1, c2


def get_crossover(name: str) -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    table = {"one_point": one_point, "two_point": two_point, "uniform": uniform}
    try:
        return table[name]
    except KeyError as exc:
        raise ValueError(f"unknown crossover {name!r}; expected one of {CROSSOVER_METHODS}") from exc


# -------------------------------------------------------------------- mutation
def point_mutate(genome: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(genome.size) < rate
    hits = int(mask.sum())
    if hits:
        genome[mask] = rng.integers(0, N_GENES, size=hits, dtype=np.int8)
    return genome


def swap_mutate(genome: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    for _ in range(max(1, int(genome.size * rate))):
        i, j = rng.integers(0, genome.size, size=2)
        genome[i], genome[j] = genome[j], genome[i]
    return genome


def scramble_mutate(genome: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Shuffle one contiguous block — a coarse move that escapes local optima."""
    span = max(2, int(genome.size * max(rate, 0.05)))
    start = int(rng.integers(0, max(1, genome.size - span)))
    block = genome[start : start + span]
    rng.shuffle(block)
    genome[start : start + span] = block
    return genome


def mixed_mutate(genome: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    genome = point_mutate(genome, rate, rng)
    if rng.random() < 0.25:
        genome = scramble_mutate(genome, rate, rng)
    return genome


def get_mutation(name: str) -> Callable[[np.ndarray, float, np.random.Generator], np.ndarray]:
    table = {
        "point": point_mutate,
        "swap": swap_mutate,
        "scramble": scramble_mutate,
        "mixed": mixed_mutate,
    }
    try:
        return table[name]
    except KeyError as exc:
        raise ValueError(f"unknown mutation {name!r}; expected one of {MUTATION_METHODS}") from exc


# ------------------------------------------------------------------- diversity
def locus_entropy(population: np.ndarray) -> float:
    """Mean per-gene Shannon entropy, normalised to ``[0, 1]``.

    1.0 means every locus still carries all four alleles uniformly; values near
    0 mean the population has converged and crossover can no longer create
    anything new.
    """
    counts = np.stack([(population == g).sum(axis=0) for g in range(N_GENES)]).astype(float)
    probs = counts / max(population.shape[0], 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probs > 0, probs * np.log2(probs), 0.0)
    return float((-terms.sum(axis=0) / np.log2(N_GENES)).mean())
