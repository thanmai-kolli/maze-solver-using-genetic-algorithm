"""Chromosome encoding and the decoders that turn genes into a walk."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .maze import DIRECTIONS, MOVES, Coord, Maze

DecodeMode = Literal["strict", "skip", "repair", "guided"]
DECODE_MODES: tuple[str, ...] = ("strict", "skip", "repair", "guided")


@dataclass
class Walk:
    """The phenotype: the trajectory a chromosome actually produces.

    Cells are held as flat ``row * cols + col`` indices because decoding runs
    millions of times; :attr:`path` materialises coordinates only when asked.
    """

    trail: list[int]
    cols: int
    reached: bool
    collisions: int
    revisits: int
    genes_used: int
    closest: float = 0.0
    genome: np.ndarray | None = field(default=None, repr=False)

    @property
    def path(self) -> list[Coord]:
        cols = self.cols
        return [(i // cols, i % cols) for i in self.trail]

    @property
    def end_index(self) -> int:
        return self.trail[-1]

    @property
    def end(self) -> Coord:
        return divmod(self.trail[-1], self.cols)

    @property
    def steps(self) -> int:
        return max(len(self.trail) - 1, 0)

    @property
    def unique_cells(self) -> int:
        return len(set(self.trail))


def random_genome(length: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, len(DIRECTIONS), size=length, dtype=np.int8)


def random_population(size: int, length: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, len(DIRECTIONS), size=(size, length), dtype=np.int8)


def to_moves(genome: np.ndarray) -> str:
    """Render a chromosome as the human-readable ``"UDLR"`` string."""
    return "".join(MOVES[int(g)] for g in genome)


def from_moves(moves: str) -> np.ndarray:
    lookup = {m: i for i, m in enumerate(MOVES)}
    return np.array([lookup[ch] for ch in moves.upper() if ch in lookup], dtype=np.int8)


def suggested_length(maze: Maze, slack: float = 2.5) -> int:
    """Pick a chromosome length that can actually encode a solution.

    A genome shorter than the shortest path makes the maze unsolvable by
    construction. Even at 1.5x the shortest path the walk has to be nearly
    perfect, so the default leaves generous slack — empirically this single
    number matters more than population size or generation count.
    """
    from .search import bfs

    exact = bfs(maze)
    floor = exact.steps if exact.solved else maze.manhattan(maze.start, maze.goal)
    return int(max(16, min(4 * maze.open_cells, round(floor * slack) + 8)))


def decode(
    genome: np.ndarray,
    maze: Maze,
    mode: DecodeMode = "repair",
    *,
    writeback: bool = False,
    distance: list[float] | None = None,
) -> Walk:
    """Execute a chromosome inside the maze and report the resulting walk.

    Four semantics, in increasing order of how much help the decoder gives:

    * ``strict`` — a blocked move ends the walk (the naive baseline: one bad
      gene throws away the whole tail of the chromosome).
    * ``skip``   — a blocked move is a no-op counted as a collision, so later
      genes still get expressed.
    * ``repair`` — a blocked move rotates clockwise to the next legal step,
      preferring an unvisited cell. Uses no knowledge of where the goal is, so
      the search is still genuinely evolutionary.
    * ``guided`` — a memetic variant: a blocked move jumps to the legal step
      closest to the goal. Powerful, but it hands the decoder a greedy solver,
      which is exactly why it is opt-in.

    With ``writeback=True`` corrections are copied back into the chromosome
    (Lamarckian inheritance) so a repaired move can be passed to offspring.
    """
    if mode not in DECODE_MODES:
        raise ValueError(f"unknown decode mode {mode!r}; expected one of {DECODE_MODES}")

    tables = maze.tables
    neighbours = tables.neighbours
    goal_distance = tables.goal_distance
    goal = tables.goal
    pos = tables.start
    n_genes = len(DIRECTIONS)
    field_ = distance if distance is not None else [float(d) for d in goal_distance]

    trail = [pos]
    visited = {pos: 1}
    closest = field_[pos]
    collisions = revisits = used = 0
    guided = mode == "guided"
    mends = guided or mode == "repair"
    strict = mode == "strict"
    repaired = genome.copy() if (writeback and mends) else None

    for i, gene in enumerate(genome.tolist()):
        used = i + 1
        options = neighbours[pos]
        nxt = options[gene]

        if nxt < 0:
            collisions += 1
            if strict:
                break
            if not mends:
                continue
            choice = -1
            if guided:
                best_key = (1 << 30, 1 << 30)
                for candidate_gene, candidate in enumerate(options):
                    if candidate < 0:
                        continue
                    key = (visited.get(candidate, 0), goal_distance[candidate])
                    if key < best_key:
                        best_key, choice, nxt = key, candidate_gene, candidate
            else:
                # Rotate clockwise from the failed gene; first unvisited wins.
                fallback = -1
                for offset in range(1, n_genes):
                    candidate_gene = (gene + offset) % n_genes
                    candidate = options[candidate_gene]
                    if candidate < 0:
                        continue
                    if fallback < 0:
                        fallback, nxt = candidate_gene, candidate
                    if candidate not in visited:
                        choice, nxt = candidate_gene, candidate
                        break
                else:
                    choice, nxt = fallback, options[fallback] if fallback >= 0 else -1
            if choice < 0:
                break
            if repaired is not None:
                repaired[i] = choice

        seen = visited.get(nxt, 0)
        if seen:
            revisits += 1
        visited[nxt] = seen + 1
        pos = nxt
        trail.append(pos)
        if field_[pos] < closest:
            closest = field_[pos]
        if pos == goal:
            return Walk(trail, tables.cols, True, collisions, revisits, used, closest, repaired)

    return Walk(
        trail, tables.cols, pos == goal, collisions, revisits, max(used, 1), closest, repaired
    )


def prune_loops(path: list[Coord]) -> list[Coord]:
    """Strip cycles from a walk, yielding the simple path between the endpoints."""
    seen: dict[Coord, int] = {}
    pruned: list[Coord] = []
    for cell in path:
        index = seen.get(cell)
        if index is None:
            seen[cell] = len(pruned)
            pruned.append(cell)
        else:
            for stale in [c for c, i in seen.items() if i > index]:
                del seen[stale]
            del pruned[index + 1 :]
    return pruned
