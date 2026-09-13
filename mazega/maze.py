"""Maze representation, procedural generation and text I/O."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

FREE = 0
WALL = 1

Coord = tuple[int, int]

# (name, delta_row, delta_col) — gene value is the index into this table.
DIRECTIONS: tuple[tuple[str, int, int], ...] = (
    ("U", -1, 0),
    ("D", 1, 0),
    ("L", 0, -1),
    ("R", 0, 1),
)
MOVES: tuple[str, ...] = tuple(d[0] for d in DIRECTIONS)
DELTAS: np.ndarray = np.array([[d[1], d[2]] for d in DIRECTIONS], dtype=np.int16)

GENERATORS = ("recursive_backtracker", "prim", "braided", "random_obstacles", "open_field")


class MazeError(ValueError):
    """Raised when a maze is structurally invalid."""


@dataclass(frozen=True)
class MazeTables:
    """Flat-index lookup tables — decoding a chromosome touches only lists."""

    cols: int
    start: int
    goal: int
    neighbours: list[list[int]]  # cell -> [4] next cell per gene, -1 if blocked
    goal_distance: list[int]  # cell -> Manhattan distance to the goal


@dataclass(eq=False)
class Maze:
    """A rectangular grid maze. ``grid[r, c] == 1`` means wall."""

    grid: np.ndarray
    start: Coord
    goal: Coord

    def __post_init__(self) -> None:
        self.grid = np.asarray(self.grid, dtype=np.uint8)
        if self.grid.ndim != 2 or self.grid.size == 0:
            raise MazeError("maze grid must be a non-empty 2-D array")
        self.start = (int(self.start[0]), int(self.start[1]))
        self.goal = (int(self.goal[0]), int(self.goal[1]))
        for label, cell in (("start", self.start), ("goal", self.goal)):
            if not self.in_bounds(cell):
                raise MazeError(f"{label} {cell} is outside the {self.rows}x{self.cols} grid")
            if self.grid[cell] == WALL:
                raise MazeError(f"{label} {cell} sits on a wall")

    # ---------------------------------------------------------------- geometry
    @property
    def rows(self) -> int:
        return int(self.grid.shape[0])

    @property
    def cols(self) -> int:
        return int(self.grid.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, self.cols

    @property
    def open_cells(self) -> int:
        return int((self.grid == FREE).sum())

    def in_bounds(self, cell: Coord) -> bool:
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, cell: Coord) -> bool:
        return self.in_bounds(cell) and self.grid[cell] == FREE

    def neighbours(self, cell: Coord) -> Iterator[tuple[int, Coord]]:
        """Yield ``(gene, neighbour)`` for every legal move out of ``cell``."""
        r, c = cell
        for gene, (_, dr, dc) in enumerate(DIRECTIONS):
            nxt = (r + dr, c + dc)
            if self.is_free(nxt):
                yield gene, nxt

    def manhattan(self, a: Coord, b: Coord) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @property
    def tables(self) -> MazeTables:
        cached = self.__dict__.get("_tables")
        if cached is None:
            cached = _build_tables(self)
            self.__dict__["_tables"] = cached
        return cached

    # ----------------------------------------------------------------- editing
    def copy(self) -> "Maze":
        return Maze(self.grid.copy(), self.start, self.goal)

    def with_cell(self, cell: Coord, value: int) -> "Maze":
        """Return a copy with ``cell`` set to FREE/WALL (start & goal stay free)."""
        if cell in (self.start, self.goal) and value == WALL:
            return self.copy()
        grid = self.grid.copy()
        grid[cell] = value
        return Maze(grid, self.start, self.goal)

    def with_endpoints(self, start: Coord | None = None, goal: Coord | None = None) -> "Maze":
        grid = self.grid.copy()
        start = tuple(start) if start is not None else self.start  # type: ignore[assignment]
        goal = tuple(goal) if goal is not None else self.goal  # type: ignore[assignment]
        grid[start] = FREE
        grid[goal] = FREE
        return Maze(grid, start, goal)  # type: ignore[arg-type]

    # --------------------------------------------------------------------- I/O
    def to_text(self) -> str:
        rows = ["".join("#" if v else "." for v in row) for row in self.grid]
        rows[self.start[0]] = _replace_at(rows[self.start[0]], self.start[1], "S")
        rows[self.goal[0]] = _replace_at(rows[self.goal[0]], self.goal[1], "G")
        return "\n".join(rows)

    @classmethod
    def from_text(cls, text: str) -> "Maze":
        """Parse an ASCII maze. ``#``/``1`` are walls, ``S`` start, ``G`` goal."""
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        if not lines:
            raise MazeError("empty maze text")
        width = max(len(ln) for ln in lines)
        lines = [ln.ljust(width, ".") for ln in lines]
        grid = np.zeros((len(lines), width), dtype=np.uint8)
        start = goal = None
        for r, line in enumerate(lines):
            for c, ch in enumerate(line):
                if ch in "#1":
                    grid[r, c] = WALL
                elif ch in "Ss":
                    start = (r, c)
                elif ch in "Gg":
                    goal = (r, c)
        if start is None or goal is None:
            raise MazeError("maze text must contain exactly one 'S' and one 'G'")
        return cls(grid, start, goal)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Maze({self.rows}x{self.cols}, start={self.start}, goal={self.goal})"


def _replace_at(line: str, index: int, char: str) -> str:
    return line[:index] + char + line[index + 1 :]


def _build_tables(maze: "Maze") -> MazeTables:
    rows, cols = maze.shape
    free = maze.grid == FREE
    neighbours = np.full((rows * cols, len(DIRECTIONS)), -1, dtype=np.int32)
    flat = np.arange(rows * cols, dtype=np.int32).reshape(rows, cols)

    for gene, (_, dr, dc) in enumerate(DIRECTIONS):
        src_r = slice(max(0, -dr), rows - max(0, dr))
        src_c = slice(max(0, -dc), cols - max(0, dc))
        dst_r = slice(max(0, dr), rows - max(0, -dr))
        dst_c = slice(max(0, dc), cols - max(0, -dc))
        step = np.where(free[dst_r, dst_c], flat[dst_r, dst_c], -1)
        neighbours[flat[src_r, src_c].ravel(), gene] = np.where(
            free[src_r, src_c], step, -1
        ).ravel()

    gr, gc = maze.goal
    rr, cc = np.indices((rows, cols))
    distance = (np.abs(rr - gr) + np.abs(cc - gc)).ravel()
    return MazeTables(
        cols=cols,
        start=maze.start[0] * cols + maze.start[1],
        goal=gr * cols + gc,
        neighbours=neighbours.tolist(),
        goal_distance=distance.tolist(),
    )


# --------------------------------------------------------------------- helpers
def _odd(n: int, minimum: int = 5) -> int:
    n = max(int(n), minimum)
    return n if n % 2 == 1 else n - 1


def _rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _carve_perfect(rows: int, cols: int, rng: np.random.Generator, algorithm: str) -> np.ndarray:
    """Carve a perfect (loop-free, fully connected) maze on an odd-sized grid."""
    grid = np.ones((rows, cols), dtype=np.uint8)
    step = ((-2, 0), (2, 0), (0, -2), (0, 2))
    grid[1, 1] = FREE

    if algorithm == "prim":
        frontier: list[Coord] = []
        seen = {(1, 1)}

        def push(cell: Coord) -> None:
            for dr, dc in step:
                nxt = (cell[0] + dr, cell[1] + dc)
                if 0 < nxt[0] < rows - 1 and 0 < nxt[1] < cols - 1 and nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)

        push((1, 1))
        while frontier:
            idx = int(rng.integers(len(frontier)))
            cell = frontier.pop(idx)
            joins = [
                (cell[0] + dr, cell[1] + dc)
                for dr, dc in step
                if 0 < cell[0] + dr < rows - 1
                and 0 < cell[1] + dc < cols - 1
                and grid[cell[0] + dr, cell[1] + dc] == FREE
            ]
            if not joins:
                continue
            other = joins[int(rng.integers(len(joins)))]
            grid[cell] = FREE
            grid[(cell[0] + other[0]) // 2, (cell[1] + other[1]) // 2] = FREE
            push(cell)
        return grid

    # depth-first "recursive backtracker" — long, winding corridors
    stack: list[Coord] = [(1, 1)]
    while stack:
        r, c = stack[-1]
        options = [
            (r + dr, c + dc)
            for dr, dc in step
            if 0 < r + dr < rows - 1 and 0 < c + dc < cols - 1 and grid[r + dr, c + dc] == WALL
        ]
        if not options:
            stack.pop()
            continue
        nr, nc = options[int(rng.integers(len(options)))]
        grid[(r + nr) // 2, (c + nc) // 2] = FREE
        grid[nr, nc] = FREE
        stack.append((nr, nc))
    return grid


def _braid(grid: np.ndarray, rng: np.random.Generator, ratio: float) -> np.ndarray:
    """Knock walls out of dead ends so the maze gains loops (many valid routes)."""
    rows, cols = grid.shape
    dead_ends = [
        (r, c)
        for r in range(1, rows - 1)
        for c in range(1, cols - 1)
        if grid[r, c] == FREE
        and sum(grid[r + dr, c + dc] == FREE for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))) == 1
    ]
    rng.shuffle(dead_ends)  # type: ignore[arg-type]
    for r, c in dead_ends[: int(len(dead_ends) * float(np.clip(ratio, 0.0, 1.0)))]:
        walls = [
            (r + dr, c + dc)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 < r + dr < rows - 1 and 0 < c + dc < cols - 1 and grid[r + dr, c + dc] == WALL
        ]
        if walls:
            grid[walls[int(rng.integers(len(walls)))]] = FREE
    return grid


def generate(
    kind: str = "recursive_backtracker",
    rows: int = 21,
    cols: int = 21,
    *,
    seed: int | None = None,
    density: float = 0.28,
    braid: float = 0.35,
) -> Maze:
    """Build a guaranteed-solvable maze.

    ``kind`` selects the topology: perfect mazes (exactly one route between any
    two cells), braided mazes (loops, so many routes) or scattered obstacles.
    """
    from .search import bfs_path  # local import: search imports nothing from maze

    if kind not in GENERATORS:
        raise ValueError(f"unknown generator {kind!r}; expected one of {GENERATORS}")
    rng = _rng(seed)

    if kind in ("recursive_backtracker", "prim", "braided"):
        rows, cols = _odd(rows), _odd(cols)
        algorithm = "prim" if kind == "prim" else "recursive_backtracker"
        grid = _carve_perfect(rows, cols, rng, algorithm)
        if kind == "braided":
            grid = _braid(grid, rng, braid)
        return Maze(grid, (1, 1), (rows - 2, cols - 2))

    rows, cols = max(int(rows), 3), max(int(cols), 3)
    start, goal = (0, 0), (rows - 1, cols - 1)
    if kind == "open_field":
        return Maze(np.zeros((rows, cols), dtype=np.uint8), start, goal)

    density = float(np.clip(density, 0.0, 0.6))
    for _ in range(60):
        grid = (rng.random((rows, cols)) < density).astype(np.uint8)
        grid[start] = grid[goal] = FREE
        maze = Maze(grid, start, goal)
        if bfs_path(maze) is not None:
            return maze
    # Fallback: force an L-shaped corridor so the instance is always solvable.
    grid[start[0], :] = FREE
    grid[:, goal[1]] = FREE
    return Maze(grid, start, goal)


PRESETS: dict[str, str] = {
    "Classic 10x10": """
        S.........
        .####.###.
        .#...#...#
        .#.#.#.#..
        ...#...#.#
        .###.###.#
        .#...#...#
        .#.#.#.##.
        ...#.....#
        .#######.G
    """,
    "Spiral trap": """
        S............
        .###########.
        .#.........#.
        .#.#######.#.
        .#.#.....#.#.
        .#.#.###.#.#.
        .#.#.#G#.#.#.
        .#.#.#.#.#.#.
        .#.#...#.#.#.
        .#.#####.#.#.
        .#.......#.#.
        .#########.#.
        ...........#.
    """,
    "Serpentine trap": """
        S..........
        ##########.
        ...........
        .##########
        ...........
        ##########.
        ...........
        .##########
        ..........G
    """,
}


def preset(name: str) -> Maze:
    """Load one of the hand-designed benchmark mazes."""
    try:
        return Maze.from_text(PRESETS[name])
    except KeyError as exc:
        raise ValueError(f"unknown preset {name!r}; expected one of {sorted(PRESETS)}") from exc


def from_rows(rows: Sequence[Sequence[int]], start: Coord, goal: Coord) -> Maze:
    """Build a maze from the classic ``0/1`` nested-list format."""
    return Maze(np.asarray(rows, dtype=np.uint8), start, goal)
