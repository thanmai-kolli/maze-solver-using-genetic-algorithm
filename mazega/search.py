"""Exact grid searches used as optimality baselines and as fitness landscapes."""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass

import numpy as np

from .maze import FREE, Coord, Maze

UNREACHABLE = -1


@dataclass(frozen=True)
class SearchResult:
    """Outcome of an exact search, used to score how close the GA got to optimal."""

    path: list[Coord] | None
    expanded: int
    algorithm: str

    @property
    def solved(self) -> bool:
        return self.path is not None

    @property
    def length(self) -> int:
        """Number of cells in the path (start inclusive); 0 when unsolved."""
        return len(self.path) if self.path else 0

    @property
    def steps(self) -> int:
        return max(self.length - 1, 0)


def _reconstruct(came_from: dict[Coord, Coord], goal: Coord, start: Coord) -> list[Coord]:
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def bfs(maze: Maze) -> SearchResult:
    """Breadth-first search — optimal on an unweighted grid."""
    start, goal = maze.start, maze.goal
    if start == goal:
        return SearchResult([start], 1, "bfs")
    frontier: deque[Coord] = deque([start])
    came_from: dict[Coord, Coord] = {}
    seen = {start}
    expanded = 0
    while frontier:
        cur = frontier.popleft()
        expanded += 1
        for _, nxt in maze.neighbours(cur):
            if nxt in seen:
                continue
            seen.add(nxt)
            came_from[nxt] = cur
            if nxt == goal:
                return SearchResult(_reconstruct(came_from, goal, start), expanded, "bfs")
            frontier.append(nxt)
    return SearchResult(None, expanded, "bfs")


def astar(maze: Maze) -> SearchResult:
    """A* with the Manhattan heuristic — same optimum as BFS, fewer expansions."""
    start, goal = maze.start, maze.goal
    counter = 0
    open_heap: list[tuple[int, int, Coord]] = [(maze.manhattan(start, goal), 0, start)]
    g_score: dict[Coord, int] = {start: 0}
    came_from: dict[Coord, Coord] = {}
    closed: set[Coord] = set()
    expanded = 0
    while open_heap:
        _, _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)
        expanded += 1
        if cur == goal:
            return SearchResult(_reconstruct(came_from, goal, start), expanded, "astar")
        for _, nxt in maze.neighbours(cur):
            tentative = g_score[cur] + 1
            if tentative < g_score.get(nxt, 1 << 30):
                g_score[nxt] = tentative
                came_from[nxt] = cur
                counter += 1
                heapq.heappush(open_heap, (tentative + maze.manhattan(nxt, goal), counter, nxt))
    return SearchResult(None, expanded, "astar")


def bfs_path(maze: Maze) -> list[Coord] | None:
    """Convenience wrapper returning just the optimal path (or ``None``)."""
    return bfs(maze).path


def distance_field(maze: Maze, origin: Coord | None = None) -> np.ndarray:
    """Flood-fill the true geodesic distance from ``origin`` (default: the goal).

    Unlike the Manhattan heuristic this respects walls, so a fitness function
    built on it has no local optima behind a barrier — the classic failure mode
    of the naive distance-to-goal reward.
    """
    origin = origin if origin is not None else maze.goal
    field = np.full(maze.shape, UNREACHABLE, dtype=np.int32)
    if maze.grid[origin] != FREE:
        return field
    field[origin] = 0
    frontier: deque[Coord] = deque([origin])
    while frontier:
        cur = frontier.popleft()
        for _, nxt in maze.neighbours(cur):
            if field[nxt] == UNREACHABLE:
                field[nxt] = field[cur] + 1
                frontier.append(nxt)
    return field


def is_solvable(maze: Maze) -> bool:
    return bfs_path(maze) is not None
