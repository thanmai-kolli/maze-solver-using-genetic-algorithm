import numpy as np
import pytest

from mazega import Maze, astar, bfs, distance_field, generate, is_solvable, preset
from mazega.maze import FREE, WALL, GENERATORS, PRESETS, MazeError
from mazega.search import UNREACHABLE


@pytest.mark.parametrize("name", list(PRESETS))
def test_presets_are_solvable(name):
    maze = preset(name)
    assert is_solvable(maze), f"preset {name!r} has no route from start to goal"


def test_from_text_roundtrip():
    maze = preset("Classic 10x10")
    assert Maze.from_text(maze.to_text()).to_text() == maze.to_text()


def test_text_requires_endpoints():
    with pytest.raises(MazeError):
        Maze.from_text("...\n...\n...")


def test_start_on_wall_rejected():
    with pytest.raises(MazeError):
        Maze(np.ones((3, 3), dtype=np.uint8), (0, 0), (2, 2))


@pytest.mark.parametrize("kind", GENERATORS)
def test_generators_are_solvable(kind):
    maze = generate(kind, 15, 15, seed=3, density=0.3)
    assert is_solvable(maze)
    assert maze.grid[maze.start] == FREE and maze.grid[maze.goal] == FREE


def test_perfect_maze_has_odd_dimensions():
    maze = generate("recursive_backtracker", 20, 16, seed=1)
    assert maze.rows % 2 == 1 and maze.cols % 2 == 1


def test_braiding_removes_dead_ends():
    seed = 5
    perfect = generate("recursive_backtracker", 21, 21, seed=seed)
    braided = generate("braided", 21, 21, seed=seed, braid=1.0)
    assert braided.open_cells > perfect.open_cells


def test_generation_is_reproducible():
    a = generate("prim", 15, 15, seed=9)
    b = generate("prim", 15, 15, seed=9)
    assert np.array_equal(a.grid, b.grid)


def test_astar_matches_bfs_and_expands_less():
    maze = generate("braided", 21, 21, seed=4)
    exact, heuristic = bfs(maze), astar(maze)
    assert exact.steps == heuristic.steps
    assert heuristic.expanded <= exact.expanded


def test_path_is_contiguous_and_legal():
    maze = generate("recursive_backtracker", 17, 17, seed=2)
    path = astar(maze).path
    assert path[0] == maze.start and path[-1] == maze.goal
    for a, b in zip(path, path[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
        assert maze.grid[b] == FREE


def test_unsolvable_maze_reports_no_path():
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[:, 2] = WALL
    maze = Maze(grid, (0, 0), (4, 4))
    assert bfs(maze).path is None
    assert astar(maze).path is None


def test_distance_field_marks_unreachable_cells():
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[:, 2] = WALL
    field = distance_field(Maze(grid, (0, 0), (0, 0)))
    assert field[0, 0] == 0
    assert field[0, 4] == UNREACHABLE


def test_editing_keeps_endpoints_open():
    maze = generate("open_field", 5, 5, seed=0)
    assert np.array_equal(maze.with_cell(maze.start, WALL).grid, maze.grid)
    moved = maze.with_endpoints(goal=(2, 2))
    assert moved.goal == (2, 2) and moved.grid[2, 2] == FREE


def test_tables_match_the_grid():
    maze = generate("braided", 13, 13, seed=6)
    tables = maze.tables
    for r in range(maze.rows):
        for c in range(maze.cols):
            index = r * maze.cols + c
            legal = {n for n in tables.neighbours[index] if n >= 0}
            expected = {nr * maze.cols + nc for _, (nr, nc) in maze.neighbours((r, c))}
            assert legal == (expected if maze.grid[r, c] == FREE else set())
