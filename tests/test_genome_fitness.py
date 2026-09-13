import numpy as np
import pytest

from mazega import FitnessEvaluator, FitnessWeights, Maze, decode, generate, preset, prune_loops
from mazega.genome import DECODE_MODES, from_moves, random_population, suggested_length, to_moves
from mazega.maze import WALL


def corridor() -> Maze:
    """A 1x5 corridor: the only solution is "RRRR"."""
    return Maze(np.zeros((1, 5), dtype=np.uint8), (0, 0), (0, 4))


def test_moves_roundtrip():
    genome = random_population(1, 32, np.random.default_rng(0))[0]
    assert np.array_equal(from_moves(to_moves(genome)), genome)


def test_perfect_genome_reaches_goal_in_every_mode():
    maze = corridor()
    for mode in DECODE_MODES:
        walk = decode(from_moves("RRRR"), maze, mode)
        assert walk.reached and walk.steps == 4 and walk.collisions == 0


def test_strict_stops_at_the_first_wall():
    walk = decode(from_moves("RULL"), corridor(), "strict")
    assert not walk.reached and walk.steps == 1 and walk.collisions == 1


def test_skip_ignores_blocked_genes():
    walk = decode(from_moves("RURURURR"), corridor(), "skip")
    assert walk.reached and walk.collisions == 3


def test_repair_substitutes_a_legal_move():
    walk = decode(from_moves("UUUU"), corridor(), "repair")
    assert walk.collisions == 4 and walk.steps > 0
    assert all(walk.trail[i] != walk.trail[i + 1] for i in range(len(walk.trail) - 1))


def test_guided_walks_toward_the_goal():
    walk = decode(from_moves("UUUU"), corridor(), "guided")
    assert walk.reached


def test_writeback_makes_the_genome_self_consistent():
    genome = from_moves("UUUU")
    walk = decode(genome, corridor(), "repair", writeback=True)
    assert walk.genome is not None and not np.array_equal(walk.genome, genome)
    assert decode(walk.genome, corridor(), "repair").collisions == 0


def test_unknown_decode_mode_rejected():
    with pytest.raises(ValueError):
        decode(from_moves("RR"), corridor(), "teleport")  # type: ignore[arg-type]


def test_prune_loops_removes_cycles():
    assert prune_loops([(0, 0), (0, 1), (0, 2), (0, 1), (1, 1)]) == [(0, 0), (0, 1), (1, 1)]
    straight = [(0, 0), (0, 1), (0, 2)]
    assert prune_loops(straight) == straight


def test_suggested_length_can_encode_the_optimum():
    for seed in range(5):
        maze = generate("recursive_backtracker", 21, 21, seed=seed)
        from mazega import astar

        assert suggested_length(maze) >= astar(maze).steps


def test_fitness_rewards_the_goal_most():
    maze = preset("Classic 10x10")
    evaluator = FitnessEvaluator(maze, decode_mode="strict")
    winner, _ = evaluator.evaluate(from_moves(_solution_moves(maze)))
    loser, _ = evaluator.evaluate(from_moves("UUUU"))
    assert winner.reached and not loser.reached
    assert winner.score > loser.score


def test_fitness_terms_sum_to_the_score():
    maze = generate("braided", 15, 15, seed=1)
    evaluator = FitnessEvaluator(maze)
    report, _ = evaluator.evaluate(random_population(1, 60, np.random.default_rng(2))[0])
    assert report.score == pytest.approx(sum(report.terms.values()))


def test_collisions_are_penalised():
    maze = corridor()
    weights = FitnessWeights(collision=1.0)
    evaluator = FitnessEvaluator(maze, weights, decode_mode="skip")
    clean, _ = evaluator.evaluate(from_moves("RRRR"))
    messy, _ = evaluator.evaluate(from_moves("RURURURR"))
    assert clean.score > messy.score


def test_geodesic_beats_manhattan_behind_a_wall():
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[1:, 3] = WALL
    maze = Maze(grid, (4, 0), (4, 4))
    geodesic = FitnessEvaluator(maze, FitnessWeights(heuristic="geodesic"))
    manhattan = FitnessEvaluator(maze, FitnessWeights(heuristic="manhattan"))
    trap = (4, 2)  # adjacent to the goal in Manhattan terms, but far around the wall
    assert manhattan.distance(trap) == 2
    assert geodesic.distance(trap) > manhattan.distance(trap)


def _solution_moves(maze: Maze) -> str:
    from mazega import astar
    from mazega.maze import DIRECTIONS

    lookup = {(dr, dc): name for name, dr, dc in DIRECTIONS}
    path = astar(maze).path
    assert path is not None
    return "".join(lookup[(b[0] - a[0], b[1] - a[1])] for a, b in zip(path, path[1:]))
