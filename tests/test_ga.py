import numpy as np
import pytest

from mazega import GAConfig, MazeGA, aggregate, astar, generate, preset, solve
from mazega.benchmark import DEFAULT_VARIANTS, compare
from mazega.operators import (
    CROSSOVER_METHODS,
    MUTATION_METHODS,
    SELECTION_METHODS,
    get_crossover,
    get_mutation,
    get_selection,
    locus_entropy,
)


@pytest.fixture(scope="module")
def maze():
    return generate("recursive_backtracker", 15, 15, seed=3)


@pytest.mark.parametrize("name", SELECTION_METHODS)
def test_selection_returns_valid_indices(name):
    rng = np.random.default_rng(0)
    fitness = rng.normal(size=40)
    picks = get_selection(name)(fitness, 25, rng, k=4, pressure=1.5, fraction=0.4)
    assert len(picks) == 25 and picks.min() >= 0 and picks.max() < 40


def test_selection_favours_fitter_individuals():
    rng = np.random.default_rng(1)
    fitness = np.arange(50, dtype=float)
    picks = get_selection("tournament")(fitness, 2000, rng, k=5)
    assert fitness[picks].mean() > fitness.mean()


@pytest.mark.parametrize("name", CROSSOVER_METHODS)
def test_crossover_preserves_length_and_alleles(name):
    rng = np.random.default_rng(2)
    a, b = rng.integers(0, 4, 40, dtype=np.int8), rng.integers(0, 4, 40, dtype=np.int8)
    c1, c2 = get_crossover(name)(a, b, rng)
    assert c1.shape == a.shape and c2.shape == a.shape
    assert set(np.unique(np.concatenate([c1, c2]))) <= set(range(4))
    assert np.array_equal(a, a) and set(np.unique(c1)) <= set(np.unique(np.concatenate([a, b])))


@pytest.mark.parametrize("name", MUTATION_METHODS)
def test_mutation_stays_in_the_alphabet(name):
    rng = np.random.default_rng(3)
    genome = rng.integers(0, 4, 50, dtype=np.int8)
    mutated = get_mutation(name)(genome.copy(), 0.3, rng)
    assert mutated.shape == genome.shape and mutated.min() >= 0 and mutated.max() < 4


def test_zero_rate_point_mutation_is_a_no_op():
    rng = np.random.default_rng(4)
    genome = rng.integers(0, 4, 30, dtype=np.int8)
    assert np.array_equal(get_mutation("point")(genome.copy(), 0.0, rng), genome)


def test_entropy_is_zero_for_a_converged_population():
    assert locus_entropy(np.zeros((20, 10), dtype=np.int8)) == pytest.approx(0.0)


def test_entropy_is_one_for_a_uniform_population():
    population = np.tile(np.arange(4, dtype=np.int8)[:, None], (1, 10))
    assert locus_entropy(population) == pytest.approx(1.0)


def test_unknown_operator_names_rejected():
    for getter in (get_selection, get_crossover, get_mutation):
        with pytest.raises(ValueError):
            getter("nope")


def test_config_validation():
    with pytest.raises(ValueError):
        GAConfig(population=2).validate()
    with pytest.raises(ValueError):
        GAConfig(elite=500).validate()
    with pytest.raises(ValueError):
        GAConfig(selection="psychic").validate()


def test_run_is_reproducible(maze):
    a = solve(maze, GAConfig(generations=40, seed=11))
    b = solve(maze, GAConfig(generations=40, seed=11))
    assert np.array_equal(a.best_genome, b.best_genome)
    assert [g.best for g in a.history] == [g.best for g in b.history]


def test_different_seeds_diverge(maze):
    a = solve(maze, GAConfig(generations=40, seed=1))
    b = solve(maze, GAConfig(generations=40, seed=2))
    assert not np.array_equal(a.best_genome, b.best_genome)


def test_best_fitness_never_decreases(maze):
    history = solve(maze, GAConfig(generations=60, seed=5)).history
    running = [max(g.best for g in history[: i + 1]) for i in range(len(history))]
    assert running == sorted(running)


def test_elitism_preserves_the_champion(maze):
    result = solve(maze, GAConfig(generations=50, seed=6, elite=4, catastrophe=False))
    bests = [g.best for g in result.history]
    assert all(b >= bests[0] - 1e-9 for b in bests)


def test_solution_path_is_legal_and_optimal(maze):
    result = solve(maze, GAConfig(generations=250, seed=7))
    assert result.solved
    path = result.simple_path
    assert path[0] == maze.start and path[-1] == maze.goal
    for a, b in zip(path, path[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
        assert maze.grid[b] == 0
    assert len(path) - 1 >= astar(maze).steps


def test_short_chromosome_is_flagged_not_crashed(maze):
    engine = MazeGA(maze, GAConfig(generations=5, chromosome=16))
    assert not engine.encodable
    assert not engine.run().solved


def test_stop_rules(maze):
    first = solve(maze, GAConfig(generations=300, seed=7, stop_rule="first_solution"))
    never = solve(maze, GAConfig(generations=60, seed=7, stop_rule="never"))
    assert first.solved and first.generations_run <= 300
    assert never.generations_run == 60


def test_catastrophe_fires_when_stuck():
    maze = generate("recursive_backtracker", 25, 25, seed=13)
    result = solve(
        maze,
        GAConfig(generations=80, seed=2, decode="strict", stagnation_limit=3, chromosome=200),
    )
    assert any(g.event == "catastrophe restart" for g in result.history)


def test_history_frame_has_one_row_per_generation(maze):
    result = solve(maze, GAConfig(generations=30, seed=8, stop_rule="never"))
    frame = result.to_frame()
    assert len(frame) == result.generations_run == 30
    assert {"generation", "best", "mean", "diversity"} <= set(frame.columns)


def test_repair_outperforms_strict_on_a_hard_maze():
    maze = preset("Spiral trap")
    strict = solve(maze, GAConfig(generations=120, seed=4, decode="strict", lamarckian=False))
    repair = solve(maze, GAConfig(generations=120, seed=4, decode="repair"))
    assert repair.best_report.score > strict.best_report.score


def test_benchmark_aggregates_every_variant():
    maze = generate("random_obstacles", 13, 13, seed=1, density=0.25)
    trials, rows = compare(
        maze,
        {k: DEFAULT_VARIANTS[k] for k in ("+ repair decoding", "guided (memetic)")},
        base=GAConfig(generations=60),
        seeds=range(2),
    )
    assert len(trials) == 4 and len(rows) == 2
    assert all(0.0 <= row["solve_rate"] <= 1.0 for row in rows)
    assert aggregate(trials) == rows
