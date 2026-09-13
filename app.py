"""MazeGA — an interactive lab for solving mazes with a genetic algorithm.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import time
from dataclasses import replace

import pandas as pd
import streamlit as st

from mazega import (
    DEFAULT_VARIANTS,
    FitnessWeights,
    GAConfig,
    Maze,
    MazeGA,
    astar,
    compare,
    generate,
    preset,
    prune_loops,
)
from mazega.maze import FREE, GENERATORS, PRESETS, WALL
from mazega.plots import (
    benchmark_figure,
    contribution_figure,
    diversity_figure,
    fitness_figure,
    maze_figure,
)

st.set_page_config(page_title="MazeGA", page_icon="🧬", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 2rem;}
      [data-testid="stMetricValue"] {font-size: 1.45rem;}
      [data-testid="stSidebar"] {min-width: 21rem;}
      h1 {letter-spacing: -0.02em;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------- state
def boot() -> None:
    st.session_state.setdefault("maze", generate("recursive_backtracker", 21, 21, seed=7))
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("bench", None)
    st.session_state.setdefault("tool", "Toggle wall")


def set_maze(maze: Maze) -> None:
    st.session_state.maze = maze
    st.session_state.result = None


boot()


# ------------------------------------------------------------------- sidebar
def maze_controls() -> None:
    st.sidebar.subheader("1 · Maze")
    source = st.sidebar.radio(
        "Source", ["Generate", "Preset", "Paste ASCII"], horizontal=True, label_visibility="collapsed"
    )

    if source == "Generate":
        kind = st.sidebar.selectbox("Algorithm", GENERATORS, index=0,
                                    format_func=lambda s: s.replace("_", " "))
        size = st.sidebar.slider("Size (cells per side)", 9, 45, 21, step=2)
        seed = st.sidebar.number_input("Maze seed", 0, 9999, 7, step=1)
        density = braid = 0.0
        if kind == "random_obstacles":
            density = st.sidebar.slider("Obstacle density", 0.05, 0.45, 0.28, step=0.01)
        if kind == "braided":
            braid = st.sidebar.slider("Braiding (loops)", 0.0, 1.0, 0.35, step=0.05)
        if st.sidebar.button("Generate maze", type="primary", width="stretch"):
            set_maze(generate(kind, size, size, seed=int(seed), density=density, braid=braid))

    elif source == "Preset":
        name = st.sidebar.selectbox("Benchmark maze", list(PRESETS))
        if st.sidebar.button("Load preset", type="primary", width="stretch"):
            set_maze(preset(name))

    else:
        text = st.sidebar.text_area(
            "`#` wall · `.` free · `S` start · `G` goal",
            value=st.session_state.maze.to_text(),
            height=190,
        )
        if st.sidebar.button("Parse maze", type="primary", width="stretch"):
            try:
                set_maze(Maze.from_text(text))
            except Exception as exc:  # surface parse errors instead of a traceback
                st.sidebar.error(str(exc))


def ga_controls() -> GAConfig:
    st.sidebar.subheader("2 · Algorithm")
    maze = st.session_state.maze

    with st.sidebar.expander("Population", expanded=True):
        population = st.slider("Population size", 20, 600, 180, step=10)
        generations = st.slider("Generations", 20, 800, 250, step=10)
        elite = st.slider("Elites carried over", 0, 30, 6)
        auto_len = st.toggle("Auto chromosome length", value=True,
                             help="Scales with the exact shortest path — the single most "
                                  "important hyper-parameter here.")
        chromosome = None
        if not auto_len:
            chromosome = st.slider("Chromosome length", 16, 1200, 200, step=8)

    with st.sidebar.expander("Operators"):
        selection = st.selectbox("Selection", ["tournament", "rank", "roulette", "sus", "truncation"])
        tournament_size = st.slider("Tournament size", 2, 20, 5) if selection == "tournament" else 5
        rank_pressure = st.slider("Rank pressure", 1.0, 2.0, 1.8, 0.05) if selection == "rank" else 1.8
        crossover = st.selectbox("Crossover", ["two_point", "one_point", "uniform"])
        crossover_rate = st.slider("Crossover rate", 0.0, 1.0, 0.9, 0.05)
        mutation = st.selectbox("Mutation", ["point", "mixed", "swap", "scramble"])
        mutation_rate = st.slider("Base mutation rate", 0.0, 0.4, 0.04, 0.01)

    with st.sidebar.expander("Adaptation"):
        adaptive = st.toggle("Adaptive mutation on stagnation", value=True)
        max_mutation_rate = st.slider("Mutation ceiling", 0.05, 0.6, 0.30, 0.05, disabled=not adaptive)
        stagnation_limit = st.slider("Stagnation limit", 3, 60, 18)
        catastrophe = st.toggle("Catastrophe restarts", value=True)
        immigrant_rate = st.slider("Random immigrants", 0.0, 0.3, 0.04, 0.01)

    with st.sidebar.expander("Decoder & fitness"):
        decode = st.select_slider(
            "Wall-collision policy",
            options=["strict", "skip", "repair", "guided"],
            value="repair",
            help="strict = walk dies · skip = ignore gene · repair = rotate to a legal "
                 "step · guided = greedy memetic step toward the goal",
        )
        lamarckian = st.toggle("Lamarckian write-back", value=True,
                               disabled=decode in ("strict", "skip"))
        heuristic = st.radio("Distance signal", ["geodesic", "manhattan"], horizontal=True,
                             help="geodesic = true shortest-path distance (no local optima "
                                  "behind walls); manhattan = the naive straight-line guess")
        goal_w = st.slider("Reward · reach goal", 0.0, 5.0, 2.0, 0.1)
        progress_w = st.slider("Reward · closest approach", 0.0, 3.0, 1.0, 0.1)
        efficiency_w = st.slider("Reward · short path", 0.0, 3.0, 1.0, 0.1)
        collision_w = st.slider("Penalty · wall hits", 0.0, 2.0, 0.35, 0.05)
        revisit_w = st.slider("Penalty · revisits", 0.0, 2.0, 0.35, 0.05)

    with st.sidebar.expander("Run"):
        stop_rule = st.selectbox("Stop when", ["optimal", "first_solution", "never"], index=0)
        patience = st.slider("Patience after solving", 5, 150, 40)
        seed = st.number_input("GA seed", 0, 9999, 42, step=1)

    return GAConfig(
        population=population,
        generations=generations,
        chromosome=chromosome,
        elite=min(elite, population - 1),
        selection=selection,
        tournament_size=tournament_size,
        rank_pressure=rank_pressure,
        crossover=crossover,
        crossover_rate=crossover_rate,
        mutation=mutation,
        mutation_rate=mutation_rate,
        adaptive_mutation=adaptive,
        max_mutation_rate=max_mutation_rate,
        stagnation_limit=stagnation_limit,
        catastrophe=catastrophe,
        immigrant_rate=immigrant_rate,
        decode=decode,
        lamarckian=lamarckian,
        weights=FitnessWeights(goal_w, progress_w, efficiency_w, collision_w, revisit_w, heuristic),
        stop_rule=stop_rule,
        patience=patience,
        seed=int(seed),
    )


maze_controls()
config = ga_controls()
maze = st.session_state.maze
optimal = astar(maze)


# -------------------------------------------------------------------- header
st.title("🧬 MazeGA")
st.caption(
    "Evolve a path through a maze, then check the answer against A\\*. "
    "Every mechanism in the algorithm can be switched off to see what it was worth."
)

head = st.columns(5)
head[0].metric("Maze", f"{maze.rows} × {maze.cols}", border=True)
head[1].metric("Open cells", maze.open_cells, border=True)
head[2].metric("A\\* optimum", f"{optimal.steps} steps" if optimal.solved else "unreachable", border=True)
result = st.session_state.result
head[3].metric("GA best", f"{max(len(result.simple_path) - 1, 0)} steps" if result and result.solved
               else ("no solution" if result else "—"), border=True)
head[4].metric(
    "Optimality",
    f"{result.optimality_ratio:.0%}" if result and result.optimality_ratio is not None else "—",
    border=True,
    help="Shortest possible path ÷ path the GA found. 100% means it matched A\\* exactly.",
)

build_tab, evolve_tab, analytics_tab, bench_tab, docs_tab = st.tabs(
    ["Build", "Evolve", "Analytics", "Benchmark", "How it works"]
)


# --------------------------------------------------------------------- build
with build_tab:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.session_state.tool = st.segmented_control(
            "Click a cell to…",
            ["Toggle wall", "Move start", "Move goal"],
            default=st.session_state.tool,
        ) or st.session_state.tool

        event = st.plotly_chart(
            maze_figure(maze, optimal=optimal.path if optimal.solved else None,
                        title="click any cell to edit", clickable=True, height=560),
            key="editor",
            on_select="rerun",
            selection_mode="points",
            theme=None,
            config={"displayModeBar": False},
        )
        points = (event or {}).get("selection", {}).get("points", [])
        if points:
            cell = (int(round(points[-1]["y"])), int(round(points[-1]["x"])))
            tool = st.session_state.tool
            try:
                if tool == "Move start":
                    set_maze(maze.with_endpoints(start=cell))
                elif tool == "Move goal":
                    set_maze(maze.with_endpoints(goal=cell))
                elif cell not in (maze.start, maze.goal):
                    set_maze(maze.with_cell(cell, FREE if maze.grid[cell] == WALL else WALL))
                st.rerun()
            except Exception as exc:
                st.warning(str(exc))

    with right:
        st.subheader("Instance")
        st.markdown(
            f"- **Reachable:** {'yes' if optimal.solved else 'no'}\n"
            f"- **Shortest path:** {optimal.steps} steps\n"
            f"- **A\\* expansions:** {optimal.expanded}\n"
            f"- **Wall ratio:** {1 - maze.open_cells / (maze.rows * maze.cols):.0%}"
        )
        if not optimal.solved:
            st.error("The goal is walled off — no algorithm can solve this. Edit the maze.")
        st.download_button("Download maze (.txt)", maze.to_text(), "maze.txt", width="stretch")
        st.code(maze.to_text(), language="text")


# -------------------------------------------------------------------- evolve
with evolve_tab:
    controls = st.columns([1, 1, 4])
    start_run = controls[0].button("▶ Run evolution", type="primary", width="stretch")
    live = controls[1].toggle("Live view", value=True)

    if start_run:
        engine = MazeGA(maze, config)
        if not engine.encodable:
            st.error(
                f"Chromosome holds {engine.length} moves but the shortest path needs "
                f"{optimal.steps}. No solution can be encoded — lengthen the chromosome."
            )
        else:
            st.caption(f"chromosome = {engine.length} moves · search space = 4^{engine.length}")
            bar = st.progress(0.0)
            board, charts = st.empty(), st.empty()
            every = max(1, config.generations // 60)
            started = time.perf_counter()

            for record in engine.evolve():
                if live and (record.index % every == 0 or record.event):
                    board.plotly_chart(
                        maze_figure(
                            maze,
                            prune_loops(record.best_path),
                            optimal=optimal.path if optimal.solved else None,
                            title=f"generation {record.index} · fitness {record.best:.3f}",
                            height=460,
                        ),
                        key=f"live-{record.index}",
                        theme=None,
                        config={"displayModeBar": False},
                    )
                    charts.plotly_chart(fitness_figure(engine.history, height=240),
                                        key=f"live-fit-{record.index}", theme=None,
                                        config={"displayModeBar": False})
                bar.progress(min((record.index + 1) / config.generations, 1.0),
                             f"generation {record.index} · best {record.best:.3f}"
                             + (f" · {record.event}" if record.event else ""))

            bar.empty()
            board.empty()
            charts.empty()
            st.session_state.result = engine.result
            st.toast(
                f"Solved in {engine.result.generation_solved} generations"
                if engine.result.solved else "No solution found",
                icon="🎯" if engine.result.solved else "⚠️",
            )
            st.rerun()  # refresh the header metrics with the finished run

    result = st.session_state.result
    if result is None:
        st.info("Set the parameters in the sidebar, then press **Run evolution**.")
    else:
        info = result.summary()
        cols = st.columns(4)
        cols[0].metric("Generations", info["generations"], border=True)
        cols[1].metric("Solved at gen",
                       info["generation_solved"] if info["solved"] else "—", border=True)
        cols[2].metric("Evaluations", f"{info['evaluations']:,}", border=True)
        cols[3].metric("Wall time", f"{info['seconds']:.2f}s", border=True)

        frame = st.slider("Replay generation", 0, result.generations_run - 1,
                          result.generations_run - 1) if result.generations_run > 1 else 0
        record = result.history[frame]
        view, side = st.columns([3, 2], gap="large")
        with view:
            st.plotly_chart(
                maze_figure(
                    maze,
                    prune_loops(record.best_path),
                    optimal=optimal.path if optimal.solved else None,
                    title=f"generation {record.index} · {'solved' if record.solved else 'searching'}",
                    height=520,
                ),
                key="replay", theme=None, config={"displayModeBar": False},
            )
        with side:
            if record.report is not None:
                st.plotly_chart(contribution_figure(record.report), key="terms", theme=None,
                                config={"displayModeBar": False})
                st.dataframe(pd.DataFrame([record.report.as_dict()]).T.rename(columns={0: "value"}),
                             width="stretch")
            st.text_input("Winning chromosome", result.moves()[:200], disabled=True)


# ----------------------------------------------------------------- analytics
with analytics_tab:
    result = st.session_state.result
    if result is None:
        st.info("Run the algorithm first — this tab plots the recorded history.")
    else:
        st.plotly_chart(fitness_figure(result.history, height=330), key="fit", theme=None)
        st.plotly_chart(diversity_figure(result.history, height=300), key="div", theme=None)

        restarts = sum(g.event == "catastrophe restart" for g in result.history)
        notes = st.columns(3)
        notes[0].metric("Catastrophe restarts", restarts, border=True)
        notes[1].metric("Final gene entropy", f"{result.history[-1].diversity:.2f}", border=True,
                        help="1.0 = fully diverse, 0 = every individual identical")
        notes[2].metric("Peak share solving", f"{max(g.success_rate for g in result.history):.0%}",
                        border=True)

        table = result.to_frame()
        st.dataframe(table, width="stretch", height=280, hide_index=True)
        st.download_button("Download history (.csv)", table.to_csv(index=False),
                           "mazega_history.csv", mime="text/csv")


# ----------------------------------------------------------------- benchmark
with bench_tab:
    st.write(
        "Each variant re-runs the **same maze** under several seeds, switching one "
        "mechanism at a time. This is the evidence behind the defaults."
    )
    picks = st.multiselect("Variants", list(DEFAULT_VARIANTS), default=list(DEFAULT_VARIANTS))
    row = st.columns(2)
    trials = row[0].slider("Seeds per variant", 2, 12, 5)
    budget = row[1].slider("Generations per trial", 20, 400, 120, step=10,
                           help="Failing variants always burn the full budget, so keep this "
                                "modest while exploring.")

    if st.button("⚖ Run benchmark", type="primary"):
        bar = st.progress(0.0)
        _, rows = compare(
            maze,
            {k: DEFAULT_VARIANTS[k] for k in picks},
            base=replace(config, generations=budget),
            seeds=range(trials),
            progress=lambda pct, label: bar.progress(pct, label),
        )
        bar.empty()
        st.session_state.bench = rows

    if st.session_state.bench:
        rows = st.session_state.bench
        st.plotly_chart(benchmark_figure(rows), key="bench", theme=None)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.download_button("Download results (.csv)", pd.DataFrame(rows).to_csv(index=False),
                           "mazega_benchmark.csv", mime="text/csv")


# ---------------------------------------------------------------------- docs
with docs_tab:
    st.markdown(
        """
### The encoding
A chromosome is a fixed-length string of moves over `{U, D, L, R}`. Decoding walks
the maze from the start, one gene at a time. The search space is `4^L`, so a
200-gene chromosome spans about `10^120` candidates.

### What happens at a wall
This is the whole design problem, and it is a switch in the sidebar:

| policy | behaviour | effect |
|---|---|---|
| `strict` | the walk ends | one bad gene discards the entire tail — the naive version |
| `skip` | gene is ignored | later genes still get expressed |
| `repair` | rotate to the next legal move, preferring unvisited cells | uses no goal knowledge, so evolution still does the work |
| `guided` | step to the legal cell nearest the goal | memetic; very strong, and it makes corridor mazes almost trivial |

### Fitness
A weighted sum of normalised terms, so every weight is interpretable:

```
score = w_goal·reached
      + w_progress·(1 − closest_distance / start_distance)
      + w_efficiency·(optimal_steps / path_steps)
      − w_collision·collision_rate
      − w_revisit·revisit_rate
```

`closest_distance` is measured with a BFS flood-fill from the goal, not straight-line
distance. Manhattan distance creates local optima on the wrong side of a wall — you
can switch to it in the sidebar and watch the search stall.

### Keeping the population alive
Elitism preserves the best individuals; **adaptive mutation** raises the rate while the
best score is stuck; **random immigrants** trickle in fresh genes every generation; and a
**catastrophe restart** rebuilds everyone but the elites after a long stall. Gene entropy
in the Analytics tab shows whether any of that is still working.

### Honest limits
The GA matches A\\* on grids up to roughly 25×25. Past that, the walk has to be
near-perfect over hundreds of genes and the plain operators stall — `guided`
decoding is what carries the larger instances. A genetic algorithm is the wrong
tool for shortest paths; it is a good way to *see* how one behaves.
"""
    )
