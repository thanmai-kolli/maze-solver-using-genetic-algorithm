import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

# -----------------------------
# INPUT MAZE
# -----------------------------

def get_maze_input():

    rows, cols = map(int, input("Enter maze size (rows cols): ").split())

    print("Enter maze row by row (0=free, 1=wall)")

    maze = []

    for i in range(rows):

        while True:
            row = list(map(int,input().split()))

            if len(row) != cols:
                print("Invalid row length, re-enter")
            else:
                maze.append(row)
                break

    start = tuple(map(int,input("Enter start (row col): ").split()))
    goal = tuple(map(int,input("Enter goal (row col): ").split()))

    return np.array(maze), start, goal


# -----------------------------
# GA PARAMETERS
# -----------------------------

POP_SIZE = 250
CHROM_LEN = 80
GENERATIONS = 120
MUT_RATE = 0.08
ELITE = 5

MOVES = ['U','D','L','R']


# -----------------------------
# MOVEMENT
# -----------------------------

def move(pos,m):

    r,c = pos

    if m=='U': r-=1
    if m=='D': r+=1
    if m=='L': c-=1
    if m=='R': c+=1

    return (r,c)


def valid(pos,maze):

    r,c = pos
    rows,cols = maze.shape

    return 0<=r<rows and 0<=c<cols and maze[r][c]==0


# -----------------------------
# PATH EXTRACTION
# -----------------------------

def extract_path(chrom,maze,start,goal):

    pos = start
    path = [start]

    for g in chrom:

        new = move(pos,g)

        if not valid(new,maze):
            break

        pos = new
        path.append(pos)

        if pos == goal:
            break

    return path


# -----------------------------
# FITNESS
# -----------------------------

def fitness(chrom,maze,start,goal):

    path = extract_path(chrom,maze,start,goal)

    last = path[-1]

    if last == goal:
        return 5000 - len(path)

    dist = abs(last[0]-goal[0]) + abs(last[1]-goal[1])

    return 1/(dist+1)


# -----------------------------
# GA OPERATORS
# -----------------------------

def create_chrom():
    return [random.choice(MOVES) for _ in range(CHROM_LEN)]


def crossover(p1,p2):

    point = random.randint(1,CHROM_LEN-1)

    return p1[:point] + p2[point:]


def mutate(chrom):

    for i in range(len(chrom)):

        if random.random() < MUT_RATE:
            chrom[i] = random.choice(MOVES)

    return chrom


# -----------------------------
# GENETIC ALGORITHM
# -----------------------------

def run_ga(maze,start,goal):

    population = [create_chrom() for _ in range(POP_SIZE)]

    best_paths = []
    best_history = []
    avg_history = []
    table_data = []

    for gen in range(GENERATIONS):

        fits = [fitness(c,maze,start,goal) for c in population]

        best_fit = max(fits)
        avg_fit = sum(fits)/len(fits)

        best_history.append(best_fit)
        avg_history.append(avg_fit)

        table_data.append([gen,best_fit,avg_fit])

        population.sort(key=lambda x:fitness(x,maze,start,goal),reverse=True)

        best = population[0]

        best_path = extract_path(best,maze,start,goal)

        best_paths.append(best_path)

        # NEW PROFESSIONAL OUTPUT
        path_length = len(best_path)

        goal_status = "Yes" if best_path[-1] == goal else "No"

        print(f"Generation {gen} | Best Fitness = {best_fit:.4f} | Path Length = {path_length} | Goal Reached = {goal_status}")

        if best_path[-1] == goal:
            print("Goal Reached Successfully!")
            break

        new_pop = population[:ELITE]

        selected = population[:POP_SIZE//2]

        while len(new_pop) < POP_SIZE:

            p1 = random.choice(selected)
            p2 = random.choice(selected)

            child = crossover(p1,p2)
            child = mutate(child)

            new_pop.append(child)

        population = new_pop

    return best_paths,best_history,avg_history,table_data


# -----------------------------
# VISUALIZATION
# -----------------------------

def show_input_maze(maze,start,goal):

    plt.figure()

    plt.imshow(maze,cmap="gray_r")

    plt.scatter(start[1],start[0])
    plt.scatter(goal[1],goal[0])

    plt.title("Input Maze Structure")

    plt.show()


def show_fitness_graph(best_history,avg_history):

    plt.figure()

    plt.plot(best_history,label="Best Fitness")
    plt.plot(avg_history,label="Average Fitness")

    plt.xlabel("Generation")
    plt.ylabel("Fitness Value")

    plt.title("Genetic Algorithm Fitness Comparison Across Generations")

    plt.legend()

    plt.show()


def show_table(table_data):

    df = pd.DataFrame(table_data,columns=[
        "Generation",
        "Best Fitness",
        "Average Fitness"
    ])

    print("\nGENERATION PERFORMANCE COMPARISON\n")

    print(df)


def animate_paths(maze,best_paths,start,goal):

    fig,ax = plt.subplots()

    def update(frame):

        ax.clear()

        ax.imshow(maze,cmap="gray_r")

        path = best_paths[frame]

        x = [p[1] for p in path]
        y = [p[0] for p in path]

        ax.plot(x,y)

        ax.scatter(start[1],start[0])
        ax.scatter(goal[1],goal[0])

        ax.set_title(f"Evolution of Best Path – Generation {frame}")

        ax.set_xticks([])
        ax.set_yticks([])

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(best_paths),
        interval=350,
        repeat=False
    )

    plt.show()


# -----------------------------
# MAIN
# -----------------------------

maze,start,goal = get_maze_input()

show_input_maze(maze,start,goal)

best_paths,best_history,avg_history,table_data = run_ga(maze,start,goal)

show_fitness_graph(best_history,avg_history)

show_table(table_data)

animate_paths(maze,best_paths,start,goal)