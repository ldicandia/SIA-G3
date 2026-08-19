# Grid World — TP1

A playable Grid World game for ITBA's *Sistemas de Inteligencia Artificial* TP1,
Ejercicio 2 (Lado A). An N×M grid holds black obstacles, numbered cars, and
numbered flags; move one car at a time until every car is parked on its
matching flag.

The game includes a rules engine, a pygame window for manual play, and an
incremental A* solver that visualizes explored cells before replaying its
optimal solution.

## Requirements

- Python 3.10–3.13 (developed and tested on 3.12)
- [pygame](https://www.pygame.org/) 2.6.1

## Installation

From the `TP1` directory:

```sh
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Running the game

From the `TP1` directory:

```sh
python -m gridworld
```

This opens a level-choice screen listing every `.json` file under
`TP1/levels/`. Press a number key to load one.

To skip the picker and load a specific level directly:

```sh
python -m gridworld levels/01-warmup.json
```

## Visualizing the optimal search

While playing any level, press `S`. The board first reveals explored cells as
blue dots while the HUD reports A*'s expanded and frontier node counts. Once
the goal is found, the game pauses briefly and replays the optimal path slowly
in yellow.

The search begins at the current board position. Press `Space` to pause or
resume and `-` / `+` to change the animation speed. The default is the
deliberately slow `0.5x` speed.

## Visualizing Search (Demo Replay)

To solve a level with any search algorithm and replay its solution step-by-step in the Pygame window:

```sh
python scripts/replay_search.py --algo <bfs|dfs|greedy|astar|iddfs> [level_path] [--heuristic <heuristic_a|heuristic_b>] [--delay <ms>]
```

Examples:

- Run **A\*** with default Manhattan heuristic on warmup level:
  ```sh
  python scripts/replay_search.py --algo astar levels/01-warmup.json
  ```
- Run **BFS** on classic level:
  ```sh
  python scripts/replay_search.py --algo bfs levels/02-classic.json
  ```
- Run **Greedy** with max Manhattan heuristic:
  ```sh
  python scripts/replay_search.py --algo greedy levels/01-warmup.json --heuristic heuristic_b
  ```
- Run **DFS** or **IDDFS** with custom animation speed (150ms per step):
  ```sh
  python scripts/replay_search.py --algo dfs levels/01-warmup.json --delay 150
  ```

In the replay window:
- Press **Space** to pause / resume the replay.
- Press **Esc** to close the window.

## Generating Comparative Benchmark Plots

To generate presentation-ready comparative benchmark charts (matching the ITBA slide styles):

```sh
python scripts/generate_plots.py [level_path] [--runs N] [--out plots/] [--compare-all] [--skip-iddfs]
```

Examples:

- Benchmark all algorithms on Warmup level (5 runs per algorithm):
  ```sh
  python scripts/generate_plots.py levels/01-warmup.json --compare-all
  ```
- Benchmark on Classic level (skipping IDDFS due to depth limit):
  ```sh
  python scripts/generate_plots.py levels/02-classic.json --skip-iddfs
  ```

This automatically exports high-resolution PNG charts in the `plots/` folder:
- **`*_desinformados.png`**: 4-panel comparison of BFS vs DFS vs IDDFS (Expanded nodes, Final frontier, Execution time with error bars, Solution cost).
- **`*_frontera_max_vs_final.png`**: Grouped bar chart comparing Final Frontier vs Maximum Peak Frontier.
- **`*_heuristicas_astar.png`**: 4-panel comparison of A* across heuristics (Manhattan vs Max Manhattan vs Euclidean).
- **`*_todos_los_algoritmos.png`**: Complete 4-panel comparison across all 5 algorithms.

## Controls

| Key | Action |
|-----|--------|
| `1`–`9` | Select a car by number (in-game), or choose a level (on the level-choice screen) |
| Arrow keys | Move the selected car one cell |
| `U` | Undo the last move |
| `R` | Reset the level to its starting position |
| `S` | Start optimal A* search from the current position |
| `Space` | Pause or resume search animation |
| `-` / `+` | Change animation speed |
| `Esc` | Quit (in-game or on the level-choice screen); go back (on a load-error screen) |

## Rules

- Exactly one selected car moves one orthogonal cell per turn.
- A move is refused, with no change to the board, when the target cell is:
  off the grid, a black obstacle, occupied by another car, or a flag
  belonging to a different car.
- A car that moves onto its own numbered flag **parks** and can never be
  selected or moved again.
- The level is won once every car is parked on its matching flag; a win
  screen then shows the total move count.
- Because parked cars lock in place and foreign flags block movement,
  parking order can leave the board in a legal-but-unwinnable state. When an
  unparked car can no longer reach its own flag, an on-screen warning names
  it and points to `U` (undo) or `R` (reset).

## Level files

Levels are JSON files under `TP1/levels/`, chosen from the picker or passed
as a command-line path. Three ship with the game at increasing complexity:

- `01-warmup.json` — 5×5, two cars
- `02-classic.json` — 7×7, three cars
- `03-gridlock.json` — 9×9, more cars and obstacles

The full field-by-field format, coordinate convention, limits, and every
validation error a malformed level can raise are documented in
[`levels/SCHEMA.md`](levels/SCHEMA.md).

## Running the tests

From the `TP1` directory:

```sh
pytest
```

The engine and level-loading modules are also proven to import and run with
no rendering library installed at all — see `scripts/check_headless.py`.
