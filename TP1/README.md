# Grid World — TP1

A playable Grid World game for ITBA's *Sistemas de Inteligencia Artificial* TP1,
Ejercicio 2 (Lado A). An N×M grid holds black obstacles, numbered cars, and
numbered flags; move one car at a time until every car is parked on its
matching flag.

This deliverable is the game base: a rules engine and a pygame window you can
play by hand. The search engine (BFS, DFS, Greedy, A*, IDDFS) that the TP
ultimately requires is a separate, later milestone.

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

## Visualizing Search (Demo Replay)

To watch `random_search` solve a level and replay its move sequence automatically in the Pygame window:

```sh
python scripts/replay_random_search.py [level_path] [--seed SEED]
```

Examples:

- Run default level (`levels/01-warmup.json`):
  ```sh
  python scripts/replay_random_search.py
  ```
- Run a specific level:
  ```sh
  python scripts/replay_random_search.py levels/02-classic.json
  ```
- Generate a new random solution on every run:
  ```sh
  python scripts/replay_random_search.py --seed random
  ```
- Use a custom numeric or text seed:
  ```sh
  python scripts/replay_random_search.py levels/01-warmup.json --seed 42
  ```

## Controls

| Key | Action |
|-----|--------|
| `1`–`9` | Select a car by number (in-game), or choose a level (on the level-choice screen) |
| Arrow keys | Move the selected car one cell |
| `U` | Undo the last move |
| `R` | Reset the level to its starting position |
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
