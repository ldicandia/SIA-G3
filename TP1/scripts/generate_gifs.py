"""Script para generar animaciones GIF de las corridas de algoritmos de búsqueda en Gridworld.

Soporta ejecutarse en modo headless (sin abrir ventana gráfica) generando GIFs
animados de la fase de exploración, el camino solución y la animación de movimientos.

Uso:
    py scripts/generate_gifs.py [level_path] [--out-dir DIR] [--algo ALGO] [--heuristic HEURISTIC] [--fps FPS]

Ejemplos:
    py scripts/generate_gifs.py levels/01-warmup.json
    py scripts/generate_gifs.py levels/01-warmup.json --algo astar --heuristic heuristic_a
    py scripts/generate_gifs.py levels/02-classic.json --out-dir gifs/classic
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence

# Asegurar que el directorio TP1 esté en sys.path
_TP1_DIR = Path(__file__).resolve().parent.parent
if str(_TP1_DIR) not in sys.path:
    sys.path.insert(0, str(_TP1_DIR))

from PIL import Image
import pygame

from gridworld.engine.board import Position
from gridworld.engine.rules import apply_move
from gridworld.engine.state import GameState
from gridworld.history import MoveHistory
from gridworld.levelfile import DEFAULT_LEVEL_PATH, LevelError, load_level
from gridworld.search.algorithms import ALGORITHMS
from gridworld.search.heuristics import HEURISTICS
from gridworld.search.problem import Problem
from gridworld.ui.app import build_solution_paths
from gridworld.ui.render import SearchHud, build_fonts, draw_frame
from gridworld.ui.sprites import build_sprites
from gridworld.ui.theme import WINDOW_SIZE, cell_size


def record_run_to_gif(
    level_path: Path | str,
    algo_name: str,
    heuristic_name: str | None,
    output_gif_path: Path | str,
    fps: int = 10,
    max_exploration_frames: int = 40,
) -> bool:
    """Ejecuta un algoritmo en un nivel y graba la simulación visual a un archivo GIF animado.

    Retorna True si tuvo éxito, False en caso contrario.
    """
    level_path = Path(level_path)
    output_gif_path = Path(output_gif_path)
    output_gif_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        level = load_level(level_path)
    except LevelError as err:
        sys.stderr.write(f"Error al cargar nivel {level_path}: {err}\n")
        return False

    problem = Problem(board=level.board, initial=level.state)
    if algo_name not in ALGORITHMS:
        sys.stderr.write(f"Algoritmo desconocido: {algo_name}\n")
        return False

    algo_fn = ALGORITHMS[algo_name]

    expansion_order: list[tuple[int, Position]] = []
    known_cells: dict[int, set[Position]] = {}

    def _record_expansion(state: GameState) -> None:
        for car, position in enumerate(state.cars, start=1):
            car_known = known_cells.setdefault(car, set())
            if position not in car_known:
                car_known.add(position)
                expansion_order.append((car, position))

    label_heuristic = f" ({heuristic_name})" if heuristic_name and algo_name in ("greedy", "astar") else ""
    print(f"Ejecutando {algo_name.upper()}{label_heuristic} en '{level.name}'...", flush=True)

    if algo_name in ("greedy", "astar"):
        if not heuristic_name or heuristic_name not in HEURISTICS:
            sys.stderr.write(f"Heurística inválida o faltante para {algo_name}: {heuristic_name}\n")
            return False
        heuristic_fn = HEURISTICS[heuristic_name]
        result = algo_fn(problem, heuristic_fn, on_expand=_record_expansion)
    else:
        result = algo_fn(problem, on_expand=_record_expansion)

    if not result.success:
        sys.stderr.write(f"El algoritmo {algo_name} no encontró solución en {level_path}.\n")
        return False

    full_solution_paths = build_solution_paths(level.board, level.state, result.path)

    # Inicializar Pygame y superficie offscreen
    if not pygame.get_init() or pygame.display.get_surface() is None:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    surface = pygame.Surface(WINDOW_SIZE)
    fonts = build_fonts()
    sprites = build_sprites(level.board, cell_size(level.board.cols, level.board.rows))

    frames: list[Image.Image] = []

    def capture_frame() -> None:
        raw_data = pygame.image.tobytes(surface, "RGB")
        img = Image.frombytes("RGB", WINDOW_SIZE, raw_data)
        frames.append(img)

    hud_status = f"{algo_name.upper()}{label_heuristic}"
    search_hud = SearchHud(
        status=hud_status,
        expanded_nodes=result.expanded_nodes,
        frontier_nodes=result.frontier_nodes,
        speed="GIF",
    )

    history = MoveHistory.start(level.state)

    # Fase 1: Exploración muestreada
    explored_cells: dict[int, set[Position]] = {}
    if expansion_order:
        total_expansions = len(expansion_order)
        if total_expansions <= max_exploration_frames:
            step_indices = list(range(total_expansions))
        else:
            step_indices = [
                int(i * (total_expansions - 1) / (max_exploration_frames - 1))
                for i in range(max_exploration_frames)
            ]

        curr_idx = 0
        for target_idx in step_indices:
            while curr_idx <= target_idx:
                car, pos = expansion_order[curr_idx]
                explored_cells.setdefault(car, set()).add(pos)
                focus = (car, pos)
                curr_idx += 1

            draw_frame(
                surface,
                fonts,
                level.board,
                history.current,
                selected=None,
                moves=0,
                sprites=sprites,
                explored_cells={car: frozenset(cells) for car, cells in explored_cells.items()},
                solution_paths={},
                focus_cell=focus,
                search=search_hud,
                show_win_overlay=False,
            )
            capture_frame()

    # Fase 2: Muestreo de ruta/solución destacada (Pause de 5 frames)
    solution_paths_map = {car: tuple(path) for car, path in full_solution_paths.items()}
    draw_frame(
        surface,
        fonts,
        level.board,
        history.current,
        selected=None,
        moves=0,
        sprites=sprites,
        explored_cells={car: frozenset(cells) for car, cells in explored_cells.items()},
        solution_paths=solution_paths_map,
        focus_cell=None,
        search=search_hud,
        show_win_overlay=False,
    )
    for _ in range(4):
        capture_frame()

    # Fase 3: Replay de movimientos de la solución
    for action in result.path:
        outcome = apply_move(level.board, history.current, action.car, action.direction)
        assert outcome.accepted
        history = history.push(outcome.state)
        selected = None if outcome.state.is_parked(action.car) else action.car

        draw_frame(
            surface,
            fonts,
            level.board,
            history.current,
            selected=selected,
            moves=history.depth,
            sprites=sprites,
            explored_cells={car: frozenset(cells) for car, cells in explored_cells.items()},
            solution_paths=solution_paths_map,
            focus_cell=None,
            search=search_hud,
            show_win_overlay=False,
        )
        capture_frame()

    # Fase 4: Solucionado (Overlay de victoria guardado por 15 frames)
    draw_frame(
        surface,
        fonts,
        level.board,
        history.current,
        selected=None,
        moves=history.depth,
        sprites=sprites,
        explored_cells={car: frozenset(cells) for car, cells in explored_cells.items()},
        solution_paths=solution_paths_map,
        focus_cell=None,
        search=search_hud,
        show_win_overlay=True,
    )
    for _ in range(12):
        capture_frame()

    # Guardar GIF animado
    duration_per_frame = int(1000 / fps)
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_per_frame,
        loop=0,
    )
    print(f"GIF guardado en: {output_gif_path} ({len(frames)} frames)", flush=True)
    return True


def generate_all_gifs(
    levels: list[Path | str] | None = None,
    out_dir: Path | str = "gifs",
    fps: int = 10,
    heuristics: list[str] | None = None,
) -> None:
    """Genera GIFs para todas las combinaciones de algoritmos y heurísticas en todos los mapas especificados."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if levels is None:
        from gridworld.levelfile import DEFAULT_LEVELS_DIR
        level_files = sorted(DEFAULT_LEVELS_DIR.glob("*.json"))
    else:
        level_files = [Path(l) for l in levels]

    if heuristics is None:
        heuristics_list = ["heuristic_a", "heuristic_b", "euclidean_distance"]
    else:
        heuristics_list = heuristics

    print("=" * 70)
    print(f"Generando suite masiva de GIFs para {len(level_files)} niveles en '{out_dir}/'...")
    print(f"Heurísticas incluidas: {', '.join(heuristics_list)}")
    print("=" * 70)

    total_attempted = 0
    total_success = 0

    for level_path in level_files:
        level_stem = level_path.stem
        print(f"\n>>> Procesando nivel: {level_stem} ({level_path.name})")

        runs: list[tuple[str, str | None, str]] = []

        # Algoritmos no informados
        for algo in ("bfs", "dfs", "iddfs"):
            # IDDFS re-explora exponencialmente en mapas de más de 6x6; BFS explora excesivamente en 03-gridlock y 04-marathon
            if level_stem in ("02-classic", "03-gridlock", "04-marathon") and algo == "iddfs":
                print(f"  [Info] Omitiendo IDDFS en {level_stem} por requerir re-exploración profunda.", flush=True)
                continue
            if level_stem in ("03-gridlock", "04-marathon") and algo == "bfs":
                print(f"  [Info] Omitiendo BFS en {level_stem} por excesivo número de estados en mapa grande.", flush=True)
                continue
            if level_stem == "04-marathon" and algo in ("bfs", "dfs"):
                print(f"  [Info] Omitiendo {algo.upper()} en {level_stem} por tamaño masivo del nivel.", flush=True)
                continue
            runs.append((algo, None, f"{level_stem}_{algo}.gif"))

        # Algoritmos informados con cada heurística
        for algo in ("greedy", "astar"):
            for heur in heuristics_list:
                runs.append((algo, heur, f"{level_stem}_{algo}_{heur}.gif"))

        for algo, heur, filename in runs:
            out_path = out_dir / filename
            total_attempted += 1
            ok = record_run_to_gif(
                level_path=level_path,
                algo_name=algo,
                heuristic_name=heur,
                output_gif_path=out_path,
                fps=fps,
            )
            if ok:
                total_success += 1

    print("=" * 70)
    print(f"Finalizado: {total_success}/{total_attempted} GIFs generados exitosamente.")
    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Genera GIFs animados de corridas de algoritmos en Gridworld para todos los mapas y heurísticas."
    )
    parser.add_argument(
        "level_path",
        nargs="?",
        default="all",
        help="Ruta al nivel JSON o 'all' para procesar todos los mapas en levels/ (por defecto 'all')",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=str,
        default="gifs",
        help="Directorio de destino para guardar los GIFs (por defecto gifs/)",
    )
    parser.add_argument(
        "--algo",
        "-a",
        type=str,
        default="all",
        choices=list(ALGORITHMS.keys()) + ["all"],
        help="Algoritmo a ejecutar ('all' para generar todos los algoritmos)",
    )
    parser.add_argument(
        "--heuristic",
        "-he",
        type=str,
        default="all",
        choices=list(HEURISTICS.keys()) + ["all"],
        help="Heurística para Greedy y A* ('heuristic_a', 'heuristic_b', 'euclidean_distance', o 'all')",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames por segundo del GIF animado (por defecto 10)",
    )

    args = parser.parse_args()

    if args.heuristic == "all":
        heuristics_to_run = ["heuristic_a", "heuristic_b", "euclidean_distance"]
    else:
        heuristics_to_run = [args.heuristic]

    if args.level_path == "all" and args.algo == "all":
        generate_all_gifs(out_dir=args.out_dir, fps=args.fps, heuristics=heuristics_to_run)
    else:
        out_dir = Path(args.out_dir)
        if args.level_path == "all":
            from gridworld.levelfile import DEFAULT_LEVELS_DIR
            levels = sorted(DEFAULT_LEVELS_DIR.glob("*.json"))
        else:
            levels = [Path(args.level_path)]

        algos_to_run = list(ALGORITHMS.keys()) if args.algo == "all" else [args.algo]

        for level_path in levels:
            for algo in algos_to_run:
                heurs = (
                    heuristics_to_run
                    if algo in ("greedy", "astar")
                    else [None]
                )
                for heur in heurs:
                    heur_suffix = f"_{heur}" if heur else ""
                    out_path = out_dir / f"{level_path.stem}_{algo}{heur_suffix}.gif"
                    record_run_to_gif(
                        level_path=level_path,
                        algo_name=algo,
                        heuristic_name=heur,
                        output_gif_path=out_path,
                        fps=args.fps,
                    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
