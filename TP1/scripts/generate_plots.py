"""Script para benchmark y generacion de graficos comparativos (estilo presentacion ITBA).

Uso:
    py scripts/generate_plots.py [level_path] [--runs N] [--out DIR] [--skip-iddfs] [--compare-all]

Ejemplos:
    py scripts/generate_plots.py levels/01-warmup.json
    py scripts/generate_plots.py levels/02-classic.json --skip-iddfs --runs 3
    py scripts/generate_plots.py levels/01-warmup.json --compare-all
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys
import time

# Ensure TP1 directory is on sys.path
_TP1_DIR = Path(__file__).resolve().parent.parent
if str(_TP1_DIR) not in sys.path:
    sys.path.insert(0, str(_TP1_DIR))

import matplotlib.pyplot as plt
import numpy as np

from gridworld.levelfile import DEFAULT_LEVEL_PATH, load_level
from gridworld.search.algorithms import ALGORITHMS
from gridworld.search.heuristics import HEURISTICS
from gridworld.search.problem import Problem


COLOR_BLUE = "#3772FF"       # BFS / Manhattan / Final
COLOR_ORANGE = "#E65F2B"     # DFS / Hungarian / Max
COLOR_GREEN = "#2EC4B6"      # IDDFS / Heuristic B
COLOR_PURPLE = "#8338EC"     # Greedy
COLOR_DARK_TEAL = "#1F7A8C"  # A*


@dataclass
class BenchmarkStats:
    algorithm: str
    heuristic_name: str | None
    success: bool
    cost: int
    expanded_nodes: int
    final_frontier: int
    max_frontier: int
    time_mean: float
    time_std: float
    times: list[float]


def benchmark_algo(
    problem: Problem,
    algo_name: str,
    heuristic_name: str | None = None,
    runs: int = 5,
) -> BenchmarkStats:
    algo_fn = ALGORITHMS[algo_name]
    times = []
    last_res = None

    for _ in range(runs):
        prob_copy = Problem(board=problem.board, initial=problem.initial)
        t0 = time.perf_counter()
        if algo_name in ("astar", "greedy"):
            h_fn = HEURISTICS[heuristic_name or "manhattan"]
            res = algo_fn(prob_copy, h_fn)
        else:
            res = algo_fn(prob_copy)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        last_res = res

    return BenchmarkStats(
        algorithm=algo_name.upper(),
        heuristic_name=heuristic_name,
        success=last_res.success,
        cost=last_res.cost if last_res.cost is not None else 0,
        expanded_nodes=last_res.expanded_nodes,
        final_frontier=last_res.frontier_nodes,
        max_frontier=last_res.max_frontier_nodes,
        time_mean=statistics.mean(times),
        time_std=statistics.stdev(times) if len(times) > 1 else 0.0,
        times=times,
    )


def apply_itba_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans", "Helvetica"],
        "figure.titlesize": 15,
        "figure.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linestyle": "--",
    })


def plot_uninformed_comparison(
    stats_list: list[BenchmarkStats],
    level_name: str,
    out_path: Path,
):
    """Generates the 4-panel comparison for uninformed algorithms (BFS, DFS, IDDFS)."""
    apply_itba_style()
    labels = [s.algorithm for s in stats_list]
    colors = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_PURPLE][:len(stats_list)]

    expanded = [s.expanded_nodes for s in stats_list]
    frontier = [s.final_frontier for s in stats_list]
    times_mean = [s.time_mean for s in stats_list]
    times_std = [s.time_std for s in stats_list]
    costs = [s.cost for s in stats_list]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    fig.suptitle(f"Metodos Desinformados - {' vs '.join(labels)} ({level_name})", fontsize=14, y=0.98)

    # 1. Nodos Expandidos
    bars1 = axes[0].bar(labels, expanded, color=colors, width=0.55, edgecolor="black", linewidth=0.5)
    axes[0].set_title("Nodos expandidos")
    axes[0].set_ylabel("Nodos expandidos")
    for bar, val in zip(bars1, expanded):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(expanded)*0.015), f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].set_ylim(0, max(expanded) * 1.15 if max(expanded) > 0 else 1)

    # 2. Nodos Frontera
    bars2 = axes[1].bar(labels, frontier, color=colors, width=0.55, edgecolor="black", linewidth=0.5)
    axes[1].set_title("Nodos frontera (final)")
    axes[1].set_ylabel("Nodos frontera")
    for bar, val in zip(bars2, frontier):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(frontier)*0.015), f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1].set_ylim(0, max(frontier) * 1.15 if max(frontier) > 0 else 1)

    # 3. Tiempo de ejecución (s) con error bars
    bars3 = axes[2].bar(labels, times_mean, yerr=times_std, capsize=4, color=colors, width=0.55, edgecolor="black", linewidth=0.5, error_kw={"elinewidth": 1.2, "capthick": 1.2})
    axes[2].set_title("Tiempo de ejecucion")
    axes[2].set_ylabel("Tiempo (s)")
    for bar, val, std in zip(bars3, times_mean, times_std):
        txt = f"{val*1000:.1f}ms" if val < 0.01 else f"{val:.3f}s"
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + (max(times_mean)*0.015), txt, ha="center", va="bottom", fontsize=9, fontweight="bold")
    max_t = max(times_mean) + (max(times_std) if times_std else 0)
    axes[2].set_ylim(0, max_t * 1.18 if max_t > 0 else 1)

    # 4. Costo de la solución
    bars4 = axes[3].bar(labels, costs, color=colors, width=0.55, edgecolor="black", linewidth=0.5)
    axes[3].set_title("Costo de la solucion")
    axes[3].set_ylabel("Costo (pasos)")
    for bar, val in zip(bars4, costs):
        axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(costs)*0.015), f"{val}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[3].set_ylim(0, max(costs) * 1.15 if max(costs) > 0 else 1)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [OK] Grafico guardado: {out_path}")


def plot_frontier_final_vs_max(
    stats_list: list[BenchmarkStats],
    level_name: str,
    out_path: Path,
):
    """Generates the grouped bar chart comparing Final Frontier vs Maximum Frontier."""
    apply_itba_style()
    labels = [s.algorithm for s in stats_list]
    final_frontier = [s.final_frontier for s in stats_list]
    max_frontier = [s.max_frontier for s in stats_list]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    fig.suptitle(f"Nodos Frontera - Cantidad Maxima vs Cantidad Final ({level_name})", fontsize=13, y=0.98)

    rects1 = ax.bar(x - width/2, final_frontier, width, label="Frontera Final", color="#3E6990", edgecolor="black", linewidth=0.5)
    rects2 = ax.bar(x + width/2, max_frontier, width, label="Maxima Frontera", color="#E07A5F", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Cantidad de Nodos")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right")

    max_val = max(max(final_frontier), max(max_frontier))
    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, h + (max_val*0.015), f"{h:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, h + (max_val*0.015), f"{h:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(0, max_val * 1.15 if max_val > 0 else 1)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [OK] Grafico guardado: {out_path}")


def plot_informed_comparison(
    stats_list: list[BenchmarkStats],
    level_name: str,
    out_path: Path,
):
    """Generates the 4-panel comparison for informed search / heuristics."""
    apply_itba_style()
    labels = [s.heuristic_name or s.algorithm for s in stats_list]
    colors = ["#2A9D8F", "#E76F51", "#264653", "#E9C46A"][:len(stats_list)]

    costs = [s.cost for s in stats_list]
    expanded = [s.expanded_nodes for s in stats_list]
    times_mean = [s.time_mean for s in stats_list]
    times_std = [s.time_std for s in stats_list]
    max_front = [s.max_frontier for s in stats_list]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    fig.suptitle(f"Comparativa de Heuristicas en A* ({level_name})", fontsize=14, y=0.98)

    # 1. Costo
    bars1 = axes[0].bar(labels, costs, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
    axes[0].set_title("Costo de solucion")
    axes[0].set_ylabel("Costo (pasos)")
    for bar, val in zip(bars1, costs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(costs)*0.015), f"{val}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, max(costs) * 1.18 if max(costs) > 0 else 1)

    # 2. Nodos expandidos
    bars2 = axes[1].bar(labels, expanded, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
    axes[1].set_title("Nodos expandidos")
    axes[1].set_ylabel("Nodos expandidos")
    for bar, val in zip(bars2, expanded):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(expanded)*0.015), f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=10)
    axes[1].tick_params(axis="x", rotation=10)
    axes[1].set_ylim(0, max(expanded) * 1.18 if max(expanded) > 0 else 1)

    # 3. Tiempo
    bars3 = axes[2].bar(labels, times_mean, yerr=times_std, capsize=4, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
    axes[2].set_title("Tiempo de ejecucion (s)")
    axes[2].set_ylabel("Tiempo (s)")
    axes[2].tick_params(axis="x", rotation=10)
    for bar, val, std in zip(bars3, times_mean, times_std):
        txt = f"{val*1000:.2f} ms" if val < 0.05 else f"{val:.3f} s"
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + (max(times_mean)*0.015), txt, ha="center", va="bottom", fontsize=9, fontweight="bold")
    max_t = max(times_mean) + (max(times_std) if times_std else 0)
    axes[2].set_ylim(0, max_t * 1.25 if max_t > 0 else 1)

    # 4. Nodos frontera max
    bars4 = axes[3].bar(labels, max_front, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
    axes[3].set_title("Nodos frontera (max)")
    axes[3].set_ylabel("Cantidad de Nodos")
    axes[3].tick_params(axis="x", rotation=10)
    for bar, val in zip(bars4, max_front):
        axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(max_front)*0.015), f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[3].set_ylim(0, max(max_front) * 1.18 if max(max_front) > 0 else 1)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [OK] Grafico guardado: {out_path}")


def plot_all_algorithms(
    stats_list: list[BenchmarkStats],
    level_name: str,
    out_path: Path,
):
    """Generates a complete comparison across all algorithms (BFS, DFS, IDDFS, Greedy, A*)."""
    apply_itba_style()
    labels = [s.algorithm for s in stats_list]
    colors = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_PURPLE, COLOR_DARK_TEAL][:len(stats_list)]

    costs = [s.cost for s in stats_list]
    expanded = [s.expanded_nodes for s in stats_list]
    times_mean = [s.time_mean for s in stats_list]
    times_std = [s.time_std for s in stats_list]
    max_front = [s.max_frontier for s in stats_list]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.suptitle(f"Comparativa General de Algoritmos ({level_name})", fontsize=14, y=0.98)

    # 1. Costo
    bars1 = axes[0].bar(labels, costs, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
    axes[0].set_title("Costo de solucion")
    axes[0].set_ylabel("Costo (pasos)")
    for bar, val in zip(bars1, costs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(costs)*0.015), f"{val}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].set_ylim(0, max(costs) * 1.18 if max(costs) > 0 else 1)

    # 2. Nodos expandidos
    bars2 = axes[1].bar(labels, expanded, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
    axes[1].set_title("Nodos expandidos")
    axes[1].set_ylabel("Nodos expandidos")
    for bar, val in zip(bars2, expanded):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(expanded)*0.015), f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1].set_ylim(0, max(expanded) * 1.18 if max(expanded) > 0 else 1)

    # 3. Tiempo
    bars3 = axes[2].bar(labels, times_mean, yerr=times_std, capsize=4, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
    axes[2].set_title("Tiempo de ejecucion")
    axes[2].set_ylabel("Tiempo (s)")
    for bar, val, std in zip(bars3, times_mean, times_std):
        txt = f"{val*1000:.1f} ms" if val < 0.05 else f"{val:.3f} s"
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + (max(times_mean)*0.015), txt, ha="center", va="bottom", fontsize=9, fontweight="bold")
    max_t = max(times_mean) + (max(times_std) if times_std else 0)
    axes[2].set_ylim(0, max_t * 1.25 if max_t > 0 else 1)

    # 4. Nodos frontera max
    bars4 = axes[3].bar(labels, max_front, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
    axes[3].set_title("Nodos frontera (max)")
    axes[3].set_ylabel("Cantidad de Nodos")
    for bar, val in zip(bars4, max_front):
        axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(max_front)*0.015), f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[3].set_ylim(0, max(max_front) * 1.18 if max(max_front) > 0 else 1)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [OK] Grafico guardado: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generar graficos comparativos para el informe y presentacion.")
    parser.add_argument("level_path", nargs="?", default=DEFAULT_LEVEL_PATH, help="Nivel a evaluar")
    parser.add_argument("--runs", "-r", type=int, default=5, help="Cantidad de repeticiones para promediar tiempos")
    parser.add_argument("--out", "-o", type=str, default="plots", help="Carpeta de salida para las imagenes")
    parser.add_argument("--skip-iddfs", action="store_true", help="Saltear IDDFS en niveles grandes")
    parser.add_argument("--compare-all", action="store_true", help="Generar grafico comparativo de todos los algoritmos")
    args = parser.parse_args()

    level = load_level(args.level_path)
    prob = Problem(board=level.board, initial=level.state)
    out_dir = Path(args.out)
    prefix = Path(args.level_path).stem

    print("=" * 65)
    print(f"GENERANDO BENCHMARKS Y GRAFICOS: '{level.name}' ({args.level_path})")
    print(f"Repeticiones por algoritmo: {args.runs}")
    print("=" * 65)

    # 1. Benchmark Desinformados (BFS, DFS, IDDFS)
    uninformed_algos = ["bfs", "dfs"] if args.skip_iddfs else ["bfs", "dfs", "iddfs"]
    print(f"\n[1/3] Evaluando metodos desinformados ({', '.join(a.upper() for a in uninformed_algos)})...")
    uninformed_stats = []
    for algo in uninformed_algos:
        print(f"  * Ejecutando {algo.upper()} ({args.runs} runs)...", end="", flush=True)
        stats = benchmark_algo(prob, algo, runs=args.runs)
        uninformed_stats.append(stats)
        print(f" Listo (tiempo medio: {stats.time_mean:.4f}s, exp: {stats.expanded_nodes})")

    plot_uninformed_comparison(uninformed_stats, level.name, out_dir / f"{prefix}_desinformados.png")
    plot_frontier_final_vs_max(uninformed_stats, level.name, out_dir / f"{prefix}_frontera_max_vs_final.png")

    # 2. Benchmark Heurísticas en A*
    print("\n[2/3] Evaluando heuristicas en A* (Manhattan vs Max Manhattan vs Euclidiana)...")
    informed_stats = []
    heuristics_to_compare = ["manhattan", "max_manhattan", "euclidean_distance"]
    for h_name in heuristics_to_compare:
        print(f"  * Ejecutando A* con {h_name} ({args.runs} runs)...", end="", flush=True)
        stats = benchmark_algo(prob, "astar", heuristic_name=h_name, runs=args.runs)
        stats.heuristic_name = h_name.replace("_", " ").title()
        informed_stats.append(stats)
        print(f" Listo (tiempo medio: {stats.time_mean:.4f}s, exp: {stats.expanded_nodes})")

    plot_informed_comparison(informed_stats, level.name, out_dir / f"{prefix}_heuristicas_astar.png")

    # 3. Comparativa General si se solicita
    if args.compare_all:
        print("\n[3/3] Evaluando todos los algoritmos (BFS, DFS, IDDFS, Greedy, A*)...")
        all_algos = ["bfs", "dfs"] + ([] if args.skip_iddfs else ["iddfs"]) + ["greedy", "astar"]
        all_stats = []
        for algo in all_algos:
            h = "manhattan" if algo in ("astar", "greedy") else None
            stats = benchmark_algo(prob, algo, heuristic_name=h, runs=args.runs)
            all_stats.append(stats)
        plot_all_algorithms(all_stats, level.name, out_dir / f"{prefix}_todos_los_algoritmos.png")

    # Resumen en consola
    print("\n" + "=" * 65)
    print("RESUMEN DE RESULTADOS:")
    print(f"{'Algoritmo / Heuristica':<25} | {'Costo':<5} | {'Expandidos':<10} | {'Frontera (Max)':<14} | {'Tiempo (s)':<12}")
    print("-" * 75)
    for s in uninformed_stats + informed_stats:
        name = f"{s.algorithm} ({s.heuristic_name})" if s.heuristic_name else s.algorithm
        print(f"{name:<25} | {s.cost:<5} | {s.expanded_nodes:<10,} | {s.max_frontier:<14,} | {s.time_mean:.5f} +- {s.time_std:.5f}")
    print("=" * 65)
    print(f"\nTodos los graficos fueron generados exitosamente en '{out_dir.resolve()}'!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
