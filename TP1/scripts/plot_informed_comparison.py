from pathlib import Path
import sys

_TP1_DIR = Path(__file__).resolve().parent.parent
if str(_TP1_DIR) not in sys.path:
    sys.path.insert(0, str(_TP1_DIR))

import matplotlib.pyplot as plt
import numpy as np

from gridworld.levelfile import load_level
from gridworld.search.problem import Problem
from scripts.generate_plots import apply_itba_style, benchmark_algo


def generate_greedy_heuristics_plot(level_path: str = "levels/02-classic.json", out_path: str = "plots/02-classic_heuristicas_greedy.png"):
    apply_itba_style()

    level = load_level(level_path)
    prob = Problem(board=level.board, initial=level.state)

    heuristics = [("manhattan", "Manhattan"), ("max_manhattan", "Max Manh."), ("euclidean_distance", "Euclidiana")]
    colors = ["#8338EC", "#E65F2B", "#2EC4B6"]

    stats_list = []
    for h_code, h_name in heuristics:
        s = benchmark_algo(prob, "greedy", heuristic_name=h_code, runs=3)
        s.heuristic_name = h_name
        stats_list.append(s)

    labels = [s.heuristic_name for s in stats_list]
    costs = [s.cost for s in stats_list]
    expanded = [s.expanded_nodes for s in stats_list]
    times_mean = [s.time_mean for s in stats_list]
    times_std = [s.time_std for s in stats_list]
    max_front = [s.max_frontier for s in stats_list]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    fig.suptitle(f"Comparativa de Heuristicas en Greedy ({level.name})", fontsize=14, y=0.98)

    # 1. Costo
    bars1 = axes[0].bar(labels, costs, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
    axes[0].set_title("Costo de solucion")
    axes[0].set_ylabel("Costo (pasos)")
    max_c = max(costs)
    for bar, val in zip(bars1, costs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max_c * 0.015), f"{val}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, max_c * 1.18 if max_c > 0 else 1)

    # 2. Nodos expandidos
    bars2 = axes[1].bar(labels, expanded, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
    axes[1].set_title("Nodos expandidos")
    axes[1].set_ylabel("Nodos expandidos")
    max_e = max(expanded)
    for bar, val in zip(bars2, expanded):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max_e * 0.015), f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[1].set_ylim(0, max_e * 1.18 if max_e > 0 else 1)

    # 3. Tiempo de ejecucion
    bars3 = axes[2].bar(labels, times_mean, yerr=times_std, capsize=4, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
    axes[2].set_title("Tiempo de ejecucion")
    axes[2].set_ylabel("Tiempo (s)")
    max_t = max(times_mean) + (max(times_std) if times_std else 0)
    for bar, val, std in zip(bars3, times_mean, times_std):
        txt = f"{val*1000:.2f} ms" if val < 0.05 else f"{val:.3f} s"
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + (max_t * 0.015), txt, ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[2].set_ylim(0, max_t * 1.25 if max_t > 0 else 1)

    # 4. Nodos frontera max
    bars4 = axes[3].bar(labels, max_front, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
    axes[3].set_title("Nodos frontera (max)")
    axes[3].set_ylabel("Cantidad de Nodos")
    max_f = max(max_front)
    for bar, val in zip(bars4, max_front):
        axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max_f * 0.015), f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[3].set_ylim(0, max_f * 1.18 if max_f > 0 else 1)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=300)
    plt.close()
    print(f"Grafico guardado exitosamente en: {out_p.resolve()}")


def generate_informed_astar_greedy_plot(level_path: str = "levels/02-classic.json", out_path: str = "plots/02-classic_heuristicas_astar_greedy.png"):
    apply_itba_style()

    level = load_level(level_path)
    prob = Problem(board=level.board, initial=level.state)

    heuristics = [("manhattan", "Manhattan"), ("max_manhattan", "Max Manh."), ("euclidean_distance", "Euclidiana")]
    
    stats_map = {}
    for a_code, a_name in [("astar", "A*"), ("greedy", "Greedy")]:
        for h_code, _ in heuristics:
            s = benchmark_algo(prob, a_code, heuristic_name=h_code, runs=3)
            stats_map[(a_code, h_code)] = s

    x = np.arange(len(heuristics))
    width = 0.35

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))
    fig.suptitle(f"Comparativa de Heuristicas en Algoritmos Informados (A* vs Greedy) - {level.name}", fontsize=14, y=0.98)

    h_labels = [h[1] for h in heuristics]

    astar_costs = [stats_map[("astar", h[0])].cost for h in heuristics]
    greedy_costs = [stats_map[("greedy", h[0])].cost for h in heuristics]

    astar_exp = [stats_map[("astar", h[0])].expanded_nodes for h in heuristics]
    greedy_exp = [stats_map[("greedy", h[0])].expanded_nodes for h in heuristics]

    astar_times = [stats_map[("astar", h[0])].time_mean for h in heuristics]
    greedy_times = [stats_map[("greedy", h[0])].time_mean for h in heuristics]

    astar_front = [stats_map[("astar", h[0])].max_frontier for h in heuristics]
    greedy_front = [stats_map[("greedy", h[0])].max_frontier for h in heuristics]

    # 1. Costo
    rects1 = axes[0].bar(x - width/2, astar_costs, width, label="A*", color="#1F7A8C", edgecolor="black", linewidth=0.5)
    rects2 = axes[0].bar(x + width/2, greedy_costs, width, label="Greedy", color="#8338EC", edgecolor="black", linewidth=0.5)
    axes[0].set_title("Costo de solucion")
    axes[0].set_ylabel("Costo (pasos)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(h_labels)
    axes[0].legend(loc="upper right")
    max_c = max(max(astar_costs), max(greedy_costs))
    for r in rects1:
        axes[0].text(r.get_x() + r.get_width()/2, r.get_height() + max_c*0.015, f"{int(r.get_height())}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for r in rects2:
        axes[0].text(r.get_x() + r.get_width()/2, r.get_height() + max_c*0.015, f"{int(r.get_height())}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].set_ylim(0, max_c * 1.18)

    # 2. Nodos Expandidos
    rects1 = axes[1].bar(x - width/2, astar_exp, width, label="A*", color="#1F7A8C", edgecolor="black", linewidth=0.5)
    rects2 = axes[1].bar(x + width/2, greedy_exp, width, label="Greedy", color="#8338EC", edgecolor="black", linewidth=0.5)
    axes[1].set_title("Nodos expandidos")
    axes[1].set_ylabel("Nodos expandidos")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(h_labels)
    axes[1].legend(loc="upper right")
    max_e = max(max(astar_exp), max(greedy_exp))
    for r in rects1:
        axes[1].text(r.get_x() + r.get_width()/2, r.get_height() + max_e*0.015, f"{int(r.get_height()):,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for r in rects2:
        axes[1].text(r.get_x() + r.get_width()/2, r.get_height() + max_e*0.015, f"{int(r.get_height()):,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1].set_ylim(0, max_e * 1.18)

    # 3. Tiempo de ejecucion
    rects1 = axes[2].bar(x - width/2, astar_times, width, label="A*", color="#1F7A8C", edgecolor="black", linewidth=0.5)
    rects2 = axes[2].bar(x + width/2, greedy_times, width, label="Greedy", color="#8338EC", edgecolor="black", linewidth=0.5)
    axes[2].set_title("Tiempo de ejecucion")
    axes[2].set_ylabel("Tiempo (s)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(h_labels)
    axes[2].legend(loc="upper right")
    max_t = max(max(astar_times), max(greedy_times))
    for r in rects1:
        v = r.get_height()
        txt = f"{v*1000:.1f}ms" if v < 0.05 else f"{v:.2f}s"
        axes[2].text(r.get_x() + r.get_width()/2, v + max_t*0.015, txt, ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for r in rects2:
        v = r.get_height()
        txt = f"{v*1000:.1f}ms" if v < 0.05 else f"{v:.2f}s"
        axes[2].text(r.get_x() + r.get_width()/2, v + max_t*0.015, txt, ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    axes[2].set_ylim(0, max_t * 1.22)

    # 4. Nodos Frontera Max
    rects1 = axes[3].bar(x - width/2, astar_front, width, label="A*", color="#1F7A8C", edgecolor="black", linewidth=0.5)
    rects2 = axes[3].bar(x + width/2, greedy_front, width, label="Greedy", color="#8338EC", edgecolor="black", linewidth=0.5)
    axes[3].set_title("Nodos frontera (max)")
    axes[3].set_ylabel("Cantidad de Nodos")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(h_labels)
    axes[3].legend(loc="upper right")
    max_f = max(max(astar_front), max(greedy_front))
    for r in rects1:
        axes[3].text(r.get_x() + r.get_width()/2, r.get_height() + max_f*0.015, f"{int(r.get_height()):,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for r in rects2:
        axes[3].text(r.get_x() + r.get_width()/2, r.get_height() + max_f*0.015, f"{int(r.get_height()):,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[3].set_ylim(0, max_f * 1.18)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=300)
    plt.close()
    print(f"Grafico guardado exitosamente en: {out_p.resolve()}")


if __name__ == "__main__":
    generate_greedy_heuristics_plot()
    generate_informed_astar_greedy_plot()
