"""Monitor the progress of ongoing GA runs in real time.

Reads in-progress `metrics.csv.tmp` and completed `metrics.csv` files to report:
- Current generation and completion percentage
- Elapsed time and estimated time remaining (ETA)
- Current best fitness
- Status (En progreso / Completado)

Usage:
    # Ver estado actual una sola vez:
    .venv/bin/python scripts/watch_progress.py --dir runs/matrix_boltzmann

    # Monitorear continuamente en vivo (actualiza cada 1 segundo):
    .venv/bin/python scripts/watch_progress.py --dir runs/matrix_boltzmann --watch
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_last_row(csv_path: Path) -> dict[str, str] | None:
    try:
        with csv_path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            if len(lines) < 2:
                return None
            header = [c.strip() for c in lines[0].strip().split(",")]
            last_line = [c.strip() for c in lines[-1].strip().split(",")]
            if len(header) != len(last_line):
                # In the middle of writing a row, fallback to previous line
                if len(lines) >= 3:
                    last_line = [c.strip() for c in lines[-2].strip().split(",")]
                else:
                    return None
            return dict(zip(header, last_line))
    except Exception:
        return None


def format_bar(percent: float, length: int = 25) -> str:
    filled = int(round(length * percent / 100.0))
    filled = max(0, min(length, filled))
    bar = "=" * filled + (">" if filled < length else "")
    return f"[{bar:<{length}}]"


def check_progress(target_dir: Path, horizon: int = 3000) -> tuple[bool, str]:
    if not target_dir.exists():
        return False, f"Directorio no encontrado: {target_dir}"

    # Search for all metrics files: completed or in-progress
    metrics_files = sorted(list(target_dir.rglob("metrics.csv")) + list(target_dir.rglob("metrics.csv.tmp")))

    if not metrics_files:
        return False, f"Esperando que comiencen las corridas en {target_dir}..."

    output_lines = [
        "=" * 78,
        f"Progreso de ejecución en: {target_dir}",
        "=" * 78,
    ]

    all_done = True
    seen_dirs: set[Path] = set()

    for path in metrics_files:
        parent_dir = path.parent
        if parent_dir in seen_dirs:
            continue
        seen_dirs.add(parent_dir)

        is_done = path.name == "metrics.csv"
        if not is_done:
            all_done = False

        row = _read_last_row(path)
        rel_name = parent_dir.relative_to(target_dir) if parent_dir != target_dir else Path("run")

        if not row:
            output_lines.append(f"• {rel_name:<30}: Iniciando...")
            continue

        gen = int(row.get("generation", 0))
        pct = min(100.0, (gen / float(horizon)) * 100.0)
        fitness = float(row.get("best_fitness", 0.0))
        elapsed = float(row.get("elapsed_s", 0.0))

        if pct > 0:
            total_est = elapsed / (pct / 100.0)
            eta = max(0.0, total_est - elapsed)
            eta_str = f"{eta:.1f}s rest" if not is_done else "listo"
        else:
            eta_str = "calculando..."

        status_tag = "✓ COMPLETADO" if is_done else "EN CURSO"
        bar = format_bar(pct, length=18)

        output_lines.append(
            f"• {str(rel_name):<22} {bar} {pct:5.1f}% | "
            f"Gen: {gen:>4}/{horizon} | Fit: {fitness:.4f} | "
            f"T: {elapsed:>5.1f}s ({eta_str}) | {status_tag}"
        )

    output_lines.append("=" * 78)
    return all_done, "\n".join(output_lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Watch progress of GA runs.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("runs/matrix_boltzmann"),
        help="Run directory to inspect (default: runs/matrix_boltzmann)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=3000,
        help="Expected total generations (default: 3000)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously watch and refresh progress every 1s",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Refresh interval in seconds when using --watch (default: 1.0)",
    )

    args = parser.parse_args(argv)
    target_dir = args.dir if args.dir.is_absolute() else PROJECT_ROOT / args.dir

    if not args.watch:
        _, msg = check_progress(target_dir, args.horizon)
        print(msg)
        return 0

    print("Monitoreando progreso en vivo (presioná Ctrl+C para salir)...")
    try:
        while True:
            all_done, msg = check_progress(target_dir, args.horizon)
            # Clear terminal screen (compatible with Linux/WSL)
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.write(msg + "\n")
            sys.stdout.flush()
            if all_done:
                print("\n¡Todas las corridas han finalizado con éxito!")
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitoreo detenido por el usuario.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
