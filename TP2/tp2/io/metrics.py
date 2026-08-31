"""Stable CSV metrics schema used by runs and later experiments."""

from __future__ import annotations

import csv
from pathlib import Path

from tp2.engine.events import GenerationEvent

METRICS_COLUMNS = [
    "generation", "renders", "elapsed_s", "best_fitness", "mean_fitness",
    "worst_fitness", "error", "diversity", "active_triangles", "stop_reason",
]


class MetricsWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle = None
        self._writer = None

    def __enter__(self) -> "MetricsWriter":
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=METRICS_COLUMNS, lineterminator="\n")
        self._writer.writeheader()
        return self

    def __exit__(self, *_: object) -> None:
        assert self._handle is not None
        self._handle.close()

    def write(self, event: GenerationEvent) -> None:
        if self._writer is None:
            raise RuntimeError("MetricsWriter must be used as a context manager")
        self._writer.writerow({
            "generation": event.generation,
            "renders": event.renders,
            "elapsed_s": event.elapsed,
            "best_fitness": event.best_fitness,
            "mean_fitness": event.mean_fitness,
            "worst_fitness": event.worst_fitness,
            "error": event.best_error,
            "diversity": event.diversity,
            "active_triangles": event.active_triangles,
            "stop_reason": event.stop_reason,
        })
