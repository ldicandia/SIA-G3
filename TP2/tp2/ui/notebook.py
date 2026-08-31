"""Inline Jupyter/Colab progress observer for the evolving best frame."""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from tp2.engine.events import GenerationEvent


class NotebookProgress:
    """Render a compact refreshed PNG for each generation in a notebook output cell."""

    def __init__(self, every: int = 1) -> None:
        if every < 1:
            raise ValueError("notebook update interval must be at least 1")
        self.every = every
        try:
            from IPython.display import Image as DisplayImage, clear_output, display
        except ImportError as exc:
            raise RuntimeError("--notebook requires Jupyter, Colab, or another IPython notebook") from exc
        self._display_image = DisplayImage
        self._clear_output = clear_output
        self._display = display

    def __call__(self, event: GenerationEvent) -> None:
        if event.generation % self.every and not event.stop_reason:
            return
        encoded = BytesIO()
        Image.fromarray(event.best_frame, "RGB").save(encoded, format="PNG")
        self._clear_output(wait=True)
        self._display(f"Generation {event.generation} · fitness {event.best_fitness:.6f} · renders {event.renders}")
        self._display(self._display_image(data=encoded.getvalue()))
