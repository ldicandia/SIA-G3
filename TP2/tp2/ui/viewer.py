"""tp2/ui/viewer.py -- the ONLY `import pygame` in the codebase outside this
file's own tests.

`Viewer` is a plain `GenerationEvent` observer: constructed by `tp2/cli.py`
only when `--viewer` is passed, driven by `for ev in run: viewer(ev)`, and
polled for `should_stop` by the composition root -- never by raising into
the engine's call stack (see ARCHITECTURE.md Anti-Pattern 6).
"""

from __future__ import annotations

import pygame

from tp2.engine.events import GenerationEvent


class Viewer:
    """Live pygame window blitting the exact frame the engine scored.

    A plain class, not a dataclass: it carries live mutable SDL state
    (`_screen`, `_font`) that a dataclass's generated `__eq__`/`__repr__`
    would have to reason about for no benefit.
    """

    def __init__(self, scale: int = 4, every: int = 1, caption: str = "TP2 — evolving") -> None:
        self.scale = scale
        self.every = every
        self.caption = caption
        self._screen: "pygame.Surface | None" = None
        self._font: "pygame.font.Font | None" = None
        self.should_stop = False
        # WR-04: tracked independently of `self._screen` so `__exit__` still
        # calls `pygame.quit()` even if `pygame.display.set_mode` raises
        # during first-frame setup (after `pygame.init()` already ran but
        # before `self._screen` is ever assigned) -- otherwise a genuinely
        # broken SDL video driver would leak initialized pygame/SDL state
        # into whatever runs next in the same process.
        self._initialized = False

    def __enter__(self) -> "Viewer":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if self._initialized:
            pygame.quit()
        return False

    def __call__(self, ev: GenerationEvent) -> None:
        # A closed viewer must never re-open on a later event.
        if self.should_stop:
            return

        if self._screen is None:
            # Lazy, on-first-frame construction is deliberate: a `Viewer`
            # object can exist and be checked for `should_stop` even before
            # any frame has arrived, and a missing/broken SDL video driver
            # raises on THIS first generation -- before any substantial
            # compute -- rather than after the population finishes
            # evaluating deep into a run.
            h, w, _ = ev.best_frame.shape
            pygame.init()
            self._initialized = True
            pygame.display.set_caption(self.caption)
            self._screen = pygame.display.set_mode((w * self.scale, h * self.scale))
            self._font = pygame.font.SysFont(None, 20)

        # Drain the event queue on EVERY call, regardless of the `every`
        # throttle below -- not only on frames that will be redrawn. This is
        # what makes "closing the window ends the run within one generation"
        # true even when `every` is 5 or 10; skipping this drain on
        # throttled generations is the most likely regression a future edit
        # introduces.
        for pygame_event in pygame.event.get():
            if pygame_event.type == pygame.QUIT:
                self.should_stop = True
        if self.should_stop:
            return

        # Always draw the final frame, whatever the throttle.
        if ev.generation % self.every != 0 and not ev.stop_reason:
            return

        # The `.transpose(1, 0, 2)` is required because pygame surfaces are
        # (width, height)-indexed while `best_frame` is (height, width,
        # channels) -- the STACK.md-measured 0.124 ms/frame path. This must
        # be bit-exact, so never reach for `pygame.draw.polygon` or any
        # independent re-render (Anti-Pattern 5): the viewer shows precisely
        # the array the engine scored, nothing else.
        surf = pygame.surfarray.make_surface(ev.best_frame.transpose(1, 0, 2))
        self._screen.blit(pygame.transform.scale(surf, self._screen.get_size()), (0, 0))
        overlay = self._font.render(
            f"gen {ev.generation}  fitness {ev.best_fitness:.4f}  renders {ev.renders}",
            True,
            (255, 0, 0),
            (255, 255, 255),
        )
        self._screen.blit(overlay, (4, 4))
        pygame.display.flip()
