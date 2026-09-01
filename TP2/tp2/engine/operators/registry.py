"""Small name-to-factory registry; the loop never branches on method names."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

# T-02-08: the recursion cap for nested specs (currently only `blend` nests).
# A depth of N means N levels of nesting below the top-level build call are
# allowed; exceeding it raises ConfigError naming the kind and the cap,
# instead of recursing until the interpreter stack fails with a traceback
# that names nothing useful. Re-exported from tp2.engine.operators.selection
# since blend is the only caller that needs it.
BLEND_MAX_DEPTH = 4


class Registry:
    def __init__(self, family: str) -> None:
        self.family = family
        self._factories: dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
            self._factories[name] = factory
            return factory
        return decorator

    def build(self, spec: dict[str, Any], depth: int = 0) -> Any:
        # Local import: config.py imports this module (via operators/__init__)
        # at module load time, so a top-level import here would be circular.
        from tp2.engine.config import ConfigError

        if depth > BLEND_MAX_DEPTH:
            raise ConfigError(
                f"{self.family} spec nested {depth} levels deep, exceeding the max depth {BLEND_MAX_DEPTH}"
            )
        method = spec.get("method")
        if method not in self._factories:
            known = ", ".join(sorted(self._factories)) or "(none)"
            raise ConfigError(f"unknown {self.family} method {method!r}; known: {known}")
        factory = self._factories[method]
        kwargs = {key: value for key, value in spec.items() if key != "method"}
        # Only a recursive factory (blend) declares `depth`; every other
        # factory's signature is untouched by this seam.
        if "depth" in inspect.signature(factory).parameters:
            kwargs["depth"] = depth
        return factory(**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


SELECTION = Registry("selection")
CROSSOVER = Registry("crossover")
MUTATION = Registry("mutation")
SURVIVAL = Registry("survival")
