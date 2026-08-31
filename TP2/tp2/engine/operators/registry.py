"""Small name-to-factory registry; the loop never branches on method names."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Registry:
    def __init__(self, family: str) -> None:
        self.family = family
        self._factories: dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
            self._factories[name] = factory
            return factory
        return decorator

    def build(self, spec: dict[str, Any]) -> Any:
        method = spec.get("method")
        if method not in self._factories:
            known = ", ".join(sorted(self._factories)) or "(none)"
            raise ValueError(f"unknown {self.family} method {method!r}; known: {known}")
        return self._factories[method](**{key: value for key, value in spec.items() if key != "method"})

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


SELECTION = Registry("selection")
CROSSOVER = Registry("crossover")
MUTATION = Registry("mutation")
SURVIVAL = Registry("survival")
