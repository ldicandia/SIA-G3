"""Exception hierarchy shared by every tp4 module.

Kept in its own module so config, runs, data and experiments can all raise
these without importing each other (no import cycles).
"""

from __future__ import annotations


class Tp4Error(Exception):
    """Base class for every error the CLI reports as `error: ...` with exit code 2."""


class ConfigError(Tp4Error, ValueError):
    """The experiment config is missing, malformed or inconsistent."""


class DataError(Tp4Error, ValueError):
    """The input data cannot be loaded or fails validation."""


class RunDirError(Tp4Error, ValueError):
    """The run directory cannot be created or replaced safely."""
