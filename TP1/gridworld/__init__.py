"""Grid World package root.

This module is intentionally import-free. Importing ``gridworld.engine``
must never transitively pull in ``gridworld.ui`` (and therefore pygame),
because the engine has to import and run in a Python session with no
rendering library installed.
"""
