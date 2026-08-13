"""Module entry point: ``python -m gridworld`` starts the game."""

import sys
from gridworld.ui.app import run

if __name__ == "__main__":
    if len(sys.argv) > 2:
        sys.stderr.write("Usage: python -m gridworld [level_path]\n")
        sys.exit(1)
    level_path = sys.argv[1] if len(sys.argv) == 2 else None
    run(level_path)
