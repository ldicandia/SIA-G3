"""Baselines that are NOT genetic algorithms.

Kept outside `tp2/engine/` specifically so a baseline cannot accidentally
import `tp2.engine.operators.selection`, `.crossover`, or `.survival` -- a
baseline that quietly grew a selection step would stop being a baseline and
start being an undisclosed second genetic algorithm.
"""
