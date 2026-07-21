"""Budgeted experimental-design methods for mapping protein epistasis.

Given a protein target and a budget of B wells, compare variant-ranking heuristics that combine
epistatic loop structure with zero-shot ESM-2 signals.

See docs/SPEC.md for the full design and docs/RESEARCH_EPISTASIS.md for the science.
"""

from __future__ import annotations

__version__ = "0.1.0"

from epibudget.acquisition import allocate, fitness_greedy
from epibudget.epistasis import (
    epsilon_pairwise,
    epsilon_third,
    ground_truth_epistasis,
    predicted_epistasis,
    wht_spectrum,
)
from epibudget.graph import EpistasisFactorGraph, selection_graph, variant_variance
from epibudget.types import Allocation, Interaction, Mutation, ScoredVariant, Variant
from epibudget.validate import infer_epistasis, map_recovery, run_validation

__all__ = [
    "Allocation",
    "EpistasisFactorGraph",
    "Interaction",
    "Mutation",
    "ScoredVariant",
    "Variant",
    "__version__",
    "allocate",
    "epsilon_pairwise",
    "epsilon_third",
    "fitness_greedy",
    "ground_truth_epistasis",
    "infer_epistasis",
    "map_recovery",
    "predicted_epistasis",
    "run_validation",
    "selection_graph",
    "variant_variance",
    "wht_spectrum",
]
