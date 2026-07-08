"""epibudget — information-optimal experimental budget allocation for mapping protein epistasis.

Given a protein target and a budget of B wells, rank the B variants whose measurement would most
reduce uncertainty about the epistatic structure of the fitness landscape (zero-shot, ESM-2).

See docs/SPEC.md for the full design and docs/RESEARCH_EPISTASIS.md for the science.
"""

from __future__ import annotations

__version__ = "0.1.0"

from epibudget.epistasis import (
    epsilon_pairwise,
    epsilon_third,
    ground_truth_epistasis,
    predicted_epistasis,
    wht_spectrum,
)
from epibudget.graph import EpistasisFactorGraph
from epibudget.types import Allocation, Interaction, Mutation, ScoredVariant, Variant

__all__ = [
    "Allocation",
    "EpistasisFactorGraph",
    "Interaction",
    "Mutation",
    "ScoredVariant",
    "Variant",
    "__version__",
    "epsilon_pairwise",
    "epsilon_third",
    "ground_truth_epistasis",
    "predicted_epistasis",
    "wht_spectrum",
]
