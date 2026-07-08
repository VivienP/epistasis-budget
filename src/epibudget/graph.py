"""Linear-Gaussian factor graph over epistatic interactions. See docs/SPEC.md#5.

Each interaction ε ~ N(ε_hat, σ²). Measuring a variant reveals a ΔG that is a fixed linear
combination in each loop the variant belongs to, so the posterior-variance update is standard
linear-Gaussian conditioning. Total uncertainty is non-increasing in the measured set and info_gain
is submodular — this is what licenses greedy selection (docs/RESEARCH_EPISTASIS.md#6).
"""

from __future__ import annotations

from collections.abc import Sequence

from epibudget.types import Interaction, Variant


class EpistasisFactorGraph:
    """Tracks posterior uncertainty of interaction terms as variants are hypothetically measured."""

    def __init__(self, interactions: Sequence[Interaction], variants: Sequence[Variant]) -> None:
        self.interactions = list(interactions)
        self.variants = list(variants)

    def posterior_variance(self, measured: frozenset[Variant]) -> dict[tuple[int, ...], float]:
        """Posterior σ² of each interaction given the set of measured variants."""
        raise NotImplementedError("Seedocs/ROADMAP.md")

    def total_uncertainty(self, measured: frozenset[Variant]) -> float:
        """Σ σ² over all interactions given ``measured`` (the objective we minimise)."""
        raise NotImplementedError("Seedocs/ROADMAP.md")

    def info_gain(self, measured: frozenset[Variant], candidate: Variant) -> float:
        """Reduction in total uncertainty from measuring ``candidate`` (≥ 0, submodular)."""
        raise NotImplementedError("Seedocs/ROADMAP.md")
