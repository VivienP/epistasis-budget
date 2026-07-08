"""Information-optimal greedy budget allocation with an exploitation slider. See docs/SPEC.md#6.

score(v) = (1 − λ)·info_gain(v) + λ·normalized_fitness(v)
  λ = 0 → pure information-optimal (the thesis)
  λ = 1 → pure fitness-greedy (the baseline to beat / current practice)
Submodular info_gain ⇒ greedy is (1 − 1/e) near-optimal.
"""

from __future__ import annotations

from collections.abc import Sequence

from epibudget.graph import EpistasisFactorGraph
from epibudget.types import Allocation, ScoredVariant, Variant


def allocate(
    graph: EpistasisFactorGraph,
    candidates: Sequence[ScoredVariant],
    budget: int,
    lambda_: float = 0.0,
    seed: int = 0,
    model_id: str = "",
) -> Allocation:
    """Greedily select ``budget`` variants maximising blended info-gain / fitness."""
    raise NotImplementedError("Week 2 — see docs/ROADMAP.md")


def fitness_greedy(candidates: Sequence[ScoredVariant], budget: int) -> list[Variant]:
    """Baseline: top-``budget`` variants by predicted ΔG (== allocate with λ=1)."""
    ranked = sorted(candidates, key=lambda s: s.delta_g, reverse=True)
    return [s.variant for s in ranked[:budget]]
