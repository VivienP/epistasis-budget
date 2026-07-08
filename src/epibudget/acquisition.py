"""Information-optimal greedy budget allocation with an exploitation slider. See docs/SPEC.md#6.

score(v) = (1 − λ)·info_gain(v) + λ·normalized_fitness(v)
  λ = 0 → pure information-optimal (the thesis)
  λ = 1 → pure fitness-greedy (the baseline to beat / current practice)
Under the v1 independent-noise model info_gain is modular (docs/SPEC.md#5), so greedy is exactly
optimal for a fixed budget — it coincides with sorting candidates by the fixed weight and taking the
top B. The (1 − 1/e) submodular bound is only the fallback for a future correlated-prior model.
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
    raise NotImplementedError("Seedocs/ROADMAP.md")


def fitness_greedy(candidates: Sequence[ScoredVariant], budget: int) -> list[Variant]:
    """Baseline: top-``budget`` variants by predicted ΔG (== allocate with λ=1)."""
    ranked = sorted(candidates, key=lambda s: s.delta_g, reverse=True)
    return [s.variant for s in ranked[:budget]]
