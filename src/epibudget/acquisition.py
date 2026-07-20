"""Modular budget allocation with an exploitation slider. See docs/SPEC.md#6.

score(v) = (1 − λ)·normalized_info_gain(v) + λ·normalized_fitness(v)
  λ = 0 → ESM-dispersion × loop-coverage heuristic
  λ = 1 → pure fitness-greedy baseline
Under the v1 independent-noise model info_gain is modular (docs/SPEC.md#5), so greedy is exactly
optimal for that stated modular objective: it coincides with sorting candidates by the fixed weight
and taking the top B. This does not make it posterior-optimal for the landscape-recovery estimand.
"""

from __future__ import annotations

from collections.abc import Sequence

from epibudget.graph import EpistasisFactorGraph
from epibudget.types import Allocation, ScoredVariant, Variant


def _minmax(values: Sequence[float]) -> list[float]:
    """Scale ``values`` to [0, 1]; a degenerate (all-equal) input maps to all-zeros (no signal)."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0] * len(values)
    span = hi - lo
    return [(v - lo) / span for v in values]


def allocate(
    graph: EpistasisFactorGraph,
    candidates: Sequence[ScoredVariant],
    budget: int,
    lambda_: float = 0.0,
    seed: int = 0,
    model_id: str = "",
) -> Allocation:
    """Select ``budget`` variants maximising ``(1−λ)·norm_info(v) + λ·norm_fit(v)``.

    ``info_gain`` is modular (graph.py), so this is a single stable sort — no iterative greedy loop.
    The λ endpoints are special-cased to bypass normalisation (which is 0/0 when a score is constant
    across the pool): λ=1 reproduces :func:`fitness_greedy` exactly (same stable sort key and input
    order ⇒ identical ordered list), λ=0 sorts by the raw info-gain weight. Selection reads only
    ESM-predicted ``delta_g`` and the factor-graph info-gain — never a measured label.
    """
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    if budget > len(candidates):
        raise ValueError(f"budget {budget} exceeds candidate count {len(candidates)}")
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}")

    info = {sv.variant: graph.info_gain(frozenset(), sv.variant) for sv in candidates}
    if lambda_ == 1.0:
        ranked = sorted(candidates, key=lambda s: s.delta_g, reverse=True)
    elif lambda_ == 0.0:
        ranked = sorted(candidates, key=lambda s: info[s.variant], reverse=True)
    else:
        norm_fit = _minmax([s.delta_g for s in candidates])
        norm_info = _minmax([info[s.variant] for s in candidates])
        blended = [
            (1.0 - lambda_) * norm_info[i] + lambda_ * norm_fit[i] for i in range(len(candidates))
        ]
        order = sorted(range(len(candidates)), key=lambda i: blended[i], reverse=True)
        ranked = [candidates[i] for i in order]

    chosen = ranked[:budget]
    return Allocation(
        budget=budget,
        selected=[s.variant for s in chosen],
        expected_info_gain=[info[s.variant] for s in chosen],
        epistasis_map=list(graph.interactions),
        seed=seed,
        model_id=model_id,
    )


def fitness_greedy(candidates: Sequence[ScoredVariant], budget: int) -> list[Variant]:
    """Baseline: top-``budget`` variants by predicted ΔG (== allocate with λ=1)."""
    ranked = sorted(candidates, key=lambda s: s.delta_g, reverse=True)
    return [s.variant for s in ranked[:budget]]
