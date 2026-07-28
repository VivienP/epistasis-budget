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
from epibudget.tie_break import DEFAULT_TIE_SEED, canonical_id, seeded_order
from epibudget.types import Allocation, ScoredVariant, Variant


def _rank(
    candidates: Sequence[ScoredVariant], scores: Sequence[float], tie_seed: int
) -> list[ScoredVariant]:
    """Rank by descending score, resolving exact ties with the shared seeded algorithm (H-1).

    Fast path: when every score is distinct there is no tie to resolve, so the ordering is fully
    determined and the (comparatively expensive) canonical identities are never built. That matters
    because a continuous score — predicted fitness, or dispersion-weighted coverage — is almost
    always tie-free, while the loop-coverage score is nothing but ties.
    """
    if len(set(scores)) == len(scores):
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return [candidates[i] for i in order]
    score_of = {id(sv): value for sv, value in zip(candidates, scores, strict=True)}
    identity_of = {id(sv): canonical_id(sv.variant) for sv in candidates}
    return seeded_order(
        list(candidates),
        lambda sv: score_of[id(sv)],
        lambda sv: identity_of[id(sv)],
        tie_seed,
    )


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
    method: str = "info",
    tie_seed: int = DEFAULT_TIE_SEED,
) -> Allocation:
    """Select ``budget`` variants maximising ``(1−λ)·norm_info(v) + λ·norm_fit(v)``.

    ``info_gain`` is modular (graph.py), so this is a single sort — no iterative greedy loop. The λ
    endpoints are special-cased to bypass normalisation (0/0 when a score is constant across the
    pool): λ=1 reproduces :func:`fitness_greedy` exactly, λ=0 sorts by the raw info-gain weight.
    Selection reads only ESM-predicted ``delta_g`` and the factor-graph info-gain — never a measured
    label.

    Exact ties are resolved by ``tie_seed`` through the shared algorithm in ``tie_break`` (audit
    H-1), so the ordering is invariant to the caller's input order and reproducible from a declared
    number. This is load-bearing for ``structural``: its loop-coverage score takes only three values
    on a four-site landscape, so the whole within-order ordering is a tie and the seed *is* the
    method. A single unseeded draw is a sample of size one; use
    ``tie_break.stratum_crosses_budget`` to detect when a budget makes that so.

    ``method`` is recorded verbatim on the result as provenance for which τ² weighting built
    ``graph`` (see :func:`epibudget.graph.selection_graph`); it does not alter this ranking.
    """
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    if budget > len(candidates):
        raise ValueError(f"budget {budget} exceeds candidate count {len(candidates)}")
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}")

    info = {sv.variant: graph.info_gain(frozenset(), sv.variant) for sv in candidates}
    if lambda_ < 1.0 and len({round(v, 12) for v in info.values()}) == 1:
        # Audit M-2: a constant weight carries no ordering information, so the "selection" collapses
        # to whatever the tie-break produces — a seeded uniform sample, not an information-weighted
        # design. This is what n_perturbations=0 does to `--method info`: every var_delta_g is 0.
        # Fail loudly rather than return a random plate under an informative-sounding name.
        raise ValueError(
            "every candidate has the same acquisition weight, so this ranking would reduce to the "
            "seeded tie-break — a uniform sample, not an information-weighted selection; check "
            "that the method actually varies across candidates (a zero masking-dispersion cache "
            "makes 'info' constant)"
        )
    if lambda_ == 1.0:
        ranked = _rank(candidates, [s.delta_g for s in candidates], tie_seed)
    elif lambda_ == 0.0:
        ranked = _rank(candidates, [info[s.variant] for s in candidates], tie_seed)
    else:
        norm_fit = _minmax([s.delta_g for s in candidates])
        norm_info = _minmax([info[s.variant] for s in candidates])
        blended = [
            (1.0 - lambda_) * norm_info[i] + lambda_ * norm_fit[i] for i in range(len(candidates))
        ]
        ranked = _rank(candidates, blended, tie_seed)

    chosen = ranked[:budget]
    return Allocation(
        budget=budget,
        selected=[s.variant for s in chosen],
        expected_info_gain=[info[s.variant] for s in chosen],
        epistasis_map=list(graph.interactions),
        seed=seed,
        model_id=model_id,
        method=method,
    )


def fitness_greedy(
    candidates: Sequence[ScoredVariant], budget: int, tie_seed: int = DEFAULT_TIE_SEED
) -> list[Variant]:
    """Baseline: top-``budget`` variants by predicted ΔG (== allocate with λ=1, same ``tie_seed``).

    Ties are resolved by the same shared algorithm, so the documented λ=1 identity still holds. A
    continuous ESM score rarely ties, so this normally takes the tie-free fast path.
    """
    ranked = _rank(candidates, [s.delta_g for s in candidates], tie_seed)
    return [s.variant for s in ranked[:budget]]
