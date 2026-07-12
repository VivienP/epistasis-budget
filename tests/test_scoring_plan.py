"""Offline tests for the pure batched-scoring planner (no ESM-2 download).

These pin the throughput-only refactor's correctness *before* the model is ever loaded: the
masked-row de-duplication counts, the per-variant RNG-draw parity with the reference
``_var_delta_g``, the edge cases, and ``finalize``'s reassembly. The slow ``test_scoring.py`` parity
test then proves the batched path reproduces the reference numbers end to end.
"""

from __future__ import annotations

import numpy as np

from epibudget.data import GB1_SITES, GB1_WT_AT_SITES, GB1_WT_SEQUENCE, enumerate_candidates
from epibudget.scoring_plan import (
    MASK_CHAR,
    dedup,
    finalize,
    plan_variant,
    variant_key,
)
from epibudget.types import Mutation, Variant

_MASK_FRACTION = 0.15

# Unique deterministic masked-rows per order over the four GB1 sites with the 20-letter alphabet
# (19 non-WT residues/site): mask one query site, the others revealed as their mutant residue.
_N_SITES = 4
_NON_WT = 19
_SINGLE_ROWS = _N_SITES  # 4: one shared "mask this site on WT" row per site
_ORDER2_ROWS = _N_SITES * (_N_SITES - 1) * _NON_WT  # 228: (query, other site) by other residue
_ORDER3_ROWS = _N_SITES * 3 * _NON_WT**2  # 4332: (query, C(3,2) other sites) by two residues
_FULL_ROWS = _SINGLE_ROWS + _ORDER2_ROWS + _ORDER3_ROWS  # 4564 unique deterministic forwards


def _det_unique_count(max_order: int, order_filter: int | None = None) -> int:
    """Number of unique deterministic (n_perturbations=0) rows over the GB1 candidate pool."""
    candidates = enumerate_candidates(
        GB1_SITES, GB1_WT_AT_SITES, allowed_aa="ACDEFGHIKLMNPQRSTVWY", max_order=max_order
    )
    rows = []
    for v in candidates:
        if order_filter is not None and len(v) != order_filter:
            continue
        _, r = plan_variant(
            GB1_WT_SEQUENCE, v, seed=0, n_perturbations=0, mask_fraction=_MASK_FRACTION
        )
        rows.extend(r)
    unique_seqs, _, _ = dedup(rows)
    return len(unique_seqs)


def test_dedup_collapses_singles_to_one_row_per_site() -> None:
    # All 19 substitutions at a GB1 site mask that site on the WT background -> one shared row.
    assert _det_unique_count(max_order=1, order_filter=1) == _SINGLE_ROWS


def test_dedup_collapses_order2_rows() -> None:
    assert _det_unique_count(max_order=2, order_filter=2) == _ORDER2_ROWS


def test_dedup_collapses_order3_rows() -> None:
    assert _det_unique_count(max_order=3, order_filter=3) == _ORDER3_ROWS


def test_dedup_full_pool_matches_4564() -> None:
    # The headline de-dup claim: the full 20-letter four-site deterministic pass is 4,564 forwards.
    assert _det_unique_count(max_order=3) == _FULL_ROWS


def _extra_sets_by_perturbation(
    wt: str, variant: Variant, seed: int, n_perturbations: int
) -> list[frozenset[int]]:
    """Recover, per perturbation pass, the set of background positions the planner masked."""
    passes, rows = plan_variant(
        wt, variant, seed=seed, n_perturbations=n_perturbations, mask_fraction=_MASK_FRACTION
    )
    extras: dict[int, frozenset[int]] = {}
    var_index_of = {pass_id: passes[pass_id][1] for pass_id in range(len(passes))}
    for masked_seq, read_pos, _mut, _wt, pass_id, _si in rows:
        if var_index_of[pass_id] < 0:
            continue
        masked_positions = {i for i, c in enumerate(masked_seq) if c == MASK_CHAR}
        extras[pass_id] = frozenset(masked_positions - {read_pos})
    ordered = sorted((var_index_of[pid], s) for pid, s in extras.items())
    return [s for _t, s in ordered]


def test_var_perturbation_draws_match_reference_rng() -> None:
    """The planner's background-mask draws equal a direct re-run of the reference RNG sequence."""
    wt = GB1_WT_SEQUENCE
    variant: Variant = frozenset(
        {(GB1_SITES[0], GB1_WT_AT_SITES[0], "A"), (GB1_SITES[1], GB1_WT_AT_SITES[1], "C")}
    )
    seed, n_pert = 0, 8
    muts = sorted(variant, key=lambda m: (m[0], m[2]))
    sites = {m[0] for m in muts}
    bg = [q for q in range(len(wt)) if q not in sites]
    n_mask = min(len(bg), max(1, round(_MASK_FRACTION * len(bg))))
    rng = np.random.default_rng(seed + variant_key(muts))
    expected = [
        frozenset(int(q) for q in rng.choice(bg, size=n_mask, replace=False)) for _ in range(n_pert)
    ]
    assert _extra_sets_by_perturbation(wt, variant, seed, n_pert) == expected


def test_n_perturbations_zero_plans_no_var_passes() -> None:
    variant: Variant = frozenset({(GB1_SITES[0], GB1_WT_AT_SITES[0], "A")})
    passes, rows = plan_variant(
        GB1_WT_SEQUENCE, variant, seed=0, n_perturbations=0, mask_fraction=_MASK_FRACTION
    )
    assert [p[1] for p in passes] == [-1]  # deterministic pass only
    assert all(len({i for i, c in enumerate(r[0]) if c == MASK_CHAR}) == 1 for r in rows)


def test_empty_background_plans_no_var_passes() -> None:
    # A 2-residue WT fully covered by a 2-site variant: no background -> var 0, no perturbations.
    wt = "AC"
    variant: Variant = frozenset({(0, "A", "C"), (1, "C", "A")})
    passes, _ = plan_variant(wt, variant, seed=0, n_perturbations=8, mask_fraction=_MASK_FRACTION)
    assert [p[1] for p in passes] == [-1]


def test_empty_variant_plans_nothing() -> None:
    passes, rows = plan_variant(
        GB1_WT_SEQUENCE, frozenset(), seed=0, n_perturbations=16, mask_fraction=_MASK_FRACTION
    )
    assert passes == [] and rows == []


def test_finalize_sums_det_and_variances_var() -> None:
    va: Variant = frozenset({(0, "A", "C")})
    passes = [(va, -1, 2), (va, 0, 2), (va, 1, 2)]
    pass_partials = [[1.0, 2.0], [0.5, 0.5], [1.5, 1.5]]
    out = finalize(passes, pass_partials)
    dg, var = out[va]
    det_expected = pass_partials[0][0] + pass_partials[0][1]
    var_expected = float(np.var(np.array([sum(pass_partials[1]), sum(pass_partials[2])])))
    assert dg == det_expected
    assert var == var_expected


def test_finalize_zero_var_when_no_perturbations() -> None:
    va: Variant = frozenset({(0, "A", "C")})
    only_det = 2.5
    out = finalize([(va, -1, 1)], [[only_det]])
    assert out[va] == (only_det, 0.0)


def test_variant_key_is_deterministic_and_discriminating() -> None:
    a: list[Mutation] = [(0, "A", "C"), (1, "D", "E")]
    b: list[Mutation] = [(0, "A", "D"), (1, "D", "E")]
    assert variant_key(a) == variant_key(a)
    assert variant_key(a) != variant_key(b)
