"""Independent oracles for the shared tie-resolution algorithm (audit finding H-1).

The loop counts are checked against a closed-form derivation and against a brute-force enumeration,
neither of which calls the factor graph; the ordering properties are checked as invariants of the
algorithm rather than against a recorded expected list.
"""

from __future__ import annotations

from itertools import combinations

import pytest

from epibudget.acquisition import allocate, fitness_greedy
from epibudget.data import enumerate_candidates
from epibudget.downstream import canonical_id as downstream_canonical_id
from epibudget.epistasis import interaction_loop
from epibudget.gate2 import canonical_id as gate2_canonical_id
from epibudget.graph import selection_graph
from epibudget.tie_break import (
    DEFAULT_TIE_SEED,
    analytic_loop_counts,
    canonical_id,
    loop_counts_over_universe,
    score_strata,
    seeded_order,
    stratum_crosses_budget,
)
from epibudget.types import ScoredVariant, Variant

_AA20 = "ACDEFGHIKLMNPQRSTVWY"
_GB1_SITES = (38, 39, 40, 53)
_GB1_WT = ("V", "D", "G", "V")
_EXPECTED_FOUR_SITE_COUNTS = {1: 1140, 2: 39, 3: 1}
_N_SINGLES = 76
_N_DOUBLES = 2166
_N_TRIPLES = 27436
_PAIRWISE_ORDER = 2


def _brute_force_counts(universe: list[Variant], max_order: int) -> dict[Variant, int]:
    """Oracle: count loop memberships by direct enumeration of every interaction's subsets."""
    counts: dict[Variant, int] = dict.fromkeys(universe, 0)
    for variant in universe:
        if not _PAIRWISE_ORDER <= len(variant) <= max_order:
            continue
        for member in interaction_loop(tuple(sorted(variant))):
            if member in counts:
                counts[member] += 1
    return counts


def test_analytic_loop_counts_match_the_documented_four_site_values() -> None:
    """n(v) is 1140 / 39 / 1 on the confirmatory universe, and constant within each order."""
    assert analytic_loop_counts(4, 20, 3) == _EXPECTED_FOUR_SITE_COUNTS


def test_analytic_loop_counts_agree_with_brute_force_on_a_tiny_universe() -> None:
    """Closed form vs direct enumeration on 3 sites x 3-letter alphabet (no formula reuse)."""
    sites, wt, alphabet = (0, 1, 2), ("M", "T", "Y"), "ACD"
    universe = enumerate_candidates(sites, wt, alphabet, 3)
    brute = _brute_force_counts(universe, 3)
    analytic = analytic_loop_counts(len(sites), len(alphabet) + 1, 3)
    for variant, count in brute.items():
        assert count == analytic[len(variant)], variant
    assert dict(loop_counts_over_universe(universe, 3)) == brute


def test_loop_counts_match_the_factor_graph_weight() -> None:
    """``loop_counts_over_universe`` reproduces the graph's tau^2 == 1 info gain exactly."""
    sites, wt, alphabet = (0, 1, 2), ("M", "T", "Y"), "ACD"
    universe = enumerate_candidates(sites, wt, alphabet, 3)
    scored = [ScoredVariant(variant=v, delta_g=0.0, var_delta_g=1.0) for v in universe]
    graph = selection_graph(scored, 3, "structural")
    counts = loop_counts_over_universe(universe, 3)
    for variant in universe:
        assert graph.info_gain(frozenset(), variant) == pytest.approx(float(counts[variant]))


def test_structural_score_is_three_valued_on_the_confirmatory_universe() -> None:
    """The degeneracy the audit found: the weight carries no within-order information at all."""
    universe = enumerate_candidates(_GB1_SITES, _GB1_WT, _AA20, 3)
    assert len(universe) == _N_SINGLES + _N_DOUBLES + _N_TRIPLES
    counts = loop_counts_over_universe(universe, 3)
    by_order: dict[int, set[int]] = {}
    for variant, count in counts.items():
        by_order.setdefault(len(variant), set()).add(count)
    assert by_order == {order: {value} for order, value in _EXPECTED_FOUR_SITE_COUNTS.items()}


def test_seeded_order_is_invariant_to_input_enumeration_order() -> None:
    """Same candidates, same seed, reversed input -> byte-identical ranking (H-1 acceptance)."""
    universe = enumerate_candidates((0, 1, 2), ("M", "T", "Y"), "ACD", 3)
    counts = loop_counts_over_universe(universe, 3)

    def score(v: Variant) -> float:
        return float(counts[v])

    forward = seeded_order(universe, score, canonical_id, tie_seed=7)
    reversed_input = seeded_order(list(reversed(universe)), score, canonical_id, tie_seed=7)
    shuffled = seeded_order(sorted(universe, key=canonical_id), score, canonical_id, tie_seed=7)
    assert forward == reversed_input == shuffled


def test_different_tie_seeds_give_different_plates() -> None:
    """A tie seed is a scientific setting: it changes which tied candidates are bought."""
    universe = enumerate_candidates((0, 1, 2), ("M", "T", "Y"), "ACD", 3)
    counts = loop_counts_over_universe(universe, 3)

    def score(v: Variant) -> float:
        return float(counts[v])

    plates = {tuple(seeded_order(universe, score, canonical_id, tie_seed=s)[:4]) for s in range(20)}
    assert len(plates) > 1


def test_seeded_order_respects_the_score_ordering_across_strata() -> None:
    """Ties are permuted only within a stratum; a higher score always precedes a lower one."""
    universe = enumerate_candidates((0, 1, 2), ("M", "T", "Y"), "ACD", 3)
    counts = loop_counts_over_universe(universe, 3)

    def score(v: Variant) -> float:
        return float(counts[v])

    ordered = seeded_order(universe, score, canonical_id, tie_seed=3)
    scores = [score(v) for v in ordered]
    assert scores == sorted(scores, reverse=True)
    # Every stratum is emitted contiguously and completely before the next one begins.
    strata = score_strata(universe, score, canonical_id)
    offset = 0
    for value, members in strata:
        block = ordered[offset : offset + len(members)]
        assert {canonical_id(v) for v in block} == {canonical_id(v) for v in members}
        assert all(score(v) == value for v in block)
        offset += len(members)


def test_stratum_crosses_budget_detects_a_seed_dependent_plate() -> None:
    """B=48 over 76 tied singles is seed-dependent; a budget on a stratum boundary is not."""
    universe = enumerate_candidates(_GB1_SITES, _GB1_WT, _AA20, 3)
    counts = loop_counts_over_universe(universe, 3)

    def score(v: Variant) -> float:
        return float(counts[v])

    assert stratum_crosses_budget(universe, score, canonical_id, 48) is True
    assert stratum_crosses_budget(universe, score, canonical_id, 192) is True
    # Exactly all singles: the cut lands on a stratum boundary, so no tie decides the plate.
    assert stratum_crosses_budget(universe, score, canonical_id, _N_SINGLES) is False


def test_default_tie_seed_is_declared_and_stable() -> None:
    """A single-draw table must be able to name its seed; the default is explicit, not implicit."""
    assert DEFAULT_TIE_SEED == 0


def test_canonical_id_is_order_independent_and_matches_the_pipelines() -> None:
    """The stratum sort key is identical to the one downstream and gate2 already use."""
    a: Variant = frozenset({(0, "A", "C"), (1, "A", "G")})
    b: Variant = frozenset({(1, "A", "G"), (0, "A", "C")})
    assert canonical_id(a) == canonical_id(b)
    assert canonical_id(a) == downstream_canonical_id(a) == gate2_canonical_id(a)


def test_loop_counts_ignore_interactions_above_max_order() -> None:
    """max_order=2 counts only pairwise loops, so a single lies in exactly its cross-site pairs."""
    sites, wt, alphabet = (0, 1, 2), ("M", "T", "Y"), "ACD"
    universe = enumerate_candidates(sites, wt, alphabet, 2)
    counts = loop_counts_over_universe(universe, 2)
    singles = [v for v in universe if len(v) == 1]
    # 2 other sites x 3 residues each = 6 pairwise interactions contain a given single.
    expected = len(sites[1:]) * len(alphabet)
    assert {counts[v] for v in singles} == {expected}
    assert analytic_loop_counts(len(sites), len(alphabet) + 1, 2)[1] == expected


def test_brute_force_and_production_agree_on_a_two_site_landscape() -> None:
    """Smallest non-trivial case: 2 sites, 2 residues -> each single lies in 2 pairwise loops."""
    universe = enumerate_candidates((0, 1), ("M", "T"), "AC", 2)
    counts = loop_counts_over_universe(universe, 2)
    for variant in universe:
        expected = sum(
            1
            for other in universe
            if len(other) == _PAIRWISE_ORDER
            and frozenset(variant) in {frozenset(c) for c in _subsets(sorted(other))}
        )
        assert counts[variant] == expected


def _subsets(mutations: list[tuple[int, str, str]]) -> list[tuple[tuple[int, str, str], ...]]:
    return [c for r in range(1, len(mutations) + 1) for c in combinations(mutations, r)]


# --------------------------------------------------------------- pipeline integration (audit H-1)


def test_allocate_is_input_order_invariant_for_a_fixed_tie_seed() -> None:
    """The selection must not depend on how the caller happened to order the candidate list."""
    universe = enumerate_candidates((0, 1, 2), ("M", "T", "Y"), "ACD", 3)
    scored = [
        ScoredVariant(variant=v, delta_g=0.0, var_delta_g=1.0 + 0.0 * i)
        for i, v in enumerate(universe)
    ]
    graph = selection_graph(scored, 3, "structural")
    forward = allocate(graph, scored, 6, lambda_=0.0, tie_seed=3).selected
    backward = allocate(graph, list(reversed(scored)), 6, lambda_=0.0, tie_seed=3).selected
    assert forward == backward
    # ...and the seed is load-bearing: a different declared seed is a different plate.
    other = allocate(graph, scored, 6, lambda_=0.0, tie_seed=4).selected
    assert forward != other


def test_validate_and_downstream_select_the_same_plate_for_the_same_inputs_and_seed() -> None:
    """The two pipelines must not disagree about what `structural` means (audit H-1).

    Before the shared algorithm, `validate` inherited the cache enumeration order while `downstream`
    pre-sorted by canonical identity, so the same named method drew different plates in each.
    """
    universe = enumerate_candidates((0, 1, 2), ("M", "T", "Y"), "ACD", 3)
    scored = [ScoredVariant(variant=v, delta_g=0.0, var_delta_g=1.0) for v in universe]
    graph = selection_graph(scored, 3, "structural")

    # validate consumes the cache/enumeration order; downstream pre-sorts by canonical identity.
    validate_like = allocate(graph, scored, 8, lambda_=0.0, tie_seed=0).selected
    downstream_like = allocate(
        graph,
        sorted(scored, key=lambda sv: downstream_canonical_id(sv.variant)),
        8,
        lambda_=0.0,
        tie_seed=0,
    ).selected
    assert validate_like == downstream_like


def test_fitness_greedy_still_equals_allocate_at_lambda_one() -> None:
    """The documented λ=1 identity must survive the shared tie-break."""
    universe = enumerate_candidates((0, 1, 2), ("M", "T", "Y"), "ACD", 3)
    scored = [
        ScoredVariant(variant=v, delta_g=float(i % 5), var_delta_g=1.0 + i)
        for i, v in enumerate(universe)
    ]
    graph = selection_graph(scored, 3, "info")
    assert allocate(graph, scored, 7, lambda_=1.0, tie_seed=2).selected == fitness_greedy(
        scored, 7, tie_seed=2
    )
