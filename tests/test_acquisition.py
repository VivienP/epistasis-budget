"""Tests for greedy budget allocation and the no-label-leakage boundary.

The exit criteria exercised here: ``allocate(lambda_=1)`` reproduces ``fitness_greedy`` exactly, and
selection is provably blind to measured fitness — it ranks only on the ESM-predicted ``delta_g`` and
the factor-graph info-gain, never a label. The label-inversion canary makes the latter behavioural.
"""

from __future__ import annotations

import inspect

import pytest

from epibudget.acquisition import _minmax, allocate, fitness_greedy
from epibudget.data import enumerate_candidates
from epibudget.epistasis import predicted_epistasis
from epibudget.graph import EpistasisFactorGraph
from epibudget.types import ScoredVariant

_BUDGET = 5


def _toy_pool(invert_delta_g: bool = False) -> tuple[EpistasisFactorGraph, list[ScoredVariant]]:
    """A small order-1..3 candidate pool over three sites, with distinct scores and a real graph."""
    variants = enumerate_candidates((0, 1, 2), ("A", "A", "A"), allowed_aa="ACG", max_order=3)
    scored: list[ScoredVariant] = []
    n = len(variants)
    for i, v in enumerate(variants):
        # Distinct delta_g and positive var_delta_g; inversion flips the delta_g ranking only.
        dg = float(n - i) if invert_delta_g else float(i)
        scored.append(ScoredVariant(variant=v, delta_g=dg, var_delta_g=0.1 + 0.01 * i))
    graph = EpistasisFactorGraph(
        predicted_epistasis(scored), {sv.variant: sv.var_delta_g for sv in scored}
    )
    return graph, scored


def test_minmax_scales_to_unit_interval_and_handles_degenerate() -> None:
    assert _minmax([1.0, 3.0, 5.0]) == pytest.approx([0.0, 0.5, 1.0])
    assert _minmax([2.0, 2.0, 2.0]) == [0.0, 0.0, 0.0]  # degenerate: no discriminating information


def test_lambda_1_reproduces_fitness_greedy_exactly() -> None:
    graph, scored = _toy_pool()
    result = allocate(graph, scored, _BUDGET, lambda_=1.0)
    assert result.selected == fitness_greedy(scored, _BUDGET)  # ordered-list equality, not just set


def test_lambda_0_sorts_by_raw_info_gain() -> None:
    graph, scored = _toy_pool()
    result = allocate(graph, scored, _BUDGET, lambda_=0.0)
    expected = sorted(scored, key=lambda s: graph.info_gain(frozenset(), s.variant), reverse=True)[
        :_BUDGET
    ]
    assert result.selected == [s.variant for s in expected]


def test_expected_info_gain_is_raw_not_blended_at_lambda_1() -> None:
    graph, scored = _toy_pool()
    result = allocate(graph, scored, _BUDGET, lambda_=1.0)
    for variant, gain in zip(result.selected, result.expected_info_gain, strict=True):
        assert gain == pytest.approx(graph.info_gain(frozenset(), variant))


def test_budget_exceeding_candidate_count_raises() -> None:
    graph, scored = _toy_pool()
    with pytest.raises(ValueError, match="exceeds"):
        allocate(graph, scored, len(scored) + 1)


def test_blend_returns_exactly_budget_variants() -> None:
    graph, scored = _toy_pool()
    result = allocate(graph, scored, _BUDGET, lambda_=0.5)
    assert len(result.selected) == _BUDGET
    assert len(set(result.selected)) == _BUDGET  # distinct


def test_selection_follows_predicted_delta_g_not_any_hidden_truth() -> None:
    # Label-inversion canary: the highest-delta_g variants are, by construction, whatever we make
    # them; fitness-greedy must track delta_g, so inverting delta_g inverts the selection. Selection
    # has no landscape/label input at all — this proves it behaviourally, not only structurally.
    graph, scored = _toy_pool(invert_delta_g=False)
    graph_inv, scored_inv = _toy_pool(invert_delta_g=True)
    normal = set(allocate(graph, scored, _BUDGET, lambda_=1.0).selected)
    inverted = set(allocate(graph_inv, scored_inv, _BUDGET, lambda_=1.0).selected)
    assert normal.isdisjoint(inverted)  # opposite delta_g ranking ⇒ disjoint top-B


def test_selection_functions_expose_no_label_parameter() -> None:
    # Structural guard: allocate / fitness_greedy must not accept a landscape/fitness argument.
    for fn in (allocate, fitness_greedy):
        params = set(inspect.signature(fn).parameters)
        assert not (params & {"landscape", "fitness", "measured", "labels"})
