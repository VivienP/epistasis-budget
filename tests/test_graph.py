"""Tests for the linear-Gaussian epistasis factor graph (docs/SPEC.md#5).

The exit criteria for the graph: ``total_uncertainty`` is non-increasing in the measured set, and
``info_gain`` is submodular on a toy. Under the v1 independent-noise model info_gain is in fact
MODULAR (a degenerate special case of submodular): the diminishing-returns inequality holds with
equality, so greedy is exactly optimal. Both the general non-strict submodularity contract and the
exact modularity are pinned below (the latter as a tripwire — it should start failing if a future
model introduces correlated priors, forcing the "modular" wording to be revisited).
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from epibudget.graph import EpistasisFactorGraph, selection_graph, variant_variance
from epibudget.types import Interaction, Mutation, ScoredVariant, Variant

MUT_I: Mutation = (0, "A", "C")
MUT_J: Mutation = (1, "D", "E")
MUT_K: Mutation = (2, "F", "G")
_TOL = 1e-9


def _v(*muts: Mutation) -> Variant:
    return frozenset(muts)


# The adjudicated toy: 3 sites, all 3 pairwise interactions plus the triple, heterogeneous τ².
_TAU2: dict[Variant, float] = {
    _v(MUT_I): 1.0,
    _v(MUT_J): 2.0,
    _v(MUT_K): 3.0,
    _v(MUT_I, MUT_J): 0.5,
    _v(MUT_I, MUT_K): 0.4,
    _v(MUT_J, MUT_K): 0.3,
    _v(MUT_I, MUT_J, MUT_K): 0.2,
}


def _toy_graph() -> EpistasisFactorGraph:
    # epsilon_hat/sigma2 on the Interaction objects are irrelevant here: the graph sources τ² from
    # var_delta_g, not from Interaction.sigma2.
    interactions = [
        Interaction.of((MUT_I, MUT_J), epsilon_hat=0.0, sigma2=0.0),
        Interaction.of((MUT_I, MUT_K), epsilon_hat=0.0, sigma2=0.0),
        Interaction.of((MUT_J, MUT_K), epsilon_hat=0.0, sigma2=0.0),
        Interaction.of((MUT_I, MUT_J, MUT_K), epsilon_hat=0.0, sigma2=0.0),
    ]
    return EpistasisFactorGraph(interactions, _TAU2)


def test_posterior_variance_prior_is_loop_sum() -> None:
    graph = _toy_graph()
    posterior = graph.posterior_variance(frozenset())
    assert posterior[(MUT_I, MUT_J)] == pytest.approx(3.5)  # 1.0 + 2.0 + 0.5
    assert posterior[(MUT_I, MUT_K)] == pytest.approx(4.4)  # 1.0 + 3.0 + 0.4
    assert posterior[(MUT_J, MUT_K)] == pytest.approx(5.3)  # 2.0 + 3.0 + 0.3
    assert posterior[(MUT_I, MUT_J, MUT_K)] == pytest.approx(7.4)  # all 7 members


def test_total_uncertainty_prior() -> None:
    assert _toy_graph().total_uncertainty(frozenset()) == pytest.approx(20.6)


def test_total_uncertainty_equals_sum_of_posterior_variance() -> None:
    graph = _toy_graph()
    for measured in (
        frozenset(),
        frozenset({_v(MUT_I)}),
        frozenset({_v(MUT_I), _v(MUT_J)}),
        frozenset({_v(MUT_I), _v(MUT_J), _v(MUT_I, MUT_J)}),
    ):
        assert graph.total_uncertainty(measured) == pytest.approx(
            sum(graph.posterior_variance(measured).values())
        )


def test_total_uncertainty_is_non_increasing_in_the_measured_set() -> None:
    graph = _toy_graph()
    chain = [
        frozenset(),
        frozenset({_v(MUT_I)}),
        frozenset({_v(MUT_I), _v(MUT_J)}),
        frozenset({_v(MUT_I), _v(MUT_J), _v(MUT_K)}),
    ]
    totals = [graph.total_uncertainty(m) for m in chain]
    assert totals == [pytest.approx(x) for x in (20.6, 17.6, 11.6, 2.6)]
    assert all(a >= b - _TOL for a, b in pairwise(totals))


def test_info_gain_is_non_negative() -> None:
    graph = _toy_graph()
    measured = frozenset({_v(MUT_I)})
    for candidate in _TAU2:
        assert graph.info_gain(measured, candidate) >= -_TOL


def test_info_gain_zero_for_already_measured_variant() -> None:
    graph = _toy_graph()
    assert graph.info_gain(frozenset({_v(MUT_J)}), _v(MUT_J)) == pytest.approx(0.0)


def test_info_gain_counts_all_loops_a_candidate_braces() -> None:
    graph = _toy_graph()
    # A single braces its 2 pairwise loops + the triple (n=3): τ²_J · 3 = 6.0.
    assert graph.info_gain(frozenset(), _v(MUT_J)) == pytest.approx(6.0)
    # A pair braces its own interaction + the triple (n=2): τ²_IJ · 2 = 1.0.
    assert graph.info_gain(frozenset(), _v(MUT_I, MUT_J)) == pytest.approx(1.0)
    # The triple braces only itself (n=1): τ²_IJK · 1 = 0.2.
    assert graph.info_gain(frozenset(), _v(MUT_I, MUT_J, MUT_K)) == pytest.approx(0.2)


def test_info_gain_is_submodular_non_strict_contract() -> None:
    # General submodularity: gain(A,v) >= gain(B,v) for A subseteq B, v not in B. Non-strict — a
    # strict ">" would be mathematically false for the (modular) v1 model.
    graph = _toy_graph()
    a: frozenset[Variant] = frozenset()
    b = frozenset({_v(MUT_I)})
    assert graph.info_gain(a, _v(MUT_J)) >= graph.info_gain(b, _v(MUT_J)) - 1e-9


def test_info_gain_is_exactly_modular_under_independent_noise() -> None:
    # Regression tripwire: under independent priors the marginal gain is M-independent (modular).
    # If correlated priors are ever introduced this equality should break, forcing the docs wording
    # about modularity to be revisited.
    graph = _toy_graph()
    gain_empty = graph.info_gain(frozenset(), _v(MUT_J))
    gain_after_i = graph.info_gain(frozenset({_v(MUT_I)}), _v(MUT_J))
    assert gain_empty == pytest.approx(6.0)
    assert gain_after_i == pytest.approx(6.0)
    assert gain_empty == pytest.approx(gain_after_i)


def test_graph_raises_when_var_delta_g_missing_a_loop_member() -> None:
    interactions = [Interaction.of((MUT_I, MUT_J), epsilon_hat=0.0, sigma2=0.0)]
    incomplete = {_v(MUT_I): 1.0, _v(MUT_I, MUT_J): 0.5}  # missing the {J} single
    with pytest.raises(KeyError, match="missing sub-variant"):
        EpistasisFactorGraph(interactions, incomplete)


def test_graph_raises_on_duplicate_interactions() -> None:
    # Duplicate mutations would make total_uncertainty and posterior_variance disagree — reject it.
    interactions = [
        Interaction.of((MUT_I, MUT_J), epsilon_hat=0.0, sigma2=0.0),
        Interaction.of((MUT_I, MUT_J), epsilon_hat=1.0, sigma2=0.0),
    ]
    with pytest.raises(ValueError, match="duplicate interactions"):
        EpistasisFactorGraph(interactions, _TAU2)


def test_info_gain_zero_and_total_unchanged_for_candidate_outside_every_loop() -> None:
    # A variant that braces no tracked interaction (n(v)=0) must give info_gain 0 and no KeyError,
    # and adding it to `measured` must not change total_uncertainty (the .get default branch).
    graph = _toy_graph()
    outsider = _v((9, "K", "L"))  # a position that appears in no interaction
    assert graph.info_gain(frozenset(), outsider) == pytest.approx(0.0)
    base = graph.total_uncertainty(frozenset())
    assert graph.total_uncertainty(frozenset({outsider})) == pytest.approx(base)


def _scored_toy() -> list[ScoredVariant]:
    """Two singles and their double, with distinct τ² so the two weightings must disagree."""
    return [
        ScoredVariant(variant=_v(MUT_I), delta_g=0.5, var_delta_g=0.25),
        ScoredVariant(variant=_v(MUT_J), delta_g=0.1, var_delta_g=4.0),
        ScoredVariant(variant=_v(MUT_I, MUT_J), delta_g=0.9, var_delta_g=1.0),
    ]


def test_variant_variance_info_carries_dispersion_and_structural_is_unit() -> None:
    scored = _scored_toy()
    assert variant_variance(scored, "info") == {sv.variant: sv.var_delta_g for sv in scored}
    assert variant_variance(scored, "structural") == {sv.variant: 1.0 for sv in scored}


def test_variant_variance_rejects_an_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown selection method"):
        variant_variance(_scored_toy(), "fitness")  # type: ignore[arg-type]


def test_structural_selection_graph_ranks_by_loops_braced_alone() -> None:
    # With τ² ≡ 1 the weight is exactly n(v): both singles brace the one pairwise loop, so they tie
    # at 1.0 regardless of their very different dispersions — which is what `info` would rank on.
    scored = _scored_toy()
    structural = selection_graph(scored, 2, "structural")
    gains = {sv.variant: structural.info_gain(frozenset(), sv.variant) for sv in scored}
    assert gains[_v(MUT_I)] == pytest.approx(1.0)
    assert gains[_v(MUT_J)] == pytest.approx(1.0)

    info = selection_graph(scored, 2, "info")
    assert info.info_gain(frozenset(), _v(MUT_J)) > info.info_gain(frozenset(), _v(MUT_I))
