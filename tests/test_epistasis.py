"""Tests for the WT-referenced epistasis terms — the mathematical heart, and invariant #1.

These pass today (pure functions). They encode the property that additive ΔG ⇒ ε ≡ 0, which is why
conjoint (non-additive) ESM-2 scoring is mandatory (docs/RESEARCH_EPISTASIS.md#5, CLAUDE.md invariant #1).
"""

from __future__ import annotations

import pytest

from epibudget.epistasis import epsilon_pairwise, epsilon_third
from epibudget.scoring import additive_delta_g
from epibudget.types import Mutation, Variant

I: Mutation = (0, "A", "C")
J: Mutation = (1, "D", "E")
K: Mutation = (2, "F", "G")


def _v(*muts: Mutation) -> Variant:
    return frozenset(muts)


def _additive_landscape(single_effects: dict[Variant, float], muts: list[Mutation]) -> dict[Variant, float]:
    """ΔG map where every variant's score is the sum of its single-mutant effects (perfectly additive)."""
    from itertools import combinations

    dg: dict[Variant, float] = {}
    for order in range(1, len(muts) + 1):
        for combo in combinations(muts, order):
            dg[frozenset(combo)] = additive_delta_g(single_effects, frozenset(combo))
    return dg


def test_epsilon_pairwise_detects_interaction() -> None:
    dg = {_v(I): 1.0, _v(J): 1.0, _v(I, J): 2.5}
    assert epsilon_pairwise(dg, I, J) == pytest.approx(0.5)


def test_epsilon_pairwise_is_zero_for_additive_landscape() -> None:
    # Invariant #1, in miniature: additivity ⇒ no epistasis signal.
    singles = {_v(I): 1.5, _v(J): -0.7}
    dg = _additive_landscape(singles, [I, J])
    assert epsilon_pairwise(dg, I, J) == pytest.approx(0.0)


def test_epsilon_third_is_zero_for_additive_landscape() -> None:
    singles = {_v(I): 1.5, _v(J): -0.7, _v(K): 0.4}
    dg = _additive_landscape(singles, [I, J, K])
    assert epsilon_third(dg, I, J, K) == pytest.approx(0.0)


def test_epsilon_third_recovers_injected_interaction() -> None:
    # Start additive, then inject a pure third-order term of +0.3 into the triple.
    singles = {_v(I): 1.5, _v(J): -0.7, _v(K): 0.4}
    dg = _additive_landscape(singles, [I, J, K])
    dg[_v(I, J, K)] += 0.3
    assert epsilon_third(dg, I, J, K) == pytest.approx(0.3)


def test_epsilon_symmetric_in_its_sites() -> None:
    dg = {_v(I): 0.2, _v(J): 0.9, _v(I, J): 1.7}
    assert epsilon_pairwise(dg, I, J) == pytest.approx(epsilon_pairwise(dg, J, I))
