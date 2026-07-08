"""Tests for the WT-referenced epistasis terms — the mathematical heart, and invariant #1.

These pass today (pure functions). They encode the property that additive ΔG ⇒ ε ≡ 0, which is why
conjoint (non-additive) ESM-2 scoring is mandatory
(docs/RESEARCH_EPISTASIS.md#5, CLAUDE.md invariant #1).
"""

from __future__ import annotations

from itertools import combinations

import pytest

from epibudget.epistasis import epsilon_pairwise, epsilon_third
from epibudget.scoring import additive_delta_g
from epibudget.types import Mutation, Variant

MUT_I: Mutation = (0, "A", "C")
MUT_J: Mutation = (1, "D", "E")
MUT_K: Mutation = (2, "F", "G")


def _v(*muts: Mutation) -> Variant:
    return frozenset(muts)


def _additive_landscape(
    single_effects: dict[Variant, float], muts: list[Mutation]
) -> dict[Variant, float]:
    """ΔG map where every variant's score is the sum of its single-mutant effects.

    The landscape is perfectly additive by construction.
    """
    dg: dict[Variant, float] = {}
    for order in range(1, len(muts) + 1):
        for combo in combinations(muts, order):
            dg[frozenset(combo)] = additive_delta_g(single_effects, frozenset(combo))
    return dg


def test_epsilon_pairwise_detects_interaction() -> None:
    dg = {_v(MUT_I): 1.0, _v(MUT_J): 1.0, _v(MUT_I, MUT_J): 2.5}
    assert epsilon_pairwise(dg, MUT_I, MUT_J) == pytest.approx(0.5)


def test_epsilon_pairwise_is_zero_for_additive_landscape() -> None:
    # Invariant #1, in miniature: additivity ⇒ no epistasis signal.
    singles = {_v(MUT_I): 1.5, _v(MUT_J): -0.7}
    dg = _additive_landscape(singles, [MUT_I, MUT_J])
    assert epsilon_pairwise(dg, MUT_I, MUT_J) == pytest.approx(0.0)


def test_epsilon_third_is_zero_for_additive_landscape() -> None:
    singles = {_v(MUT_I): 1.5, _v(MUT_J): -0.7, _v(MUT_K): 0.4}
    dg = _additive_landscape(singles, [MUT_I, MUT_J, MUT_K])
    assert epsilon_third(dg, MUT_I, MUT_J, MUT_K) == pytest.approx(0.0)


def test_epsilon_third_recovers_injected_interaction() -> None:
    # Start additive, then inject a pure third-order term of +0.3 into the triple.
    singles = {_v(MUT_I): 1.5, _v(MUT_J): -0.7, _v(MUT_K): 0.4}
    dg = _additive_landscape(singles, [MUT_I, MUT_J, MUT_K])
    dg[_v(MUT_I, MUT_J, MUT_K)] += 0.3
    assert epsilon_third(dg, MUT_I, MUT_J, MUT_K) == pytest.approx(0.3)


def test_epsilon_symmetric_in_its_sites() -> None:
    dg = {_v(MUT_I): 0.2, _v(MUT_J): 0.9, _v(MUT_I, MUT_J): 1.7}
    assert epsilon_pairwise(dg, MUT_I, MUT_J) == pytest.approx(epsilon_pairwise(dg, MUT_J, MUT_I))
