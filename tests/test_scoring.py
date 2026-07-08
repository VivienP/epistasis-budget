"""Tests for the scoring path and INVARIANT #1 (conjoint, never additive).

The pure-function test below documents the failure mode: additive ΔG ⇒ ε ≡ 0. The ESM-2 integration
test is the real guard — it is skipped until scoring.py is implemented (Week 0), then it must assert
Var[ε] > 0 on a real GB1 slice. Do NOT delete or weaken it.
"""

from __future__ import annotations

import pytest

from epibudget.data import apply_mutations
from epibudget.epistasis import epsilon_pairwise
from epibudget.scoring import ConjointScorer, additive_delta_g
from epibudget.types import Mutation, Variant

I: Mutation = (0, "A", "C")
J: Mutation = (1, "D", "E")


def _v(*muts: Mutation) -> Variant:
    return frozenset(muts)


def test_additive_scoring_yields_zero_epistasis() -> None:
    """Why conjoint scoring is required: the forbidden additive shortcut kills all signal."""
    singles = {_v(I): 1.1, _v(J): -0.4}
    dg = {
        _v(I): additive_delta_g(singles, _v(I)),
        _v(J): additive_delta_g(singles, _v(J)),
        _v(I, J): additive_delta_g(singles, _v(I, J)),
    }
    assert epsilon_pairwise(dg, I, J) == pytest.approx(0.0)


def test_apply_mutations_puts_all_mutations_on_the_background() -> None:
    # Conjoint context = the fully-mutated sequence; this is what scoring must condition on.
    wt = "AD"
    assert apply_mutations(wt, _v(I, J)) == "CE"


def test_apply_mutations_rejects_wt_mismatch() -> None:
    with pytest.raises(ValueError, match="WT mismatch"):
        apply_mutations("AD", _v((0, "Q", "C")))  # WT says Q but sequence has A


@pytest.mark.slow
@pytest.mark.skip(reason="Enable once ConjointScorer is implemented (Week 0). Guard for invariant #1.")
def test_epsilon_not_identically_zero() -> None:
    """THE guard: on a real GB1 slice, conjoint ESM-2 scoring must produce non-zero epistasis.

    Implementation sketch (Week 0):
        scorer = ConjointScorer("facebook/esm2_t12_35M_UR50D", seed=0)
        # score singles + doubles over the four GB1 sites on a small slice, build dg, compute ε(i,j)
        # assert numpy.var(list_of_epsilons) > 0
    """
    scorer = ConjointScorer("facebook/esm2_t12_35M_UR50D", seed=0)
    assert scorer is not None  # placeholder until implemented
