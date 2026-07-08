"""Tests for the scoring path and INVARIANT #1 (conjoint, never additive).

Two pure-function tests pin the failure mode from both sides — additive ΔG ⇒ ε ≡ 0, and a
non-additive ΔG ⇒ ε ≠ 0 — so the invariant-#1 math (``epsilon_pairwise``) is gated offline on every
commit. The ESM-2 integration test is the end-to-end guard: it scores a small real GB1 slice
conjointly and asserts Var[ε] > 0. It is marked ``slow`` (needs an ESM-2 forward pass), so the
default offline run skips it. Do NOT delete or weaken any of them.
"""

from __future__ import annotations

import numpy as np
import pytest

from epibudget.data import GB1_SITES, GB1_WT_AT_SITES, GB1_WT_SEQUENCE, apply_mutations
from epibudget.epistasis import epsilon_pairwise
from epibudget.scoring import ConjointScorer, additive_delta_g
from epibudget.types import Mutation, Variant

MUT_I: Mutation = (0, "A", "C")
MUT_J: Mutation = (1, "D", "E")


def _v(*muts: Mutation) -> Variant:
    return frozenset(muts)


def test_additive_scoring_yields_zero_epistasis() -> None:
    """Why conjoint scoring is required: the forbidden additive shortcut kills all signal."""
    singles = {_v(MUT_I): 1.1, _v(MUT_J): -0.4}
    dg = {
        _v(MUT_I): additive_delta_g(singles, _v(MUT_I)),
        _v(MUT_J): additive_delta_g(singles, _v(MUT_J)),
        _v(MUT_I, MUT_J): additive_delta_g(singles, _v(MUT_I, MUT_J)),
    }
    assert epsilon_pairwise(dg, MUT_I, MUT_J) == pytest.approx(0.0)


def test_nonadditive_landscape_yields_nonzero_epistasis() -> None:
    """The offline guard for invariant #1: a genuinely non-additive ΔG map must produce ε ≠ 0.

    Mirror of the additive test above. Conjoint scoring exists precisely to make the double differ
    from the sum of singles; this asserts ``epsilon_pairwise`` reports that gap when present, so a
    regression collapsing scoring to additive is caught even in the offline suite (the ESM-2
    end-to-end guard below stays ``slow``).
    """
    dg = {
        _v(MUT_I): 1.1,
        _v(MUT_J): -0.4,
        _v(MUT_I, MUT_J): 1.1 + -0.4 + 0.7,  # +0.7 pure interaction on top of the additive sum
    }
    assert epsilon_pairwise(dg, MUT_I, MUT_J) == pytest.approx(0.7)


def test_apply_mutations_puts_all_mutations_on_the_background() -> None:
    # Conjoint context = the fully-mutated sequence; this is what scoring must condition on.
    wt = "AD"
    assert apply_mutations(wt, _v(MUT_I, MUT_J)) == "CE"


def test_apply_mutations_rejects_wt_mismatch() -> None:
    with pytest.raises(ValueError, match="WT mismatch"):
        apply_mutations("AD", _v((0, "Q", "C")))  # WT says Q but sequence has A


@pytest.mark.slow
def test_epsilon_not_identically_zero() -> None:
    """THE guard: on a real GB1 slice, conjoint ESM-2 scoring must produce non-zero epistasis.

    Score the singles and doubles over the four GB1 sites (a fixed non-WT residue per site) with
    the fast 35M model, build the ΔG map, and assert the pairwise ε terms are not all zero — i.e.
    conjoint scoring is genuinely non-additive (invariant #1). ``var_delta_g`` is not exercised
    here (the gate needs only ΔG), so perturbations are switched off for speed.
    """
    scorer = ConjointScorer("facebook/esm2_t12_35M_UR50D", seed=0, n_perturbations=0)

    # One deliberate substitution per GB1 site (all non-WT), then all singles and pairwise combos.
    sites_wt = list(zip(GB1_SITES, GB1_WT_AT_SITES, strict=True))
    muts: dict[int, Mutation] = {p: (p, wt, "A" if wt != "A" else "S") for p, wt in sites_wt}
    singles = [frozenset({m}) for m in muts.values()]
    pairs = [
        frozenset({muts[a], muts[b]}) for i, a in enumerate(GB1_SITES) for b in GB1_SITES[i + 1 :]
    ]
    scored = scorer.score_batch(GB1_WT_SEQUENCE, [*singles, *pairs])
    dg: dict[Variant, float] = {sv.variant: sv.delta_g for sv in scored}

    epsilons = [
        epsilon_pairwise(dg, muts[a], muts[b])
        for i, a in enumerate(GB1_SITES)
        for b in GB1_SITES[i + 1 :]
    ]
    assert float(np.var(epsilons)) > 0.0, "conjoint scoring collapsed to additive (ε ≡ 0)"


@pytest.mark.slow
def test_var_delta_g_is_positive_and_deterministic() -> None:
    """The masking-perturbation dispersion is a real, reproducible uncertainty proxy.

    Background-context masking must produce non-zero dispersion (ESM-2 dropout is 0, so this — not
    MC-dropout — is the uncertainty signal), and it must be deterministic given the seed.
    """
    variant = frozenset(
        {(GB1_SITES[0], GB1_WT_AT_SITES[0], "A"), (GB1_SITES[1], GB1_WT_AT_SITES[1], "C")}
    )
    a = ConjointScorer("facebook/esm2_t12_35M_UR50D", seed=0, n_perturbations=8)
    b = ConjointScorer("facebook/esm2_t12_35M_UR50D", seed=0, n_perturbations=8)
    sv_a = a.score(GB1_WT_SEQUENCE, variant)
    sv_b = b.score(GB1_WT_SEQUENCE, variant)
    assert sv_a.var_delta_g > 0.0  # perturbations genuinely disperse the score
    assert sv_a.var_delta_g == sv_b.var_delta_g  # deterministic given seed
    assert sv_a.delta_g == sv_b.delta_g
