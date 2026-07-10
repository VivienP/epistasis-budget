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

from epibudget.data import (
    GB1_SITES,
    GB1_WT_AT_SITES,
    GB1_WT_SEQUENCE,
    apply_mutations,
    enumerate_candidates,
)
from epibudget.epistasis import epsilon_pairwise
from epibudget.scoring import ConjointScorer, additive_delta_g, resolve_device
from epibudget.types import Mutation, ScoredVariant, Variant

# Max scalar drift tolerated between the batched and reference paths. Both forward the same masked
# rows; the only possible divergence is CPU-BLAS batch non-invariance (the batched path packs
# unrelated rows). Measured on this machine: the gap is exactly 0.0 (bit-exact) at both 1 and 12
# threads (scripts/bench_scoring.py). This tiny tolerance only guards against BLAS variation on
# other CPUs; on the GPU path (not bit-parity with the CPU oracle) parity is by selection identity.
_PARITY_ATOL = 1e-6

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


def test_resolve_device_pass_through_and_auto() -> None:
    """Explicit devices pass through; ``auto`` selects CUDA only when a GPU is present, else CPU."""
    import torch  # noqa: PLC0415  # offline: only queries cuda availability, no model download

    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("auto") == ("cuda" if torch.cuda.is_available() else "cpu")


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


def _info_selection(scored: list[ScoredVariant], budget: int) -> set[Variant]:
    """Info-optimal (λ=0) selection from a set of scored variants — the decision-level output."""
    from epibudget.acquisition import allocate  # noqa: PLC0415
    from epibudget.epistasis import predicted_epistasis  # noqa: PLC0415
    from epibudget.graph import EpistasisFactorGraph  # noqa: PLC0415

    graph = EpistasisFactorGraph(
        predicted_epistasis(scored, max_order=3),
        {sv.variant: sv.var_delta_g for sv in scored},
    )
    return set(allocate(graph, scored, budget, lambda_=0.0).selected)


@pytest.mark.slow
def test_optimized_batch_matches_reference() -> None:
    """score_batch (de-duped, cross-variant batched) reproduces score (the per-variant reference).

    Throughput-only: on a real GB1 slice with the full ε machinery, the batched delta_g/var_delta_g
    must equal the per-variant oracle within a tight tolerance, and — the claim that actually
    matters — the info-optimal and fitness-greedy selections built from either must be identical.
    Run single-threaded so CPU BLAS is batch-invariant; the only allowed divergence is a residual
    float gap from packing unrelated rows, which must not move any selection.
    """
    import torch  # noqa: PLC0415

    from epibudget.acquisition import fitness_greedy  # noqa: PLC0415

    # A reduced-alphabet slice: cross-variant de-dup is active (shared masked rows) and there are
    # enough order-2/3 terms to build a real selection. pool = 8 singles + 24 doubles + 32 triples.
    candidates = enumerate_candidates(GB1_SITES, GB1_WT_AT_SITES, allowed_aa="ACD", max_order=3)
    scorer = ConjointScorer("facebook/esm2_t12_35M_UR50D", seed=0, n_perturbations=4)

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        reference = [scorer.score(GB1_WT_SEQUENCE, v) for v in candidates]
        optimized = scorer.score_batch(GB1_WT_SEQUENCE, candidates)
    finally:
        torch.set_num_threads(old_threads)

    ref_by = {sv.variant: sv for sv in reference}
    max_dg = max(abs(o.delta_g - ref_by[o.variant].delta_g) for o in optimized)
    max_var = max(abs(o.var_delta_g - ref_by[o.variant].var_delta_g) for o in optimized)
    assert max_dg <= _PARITY_ATOL, f"delta_g drift {max_dg:.2e} exceeds {_PARITY_ATOL:.0e}"
    assert max_var <= _PARITY_ATOL, f"var_delta_g drift {max_var:.2e} exceeds {_PARITY_ATOL:.0e}"

    for budget in (8, 16):
        assert _info_selection(reference, budget) == _info_selection(optimized, budget)
        assert set(fitness_greedy(reference, budget)) == set(fitness_greedy(optimized, budget))


@pytest.mark.slow
def test_score_batch_is_deterministic() -> None:
    """Two score_batch runs at a fixed thread count give identical numbers (seeded determinism)."""
    import torch  # noqa: PLC0415

    candidates = enumerate_candidates(GB1_SITES, GB1_WT_AT_SITES, allowed_aa="AC", max_order=2)
    scorer = ConjointScorer("facebook/esm2_t12_35M_UR50D", seed=0, n_perturbations=4)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        first = scorer.score_batch(GB1_WT_SEQUENCE, candidates)
        second = scorer.score_batch(GB1_WT_SEQUENCE, candidates)
    finally:
        torch.set_num_threads(old_threads)
    assert all(
        x.delta_g == y.delta_g and x.var_delta_g == y.var_delta_g
        for x, y in zip(first, second, strict=True)
    )
