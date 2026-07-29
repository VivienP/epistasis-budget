"""Independent oracles for the corrected contrast evaluation (audit C-1, C-2, H-2, H-4, M-3).

Every expected value here is derived by hand or by an inline reference implementation that does not
call the corresponding production function. Landscapes are constructed with *known* additive,
pairwise and third-order structure, so the expected contrast is known before the code runs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import pearsonr, rankdata, spearmanr

from epibudget.data import enumerate_candidates
from epibudget.downstream import FeatureSpace, fit_ridge
from epibudget.recovery import (
    CALIBRATION_POLICIES,
    DECISION_ELIGIBLE_POLICIES,
    SCHEMA_VERSION,
    SEED_KINDS,
    TERM_SUBSETS,
    CorrectedRecoveryReport,
    FloatArray,
    Term,
    common_term_subset,
    contrast,
    evaluate_order,
    fisher_z_mean,
    is_effectively_constant,
    measured_skeleton,
    paired_difference_ci,
    partial_correlation,
    prior_mu,
    relative_sse_gain,
    residualise,
    safe_corr,
    term_sha256,
)
from epibudget.types import Mutation, Variant

_SITES = (0, 1, 2, 3)
_WT = ("A", "A", "A", "A")
_ALPHABET = "CD"  # two non-WT residues per site
_TOL = 1e-12
_N_SITES = 4
_PAIRWISE = 2
_THIRD = 3
_STRONG_CORR = 0.5
_VERY_STRONG_CORR = 0.9
_NEGLIGIBLE = 0.05
_SMALL_PARTIAL = 0.15
_NOISE_TOL = 1e-9
_SPURIOUS_CORR = 0.1
_REAL_SPREAD = 0.5


def _mut(site: int, aa: str) -> Mutation:
    return (site, "A", aa)


def _all_variants(max_order: int = 3) -> list[Variant]:
    """Every order-1..max_order variant over 4 sites x 2 residues (independent enumeration)."""
    out: list[Variant] = []
    for order in range(1, max_order + 1):
        for sites in combinations(_SITES, order):
            for residues in product(_ALPHABET, repeat=order):
                out.append(frozenset(_mut(s, a) for s, a in zip(sites, residues, strict=True)))
    return out


# --------------------------------------------------------------------- synthetic landscapes


def _additive_dg() -> dict[Variant, float]:
    """Purely additive DG: every pairwise and third-order contrast must be exactly 0."""
    main = {(s, a): 0.3 * s + (0.7 if a == "C" else -0.4) for s in _SITES for a in _ALPHABET}
    dg = {frozenset(): 0.0}
    for v in _all_variants():
        dg[v] = sum(main[(s, a)] for s, _w, a in v)
    return dg


_PAIR_COUPLING = {(0, 1): 1.5, (0, 2): -0.8, (0, 3): 0.4, (1, 2): 0.9, (1, 3): -1.1, (2, 3): 0.6}
_TRIPLE_COUPLING = 2.25


def _pairwise_dg() -> dict[Variant, float]:
    """Additive + known pairwise couplings: eps2 == the coupling, eps3 == 0 exactly."""
    dg = _additive_dg()
    for v in _all_variants():
        sites = tuple(sorted(s for s, _w, _a in v))
        extra = sum(
            _PAIR_COUPLING[pair] for pair in _PAIR_COUPLING if pair[0] in sites and pair[1] in sites
        )
        dg[v] += extra
    return dg


def _third_order_dg() -> dict[Variant, float]:
    """Additive + pairwise + one constant third-order term: eps3 == that constant exactly."""
    dg = _pairwise_dg()
    for v in _all_variants():
        if len(v) == _THIRD:
            dg[v] += _TRIPLE_COUPLING
    return dg


def _global_epistasis_dg() -> dict[Variant, float]:
    """Additive latent effects passed through a saturating link: the real-landscape confound regime.

    Under nonspecific ("global") epistasis DG = g(a) with a = sum of latent additive effects and g
    concave, the pairwise contrast is

        eps(ij) = g(a_i + a_j) - g(a_i) - g(a_j),

    which for g(a) = a - c*a^2 is exactly -2c*a_i*a_j. So eps is a deterministic function of the
    same latent effects that make up the purchased skeleton k(S) = -DG(i) - DG(j). That coupling —
    not any algebraic identity — is why the raw correlation is large on real landscapes: GB1 and
    TrpB both show strong global epistasis. A fixture that omits it (independent couplings) cannot
    exhibit the confound at all, which is itself the useful statement of when the metric misleads.
    """
    # Deleterious latent effects (as in a real DMS: most substitutions lose function) plus concave
    # saturation. With same-signed a_i the sum and the product move together, so eps and k(S) are
    # strongly associated and the raw correlation inflates -- the regime GB1 and TrpB are in.
    latent = {
        (0, "C"): -0.3,
        (0, "D"): -2.6,
        (1, "C"): -0.8,
        (1, "D"): -2.1,
        (2, "C"): -1.4,
        (2, "D"): -0.5,
        (3, "C"): -1.1,
        (3, "D"): -1.8,
    }
    curvature = 0.18
    dg: dict[Variant, float] = {frozenset(): 0.0}
    for v in _all_variants():
        additive = sum(latent[(site, aa)] for site, _w, aa in v)
        dg[v] = additive - curvature * additive**2
    return dg


def _oracle_eps2(dg: dict[Variant, float], i: Mutation, j: Mutation) -> float:
    """Hand formula, no production call: eps = DG(ij) - DG(i) - DG(j) (+ DG(WT) = 0)."""
    return dg[frozenset({i, j})] - dg[frozenset({i})] - dg[frozenset({j})]


def _oracle_eps3(dg: dict[Variant, float], i: Mutation, j: Mutation, k: Mutation) -> float:
    """Hand formula: DG(ijk) - DG(ij) - DG(ik) - DG(jk) + DG(i) + DG(j) + DG(k)."""
    return (
        dg[frozenset({i, j, k})]
        - dg[frozenset({i, j})]
        - dg[frozenset({i, k})]
        - dg[frozenset({j, k})]
        + dg[frozenset({i})]
        + dg[frozenset({j})]
        + dg[frozenset({k})]
    )


# --------------------------------------------------------------------- contrast algebra


def test_contrast_matches_the_hand_formula_on_all_three_landscapes() -> None:
    for dg in (_additive_dg(), _pairwise_dg(), _third_order_dg()):
        for v in _all_variants():
            if len(v) == 1:
                continue
            term = tuple(sorted(v))
            expected = (
                _oracle_eps2(dg, *term) if len(term) == _PAIRWISE else _oracle_eps3(dg, *term)
            )
            assert contrast(dg, term) == pytest.approx(expected, abs=_TOL)


def test_additive_landscape_has_exactly_zero_contrast_at_every_order() -> None:
    dg = _additive_dg()
    for v in _all_variants():
        if len(v) == 1:
            continue
        assert contrast(dg, tuple(sorted(v))) == pytest.approx(0.0, abs=_TOL)


def test_pairwise_landscape_recovers_the_injected_coupling_and_zero_third_order() -> None:
    dg = _pairwise_dg()
    for v in _all_variants():
        term = tuple(sorted(v))
        sites = tuple(sorted(s for s, _w, _a in v))
        if len(term) == _PAIRWISE:
            assert contrast(dg, term) == pytest.approx(_PAIR_COUPLING[sites], abs=_TOL)
        elif len(term) == _THIRD:
            assert contrast(dg, term) == pytest.approx(0.0, abs=_TOL)


def test_third_order_landscape_recovers_the_injected_triple_term() -> None:
    dg = _third_order_dg()
    for v in _all_variants():
        term = tuple(sorted(v))
        if len(term) == _THIRD:
            assert contrast(dg, term) == pytest.approx(_TRIPLE_COUPLING, abs=_TOL)


def test_measured_skeleton_is_the_shared_component_by_construction() -> None:
    """k(S) equals eps(S) minus the contrast built from the unmeasured members alone."""
    dg = _third_order_dg()
    singles = [v for v in _all_variants() if len(v) == 1]
    measured = frozenset(singles)
    for v in _all_variants():
        if len(v) != _PAIRWISE:
            continue
        term = tuple(sorted(v))
        i, j = term
        # both singles measured, the double is not -> k(S) = -DG(i) - DG(j)
        assert measured_skeleton(term, measured, dg) == pytest.approx(
            -dg[frozenset({i})] - dg[frozenset({j})], abs=_TOL
        )


# --------------------------------------------------------------------- the C-1 confound


def test_all_singles_zero_prior_gives_high_raw_correlation_but_no_skeleton_free_signal() -> None:
    """C-1: the decisive counterexample, in miniature.

    All singles measured, every unmeasured variant given prior 0. The estimate is then exactly
    eps_hat = k(S) = -DG(i) - DG(j), which shares its whole content with the truth. The raw
    correlation is large; the skeleton-controlled association is ~0 because after removing k(S)
    nothing is left; and the SSE gain is negative because the estimate is wrong by DG(ij).
    """
    dg = _global_epistasis_dg()
    singles = [v for v in _all_variants() if len(v) == 1]
    revealed = {v: dg[v] for v in singles}
    terms = [tuple(sorted(v)) for v in _all_variants() if len(v) == _PAIRWISE]
    truth = {t: contrast(dg, t) for t in terms}
    esm = dict.fromkeys(dg, 0.0)  # the model knows nothing at all

    # Precondition: this fixture really is in the skeleton-dominated regime.
    skeleton_sd = float(np.std([measured_skeleton(t, frozenset(revealed), dg) for t in terms]))
    eps_sd = float(np.std([truth[t] for t in terms]))
    assert skeleton_sd > eps_sd

    prior, record = prior_mu(esm, revealed, "zero_prior")
    result = evaluate_order(terms, truth, dg, prior, revealed, "pairwise")

    assert record.decision_eligible and record.n_calibration_labels == 0
    assert result.census.n_pinned == 0
    assert result.census.n_informed_not_pinned == len(terms)
    assert result.census.n_uninformed == 0

    raw = result.raw_pearson_with_skeleton
    assert raw is not None and abs(raw) > _STRONG_CORR  # large "recovery", no model at all
    # ...and it is entirely the skeleton: the estimate IS k(S).
    assert result.skeleton_pearson == pytest.approx(raw, abs=_NOISE_TOL)
    # Residualising on k(S) leaves only floating-point noise, because the estimate IS k(S): the
    # skeleton-free association is negligible next to the headline number it produced.
    partial = result.partial_pearson
    assert partial is None or (abs(partial) < _NEGLIGIBLE and abs(partial) < 0.1 * abs(raw))
    # The plate did not reduce contrast error, so "recovery" wording is refused outright.
    assert result.relative_sse_gain is not None and result.relative_sse_gain < 0.0
    assert not result.recovery_wording_permitted


def test_correlation_can_rise_while_squared_error_worsens() -> None:
    """C-2: a hand-built case where the correlation improves and the SSE gain is negative."""
    truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    prior = np.array([3.0, 3.0, 3.0, 3.0, 3.1])  # nearly flat: poor correlation, modest error
    post = np.array([10.0, 20.0, 30.0, 40.0, 50.0])  # perfectly correlated, wildly miscalibrated

    corr_prior = pearsonr(prior, truth).statistic
    corr_post = pearsonr(post, truth).statistic
    assert corr_post > corr_prior  # correlation improved
    gain = relative_sse_gain(prior, post, truth)
    assert gain is not None and gain < 0.0  # yet the error got much worse

    # The oracle value, computed by hand from the two residual sums.
    expected = 1.0 - float(((post - truth) ** 2).sum()) / float(((prior - truth) ** 2).sum())
    assert gain == pytest.approx(expected)


def test_zero_sse_gain_does_not_permit_recovery_wording() -> None:
    """An unchanged estimator is an honest null, not evidence that anything was recovered."""
    dg = _pairwise_dg()
    terms = [tuple(sorted(v)) for v in _all_variants() if len(v) == _PAIRWISE]
    truth = {term: contrast(dg, term) for term in terms}
    zero_prior = dict.fromkeys(dg, 0.0)

    result = evaluate_order(terms, truth, dg, zero_prior, {}, "pairwise")

    assert result.relative_sse_gain == pytest.approx(0.0)
    assert not result.recovery_wording_permitted


def test_skeleton_controlled_association_is_zero_when_only_the_skeleton_carries_signal() -> None:
    """A high raw correlation with an approximately zero partial correlation, by construction."""
    rng = np.random.default_rng(0)
    skeleton = rng.normal(size=400)
    noise = rng.normal(size=400)
    truth = skeleton + 0.15 * noise
    pred = skeleton + 0.15 * rng.normal(size=400)  # shares only the skeleton with the truth

    raw = pearsonr(pred, truth).statistic
    assert raw > _VERY_STRONG_CORR
    partial = partial_correlation(pred, truth, skeleton, "pearson")
    assert partial is not None and abs(partial) < _SMALL_PARTIAL


def test_residualise_and_partial_correlation_against_a_closed_form_oracle() -> None:
    """Partial correlation of x,y given z equals the standard three-correlation formula."""
    rng = np.random.default_rng(3)
    z = rng.normal(size=200)
    x = 0.8 * z + rng.normal(size=200)
    y = -0.5 * z + rng.normal(size=200)

    rxy = pearsonr(x, y).statistic
    rxz = pearsonr(x, z).statistic
    ryz = pearsonr(y, z).statistic
    expected = (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))
    assert partial_correlation(x, y, z, "pearson") == pytest.approx(expected, abs=1e-10)

    # A constant control cannot explain anything, so partialling it out only centres.
    constant = np.full(200, 4.0)
    assert residualise(x, constant) == pytest.approx(x - x.mean())
    assert partial_correlation(x, y, constant, "pearson") == pytest.approx(rxy, abs=1e-10)


def test_partial_spearman_is_partial_pearson_on_ranks() -> None:
    rng = np.random.default_rng(11)
    z = rng.normal(size=120)
    x = z**3 + rng.normal(size=120)
    y = np.exp(z) + rng.normal(size=120)
    expected = partial_correlation(rankdata(x), rankdata(y), rankdata(z), "pearson")
    assert partial_correlation(x, y, z, "spearman") == pytest.approx(expected)


# --------------------------------------------------------------------- calibration (H-2)


def test_label_free_policies_are_identical_across_methods_and_read_no_labels() -> None:
    """H-2: the decision-eligible policies cannot differ by method because they see no labels."""
    esm = {frozenset({_mut(0, "C")}): 2.0, frozenset({_mut(1, "C")}): -3.0}
    revealed_a = {frozenset({_mut(0, "C")}): 5.0}
    revealed_b = {frozenset({_mut(1, "C")}): -7.0}
    for policy in ("zero_prior", "fixed_unit"):
        mu_a, rec_a = prior_mu(esm, revealed_a, policy)
        mu_b, rec_b = prior_mu(esm, revealed_b, policy)
        assert mu_a == mu_b, policy  # identical prior despite different plates
        assert rec_a.n_calibration_labels == 0 and not rec_a.labels_are_method_specific
        assert rec_a.decision_eligible and rec_b.decision_eligible
        assert policy in DECISION_ELIGIBLE_POLICIES


def test_per_method_policy_is_method_specific_and_not_decision_eligible() -> None:
    """H-2: the historical policy really does produce a different, method-dependent prior."""
    esm = {frozenset({_mut(0, "C")}): 2.0, frozenset({_mut(1, "C")}): -3.0}
    mu_a, rec_a = prior_mu(esm, {frozenset({_mut(0, "C")}): 4.0}, "per_method")
    mu_b, rec_b = prior_mu(esm, {frozenset({_mut(1, "C")}): 6.0}, "per_method")
    assert rec_a.slope == pytest.approx(2.0)  # 2*4 / 2^2
    assert rec_b.slope == pytest.approx(-2.0)  # (-3)*6 / (-3)^2
    assert rec_a.slope * rec_b.slope < 0.0  # opposite signs: the audit's sign-flip, reproduced
    assert mu_a != mu_b
    assert rec_a.labels_are_method_specific and not rec_a.decision_eligible


def test_calibration_policies_are_exactly_the_declared_set() -> None:
    assert set(CALIBRATION_POLICIES) == {"zero_prior", "fixed_unit", "per_method"}
    assert set(CALIBRATION_POLICIES) > DECISION_ELIGIBLE_POLICIES


# --------------------------------------------------------------------- comparability (M-3)


def test_common_term_subset_gives_both_methods_the_same_estimand() -> None:
    """M-3: comparisons run on terms in the same state for both plates, not each plate's own.

    The previous version of this test built plates for which BOTH common subsets were empty, so its
    loops were vacuous and the only surviving assertion compared 0 against 24. The plates below are
    chosen so each common subset is non-empty, and the test asserts that explicitly before using it.
    """
    dg = _pairwise_dg()
    terms = [tuple(sorted(v)) for v in _all_variants() if len(v) == _PAIRWISE]
    singles = [v for v in _all_variants() if len(v) == 1]
    # A buys the singles at site 0, B the singles at site 1. Then the (0,1) pairs are informed by
    # both and pinned by neither, while the (2,3) pairs are untouched by both -- so both common
    # subsets are non-empty and the comparison has something to run on.
    revealed_a = {v: dg[v] for v in singles if next(iter(v))[0] == 0}
    revealed_b = {v: dg[v] for v in singles if next(iter(v))[0] == 1}

    common = common_term_subset(terms, revealed_a, revealed_b, "common_informed_not_pinned")
    uninformed = common_term_subset(terms, revealed_a, revealed_b, "common_uninformed")
    assert common, "fixture must produce a non-empty common informed set"
    assert uninformed, "fixture must produce a non-empty common uninformed set"

    for term in common:
        loop = [frozenset({term[0]}), frozenset({term[1]}), frozenset(term)]
        assert any(m in revealed_a for m in loop) and any(m in revealed_b for m in loop)
        assert not all(m in revealed_a for m in loop) and not all(m in revealed_b for m in loop)
    for term in uninformed:
        loop = [frozenset({term[0]}), frozenset({term[1]}), frozenset(term)]
        assert not any(m in revealed_a for m in loop) and not any(m in revealed_b for m in loop)

    # The honest common set is strictly smaller than either method's own informed set.
    own_a = [
        t
        for t in terms
        if any(m in revealed_a for m in (frozenset({t[0]}), frozenset({t[1]}), frozenset(t)))
    ]
    assert 0 < len(common) < len(own_a)
    # The three classes are disjoint and never overlap.
    assert not set(common) & set(uninformed)


def test_term_sha256_identifies_the_set_and_ignores_ordering() -> None:
    terms = [tuple(sorted(v)) for v in _all_variants() if len(v) == _PAIRWISE]
    assert term_sha256(terms) == term_sha256(list(reversed(terms)))
    assert term_sha256(terms) != term_sha256(terms[:-1])


def test_term_census_is_exhaustive_and_disjoint() -> None:
    dg = _pairwise_dg()
    terms = [tuple(sorted(v)) for v in _all_variants() if len(v) == _PAIRWISE]
    truth = {t: contrast(dg, t) for t in terms}
    esm = dict.fromkeys(dg, 0.0)
    revealed = {v: dg[v] for v in [t for t in _all_variants() if len(t) == 1][:3]}
    prior, _ = prior_mu(esm, revealed, "zero_prior")
    result = evaluate_order(terms, truth, dg, prior, revealed, "pairwise")
    result.census.check()  # must not raise
    assert (
        result.census.n_pinned + result.census.n_informed_not_pinned + result.census.n_uninformed
        == len(terms)
    )


# --------------------------------------------------------------------- paired inference (H-4)


def test_paired_difference_uses_the_same_resampled_terms_for_both_methods() -> None:
    """H-4: a paired difference CI, not the non-overlap of two marginal intervals."""
    rng = np.random.default_rng(5)
    truth = rng.normal(size=300)
    pred_a = truth + 0.5 * rng.normal(size=300)
    pred_b = truth + 2.0 * rng.normal(size=300)
    delta, ci = paired_difference_ci(pred_a, pred_b, truth, "pearson", seed=1, n_bootstrap=300)
    assert delta is not None and ci is not None
    expected = pearsonr(pred_a, truth).statistic - pearsonr(pred_b, truth).statistic
    assert delta == pytest.approx(expected)
    assert ci[0] < delta < ci[1]
    assert ci[0] > 0.0  # A really is better here, and the paired interval says so

    # Two identical methods must give a delta of exactly zero with a zero-width interval.
    same_delta, same_ci = paired_difference_ci(
        pred_a, pred_a, truth, "pearson", seed=1, n_bootstrap=100
    )
    assert same_delta == pytest.approx(0.0)
    assert same_ci is not None and same_ci[0] == pytest.approx(0.0, abs=1e-12)


def test_fisher_z_mean_differs_from_the_arithmetic_mean_and_is_exact_on_a_hand_case() -> None:
    """L-2: correlations average on the z scale, not the r scale."""
    values = [0.2, 0.9, 0.95]
    expected = float(np.tanh(np.mean(np.arctanh(values))))
    assert fisher_z_mean(values) == pytest.approx(expected)
    assert fisher_z_mean(values) != pytest.approx(float(np.mean(values)))
    assert fisher_z_mean([0.5]) == pytest.approx(0.5)
    assert fisher_z_mean([]) is None
    # A perfect correlation must not send the mean to infinity.
    assert fisher_z_mean([1.0, 0.0]) is not None
    assert np.isfinite(fisher_z_mean([1.0, 0.0]) or np.inf)


# --------------------------------------------------------------------- schema separation


def test_schema_version_is_bumped_so_old_readers_cannot_misread_new_fields() -> None:
    assert SCHEMA_VERSION >= _THIRD
    # The raw correlation is only reachable under a name that says what it contains.
    dg = _pairwise_dg()
    terms = [tuple(sorted(v)) for v in _all_variants() if len(v) == _PAIRWISE]
    truth = {t: contrast(dg, t) for t in terms}
    esm = dict.fromkeys(dg, 0.0)
    revealed = {v: dg[v] for v in [t for t in _all_variants() if len(t) == 1]}
    prior, _ = prior_mu(esm, revealed, "zero_prior")
    result = evaluate_order(terms, truth, dg, prior, revealed, "pairwise")
    fields = set(result.model_dump())
    assert "pearson" not in fields and "spearman" not in fields
    assert "raw_pearson_with_skeleton" in fields
    assert "relative_sse_gain" in fields and "recovery_wording_permitted" in fields


def test_the_withdrawn_held_out_estimand_is_absent_and_its_census_is_kept() -> None:
    """The held-out contrast estimand is withdrawn; no field may offer it, and why is recorded.

    The withdrawal (prospective-amendment-2 S4.1) rests on two facts, both asserted here rather
    than merely asserted in prose. A schema that still carried ``held_out_contrast_*`` would report
    ``null`` for every plate, which reads as "measured, found nothing" instead of "withdrawn as
    degenerate" -- the opposite of what the audit established.
    """
    dg = _pairwise_dg()
    terms = [tuple(sorted(v)) for v in _all_variants() if len(v) == _PAIRWISE]
    truth = {t: contrast(dg, t) for t in terms}
    esm = dict.fromkeys(dg, 0.0)
    # Measure three triples only: no pairwise loop member is touched, so every k(S) is 0.
    triples = [v for v in _all_variants() if len(v) == _THIRD]
    revealed = {v: dg[v] for v in triples[:_THIRD]}
    prior, _ = prior_mu(esm, revealed, "zero_prior")
    result = evaluate_order(terms, truth, dg, prior, revealed, "pairwise")

    fields = set(result.model_dump())
    assert not [f for f in fields if "held_out" in f], (
        "a withdrawn estimand must not survive as a field that reports null for every plate"
    )

    # (a) The census of the population it would have scored is retained; it IS the evidence.
    assert result.census.n_uninformed == len(terms)
    for term in terms:
        assert measured_skeleton(term, frozenset(revealed), dg) == pytest.approx(0.0, abs=_TOL)

    # (b) Degeneracy: with no loop member measured and a zero prior, every predicted contrast is
    # the same constant, so no correlation exists over that population -- there is nothing to score.
    post = dict(prior)
    post.update(revealed)
    predicted = np.array([contrast(post, t) for t in terms], dtype=np.float64)
    observed = np.array([truth[t] for t in terms], dtype=np.float64)
    assert is_effectively_constant(predicted)
    assert safe_corr(predicted, observed, "spearman") is None
    assert not is_effectively_constant(
        observed
    )  # the truth genuinely varies; only the guess does not


# --------------------------------------------------------------------- identifiability


def _dense_landscape(space_sites: tuple[int, ...]) -> tuple[dict[Variant, float], dict, dict]:
    """A 4-site/3-letter landscape with known main effects and known per-residue-pair couplings."""
    rng = np.random.default_rng(0)
    alphabet = "CD"
    main = {(s, a): float(rng.normal()) for s in space_sites for a in alphabet}
    pair = {
        ((i, a), (j, b)): float(rng.normal() * 2.0)
        for i in space_sites
        for j in space_sites
        if i < j
        for a in alphabet
        for b in alphabet
    }
    dg: dict[Variant, float] = {frozenset(): 0.0}
    for v in _all_variants():
        muts = sorted(v)
        total = sum(main[(p, m)] for p, _w, m in muts)
        for (pi, _wi, mi), (pj, _wj, mj) in combinations(muts, 2):
            total += pair[((pi, mi), (pj, mj))]
        dg[v] = total
    return dg, main, pair


def test_a_contrast_over_an_unassayed_residue_pair_is_exactly_unidentifiable() -> None:
    """The decisive identifiability result behind the withdrawn map-recovery claim.

    In the reference-coded main-effects-plus-pairwise basis, the column for a specific residue pair
    is active only in variants that contain exactly that pair. If no training row does, the centered
    column is identically zero, so the fitted coefficient is exactly 0 — not merely poorly
    estimated. The predicted contrast then collapses to the same constant for every such term:

        eps_hat(ij) = (c + m_i + m_j + 0) - (c + m_i) - (c + m_j) = -c

    So no ranking signal can exist over unassayed pairs, whatever the acquisition strategy. Any
    apparent "recovery" over them comes from the measured skeleton k(S) or from the prior, never
    from learning. This is why restricting the estimand to entirely-unmeasured loops does not
    rescue it: that restriction makes the prediction constant by construction.
    """
    sites, wt, alphabet = _SITES, _WT, "CD"
    space = FeatureSpace(sites, wt, alphabet)
    dg, _main, _pair = _dense_landscape(sites)
    universe = enumerate_candidates(sites, wt, alphabet, 3)

    # Plate: every single, plus the doubles at site pair (0, 1) only.
    singles = [v for v in universe if len(v) == 1]
    doubles = [v for v in universe if len(v) == _PAIRWISE]
    assayed_site_pair = (0, 1)
    train = singles + [
        v for v in doubles if tuple(sorted(p for p, _w, _m in v)) == assayed_site_pair
    ]
    response = np.array([dg[v] for v in train], dtype=np.float64)
    model = fit_ridge(space.design_matrix(train), response, space.penalties(1e-6, 1e-6))

    def predict(v: Variant) -> float:
        return model.intercept + float(model.coef[list(space.active_columns(v))].sum())

    def predicted_contrast(term: tuple[Mutation, ...]) -> float:
        i, j = term
        return predict(frozenset(term)) - predict(frozenset({i})) - predict(frozenset({j}))

    seen = [
        tuple(sorted(v))
        for v in doubles
        if tuple(sorted(p for p, _w, _m in v)) == assayed_site_pair
    ]
    unseen = [
        tuple(sorted(v))
        for v in doubles
        if tuple(sorted(p for p, _w, _m in v)) != assayed_site_pair
    ]
    assert seen and unseen

    # Assayed residue pairs: the contrast is recovered essentially exactly.
    seen_pred = np.array([predicted_contrast(t) for t in seen])
    seen_true = np.array([contrast(dg, t) for t in seen])
    assert seen_pred == pytest.approx(seen_true, abs=1e-4)

    # Unassayed residue pairs: the prediction is the SAME CONSTANT for every term.
    unseen_pred = np.array([predicted_contrast(t) for t in unseen])
    unseen_true = np.array([contrast(dg, t) for t in unseen])
    assert float(unseen_pred.std()) < _NOISE_TOL
    assert unseen_pred == pytest.approx(np.full(len(unseen), -model.intercept), abs=_NOISE_TOL)
    assert float(unseen_true.std()) > _REAL_SPREAD  # the truth varies; the model cannot see it
    # ...so no correlation is even defined over them.
    assert is_effectively_constant(unseen_pred)
    assert not is_effectively_constant(unseen_true)


def test_a_near_constant_vector_yields_no_correlation_instead_of_rounding_noise() -> None:
    """An exact ``std == 0`` guard manufactures a correlation out of floating-point noise.

    Discovered while writing the identifiability test above: the unidentifiable contrasts are
    algebraically identical but differ in their last bits, and ranking those bits produced a
    spurious Spearman of +0.33 over 20 terms. Since unidentifiable terms are the large majority of
    the term universe, that noise would be reported as signal.
    """
    constant_with_noise = np.full(20, -0.65984618) + np.arange(20) * 1e-17
    varying = np.linspace(-4.0, 2.0, 20)
    assert float(np.std(constant_with_noise)) > 0.0  # the old guard would let this through
    # ...and scipy would report this from pure noise:
    assert abs(float(spearmanr(constant_with_noise, varying).statistic)) > _SPURIOUS_CORR
    assert is_effectively_constant(constant_with_noise)
    assert _corr_via_report(constant_with_noise, varying) is None
    # A genuinely varying vector is untouched.
    assert not is_effectively_constant(varying)
    assert _corr_via_report(varying, np.linspace(0.0, 1.0, 20)) == pytest.approx(1.0)


def _corr_via_report(a: FloatArray, b: FloatArray) -> float | None:
    return safe_corr(a, b, "spearman")


# --------------------------------------------------------------------- emitter integration


_SHA256_HEX_LEN = 64


def test_emitter_writes_a_file_that_validates_against_its_own_schema(  # noqa: PLR0915
    tmp_path: Path,
) -> None:
    """The integration debt this closes: the emitted JSON must satisfy CorrectedRecoveryReport.

    A schema no emitter constructs enforces nothing. This runs the real script end to end over a
    tiny synthetic landscape and parses the result back through the model, so a field the emitter
    forgets -- or invents -- fails here rather than after a three-hour run.
    """
    repo = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "src")
    driver = repo / "tests" / "_corrected_recovery_driver.py"

    proc = subprocess.run(
        [sys.executable, str(driver), str(repo), str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo),
    )
    assert proc.returncode == 0, f"emitter failed:\n{proc.stdout}\n{proc.stderr}"

    written = tmp_path / "out" / "corrected_recovery_synthetic_tiny.json"
    report = CorrectedRecoveryReport.model_validate_json(written.read_text(encoding="utf-8"))

    assert report.schema_version == SCHEMA_VERSION
    assert report.model_id == "synthetic/test-scorer"
    assert not report.decision_eligible and report.reason_not_decision_eligible
    assert report.provenance["argv"][0].endswith("corrected_recovery.py")
    assert report.provenance["exact_command"]
    assert report.provenance["input_hashes_at_start"] == report.provenance["input_hashes_at_end"]
    assert report.provenance["workspace_stable"] is True

    # Every declared method is present, and `random`/`practice` are among them -- their absence was
    # the other half of the integration debt.
    assert {"random", "practice"}.issubset({m.method for m in report.methods})
    assert set(report.methods_evaluated) == {m.method for m in report.methods}

    for method in report.methods:
        assert method.seed_kind in SEED_KINDS
        assert (method.seed is None) == (method.seed_kind == "none")
        assert len(method.selected_identity_sha256) == _SHA256_HEX_LEN
        assert method.calibration.policy in CALIBRATION_POLICIES
        for order in method.orders:
            order.census.check()  # exhaustive: the three classes sum to n_terms
            assert order.census.n_terms == order.n_terms

    # Seeded methods are serialized one realisation at a time. No seed-0 record may stand in for
    # the distribution, and every declared seed must be present for every emitted cell.
    structural = [m for m in report.methods if m.method == "structural"]
    random = [m for m in report.methods if m.method == "random"]
    for budget in report.budgets:
        for policy in CALIBRATION_POLICIES:
            structural_seeds = {
                m.seed for m in structural if m.budget == budget and m.calibration.policy == policy
            }
            assert structural_seeds in ({None}, set(range(report.tie_seeds)))
            assert {
                m.seed for m in random if m.budget == budget and m.calibration.policy == policy
            } == set(range(report.random_seeds))

    # Paired contrasts exist, run only under the label-free policies, and each records the hash of
    # the exact term set it was computed on (audit M-3).
    assert report.paired_contrasts
    for result in report.paired_contrasts:
        assert result.calibration_policy in DECISION_ELIGIBLE_POLICIES
        assert result.term_subset in TERM_SUBSETS
        assert len(result.term_sha256) == _SHA256_HEX_LEN
        assert len(result.selected_identity_sha256_a) == _SHA256_HEX_LEN
        assert len(result.selected_identity_sha256_b) == _SHA256_HEX_LEN
        assert result.seed_kind_a in SEED_KINDS
        assert result.seed_kind_b in SEED_KINDS
        # A missing difference always says why, and never counts as evidence of a difference.
        assert (result.delta is None) == bool(result.reason)
        if result.delta is None:
            assert not result.term_leverage_excludes_zero

    # Different stochastic mechanisms have no natural index-wise pairing. The registered recovery
    # comparison therefore emits their Cartesian product instead of silently reusing random seed
    # 0 against structural seeds 0, 3, 6, ... via modulo arithmetic.
    structural_random = [
        result
        for result in report.paired_contrasts
        if (result.method_a, result.method_b) == ("structural", "random")
    ]
    structural_draws_by_budget = {
        budget: len(
            {
                (m.seed_kind, m.seed)
                for m in structural
                if m.budget == budget and m.calibration.policy == CALIBRATION_POLICIES[0]
            }
        )
        for budget in report.budgets
    }
    expected = sum(structural_draws_by_budget.values()) * (
        len(report.paired_contrast_policies)
        * 2  # pairwise + third order
        * len(TERM_SUBSETS)
        * report.random_seeds
    )
    assert len(structural_random) == expected

    # Two tie-dominated methods use the same declared tie seed, so their draw identity is paired
    # rather than crossed. This is the only seeded pairing with a shared stochastic mechanism.
    structural_singles = [
        result
        for result in report.paired_contrasts
        if (result.method_a, result.method_b) == ("structural", "singles_zero_prior")
    ]
    assert structural_singles
    assert all(result.seed_a == result.seed_b for result in structural_singles)

    # Publication is create-only: rerunning into the same directory must fail without changing the
    # first validated artifact.
    original_bytes = written.read_bytes()
    second = subprocess.run(
        [sys.executable, str(driver), str(repo), str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo),
    )
    assert second.returncode != 0
    assert written.read_bytes() == original_bytes


def test_a_paired_contrast_on_a_common_subset_is_smaller_than_the_confounded_one() -> None:
    """M-3 in one assertion: the honest comparison is not the all-eligible one.

    On `all_eligible` both arms carry a purchased skeleton, and the arm that bought more of it
    scores higher for that reason alone. Restricting to terms in the same state for both plates
    removes that shared component, and the apparent gap shrinks.
    """
    dg = _pairwise_dg()
    terms = [tuple(sorted(v)) for v in _all_variants() if len(v) == _PAIRWISE]
    truth_by_term = {t: contrast(dg, t) for t in terms}
    singles = [v for v in _all_variants() if len(v) == 1]
    revealed_a = {v: dg[v] for v in singles if next(iter(v))[0] in (0, 1)}
    revealed_b = {v: dg[v] for v in singles if next(iter(v))[0] == 1}

    common = common_term_subset(terms, revealed_a, revealed_b, "common_informed_not_pinned")
    assert common, "fixture must leave a non-empty common set"
    assert len(common) < len(terms)

    def post_eps(revealed: dict[Variant, float], subset: list[Term]) -> FloatArray:
        prior, _record = prior_mu(dict.fromkeys(dg, 0.0), revealed, "zero_prior")
        post = dict(prior)
        post.update(revealed)
        return np.array([contrast(post, t) for t in subset], dtype=np.float64)

    deltas: dict[str, float] = {}
    for label, subset in (("all_eligible", terms), ("common", common)):
        truth = np.array([truth_by_term[t] for t in subset], dtype=np.float64)
        delta, _ci = paired_difference_ci(
            post_eps(revealed_a, subset),
            post_eps(revealed_b, subset),
            truth,
            "spearman",
            seed=0,
            n_bootstrap=200,
        )
        assert delta is not None
        deltas[label] = delta

    assert abs(deltas["common"]) < abs(deltas["all_eligible"])


def test_emitter_rejects_a_cache_from_the_wrong_scorer(tmp_path: Path) -> None:
    """The emitter must not read a cache it cannot prove is the one it claims to read.

    Parsing the sidecar alone can never fail -- it compares the sidecar against itself. The expected
    identity has to come from the caller, so a cache produced by a different model (or WT, alphabet,
    perturbation count, or candidate set) is refused rather than silently reanalysed.
    """
    repo = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "src")
    env["EPIBUDGET_DRIVER_MODEL_ID"] = "facebook/esm2_t33_650M_UR50D"  # not what the cache carries
    driver = repo / "tests" / "_corrected_recovery_driver.py"

    proc = subprocess.run(
        [sys.executable, str(driver), str(repo), str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo),
    )
    assert proc.returncode != 0, "a wrong-model cache must not be accepted"
    assert "model_id" in (proc.stdout + proc.stderr)
    assert not (tmp_path / "out").exists(), "nothing may be written when the cache is rejected"


@pytest.mark.parametrize("flag", ["--tie-seeds", "--random-seeds", "--bootstrap"])
def test_emitter_rejects_non_positive_sampling_counts(tmp_path: Path, flag: str) -> None:
    script = Path(__file__).resolve().parent.parent / "scripts" / "corrected_recovery.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset",
            "gb1_wu2016",
            "--scored-cache",
            str(tmp_path / "missing.jsonl"),
            flag,
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert f"{flag} must be >= 1" in (proc.stdout + proc.stderr)
