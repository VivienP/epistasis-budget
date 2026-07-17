"""Offline tests for the Gate-3 correlated-error inference probe (src/epibudget/gate3.py).

All tests are synthetic and offline: no score cache, no GB1 fetch, no GPU. They pin the pure-math
invariants (ridge BLUP == Gaussian conditioning; the exact single-measured additive formula; λ=∞
reproduces the pin baseline) and reproduce, on a tiny 4-site universe, the Gate-2 phenomenon that
pinning doubles the squared error while a correlated prior recovers calibration.
"""

from __future__ import annotations

from math import exp

import numpy as np
import pytest

from epibudget.data import enumerate_candidates
from epibudget.gate3 import (
    _effect_index,
    _incidence,
    _partial_pearson,
    _partial_spearman,
    _residualize,
    _ridge_blup,
    _sub_effects,
    run_gate3,
)
from epibudget.types import ScoredVariant, Variant

_SITES = (0, 1, 2, 3)
_WT = ("A", "A", "A", "A")
_ALPHABET = "ACD"  # WT 'A' + two non-WT residues => 8 singles, 24 pairs, 32 triples
_PAIR_ORDER = 2
_TRIPLE_ORDER = 3
_N_SINGLES = 8
_HALF = 0.5
_TOL = 1e-9
_HIGH_CORR = 0.8
_LOW_CORR = 0.1


def _mutation_effect(variant: Variant, weights: dict[tuple[int, str], float]) -> float:
    return sum(weights[(site, aa)] for site, _wt, aa in variant)


def _build_universe(
    seed: int, additive_error_scale: float, residual_scale: float
) -> tuple[list[ScoredVariant], dict[Variant, float]]:
    """A 4-site landscape with genuine epistasis whose ESM error is dominantly additive."""
    rng = np.random.default_rng(seed)
    variants = enumerate_candidates(_SITES, _WT, allowed_aa=_ALPHABET, max_order=3)
    mutations = sorted({m for variant in variants for m in variant})
    main = {(s, a): float(rng.normal(0.0, 1.0)) for s, _w, a in mutations}
    additive_err = {(s, a): float(rng.normal(0.0, additive_error_scale)) for s, _w, a in mutations}

    landscape: dict[Variant, float] = {frozenset(): 1.0}
    scored: list[ScoredVariant] = []
    for variant in variants:
        # True ΔG = additive main effects + real (non-additive) pairwise interaction.
        epistasis = 0.6 * float(rng.normal(0.0, 1.0)) if len(variant) >= _PAIR_ORDER else 0.0
        true_dg = _mutation_effect(variant, main) + epistasis
        landscape[variant] = exp(true_dg)
        # ESM prior error e = esm - true is additive plus a small residual (slope b ~ 1).
        esm = (
            true_dg
            + _mutation_effect(variant, additive_err)
            + float(rng.normal(0.0, residual_scale))
        )
        scored.append(ScoredVariant(variant=variant, delta_g=esm, var_delta_g=1.0))
    return scored, landscape


def test_sub_effects_and_incidence() -> None:
    variants = enumerate_candidates(_SITES, _WT, allowed_aa=_ALPHABET, max_order=3)
    triple = next(v for v in variants if len(v) == _TRIPLE_ORDER)
    assert len(_sub_effects(triple, 1)) == _TRIPLE_ORDER  # three singles
    assert len(_sub_effects(triple, 2)) == _TRIPLE_ORDER + _TRIPLE_ORDER  # + three pairs
    singles = [v for v in variants if len(v) == 1]
    index = _effect_index(singles, 1)
    assert len(index) == _N_SINGLES
    design = _incidence(singles, index, 1)
    assert design.shape == (_N_SINGLES, _N_SINGLES)
    assert np.allclose(design, np.eye(_N_SINGLES))  # each single hits its own effect column


def test_ridge_blup_equals_gaussian_conditioning() -> None:
    rng = np.random.default_rng(0)
    n, p, n_unmeasured = 12, 5, 7
    design_m = (rng.random((n, p)) < _HALF).astype(np.float64)
    error_m = rng.normal(size=n)
    design_u = (rng.random((n_unmeasured, p)) < _HALF).astype(np.float64)
    tau2, sigma2 = 0.7, 0.3
    lam = sigma2 / tau2

    a_hat = _ridge_blup(design_m, error_m, lam)
    pred_ridge = design_u @ a_hat

    cov_mm = tau2 * design_m @ design_m.T + sigma2 * np.eye(n)
    cov_um = tau2 * design_u @ design_m.T
    pred_gauss = cov_um @ np.linalg.solve(cov_mm, error_m)
    assert np.allclose(pred_ridge, pred_gauss, atol=1e-10)


def test_single_measured_pair_correction_matches_additive_formula() -> None:
    # Two measured singles i, j; unmeasured pair {i,j}. With the single basis GᵀG = I, so
    # â = e/(1+λ) and the predicted pair error is (e_i + e_j)/(1+λ).
    variants = enumerate_candidates(_SITES, _WT, allowed_aa=_ALPHABET, max_order=3)
    singles = sorted((v for v in variants if len(v) == 1), key=sorted)
    i, j = singles[0], singles[1]
    index = _effect_index([i, j], 1)
    design = _incidence([i, j], index, 1)
    e = np.array([1.3, -0.7], dtype=np.float64)
    lam = 2.0
    a_hat = _ridge_blup(design, e, lam)
    pair = i | j
    pred = _incidence([pair], index, 1) @ a_hat
    assert pred[0] == pytest.approx((1.3 + -0.7) / (1.0 + lam))


def test_lambda_inf_frontier_point_equals_pin() -> None:
    scored, landscape = _build_universe(seed=1, additive_error_scale=1.5, residual_scale=0.05)
    result = run_gate3(
        scored,
        landscape,
        budgets=(6,),
        bases=("single",),
        n_folds=4,
        bootstrap_iterations=200,
    )
    cell = next(c for c in result.results[0].cells if c.order == "pairwise")
    inf_point = next(point for point in cell.frontier if not np.isfinite(point[0]))
    assert inf_point[1] == pytest.approx(cell.pin_sse_gain)  # sse_gain at λ=∞ is the pin baseline


def test_residualize_removes_linear_control() -> None:
    rng = np.random.default_rng(0)
    n = 500
    k = rng.normal(size=n)
    y = 3.0 + 2.0 * k + rng.normal(size=n)
    resid = _residualize(y, k)
    assert abs(float(np.corrcoef(resid, k)[0, 1])) < _TOL  # residual orthogonal to the control


def test_partial_corr_strips_a_purely_shared_control() -> None:
    # pred and truth share ONLY the large control k -> raw corr high, partial corr ~ 0.
    rng = np.random.default_rng(1)
    n = 3000
    k = rng.normal(scale=5.0, size=n)
    pred = k + rng.normal(size=n)
    truth = k + rng.normal(size=n)
    raw = float(np.corrcoef(pred, truth)[0, 1])
    partial_p = _partial_pearson(pred, truth, k)
    partial_s = _partial_spearman(pred, truth, k)
    assert raw > _HIGH_CORR  # the shared control inflates the raw correlation
    assert partial_p is not None and abs(partial_p) < _LOW_CORR  # ...and residualization removes it
    assert partial_s is not None and abs(partial_s) < _LOW_CORR


def test_partial_corr_preserves_signal_beyond_control() -> None:
    # pred and truth share k AND a genuine signal s independent of k -> partial corr detects s.
    rng = np.random.default_rng(2)
    n = 3000
    k = rng.normal(scale=5.0, size=n)
    s = rng.normal(size=n)
    pred = k + s + 0.3 * rng.normal(size=n)
    truth = k + s + 0.3 * rng.normal(size=n)
    partial_p = _partial_pearson(pred, truth, k)
    assert partial_p is not None and partial_p > _HALF  # genuine signal survives residualization


def test_residualized_fields_are_populated_on_a_real_small_run() -> None:
    scored, landscape = _build_universe(seed=3, additive_error_scale=1.0, residual_scale=0.1)
    result = run_gate3(
        scored, landscape, budgets=(6,), bases=("single",), n_folds=4, bootstrap_iterations=200
    )
    cell = next(c for c in result.results[0].cells if c.order == "pairwise")
    assert cell.residual_corr_delta_spearman is not None
    assert cell.residual_corr_delta_spearman_ci95 is not None
    assert cell.skeleton_confound_spearman is not None


def test_correlated_prior_beats_pin_on_additive_error_landscape() -> None:
    # Dominantly additive ESM error: pinning measured singles into pair loops breaks the error
    # cancellation (SSE up), and the additive correlated prior restores calibration.
    scored, landscape = _build_universe(seed=7, additive_error_scale=2.0, residual_scale=0.05)
    result = run_gate3(
        scored,
        landscape,
        budgets=(6,),
        bases=("single",),
        n_folds=4,
        bootstrap_iterations=200,
    )
    cell = next(c for c in result.results[0].cells if c.order == "pairwise")
    assert cell.pin_sse_gain is not None and cell.pin_sse_gain < 0.0  # doubling reproduces
    best_gain = max(
        point[1] for point in cell.frontier if point[1] is not None and np.isfinite(point[0])
    )
    assert best_gain > cell.pin_sse_gain  # a correlated λ improves calibration over pin
