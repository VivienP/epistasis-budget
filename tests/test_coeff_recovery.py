"""Offline tests for the Step-6A compressed-sensing recovery module (coeff_recovery.py).

All synthetic, no cache/GPU/network. They pin the load-bearing numerics: the coordinate-descent
LASSO (== soft-thresholded OLS on an orthonormal design; sparse-support recovery), the closed-form
restricted-order Fourier kernel (== the explicit character Gram matrix), the multiallelic Fourier
design (exact round-trip + Parseval vs ``wht_spectrum``), and that Fourier-LASSO extrapolates to
unmeasured variants when the landscape is sparse but not when it is dense.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from epibudget.coeff_recovery import (
    _build_fourier_config,
    _cd_lasso_path,
    _design_matrix,
    _doptimal_order,
    _kernel_cross,
    _order_symmetric_kernel,
    _reconstruct,
    _site_indices,
    _soft_threshold,
)
from epibudget.epistasis import wht_spectrum
from epibudget.types import Variant

_TOL = 1e-6
_TIGHT = 1e-9
_HIGH_CORR = 0.9
_MAX_SPARSE_SUPPORT = 12  # a 6-sparse recovery must not blow up the support
_Q3 = "ACD"  # 3-letter alphabet: WT 'A' + two contrasts


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def test_soft_threshold() -> None:
    assert _soft_threshold(2.0, 0.5) == 2.0 - 0.5
    assert _soft_threshold(-2.0, 0.5) == -(2.0 - 0.5)
    assert _soft_threshold(0.3, 0.5) == 0.0


def test_lasso_equals_soft_thresholded_ols_on_orthonormal_design() -> None:
    rng = np.random.default_rng(0)
    n, k = 60, 8
    design, _ = np.linalg.qr(rng.normal(size=(n, k)))  # orthonormal columns, ‖col‖²=1
    y = rng.normal(size=n)
    lam = 1.0
    beta = _cd_lasso_path(design, y, [lam])[0]
    expected = np.array(
        [_soft_threshold(float(design[:, j] @ y), lam / 2.0) for j in range(k)], dtype=np.float64
    )
    assert np.allclose(beta, expected, atol=_TOL)


def test_lasso_recovers_a_sparse_support() -> None:
    rng = np.random.default_rng(1)
    n, p, k_true = 80, 120, 5
    design = rng.normal(size=(n, p))
    beta_true = np.zeros(p, dtype=np.float64)
    support = rng.choice(p, size=k_true, replace=False)
    beta_true[support] = rng.normal(size=k_true) * 3.0
    y = design @ beta_true  # noiseless
    design_c = design - design.mean(axis=0)
    y_c = y - y.mean()
    path = [4.0 * ratio for ratio in np.geomspace(1.0, 1e-3, 20)]
    beta_hat = _cd_lasso_path(design_c, y_c, path)[-1]
    assert _corr(beta_hat, beta_true) > _HIGH_CORR
    top = set(np.argsort(np.abs(beta_hat))[-k_true:].tolist())
    assert set(support.tolist()) == top  # exact support recovered at small λ


def test_closed_form_kernel_equals_explicit_character_gram() -> None:
    # 3 sites, q=3 ⇒ the closed-form kernel covers orders 1..(n-1)=1..2; build that design.
    sites, wt = (0, 1, 2), ("A", "A", "A")
    config = _build_fourier_config(sites, wt, _Q3, max_order=2)
    genotypes = _all_genotypes(sites, wt, _Q3)
    site_idx = _site_indices(config, genotypes)
    design = _design_matrix(config, site_idx)  # orders 1..2 characters
    explicit = design @ design.T
    formula = _kernel_cross(site_idx, site_idx, config.q)
    assert np.allclose(explicit, formula, atol=_TIGHT)


def test_fourier_design_round_trips_and_matches_wht_spectrum() -> None:
    rng = np.random.default_rng(2)
    sites, wt = (0, 1, 2), ("A", "A", "A")
    genotypes = _all_genotypes(sites, wt, _Q3)  # complete landscape (27 genotypes)
    dg_values = rng.normal(size=len(genotypes))
    config = _build_fourier_config(sites, wt, _Q3, max_order=3)  # all non-constant orders
    site_idx = _site_indices(config, genotypes)
    design = _design_matrix(config, site_idx)  # 27 x 26, orthonormal over the full landscape
    y_c = dg_values - dg_values.mean()
    beta = np.linalg.lstsq(design, y_c, rcond=None)[0]
    recon = design @ beta + dg_values.mean()
    assert np.allclose(recon, dg_values, atol=_TOL)  # exact round-trip
    assert np.isclose(float(np.sum(beta**2)), float(np.sum(y_c**2)), atol=_TOL)  # Parseval
    spectrum = wht_spectrum({g: float(dg_values[i]) for i, g in enumerate(genotypes)}, sites)
    assert np.isclose(sum(spectrum.values()) * len(genotypes), float(np.sum(beta**2)), atol=_TOL)


def test_fourier_lasso_reconstruction_extrapolates_a_sparse_spectrum() -> None:
    # The compressed-sensing property (design → LASSO fit → reconstruct extrapolates to UNMEASURED
    # variants), isolated from the data-dependent CV λ-selection that ``_fourier_lasso_fit`` wraps
    # (the cache run exercises that). On a well-posed orthonormal design (60 of a complete 4-site
    # p=32 order-1..2 landscape), a sparse spectrum is recovered and extrapolated to the held-out.
    sites, wt = (0, 1, 2, 3), ("A", "A", "A", "A")
    config = _build_fourier_config(sites, wt, _Q3, max_order=2)
    genotypes = _all_genotypes(sites, wt, _Q3)  # complete landscape, 81 genotypes
    design_all = _design_matrix(config, _site_indices(config, genotypes))
    p = design_all.shape[1]
    rng = np.random.default_rng(3)
    order = rng.permutation(len(genotypes)).tolist()
    measured = [genotypes[i] for i in sorted(order[:60])]
    needed = [genotypes[i] for i in sorted(order[60:])]
    beta_true = np.zeros(p, dtype=np.float64)
    beta_true[rng.choice(p, size=6, replace=False)] = rng.normal(size=6) * 3.0
    dg = design_all @ beta_true
    dg_by_v = {v: float(dg[i]) for i, v in enumerate(genotypes)}
    y_m = np.array([dg_by_v[v] for v in measured], dtype=np.float64)

    design_m = _design_matrix(config, _site_indices(config, measured))
    col_mean, col_std = design_m.mean(axis=0), design_m.std(axis=0)
    scale = np.where(col_std > 0.0, col_std, 1.0)
    design_std = np.where(col_std > 0.0, (design_m - col_mean) / scale, 0.0)
    beta_std = _cd_lasso_path(design_std, y_m - y_m.mean(), [1e-2])[-1]  # fixed small λ
    raw_beta = beta_std / scale
    support = np.flatnonzero(raw_beta).tolist()
    intercept = float(y_m.mean()) - float(raw_beta @ col_mean)
    pred = _reconstruct(config, _site_indices(config, needed), support, raw_beta, intercept)
    true = np.array([dg_by_v[v] for v in needed], dtype=np.float64)
    assert _corr(pred, true) > _HIGH_CORR  # sparse spectrum recovered and extrapolated to held-out
    assert len(support) <= _MAX_SPARSE_SUPPORT  # stays sparse


def test_doptimal_order_is_distinct_prefix_consistent_and_label_free() -> None:
    # Label-free by signature (no landscape/labels), returns distinct variants, and is
    # prefix-consistent so the budget-B design is the first B of a longer greedy run (Step 6B).
    sites, wt = (0, 1, 2, 3), ("A", "A", "A", "A")
    config = _build_fourier_config(sites, wt, _Q3, max_order=3)
    candidates = [g for g in _all_genotypes(sites, wt, _Q3) if g]  # non-WT genotypes
    count, prefix = 20, 8
    long = _doptimal_order(config, candidates, count)
    assert len(long) == count
    assert len(set(long)) == count  # distinct
    assert _doptimal_order(config, candidates, prefix) == long[:prefix]  # prefix-consistent


def test_order_symmetric_kernel_decomposes_and_matches_kernel_cross() -> None:
    # The order-restricted kernel is additive over orders and its full range (1..n-1) reproduces
    # the closed-form ``_kernel_cross`` to machine precision (Step 6C's kernel generalises 6B's).
    sites, wt = (0, 1, 2, 3), ("A", "A", "A", "A")  # n=4 ⇒ _kernel_cross covers orders 1..3
    config = _build_fourier_config(sites, wt, _Q3, max_order=3)
    genotypes = _all_genotypes(sites, wt, _Q3)
    idx = _site_indices(config, genotypes)
    pair = _order_symmetric_kernel(idx, idx, config.q, (1, 2))
    third = _order_symmetric_kernel(idx, idx, config.q, (3,))
    full = _order_symmetric_kernel(idx, idx, config.q, (1, 2, 3))
    assert np.allclose(pair + third, full, atol=_TIGHT)  # order-additive
    assert np.allclose(full, _kernel_cross(idx, idx, config.q), atol=_TIGHT)  # == 6B's kernel
    assert not np.allclose(pair, full, atol=_TOL)  # restriction genuinely drops the order-3 block


def test_order_symmetric_kernel_matches_explicit_lower_order_gram() -> None:
    # Pin the (1,2) sub-kernel to ground truth directly (not just via the additive
    # full==_kernel_cross identity): it must equal the explicit Gram of the order-1,2 character
    # design at n=4, where an order-3 block exists and must be excluded — guarding the per-order
    # split Step 6C relies on.
    sites, wt = (0, 1, 2, 3), ("A", "A", "A", "A")
    genotypes = _all_genotypes(sites, wt, _Q3)
    config_full = _build_fourier_config(sites, wt, _Q3, max_order=3)
    idx = _site_indices(config_full, genotypes)
    config_12 = _build_fourier_config(sites, wt, _Q3, max_order=2)  # design of ONLY order-1,2 chars
    design_12 = _design_matrix(config_12, _site_indices(config_12, genotypes))
    gram_12 = design_12 @ design_12.T
    kernel_12 = _order_symmetric_kernel(idx, idx, config_full.q, (1, 2))
    assert np.allclose(gram_12, kernel_12, atol=_TIGHT)  # (1,2) sub-kernel == lower-order Gram


def test_doptimal_pairs_targets_a_different_subspace_than_isotropic() -> None:
    # The pairwise-targeted design (orders 1..2, Step 6C) stays well-formed (distinct,
    # prefix-consistent, label-free) and selects a genuinely different set than the isotropic
    # orders-1..3 design (Step 6B) — evidence the target subspace actually steers acquisition.
    sites, wt = (0, 1, 2, 3), ("A", "A", "A", "A")
    config = _build_fourier_config(sites, wt, _Q3, max_order=3)
    candidates = [g for g in _all_genotypes(sites, wt, _Q3) if g]
    count, prefix = 20, 8
    pairs = _doptimal_order(config, candidates, count, orders=(1, 2))
    assert len(pairs) == count
    assert len(set(pairs)) == count  # distinct
    assert _doptimal_order(config, candidates, prefix, orders=(1, 2)) == pairs[:prefix]  # prefix
    isotropic = _doptimal_order(config, candidates, count)
    assert set(pairs) != set(isotropic)  # pairwise targeting changes the design


def _all_genotypes(sites: tuple[int, ...], wt: tuple[str, ...], alphabet: str) -> list[Variant]:
    """Every genotype over ``sites`` (a complete landscape), as WT-diff Variants (WT = empty)."""
    genotypes: list[Variant] = []
    for residues in product(alphabet, repeat=len(sites)):
        genotypes.append(
            frozenset(
                (site, wt[i], aa)
                for i, (site, aa) in enumerate(zip(sites, residues, strict=True))
                if aa != wt[i]
            )
        )
    return genotypes
