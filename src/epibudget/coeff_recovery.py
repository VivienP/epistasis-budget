"""Step 6A — compressed-sensing (Fourier) epistasis-recovery baseline (cache-only, zero-GPU).

See docs/specs/step6-coefficient-recovery.md. Gate 3 found the ESM inclusion-exclusion pipeline
recovers GB1's epistasis map weakly. This module answers a narrower question: does a standard
compressed-sensing estimator (L1/L2 on the multiallelic Walsh-Hadamard character basis), fit
directly on the SAME frozen selections and scored with Gate 3's residualized metric, do any better?
Pure numpy, no scikit-learn. It never feeds selection (fitting reads only revealed labels via
``data.reveal_measured_fitness``, after the ESM-only/label-free selection is fixed) and its report
is ``public_claim_eligible = False`` — a diagnostic baseline, not a production replacement for
``infer_epistasis``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import combinations, product

import numpy as np
import numpy.typing as npt

from epibudget.data import (
    GB1_SITES,
    GB1_WT_AT_SITES,
    reveal_measured_fitness,
)
from epibudget.epistasis import _orthonormal_contrast_basis, interaction_loop
from epibudget.gate2 import (
    Term,
    _canonical_scored,
    _center_positive,
    _epsilon,
    _reveal_selection,
    _truth_terms,
)
from epibudget.gate3 import (
    _bootstrap_residual_delta_ci,
    _info_selection,
    _partial_spearman,
    _safe_corr,
    _skeleton,
)
from epibudget.robustness import variant_fold
from epibudget.types import ScoredVariant, Variant
from epibudget.validate import random_selection

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

logger = logging.getLogger(__name__)

AA20 = "ACDEFGHIKLMNPQRSTVWY"
_PAIRWISE_ORDER = 2
_THIRD_ORDER = 3
_ORDERS: tuple[tuple[int, str], ...] = ((_PAIRWISE_ORDER, "pairwise"), (_THIRD_ORDER, "third"))

_DEFAULT_MODEL_ID = "facebook/esm2_t33_650M_UR50D"
_DEFAULT_DATASET = "gb1_wu2016"

# Coordinate-descent LASSO: warm-started descending λ-path, active-set restricted sweeps.
_N_LAMBDA = 20
_LAMBDA_RATIO_MIN = 1e-3
_MAX_ACTIVE_SET_ROUNDS = 50
_CD_MAX_SWEEPS = 200
_CD_TOL = 1e-7

# Kernel-ridge: λ by K-fold CV over a fixed log-spaced grid.
_RIDGE_LAMBDA_GRID: tuple[float, ...] = tuple(float(value) for value in np.logspace(-3, 4, 15))

# Decision rule (operational budgets B>=96): the ESM pipeline's Gate-3 residualized recovery,
# a fixed constant reference (docs/VALIDATION.md), not re-derived here.
_OPERATIONAL_BUDGET = 96
_ESM_REFERENCE: dict[str, dict[int, float]] = {
    "pairwise": {96: 0.20, 192: 0.29},
    "third": {96: 0.0, 192: 0.0},
}
_WEAK_THRESHOLD = 0.05


# --------------------------------------------------------------------- Fourier basis / design


@dataclass(frozen=True)
class _FourierConfig:
    """Fixed per-site alphabet/basis/mode bookkeeping for the restricted-order Fourier design.

    ``modes[col]`` is the mode tuple for design column ``col`` — the single source of the
    column<->mode correspondence, shared by design construction and support reconstruction so the
    two can never drift out of alignment.
    """

    sites: tuple[int, ...]
    alphabet_index: tuple[dict[str, int], ...]  # per site: residue -> idx_s (0 = WT)
    q: int
    basis: FloatArray  # q x q orthonormal contrast basis (row 0 = constant mode), shared per site
    max_order: int
    modes: tuple[tuple[int, ...], ...]


def _build_fourier_config(
    sites: Sequence[int], wt_at_sites: Sequence[str], alphabet: str, max_order: int
) -> _FourierConfig:
    """Build the per-site alphabets (WT first), orthonormal basis, and canonical mode ordering."""
    if len(sites) != len(wt_at_sites):
        raise ValueError(
            f"sites and wt_at_sites length mismatch: {len(sites)} vs {len(wt_at_sites)}"
        )
    if not 1 <= max_order <= len(sites):
        raise ValueError(f"max_order must be in 1..{len(sites)}, got {max_order}")
    for wt in wt_at_sites:
        if wt not in alphabet:
            raise ValueError(f"alphabet {alphabet!r} is missing WT residue {wt!r}")
    alphabets = [[wt, *sorted(aa for aa in alphabet if aa != wt)] for wt in wt_at_sites]
    q = len(alphabets[0])
    alphabet_index = tuple({aa: i for i, aa in enumerate(a)} for a in alphabets)
    basis = _orthonormal_contrast_basis(q)
    modes = tuple(_full_modes(len(sites), q, max_order))
    return _FourierConfig(
        sites=tuple(sites),
        alphabet_index=alphabet_index,
        q=q,
        basis=basis,
        max_order=max_order,
        modes=modes,
    )


def _order_combos(n_sites: int, max_order: int) -> list[tuple[int, ...]]:
    """Site combinations in canonical column-block order: ascending order, then combinations()."""
    return [
        combo for order in range(1, max_order + 1) for combo in combinations(range(n_sites), order)
    ]


def _full_modes(n_sites: int, q: int, max_order: int) -> list[tuple[int, ...]]:
    """Every mode tuple of order 1..max_order, in the exact order ``_design_matrix`` builds."""
    modes: list[tuple[int, ...]] = []
    for combo in _order_combos(n_sites, max_order):
        nonzero_ranges = [range(1, q)] * len(combo)
        for assignment in product(*nonzero_ranges):
            mode = [0] * n_sites
            for site, value in zip(combo, assignment, strict=True):
                mode[site] = value
            modes.append(tuple(mode))
    return modes


def _site_indices(config: _FourierConfig, variants: Sequence[Variant]) -> IntArray:
    """idx_s(v) per site, per variant: the residue index (0 = WT) at each of ``config.sites``."""
    n_sites = len(config.sites)
    site_pos = {site: i for i, site in enumerate(config.sites)}
    out = np.zeros((len(variants), n_sites), dtype=np.int64)
    for row, variant in enumerate(variants):
        for pos, _wt_aa, mut_aa in variant:
            col = site_pos.get(pos)
            if col is not None:
                out[row, col] = config.alphabet_index[col][mut_aa]
    return out


def _batched_outer(cols: Sequence[FloatArray]) -> FloatArray:
    """Batched outer product: cols[i] is (B, k_i) -> (B, prod(k_i)), row b = kron of cols[i][b]."""
    if len(cols) == 1:
        return cols[0]
    b = cols[0].shape[0]
    letters = "cdefghijklmnopqrstuvwxyz"
    if len(cols) > len(letters):
        raise ValueError("too many sites for a batched outer product")
    subs_in = ",".join(f"b{letters[i]}" for i in range(len(cols)))
    subs_out = "b" + "".join(letters[: len(cols)])
    result = np.einsum(f"{subs_in}->{subs_out}", *cols)
    return np.asarray(result.reshape(b, -1), dtype=np.float64)


def _design_matrix(config: _FourierConfig, site_idx: IntArray) -> FloatArray:
    """The explicit character design X[v, m] = chi_m(v) over orders 1..max_order (dense, B x p)."""
    n_sites = len(config.sites)
    b = site_idx.shape[0]
    cols = [config.basis[:, site_idx[:, s]].T for s in range(n_sites)]  # cols[s]: (B, q)
    # Off-support sites carry the constant mode B_s[0, ·] (a scalar = ±1/√q; QR fixes the sign), so
    # an order-r column is B_s[0,0]^(n-r) · ∏_{s∈combo} B_s[m_s, ·]. Using the ACTUAL B_s[0,0] (not
    # +1/√q) makes the design the FULL orthonormal character χ_m with the same sign as
    # _character/_reconstruct/_kernel_cross — else LASSO fit ≠ predict for odd (n-r) (orders 1, 3).
    const = float(config.basis[0, 0])
    blocks = [
        _batched_outer([cols[s][:, 1:] for s in combo]) * (const ** (n_sites - len(combo)))
        for combo in _order_combos(n_sites, config.max_order)
    ]
    if not blocks:
        return np.zeros((b, 0), dtype=np.float64)
    return np.asarray(np.concatenate(blocks, axis=1), dtype=np.float64)


def _character(mode: tuple[int, ...], site_idx: tuple[int, ...], basis: FloatArray) -> float:
    """chi_m(v) = prod_s B_s[m_s, idx_s(v)] — full product over every site, WT-site factors too."""
    value = 1.0
    for m_s, idx_s in zip(mode, site_idx, strict=True):
        value *= float(basis[m_s, idx_s])
    return value


def _reconstruct(
    config: _FourierConfig,
    site_idx: IntArray,
    support_idx: Sequence[int],
    raw_beta: FloatArray,
    intercept: float,
) -> FloatArray:
    """Reconstruct ``intercept + sum(raw_beta[col] * chi_{modes[col]}(v))`` for every row of v."""
    n = site_idx.shape[0]
    predicted = np.full(n, intercept, dtype=np.float64)
    for col in support_idx:
        mode = config.modes[col]
        factor = np.ones(n, dtype=np.float64)
        for s, m_s in enumerate(mode):
            factor = factor * config.basis[m_s, site_idx[:, s]]
        predicted = predicted + float(raw_beta[col]) * factor
    return predicted


def _kernel_cross(u_idx: IntArray, v_idx: IntArray, q: int) -> FloatArray:
    """Closed-form restricted-order (orders 1..n_sites-1) Fourier kernel between two variant sets.

    K(u,v) = agree_all(u,v) - (1/q)^n - prod_s(agree_s(u,v) - 1/q): the full multiplicative kernel
    over ALL modes factors as prod_s agree_s; subtracting the order-0 term (1/q)^n and the top,
    order-n term prod_s(agree_s - 1/q) leaves exactly orders 1..n-1. Computed one site at a time
    (not a single (N,M,n) broadcast) to bound peak memory to O(N*M).
    """
    n_sites = u_idx.shape[1]
    agree_all = np.ones((u_idx.shape[0], v_idx.shape[0]), dtype=np.float64)
    order_top = np.ones((u_idx.shape[0], v_idx.shape[0]), dtype=np.float64)
    for s in range(n_sites):
        agree_s = (u_idx[:, s, None] == v_idx[None, :, s]).astype(np.float64)
        agree_all = agree_all * agree_s
        order_top = order_top * (agree_s - 1.0 / q)
    order0 = (1.0 / q) ** n_sites
    return np.asarray(agree_all - order0 - order_top, dtype=np.float64)


def _order_symmetric_kernel(
    u_idx: IntArray, v_idx: IntArray, q: int, orders: Sequence[int]
) -> FloatArray:
    """Closed-form Fourier kernel restricted to a set of interaction ``orders``.

    Sum over every mode whose non-constant support size is in ``orders`` of chi_m(u)·chi_m(v). The
    full kernel factors as prod_s(agree_s); grouping modes by their support T gives
    prod_{s in T}(agree_s - 1/q) · (1/q)^(n-|T|), so the order-r contribution is e_r · (1/q)^(n-r),
    where e_r is the r-th elementary symmetric polynomial of the per-site non-constant agreements
    (agree_s - 1/q). Generalises ``_kernel_cross`` (which is exactly orders 1..n-1) to any order
    subset — the basis for the pairwise-targeted (orders 1..2) acquisition of Step 6C. Built via the
    elementary-symmetric recurrence, holding n_sites+1 accumulators of shape (N, M) — O(n_sites*N*M)
    peak — without ever materialising the full mode set.
    """
    n_sites = u_idx.shape[1]
    shape = (u_idx.shape[0], v_idx.shape[0])
    esym = [np.ones(shape, dtype=np.float64)] + [
        np.zeros(shape, dtype=np.float64) for _ in range(n_sites)
    ]
    for s in range(n_sites):
        non_const = (u_idx[:, s, None] == v_idx[None, :, s]).astype(np.float64) - 1.0 / q
        for r in range(n_sites, 0, -1):
            esym[r] = esym[r] + non_const * esym[r - 1]
    out = np.zeros(shape, dtype=np.float64)
    for r in orders:
        out = out + esym[r] * (1.0 / q) ** (n_sites - r)
    return np.asarray(out, dtype=np.float64)


# --------------------------------------------------------------------- LASSO (coordinate descent)


def _soft_threshold(z: float, gamma: float) -> float:
    """sign(z)*max(|z|-gamma, 0) — the coordinate-descent proximal operator for the L1 penalty."""
    if z > gamma:
        return z - gamma
    if z < -gamma:
        return z + gamma
    return 0.0


def _cd_lasso_path(
    design: FloatArray, y: FloatArray, lambda_path: Sequence[float]
) -> list[FloatArray]:
    """Warm-started coordinate-descent LASSO along a descending lambda path.

    ``min ||y - X beta||^2 + lambda*||beta||_1``, no intercept term (both ``design`` columns and
    ``y`` are expected already centered by the caller). Active-set strategy: a full KKT check
    (``design.T @ r``, the one O(n*p) step) finds columns violating stationarity; cyclic coordinate
    descent then sweeps only that (typically small) active set until converged, and the KKT check
    repeats until no inactive column violates it — exact cyclic CD without ever sweeping every
    column on every iteration. Returns one beta (length p) per lambda, warm-started from the
    previous lambda's solution.
    """
    p = design.shape[1]
    z = np.sum(design * design, axis=0)
    beta = np.zeros(p, dtype=np.float64)
    r = y.copy()
    active: set[int] = set()
    betas: list[FloatArray] = []
    for lam in lambda_path:
        gamma = lam / 2.0
        for _round in range(_MAX_ACTIVE_SET_ROUNDS):
            rho_full = design.T @ r
            violators = set(np.flatnonzero((np.abs(rho_full) > gamma) & (z > 0.0)).tolist())
            if violators <= active:
                break
            active |= violators
            for _sweep in range(_CD_MAX_SWEEPS):
                max_delta = 0.0
                for j in active:
                    col = design[:, j]
                    rho_j = float(col @ r) + z[j] * beta[j]
                    new_beta_j = _soft_threshold(rho_j, gamma) / z[j]
                    delta = new_beta_j - beta[j]
                    if delta != 0.0:
                        r = r - col * delta
                        beta[j] = new_beta_j
                        max_delta = max(max_delta, abs(delta))
                if max_delta < _CD_TOL:
                    break
        else:
            logger.warning(
                "coordinate-descent active-set did not converge within %d rounds at lambda=%g",
                _MAX_ACTIVE_SET_ROUNDS,
                lam,
            )
        betas.append(beta.copy())
    return betas


# --------------------------------------------------------------------- estimators


def _kfold_indices(measured: Sequence[Variant], n_folds: int) -> IntArray:
    return np.array([variant_fold(v, n_folds) for v in measured], dtype=np.int64)


def _fourier_ridge_fit(
    config: _FourierConfig,
    measured: Sequence[Variant],
    site_idx_m: IntArray,
    y: FloatArray,
    needed: Sequence[Variant],
    site_idx_u: IntArray,
    n_folds: int,
) -> tuple[dict[Variant, float], int]:
    """Kernel-ridge on the restricted-order Fourier kernel; λ by K-fold CV (dense, no L1)."""
    n_sites = len(config.sites)
    if config.max_order != n_sites - 1:
        raise ValueError(
            "fourier_ridge's closed-form kernel is derived for max_order == n_sites - 1 "
            f"(orders 1..{n_sites - 1}); got max_order={config.max_order}, n_sites={n_sites}"
        )
    k_mm = _kernel_cross(site_idx_m, site_idx_m, config.q)
    folds = _kfold_indices(measured, n_folds)
    n_measured = len(measured)
    identity = np.eye(n_measured, dtype=np.float64)

    def _cv_sse(lam: float) -> float:
        total = 0.0
        counted = 0
        for fold in range(n_folds):
            train = folds != fold
            test = folds == fold
            if not np.any(train) or not np.any(test):
                continue
            y_train = y[train]
            mean_train = float(np.mean(y_train))
            k_train = k_mm[np.ix_(train, train)]
            alpha = np.linalg.solve(
                k_train + lam * np.eye(k_train.shape[0], dtype=np.float64), y_train - mean_train
            )
            pred = mean_train + k_mm[np.ix_(test, train)] @ alpha
            total += float(np.sum(np.square(pred - y[test])))
            counted += 1
        return total if counted else float("inf")

    lambda_star = min(_RIDGE_LAMBDA_GRID, key=_cv_sse)
    mean_all = float(np.mean(y))
    alpha_all = np.asarray(
        np.linalg.solve(k_mm + lambda_star * identity, y - mean_all), dtype=np.float64
    )
    k_um = _kernel_cross(site_idx_u, site_idx_m, config.q)
    predicted = mean_all + k_um @ alpha_all
    support = int(np.count_nonzero(alpha_all))
    return {v: float(predicted[i]) for i, v in enumerate(needed)}, support


def _fourier_lasso_fit(
    config: _FourierConfig,
    measured: Sequence[Variant],
    site_idx_m: IntArray,
    y: FloatArray,
    needed: Sequence[Variant],
    site_idx_u: IntArray,
    n_folds: int,
) -> tuple[dict[Variant, float], int]:
    """Coordinate-descent LASSO on the explicit standardized design; lambda by K-fold CV."""
    design_raw = _design_matrix(config, site_idx_m)
    mean_y = float(np.mean(y))
    y_c = y - mean_y

    col_mean = np.asarray(design_raw.mean(axis=0), dtype=np.float64)
    col_std = np.asarray(design_raw.std(axis=0), dtype=np.float64)
    scale = np.where(col_std > 0.0, col_std, 1.0)
    design_std = np.where(col_std > 0.0, (design_raw - col_mean) / scale, 0.0)

    p = design_std.shape[1]
    max_rho = float(np.max(np.abs(design_std.T @ y_c))) if p else 0.0
    if p == 0 or max_rho == 0.0:
        raw_beta = np.zeros(p, dtype=np.float64)
    else:
        lambda_max = 2.0 * max_rho
        ratios = np.asarray(np.geomspace(1.0, _LAMBDA_RATIO_MIN, _N_LAMBDA), dtype=np.float64)
        lambda_path = [lambda_max * float(ratio) for ratio in ratios]
        folds = _kfold_indices(measured, n_folds)
        cv_sse = np.zeros(_N_LAMBDA, dtype=np.float64)
        cv_seen = np.zeros(_N_LAMBDA, dtype=np.bool_)
        for fold in range(n_folds):
            train = folds != fold
            test = folds == fold
            if not np.any(train) or not np.any(test):
                continue
            betas_path = _cd_lasso_path(design_std[train], y_c[train], lambda_path)
            for i, beta_i in enumerate(betas_path):
                pred = design_std[test] @ beta_i
                cv_sse[i] += float(np.sum(np.square(pred - y_c[test])))
                cv_seen[i] = True
        best = int(np.argmin(np.where(cv_seen, cv_sse, np.inf))) if np.any(cv_seen) else 0
        beta_std = _cd_lasso_path(design_std, y_c, lambda_path[: best + 1])[-1]
        raw_beta = beta_std / scale

    support_idx = np.flatnonzero(raw_beta)
    intercept = mean_y - float(np.dot(raw_beta, col_mean))
    predicted = _reconstruct(config, site_idx_u, support_idx.tolist(), raw_beta, intercept)
    return {v: float(predicted[i]) for i, v in enumerate(needed)}, len(support_idx)


def _fit_estimator(
    estimator: str,
    config: _FourierConfig,
    measured: Sequence[Variant],
    site_idx_m: IntArray,
    y: FloatArray,
    needed: Sequence[Variant],
    site_idx_u: IntArray,
    n_folds: int,
) -> tuple[dict[Variant, float], int]:
    if estimator == "fourier_lasso":
        return _fourier_lasso_fit(config, measured, site_idx_m, y, needed, site_idx_u, n_folds)
    if estimator == "fourier_ridge":
        return _fourier_ridge_fit(config, measured, site_idx_m, y, needed, site_idx_u, n_folds)
    raise ValueError(
        f"unknown estimator {estimator!r}; expected 'fourier_lasso' or 'fourier_ridge'"
    )


# --------------------------------------------------------------------- scoring


@dataclass(frozen=True)
class CoeffOrderCell:
    """Recovery of one interaction order from one (estimator, budget, selection) fit.

    ``pin`` is the trivial zero-imputation baseline (the true label where measured, 0 elsewhere, no
    Fourier fit at all) so ``residual_delta_spearman`` isolates what the compressed-sensing
    extrapolation adds beyond simply having the measured labels.
    """

    order: str
    n_terms: int
    raw_pearson: float | None
    raw_spearman: float | None
    sse: float
    pin_spearman: float | None
    residual_pin_spearman: float | None
    residual_spearman: float | None
    residual_delta_spearman: float | None
    residual_delta_spearman_ci95: tuple[float, float] | None


@dataclass(frozen=True)
class CoeffFitResult:
    """One (estimator, budget, selection[, seed]) fit: recovery per order plus support size."""

    estimator: str
    budget: int
    selection: str  # "info" | "random" | "doptimal" | "doptimal_pairs"
    seed: int | None  # None for "info"; the draw index for "random"
    n_measured: int
    support_size: int
    cells: list[CoeffOrderCell] = field(default_factory=list)


@dataclass(frozen=True)
class Gate6Result:
    dataset: str
    model_id: str
    budgets: list[int]
    estimators: list[str]
    random_seeds: int
    max_order: int
    public_claim_eligible: bool = False
    fits: list[CoeffFitResult] = field(default_factory=list)
    decision: dict[str, str] = field(default_factory=dict)
    decision_reason: dict[str, str] = field(default_factory=dict)


def _seed_for(*parts: object) -> int:
    """Stable, salt-free seed from a bootstrap cell's identity (deterministic across processes)."""
    key = 0
    for part in parts:
        for ch in str(part):
            key = (key * 131 + ord(ch)) % (2**31)
    return key


def _evaluate_fit(
    predicted_map: Mapping[Variant, float],
    revealed: Mapping[Variant, float],
    needed: Sequence[Variant],
    support_size: int,
    estimator: str,
    budget: int,
    selection: str,
    seed: int | None,
    truth_by_order: Mapping[int, Mapping[Term, float]],
    true_dg: Mapping[Variant, float],
    bootstrap_iterations: int,
) -> CoeffFitResult:
    measured_set = frozenset(revealed)
    mu_pin = {v: revealed.get(v, 0.0) for v in needed}
    cells: list[CoeffOrderCell] = []
    for order, order_name in _ORDERS:
        truth_terms = truth_by_order.get(order, {})
        if not truth_terms:
            continue
        terms = sorted(truth_terms)
        truth = np.array([truth_terms[t] for t in terms], dtype=np.float64)
        pred = np.array([_epsilon(predicted_map, t) for t in terms], dtype=np.float64)
        pin = np.array([_epsilon(mu_pin, t) for t in terms], dtype=np.float64)
        raw_pearson, raw_spearman = _safe_corr(pred, truth)
        _pin_pearson, pin_spearman = _safe_corr(pin, truth)
        sse = float(np.sum(np.square(pred - truth)))
        skeleton = np.array([_skeleton(t, measured_set, true_dg) for t in terms], dtype=np.float64)
        residual_spearman = _partial_spearman(pred, truth, skeleton)
        residual_pin_spearman = _partial_spearman(pin, truth, skeleton)
        residual_delta = (
            None
            if residual_spearman is None or residual_pin_spearman is None
            else residual_spearman - residual_pin_spearman
        )
        seed_for_ci = _seed_for(estimator, budget, selection, seed, order_name)
        residual_delta_ci = _bootstrap_residual_delta_ci(
            pin, pred, truth, skeleton, bootstrap_iterations, seed_for_ci
        )
        cells.append(
            CoeffOrderCell(
                order=order_name,
                n_terms=len(terms),
                raw_pearson=raw_pearson,
                raw_spearman=raw_spearman,
                sse=sse,
                pin_spearman=pin_spearman,
                residual_pin_spearman=residual_pin_spearman,
                residual_spearman=residual_spearman,
                residual_delta_spearman=residual_delta,
                residual_delta_spearman_ci95=residual_delta_ci,
            )
        )
    return CoeffFitResult(
        estimator=estimator,
        budget=budget,
        selection=selection,
        seed=seed,
        n_measured=len(revealed),
        support_size=support_size,
        cells=cells,
    )


# --------------------------------------------------------------------- decision


def _classify_cell(residual_spearman: float | None, reference: float) -> str:
    """Compare one (order, budget) cell's own residualized Spearman to the fixed ESM reference.

    Simplified per spec ("keep the decision logic simple"): the ESM pipeline's reference is a fixed
    published constant, not a resampled distribution here, so "CI-overlapping or higher" reduces to
    a direct point comparison against that constant.
    """
    if residual_spearman is None:
        return "inconclusive"
    if abs(residual_spearman) < _WEAK_THRESHOLD and abs(reference) < _WEAK_THRESHOLD:
        return "both_weak"
    if residual_spearman >= reference:
        return "compressed_sensing_competitive"
    return "esm_pipeline_ahead"


def _decide(fits: Sequence[CoeffFitResult]) -> tuple[dict[str, str], dict[str, str]]:
    """Per order, compare the Fourier-LASSO/info residualized recovery to the ESM reference.

    Only budgets >= ``_OPERATIONAL_BUDGET`` with a registered reference value are considered
    (docs/specs/step6-coefficient-recovery.md). All qualifying budgets must agree on a category or
    the order is reported "inconclusive".
    """
    decisions: dict[str, str] = {}
    reasons: dict[str, str] = {}
    for order_name, reference_by_budget in _ESM_REFERENCE.items():
        cells_by_budget = {
            fit.budget: cell
            for fit in fits
            if fit.estimator == "fourier_lasso" and fit.selection == "info"
            for cell in fit.cells
            if cell.order == order_name
        }
        operational_budgets = sorted(
            budget
            for budget in reference_by_budget
            if budget >= _OPERATIONAL_BUDGET and budget in cells_by_budget
        )
        if not operational_budgets:
            decisions[order_name] = "inconclusive"
            reasons[order_name] = "no operational-budget Fourier-LASSO/info fit available"
            continue
        categories = {
            _classify_cell(cells_by_budget[budget].residual_spearman, reference_by_budget[budget])
            for budget in operational_budgets
        }
        decisions[order_name] = next(iter(categories)) if len(categories) == 1 else "inconclusive"
        reasons[order_name] = (
            "Fourier-LASSO/info residualized recovery vs the ESM reference at budgets "
            f"{operational_budgets}: {sorted(categories)}"
        )
    return decisions, reasons


# --------------------------------------------------------------------- D-optimal acquisition (6B)

_DOPTIMAL_OBS_VAR = 1e-2


def _doptimal_order(
    config: _FourierConfig,
    candidates: Sequence[Variant],
    budget: int,
    *,
    orders: Sequence[int] | None = None,
) -> list[Variant]:
    """Greedy Bayesian D-optimal (max posterior-variance) design over the Fourier kernel.

    Selects the ``budget`` variants that most reduce coefficient uncertainty under a N(0, I) prior
    on the Fourier coefficients with observation noise ``_DOPTIMAL_OBS_VAR``, using ONLY the design
    (the closed-form kernel) — never a measured label (the label barrier). Prefix-consistent (the
    first B of a longer run is the budget-B design). Efficient GP-style rank-1 update: O(N·B²).

    ``orders`` sets the target subspace. ``None`` (Step 6B) uses the full orders-1..n-1 kernel — an
    isotropic design that, on GB1, spends most of its budget resolving the vast order-3 subspace.
    ``orders=(1, 2)`` (Step 6C) targets only singles+pairwise structure: the weighted-pairs design
    that concentrates the budget where the map is plausibly recoverable at these budgets.
    """
    idx = _site_indices(config, candidates)
    n = len(candidates)
    if n == 0:
        return []
    n_sites = len(config.sites)
    if orders is None:
        # K(v,v), constant across candidates (every site agrees with itself).
        kvv = 1.0 - (1.0 / config.q) ** n_sites - (1.0 - 1.0 / config.q) ** n_sites
    else:
        kvv = float(_order_symmetric_kernel(idx[[0]], idx[[0]], config.q, orders)[0, 0])
    post_var = np.full(n, kvv, dtype=np.float64)
    updates = np.zeros((n, min(budget, n)), dtype=np.float64)
    selected: list[int] = []
    for step in range(min(budget, n)):
        masked = post_var.copy()
        masked[selected] = -np.inf
        pick = int(np.argmax(masked))
        selected.append(pick)
        prior_cov = (
            _kernel_cross(idx, idx[[pick]], config.q)[:, 0]
            if orders is None
            else _order_symmetric_kernel(idx, idx[[pick]], config.q, orders)[:, 0]
        )
        cov = prior_cov - updates[:, :step] @ updates[pick, :step]
        denom = _DOPTIMAL_OBS_VAR + max(float(cov[pick]), 0.0)
        updates[:, step] = cov / np.sqrt(denom)
        post_var = np.maximum(post_var - updates[:, step] ** 2, 0.0)
    return [candidates[i] for i in selected]


# --------------------------------------------------------------------- orchestration


def run_coeff_recovery(
    scored: Sequence[ScoredVariant],
    landscape: dict[Variant, float],
    *,
    budgets: Sequence[int] = (48, 96, 192),
    estimators: Sequence[str] = ("fourier_lasso", "fourier_ridge"),
    random_seeds: int = 5,
    max_order: int = 3,
    n_folds: int = 5,
    bootstrap_iterations: int = 1000,
    dataset: str = _DEFAULT_DATASET,
    model_id: str = _DEFAULT_MODEL_ID,
) -> Gate6Result:
    """Compressed-sensing recovery on GB1's Fourier basis, on the frozen Gate-2/3 selections.

    Selection (``info`` via the frozen Gate-2 allocation, ``random`` via uniform sampling) is
    ESM-only / label-free; only after selection is fixed does ``reveal_measured_fitness`` read the
    measured labels that fitting and CV use (the label barrier, docs/AI_CONTRACT.md).
    """
    canonical = _canonical_scored(scored)
    wild_type: Variant = frozenset()
    true_dg = _center_positive(
        reveal_measured_fitness(
            landscape,
            [wild_type, *(item.variant for item in canonical if item.variant != wild_type)],
        )
    )
    truth_by_order = _truth_terms(canonical, landscape, max_order)
    needed: set[Variant] = set()
    for terms in truth_by_order.values():
        for term in terms:
            needed.update(interaction_loop(term))
    needed_list = sorted(needed, key=lambda v: (len(v), sorted(v)))

    config = _build_fourier_config(GB1_SITES, GB1_WT_AT_SITES, AA20, max_order)
    site_idx_u = _site_indices(config, needed_list)
    candidate_variants = [item.variant for item in canonical]
    doptimal_order = _doptimal_order(config, candidate_variants, max(budgets)) if budgets else []
    doptimal_pairs_order = (
        _doptimal_order(config, candidate_variants, max(budgets), orders=(1, 2)) if budgets else []
    )

    fits: list[CoeffFitResult] = []
    for budget in budgets:
        selections: list[tuple[str, int | None, list[Variant]]] = [
            ("info", None, _info_selection(canonical, budget, max_order))
        ]
        selections.extend(
            ("random", seed, random_selection(canonical, budget, seed))
            for seed in range(random_seeds)
        )
        selections.append(("doptimal", None, doptimal_order[:budget]))
        selections.append(("doptimal_pairs", None, doptimal_pairs_order[:budget]))
        for selection_name, seed, selected in selections:
            revealed, *_ = _reveal_selection(landscape, selected)
            measured = sorted(revealed, key=sorted)
            if not measured:
                continue
            y = np.array([revealed[v] for v in measured], dtype=np.float64)
            site_idx_m = _site_indices(config, measured)
            for estimator in estimators:
                predicted, support = _fit_estimator(
                    estimator, config, measured, site_idx_m, y, needed_list, site_idx_u, n_folds
                )
                fits.append(
                    _evaluate_fit(
                        predicted,
                        revealed,
                        needed_list,
                        support,
                        estimator,
                        budget,
                        selection_name,
                        seed,
                        truth_by_order,
                        true_dg,
                        bootstrap_iterations,
                    )
                )
    decision, reason = _decide(fits)
    return Gate6Result(
        dataset=dataset,
        model_id=model_id,
        budgets=list(budgets),
        estimators=list(estimators),
        random_seeds=random_seeds,
        max_order=max_order,
        fits=fits,
        decision=decision,
        decision_reason=reason,
    )


def result_to_dict(result: Gate6Result) -> dict[str, object]:
    return asdict(result)
