"""Downstream-impact benchmark (docs/specs/downstream.md, docs/VALIDATION.md protocol amendment 1).

Escapes the map-recovery tautology: for each selection method, train a fixed supervised learner ONLY
on the fitness labels its budget reveals, then rank held-out double/triple mutants. The primary
learner never sees a held-out variant's own ESM score and never calls ``infer_epistasis`` (whose
posterior mean keeps the ESM prior for unmeasured terms), so a high score cannot restate the prior;
the three ESM diagnostics below are a separate, explicitly-labelled, non-primary path that may.

Post-hoc and offline: runs on the already-computed ``ScoredVariant`` cache and the measured
landscape; no torch, no model, no network. It never alters the frozen historical decision rule.
Every summary, effect, partition aggregate, and decision field is a pure function of the immutable
raw per-fold records this module serializes first (docs/specs/downstream.md "Raw record schema") —
never computed on a second, parallel path.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from math import isfinite, isnan, log2, sqrt
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from scipy.stats import pearsonr, rankdata, spearmanr
from scipy.stats import t as student_t

from epibudget.acquisition import allocate, fitness_greedy
from epibudget.data import GB1_SITES, GB1_WT_AT_SITES, reveal_measured_fitness
from epibudget.epistasis import predicted_epistasis
from epibudget.graph import EpistasisFactorGraph
from epibudget.provenance import write_json_atomic
from epibudget.scored_cache import candidate_sha256
from epibudget.types import Interaction, ScoredVariant, Variant
from epibudget.validate import esm_prior_mu, practice_heuristic, random_selection

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

_PAIRWISE_ORDER = 2
_THIRD_ORDER = 3
_MIN_POINTS_FOR_CORR = 3
_SALT_PREFIX = "epibudget-downstream-v1:"
_INNER_SALT = hashlib.sha256(b"epibudget-downstream-inner:v1").hexdigest()

# --------------------------------------------------------------------- frozen protocol amendment 1

PROTOCOL_VERSION = "epibudget-downstream-v1"
AMENDMENT_VERSION = "protocol-amendment-1"
N_INNER_FOLDS = 3
GRID_MAIN: tuple[float, ...] = (0.1, 1.0, 10.0)
GRID_PAIR: tuple[float, ...] = (1.0, 10.0, 100.0)
EXPECTED_PARTITIONS = (
    20  # frozen registered R; independent of the --partitions argument of a given run
)
SIGN_THRESHOLD = 16  # >= 16 of the 20 expected partition means must be strictly positive
MIN_STRUCTURAL_EFFECT_SIZE = (
    0.0  # no non-zero threshold is defensible pre-result; see spec amendment 1
)


# --------------------------------------------------------------------- folds (label-free, SHA-256)


def canonical_id(variant: Variant) -> str:
    """Order-independent identity string of a variant (sorted mutation lists, compact JSON).

    The single canonical form fed to every hash and every seeded step, so fold assignment and any
    resampling are byte-identical across processes regardless of ``PYTHONHASHSEED``.
    """
    return json.dumps(
        sorted([list(mutation) for mutation in variant]), separators=(",", ":"), ensure_ascii=True
    )


def partition_salt(index: int) -> str:
    """Frozen per-partition salt; the fold partition is a pure function of it and identity."""
    return hashlib.sha256(f"{_SALT_PREFIX}{index}".encode("ascii")).hexdigest()


def _fold_hash(salt: str, ident: str) -> str:
    return hashlib.sha256(f"{salt}:{ident}".encode("ascii")).hexdigest()


def assign_outer_folds(
    eval_variants: Sequence[Variant], n_folds: int, salt: str
) -> dict[Variant, int]:
    """Assign each order-2/3 variant a fold, stratified by order via a documented SHA-256 rule.

    Within each mutation order the variants are ranked by ``(sha256(f"{salt}:{id}"), id)`` and
    given ``fold = rank % n_folds``. Balanced +/-1 per order, reorder-stable, and from identity
    ONLY: never a fitness value, live/dead status, or missingness. Singles are never held out.
    """
    if n_folds < 2:  # noqa: PLR2004 — at least two folds so a held-out set exists
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    by_order: dict[int, list[tuple[str, str, Variant]]] = defaultdict(list)
    for variant in eval_variants:
        ident = canonical_id(variant)
        by_order[len(variant)].append((_fold_hash(salt, ident), ident, variant))
    folds: dict[Variant, int] = {}
    for items in by_order.values():
        items.sort(key=lambda triple: (triple[0], triple[1]))
        for rank, (_hash, _ident, variant) in enumerate(items):
            folds[variant] = rank % n_folds
    return folds


def _inner_folds_balanced(variants: Sequence[Variant], n_inner: int, salt: str) -> list[int] | None:
    """Balanced identity-sorted inner-fold labels; ``None`` if fewer than ``n_inner`` can be formed.

    Sorts training identities by ``(sha256(f"{salt}:{id}"), id)`` and assigns ``fold = rank %
    n_inner`` — the same balancing mechanism as :func:`assign_outer_folds`, without the outer
    per-order stratification (a training set mixes whatever orders selection produced). Invariant to
    the input order of ``variants``. Returns ``None`` (never a smaller fold count) if fewer than
    ``n_inner`` distinct non-empty labels result, so the caller can fall back explicitly rather than
    silently run fewer-fold CV.
    """
    n = len(variants)
    if n < n_inner:
        return None
    ids = [canonical_id(v) for v in variants]
    order = sorted(range(n), key=lambda i: (_fold_hash(salt, ids[i]), ids[i]))
    labels = [0] * n
    for rank, i in enumerate(order):
        labels[i] = rank % n_inner
    if len(set(labels)) < n_inner:  # defensive: unreachable for n>=n_inner distinct identities
        return None
    return labels


# ------------------------------------------------------------------ feature space (global, fixed)


class FeatureSpace:
    """Reference-coded (WT=reference) main-effect + pairwise indicator columns over the sites.

    A single global dictionary shared by every method / fold / budget, so methods differ only in
    their training data, never in model capacity. No third-order columns; no ESM feature. A variant
    maps to the columns of its non-WT residues (main) and each cross-site residue pair (pairwise).
    """

    __slots__ = ("main_index", "n_features", "pair_index", "penalty_is_main", "sites", "wt_of")

    def __init__(self, sites: Sequence[int], wt_at_sites: Sequence[str], alphabet: str) -> None:
        self.sites: tuple[int, ...] = tuple(sorted(sites))
        self.wt_of: dict[int, str] = dict(zip(sorted(sites), wt_at_sites, strict=True))
        self.main_index: dict[tuple[int, str], int] = {}
        self.pair_index: dict[tuple[tuple[int, str], tuple[int, str]], int] = {}
        column = 0
        for pos in self.sites:
            for aa in alphabet:
                if aa != self.wt_of[pos]:
                    self.main_index[(pos, aa)] = column
                    column += 1
        n_main = column
        for pos_i, pos_j in combinations(self.sites, 2):
            for aa_i in alphabet:
                if aa_i == self.wt_of[pos_i]:
                    continue
                for aa_j in alphabet:
                    if aa_j == self.wt_of[pos_j]:
                        continue
                    self.pair_index[((pos_i, aa_i), (pos_j, aa_j))] = column
                    column += 1
        self.n_features = column
        self.penalty_is_main: IntArray = np.zeros(column, dtype=np.int64)
        self.penalty_is_main[:n_main] = 1

    def active_columns(self, variant: Variant, *, include_pairs: bool = True) -> list[int]:
        """Columns a variant activates: its main effects and (optionally) its residue pairs."""
        muts = sorted(variant)
        cols = [self.main_index[(pos, mut)] for pos, _wt, mut in muts]
        if include_pairs:
            for (pi, _wi, mi), (pj, _wj, mj) in combinations(muts, 2):
                cols.append(self.pair_index[((pi, mi), (pj, mj))])
        return cols

    def design_matrix(
        self, variants: Sequence[Variant], *, include_pairs: bool = True
    ) -> FloatArray:
        """Dense n×p 0/1 design over ``variants`` (used for the small training fits)."""
        matrix: FloatArray = np.zeros((len(variants), self.n_features), dtype=np.float64)
        for row, variant in enumerate(variants):
            for col in self.active_columns(variant, include_pairs=include_pairs):
                matrix[row, col] = 1.0
        return matrix

    def penalties(self, alpha_main: float, alpha_pair: float) -> FloatArray:
        """Per-feature ridge penalty vector (strictly positive); intercept handled by centering."""
        if alpha_main <= 0.0 or alpha_pair <= 0.0:
            raise ValueError(f"penalties must be > 0, got main={alpha_main}, pair={alpha_pair}")
        return np.where(self.penalty_is_main == 1, alpha_main, alpha_pair).astype(np.float64)


# --------------------------------------------------------------------- generalized-dual ridge


class RidgeModel(BaseModel):
    """A fitted ridge (coef + intercept); prediction is sparse over a variant's active columns."""

    model_config = {"arbitrary_types_allowed": True}

    coef: FloatArray
    intercept: float
    degenerate: bool

    def predict_active(self, active_columns: Sequence[Sequence[int]]) -> FloatArray:
        """Predict for held-out variants given each one's active column indices (sparse, exact)."""
        out: FloatArray = np.empty(len(active_columns), dtype=np.float64)
        for row, cols in enumerate(active_columns):
            out[row] = self.intercept + float(self.coef[list(cols)].sum())
        return out


def fit_ridge(design: FloatArray, response: FloatArray, penalties: FloatArray) -> RidgeModel:
    """Fit ridge with a per-feature penalty via the generalized dual (an n×n solve, never p×p).

    Minimises ``‖y − Xβ − c‖² + Σ_k Λ_k β_k²`` with the intercept ``c`` unpenalised (via centering).
    Using ``β = Λ⁻¹Xᵀ(I_n + XΛ⁻¹Xᵀ)⁻¹ y_c`` keeps the solve at size n = #training rows (≤ B), so an
    all-singles design (which is rank-deficient in the primal with unpenalised mains) stays PD here.
    """
    n = design.shape[0]
    if n == 0:
        return RidgeModel(
            coef=np.zeros(design.shape[1], dtype=np.float64), intercept=0.0, degenerate=True
        )
    x_mean = design.mean(axis=0)
    y_mean = float(response.mean())
    centered = design - x_mean
    yc = response - y_mean
    lam_inv = 1.0 / penalties
    scaled = centered * lam_inv  # (n, p): each column divided by its penalty
    gram = scaled @ centered.T + np.eye(n, dtype=np.float64)  # (n, n), PD
    weights = np.linalg.solve(gram, yc)
    coef = lam_inv * (centered.T @ weights)
    intercept = y_mean - float(x_mean @ coef)
    return RidgeModel(coef=coef, intercept=intercept, degenerate=False)


class AlphaChoice(BaseModel):
    """The inner-CV-selected penalty/ies for one regime, and whether it fell back to shrinkage."""

    alpha_main: float
    # None only for the main-effects-only regime, which has no active pairwise column.
    alpha_pair: float | None
    # False marks alpha_pair as structurally not applicable (main-only), not missing data.
    applicable: bool = True
    fell_back: bool
    fallback_reason: str | None
    n_inner_folds_used: int  # 0 if inner CV did not run (fallback), else N_INNER_FOLDS
    # Set only by the ESM-offset regime: the through-origin slope refit on all outer-training
    # labels after alpha selection (never on an inner- or outer-held-out label). None elsewhere.
    b: float | None = None


def _through_origin_slope(x: FloatArray, y: FloatArray) -> float:
    """Through-origin least-squares slope of ``y`` on ``x``; degenerate ``x`` falls back to 1.0.

    Anchors the fit at the shared WT reference (ΔG/ESM both 0 there), the same convention
    ``validate._calibrate_slope`` uses on the log-fitness scale; this is the ``log1p``-scale
    analogue used to build the ESM-offset diagnostic's response.
    """
    denom = float(np.dot(x, x))
    return float(np.dot(x, y) / denom) if denom != 0.0 else 1.0


def select_alpha(
    space: FeatureSpace,
    train_variants: Sequence[Variant],
    response: FloatArray,
    grid_main: Sequence[float],
    grid_pair: Sequence[float],
    n_inner: int,
    inner_salt: str,
) -> AlphaChoice:
    """Pick (alpha_main, alpha_pair) by held-out inner-fold ``log1p`` MSE; tie-break to shrinkage.

    The criterion is the mean error on the *held-out* inner fold (never the training-fit error,
    which is monotone in alpha and would always pick the smallest). A training set too small to form
    ``n_inner`` non-empty balanced folds falls back to the strongest-shrinkage grid corner, flagged
    with an explicit reason (never silently run on fewer folds).
    """
    strongest = AlphaChoice(
        alpha_main=max(grid_main),
        alpha_pair=max(grid_pair),
        fell_back=True,
        fallback_reason="training_set_too_small",
        n_inner_folds_used=0,
    )
    n = len(train_variants)
    if n < n_inner:
        return strongest
    labels = _inner_folds_balanced(train_variants, n_inner, inner_salt)
    if labels is None:
        return strongest.model_copy(update={"fallback_reason": "insufficient_distinct_inner_folds"})
    design = space.design_matrix(train_variants)
    active = [space.active_columns(v) for v in train_variants]
    label_arr = np.array(labels)
    best: tuple[float, float, float] | None = None  # (mse, -alpha_main, -alpha_pair), min-first
    for alpha_main in grid_main:
        for alpha_pair in grid_pair:
            penalties = space.penalties(alpha_main, alpha_pair)
            errors: list[float] = []
            for fold in range(n_inner):
                test_mask = label_arr == fold
                train_mask = ~test_mask
                model = fit_ridge(design[train_mask], response[train_mask], penalties)
                pred = model.predict_active([active[i] for i in np.nonzero(test_mask)[0]])
                errors.extend((pred - response[test_mask]) ** 2)
            key = (float(np.mean(errors)), -alpha_main, -alpha_pair)
            if best is None or key < best:
                best = key
    assert best is not None  # n_inner non-empty folds guarantee at least one scored grid point
    return AlphaChoice(
        alpha_main=-best[1],
        alpha_pair=-best[2],
        fell_back=False,
        fallback_reason=None,
        n_inner_folds_used=n_inner,
    )


def select_alpha_main_only(
    space: FeatureSpace,
    train_variants: Sequence[Variant],
    response: FloatArray,
    grid_main: Sequence[float],
    n_inner: int,
    inner_salt: str,
) -> AlphaChoice:
    """Dedicated 1-D inner-CV alpha search for the main-effects-only regime (no pairwise column).

    Searching ``alpha_pair`` would be wasted computation: with ``include_pairs=False`` no pairwise
    column is ever active, so no prediction depends on it (spec amendment 1).
    """
    strongest = AlphaChoice(
        alpha_main=max(grid_main),
        alpha_pair=None,
        applicable=False,
        fell_back=True,
        fallback_reason="training_set_too_small",
        n_inner_folds_used=0,
    )
    n = len(train_variants)
    if n < n_inner:
        return strongest
    labels = _inner_folds_balanced(train_variants, n_inner, inner_salt)
    if labels is None:
        return strongest.model_copy(update={"fallback_reason": "insufficient_distinct_inner_folds"})
    design = space.design_matrix(train_variants, include_pairs=False)
    active = [space.active_columns(v, include_pairs=False) for v in train_variants]
    label_arr = np.array(labels)
    best: tuple[float, float] | None = None  # (mse, -alpha_main)
    for alpha_main in grid_main:
        penalties = space.penalties(alpha_main, max(grid_main))  # pairwise entries unused/inert
        errors: list[float] = []
        for fold in range(n_inner):
            test_mask = label_arr == fold
            train_mask = ~test_mask
            model = fit_ridge(design[train_mask], response[train_mask], penalties)
            pred = model.predict_active([active[i] for i in np.nonzero(test_mask)[0]])
            errors.extend((pred - response[test_mask]) ** 2)
        key = (float(np.mean(errors)), -alpha_main)
        if best is None or key < best:
            best = key
    assert best is not None
    return AlphaChoice(
        alpha_main=-best[1],
        alpha_pair=None,
        applicable=False,
        fell_back=False,
        fallback_reason=None,
        n_inner_folds_used=n_inner,
    )


def _fit_offset_fold(
    design: FloatArray, y: FloatArray, esm: FloatArray, train_mask: BoolArray, penalties: FloatArray
) -> tuple[float, RidgeModel]:
    """Fit the offset ``b_inner`` and the residual ridge from the ``train_mask`` rows only.

    A pure function of the rows selected by ``train_mask``: the complementary (validation) rows
    never enter ``b_inner`` or the residual model here, only a caller's later loss computation.
    """
    b_inner = _through_origin_slope(esm[train_mask], y[train_mask])
    offset_train = y[train_mask] - b_inner * esm[train_mask]
    model = fit_ridge(design[train_mask], offset_train, penalties)
    return b_inner, model


def select_alpha_esm_offset(
    space: FeatureSpace,
    train_variants: Sequence[Variant],
    y: FloatArray,
    esm: FloatArray,
    grid_main: Sequence[float],
    grid_pair: Sequence[float],
    n_inner: int,
    inner_salt: str,
) -> AlphaChoice:
    """Nested inner-CV alpha selection for the ESM-offset regime, refitting ``b`` per inner fold.

    For every grid point and every inner fold, ``b_inner`` is fit from the inner-training rows
    ONLY (:func:`_fit_offset_fold`), the residual ridge trains on the same inner-training rows,
    and the validation loss is measured on the untouched inner-validation ``y`` (after adding
    ``b_inner * esm`` back) — an inner-validation label can move its own fold's loss but never
    ``b_inner`` or the fitted residual model, since fitting ``b`` once on the full outer-training
    set before running inner CV would leak each inner-validation label into the ``b`` its own fold
    is later scored against. After alpha selection, ``b`` is refit once on the full outer-training
    set for the final model; this function never sees an outer-held-out label.
    """
    b_final = _through_origin_slope(esm, y)
    strongest = AlphaChoice(
        alpha_main=max(grid_main),
        alpha_pair=max(grid_pair),
        fell_back=True,
        fallback_reason="training_set_too_small",
        n_inner_folds_used=0,
        b=b_final,
    )
    n = len(train_variants)
    if n < n_inner:
        return strongest
    labels = _inner_folds_balanced(train_variants, n_inner, inner_salt)
    if labels is None:
        return strongest.model_copy(update={"fallback_reason": "insufficient_distinct_inner_folds"})
    design = space.design_matrix(train_variants)
    active = [space.active_columns(v) for v in train_variants]
    label_arr = np.array(labels)
    best: tuple[float, float, float] | None = None  # (mse, -alpha_main, -alpha_pair), min-first
    for alpha_main in grid_main:
        for alpha_pair in grid_pair:
            penalties = space.penalties(alpha_main, alpha_pair)
            errors: list[float] = []
            for fold in range(n_inner):
                val_mask = label_arr == fold
                train_mask = ~val_mask
                b_inner, model = _fit_offset_fold(design, y, esm, train_mask, penalties)
                val_active = [active[i] for i in np.nonzero(val_mask)[0]]
                pred = model.predict_active(val_active) + b_inner * esm[val_mask]
                errors.extend((pred - y[val_mask]) ** 2)
            key = (float(np.mean(errors)), -alpha_main, -alpha_pair)
            if best is None or key < best:
                best = key
    assert best is not None  # n_inner non-empty folds guarantee at least one scored grid point
    return AlphaChoice(
        alpha_main=-best[1],
        alpha_pair=-best[2],
        fell_back=False,
        fallback_reason=None,
        n_inner_folds_used=n_inner,
        b=b_final,
    )


# ------------------------------------------------------------------ metrics (order-stratified)


def _corr(pred: FloatArray, true: FloatArray, kind: str) -> float | None:
    """Spearman or Pearson of two aligned arrays, or None if undefined (too few / constant)."""
    if len(pred) < _MIN_POINTS_FOR_CORR:
        return None
    if float(np.std(pred)) == 0.0 or float(np.std(true)) == 0.0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # constant/near-constant caution — degeneracy handled above
        stat = (
            spearmanr(pred, true).statistic
            if kind == "spearman"
            else pearsonr(pred, true).statistic
        )
    value = float(stat)
    return None if isnan(value) else value


def macro_spearman(rho_doubles: float | None, rho_triples: float | None) -> float | None:
    """Order-stratified primary statistic ``½(ρ_doubles + ρ_triples)``; None if either is undefined.

    Averaging the two per-order Spearmans removes a between-order separation that could inflate a
    pooled correlation even when neither order is ranked well.
    """
    if rho_doubles is None or rho_triples is None:
        return None
    return 0.5 * (rho_doubles + rho_triples)


def percentile_relevance(fitness: FloatArray) -> FloatArray:
    """Deterministic percentile rank of raw fitness in [0, 1] (zeros included, ties averaged)."""
    n = len(fitness)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    ranks: FloatArray = np.asarray(rankdata(fitness, method="average"), dtype=np.float64)
    return (ranks - 1.0) / (n - 1)


def ndcg_at_k(pred: FloatArray, relevance: FloatArray, k: int, tiebreak: Sequence[str]) -> float:
    """NDCG@k; gain = relevance, discount ``1/log2(rank_0indexed+2)``, ties by ``tiebreak`` (id).

    All-tied relevance (including all-zero) is a convention case, not a degeneracy: every ordering
    of a constant-relevance set achieves the unique reachable DCG, so NDCG is defined as ``1.0``.
    """
    n = len(pred)
    if n == 0 or k <= 0:
        return 0.0
    k = min(k, n)
    if float(np.ptp(relevance)) == 0.0:
        return 1.0
    ranked = sorted(range(n), key=lambda i: (-float(pred[i]), tiebreak[i]))
    ideal = sorted(range(n), key=lambda i: (-float(relevance[i]), tiebreak[i]))

    def _dcg(order: list[int]) -> float:
        return sum(float(relevance[order[r]]) / log2(r + 2) for r in range(k))

    idcg = _dcg(ideal)
    return _dcg(ranked) / idcg if idcg > 0.0 else 0.0


def learning_curve_auc(values: Sequence[float | None]) -> float | None:
    """Equal-weight trapezoidal average over ordered-budget values (None if any point undefined)."""
    if any(v is None for v in values) or len(values) < _PAIRWISE_ORDER:
        return None
    vals = [float(v) for v in values if v is not None]
    m = len(vals)
    weighted = vals[0] + vals[-1] + 2.0 * sum(vals[1:-1])
    return weighted / (2.0 * (m - 1))


# ------------------------------------------------------------------ corrected-CV sensitivity


class SensitivityInterval(BaseModel):
    """One labelled corrected-CV sensitivity convention (never the primary decision gate)."""

    convention: str  # "pool_ratio" | "effective_label_ratio"
    status: str  # "sensitivity_only" | "unavailable"
    n_test: float | None
    n_train: float | None
    ratio: float | None
    n_valid_effects: int
    delta_mean: float | None
    sample_variance: float | None
    df: int | None
    se: float | None
    t_critical: float | None
    ci95: tuple[float, float] | None


def _corrected_cv_formula(
    deltas: Sequence[float | None],
    n_test: float | None,
    n_train: float | None,
    convention: str,
    alpha: float = 0.05,
) -> SensitivityInterval:
    """Nadeau-Bengio corrected-resampled t over paired per-fold-instance differences (companion).

    ``n_test``/``n_train`` are the mean per-fold-instance held-out/selectable-pool (or effective-
    training) sizes for this convention; ``ratio = n_test/n_train`` feeds the Nadeau-Bengio variance
    inflation. Both are recorded explicitly (not only the derived ratio) so the interval is exactly
    reconstructable from the report.
    """
    valid = [float(d) for d in deltas if d is not None]
    n = len(valid)
    ratio = (
        n_test / n_train if (n_test is not None and n_train is not None and n_train > 0.0) else None
    )
    if ratio is None or n == 0:
        return SensitivityInterval(
            convention=convention,
            status="unavailable",
            n_test=n_test,
            n_train=n_train,
            ratio=ratio,
            n_valid_effects=n,
            delta_mean=float(np.mean(valid)) if valid else None,
            sample_variance=None,
            df=None,
            se=None,
            t_critical=None,
            ci95=None,
        )
    mean = float(np.mean(valid))
    if n < _PAIRWISE_ORDER:
        return SensitivityInterval(
            convention=convention,
            status="sensitivity_only",
            n_test=n_test,
            n_train=n_train,
            ratio=ratio,
            n_valid_effects=n,
            delta_mean=mean,
            sample_variance=None,
            df=max(n - 1, 0),
            se=None,
            t_critical=None,
            ci95=None,
        )
    var = float(np.var(valid, ddof=1))
    df = n - 1
    se = sqrt(var * (1.0 / n + ratio))
    if se > 0.0:
        crit = float(student_t.ppf(1.0 - alpha / 2.0, df))
        ci = (mean - crit * se, mean + crit * se)
    else:  # zero between-fold-instance variance — knife-edge, no width
        crit = None
        ci = (mean, mean)
    return SensitivityInterval(
        convention=convention,
        status="sensitivity_only",
        n_test=n_test,
        n_train=n_train,
        ratio=ratio,
        n_valid_effects=n,
        delta_mean=mean,
        sample_variance=var,
        df=df,
        se=se,
        t_critical=crit,
        ci95=ci,
    )


class CorrectedCVCompanion(BaseModel):
    """A labelled, non-decisional corrected-CV sensitivity companion with two ratio conventions."""

    status: str = "sensitivity_only"
    assumption_warning: str = (
        "descriptive corrected-CV sensitivity analysis on paired per-fold-instance differences; "
        "folds are not i.i.d. (shared selectable pool, salted partitions of one landscape, one "
        "model per fold-instance); the train/test ratio is not naturally identified in this "
        "selection-then-training protocol, so two explicit conventions are reported instead of one "
        "authoritative ratio; neither is a frequentist CI over future wet-lab campaigns or proteins"
    )
    method_a: str
    method_b: str
    estimand: str
    regime: str
    statistic: str
    pool_ratio: SensitivityInterval
    effective_label_ratio: SensitivityInterval


# ------------------------------------------------------------------ raw record schema


class DeterministicFoldRecord(BaseModel):
    """One immutable raw record for a deterministic method at one (partition, fold, budget)."""

    model_config = {"frozen": True}

    protocol_version: str
    estimand: str
    missingness_regime: str
    partition_index: int
    partition_salt: str
    fold_index: int
    fold_identity_hash: str
    method: str
    budget: int
    random_seed: None = None

    selected_count: int
    selected_identity_hash: str
    selectable_pool_size: int
    revealed_count: int
    live_count: int
    dead_count: int
    missing_count: int
    unusable_count: int
    effective_train_size: int
    train_live_fraction: float | None

    selected_singles: int
    selected_doubles: int
    selected_triples: int
    train_singles: int
    train_doubles: int
    train_triples: int

    alpha_full: AlphaChoice
    alpha_main_only: AlphaChoice
    alpha_no_triples: AlphaChoice
    alpha_esm_offset: AlphaChoice

    n_eval: int
    s_macro: float | None
    rho_doubles: float | None
    rho_triples: float | None
    pooled_spearman: float | None
    pearson: float | None
    rmse: float | None

    ndcg: float | None
    hit_rate: float | None
    best_true_top_b: float | None
    regret: float | None
    live_fraction_top_b: float | None
    top_b_order_diversity: int | None
    top_b_identity_diversity: int | None

    uplift: float | None
    transfer_rho_triples: float | None
    transfer_train_singles: int
    transfer_train_doubles: int
    transfer_degenerate_double_coverage: bool

    esm_circular_s_macro: float | None
    esm_zero_shot_s_macro: float | None
    esm_offset_s_macro: float | None

    status: str
    warnings: list[str]


class RandomFoldSeedRecord(BaseModel):
    """One immutable raw record for the random baseline at one (partition, fold, budget, seed)."""

    model_config = {"frozen": True}

    protocol_version: str
    estimand: str
    missingness_regime: str
    partition_index: int
    partition_salt: str
    fold_index: int
    fold_identity_hash: str
    method: str = "random"
    budget: int
    random_seed: int

    selected_count: int
    selected_identity_hash: str
    selectable_pool_size: int
    revealed_count: int
    live_count: int
    dead_count: int
    missing_count: int
    unusable_count: int
    effective_train_size: int
    train_live_fraction: float | None

    selected_singles: int
    selected_doubles: int
    selected_triples: int
    train_singles: int
    train_doubles: int
    train_triples: int

    alpha_full: AlphaChoice
    alpha_main_only: AlphaChoice
    alpha_no_triples: AlphaChoice
    alpha_esm_offset: AlphaChoice

    n_eval: int
    s_macro: float | None
    rho_doubles: float | None
    rho_triples: float | None
    pooled_spearman: float | None
    pearson: float | None
    rmse: float | None

    ndcg: float | None
    hit_rate: float | None
    best_true_top_b: float | None
    regret: float | None
    live_fraction_top_b: float | None
    top_b_order_diversity: int | None
    top_b_identity_diversity: int | None

    uplift: float | None
    transfer_rho_triples: float | None
    transfer_train_singles: int
    transfer_train_doubles: int
    transfer_degenerate_double_coverage: bool

    esm_circular_s_macro: float | None
    esm_zero_shot_s_macro: float | None
    esm_offset_s_macro: float | None

    status: str
    warnings: list[str]


FoldRecord = DeterministicFoldRecord | RandomFoldSeedRecord


# ------------------------------------------------------------------ orchestration context

_AA20 = "ACDEFGHIKLMNPQRSTVWY"
_ESTIMANDS = ("target_blind", "target_aware")
_REGIMES = ("attempted_budget", "measured_available")
_METHODS = ("info", "structural", "fitness", "random", "practice")
_DETERMINISTIC_METHODS = ("info", "structural", "fitness", "practice")
_PRIMARY_ESTIMAND = "target_blind"
_PRIMARY_REGIME = "attempted_budget"
_REPORT_NOTE = (
    "post-registration downstream-impact benchmark, protocol amendment 1; escapes the map-recovery "
    "tautology (the primary learner never sees a held-out variant's ESM score and never calls "
    "infer_epistasis); the corrected-CV interval is a labelled sensitivity companion, not the "
    "primary gate; does not alter the frozen historical GB1 map-recovery decision rule"
)

_PAIR_SPECS: tuple[tuple[str, str, str], ...] = (
    ("structural", "fitness", "auc"),  # primary contrast
    ("info", "structural", "at_max"),  # mechanistic ablation
    ("structural", "random", "auc"),  # companion
    ("practice", "structural", "auc"),  # companion
)


# ------------------------------------------------------------ confirmatory protocol profile


class ConfirmatoryProfile(BaseModel):
    """The single authoritative frozen confirmatory-execution profile (protocol amendment 1).

    A report may only become ``decision_eligible`` when its executed configuration matches this
    profile exactly, in addition to the pre-existing partition-coverage requirement — 20 partitions
    of the *wrong* recipe (alphabet, budgets, ...) must never be mistaken for a confirmatory result.
    """

    model_config = {"frozen": True}

    protocol_version: str
    partitions: int
    outer_folds: int
    budgets: tuple[int, ...]  # order-sensitive: learning_curve_auc trapezoidal-integrates in order
    alphabet: str
    max_order: int
    n_perturbations: int  # scoring recipe: the cache's masking passes (0 disables the info prior)
    random_seeds: tuple[int, ...]
    inner_folds: int
    estimands: tuple[str, ...]
    missingness_regimes: tuple[str, ...]
    methods: tuple[str, ...]


CONFIRMATORY_PROFILE = ConfirmatoryProfile(
    protocol_version=PROTOCOL_VERSION,
    partitions=EXPECTED_PARTITIONS,
    outer_folds=5,
    budgets=(48, 96, 192),
    alphabet=_AA20,
    max_order=3,
    n_perturbations=16,
    random_seeds=tuple(range(20)),
    inner_folds=N_INNER_FOLDS,
    estimands=_ESTIMANDS,
    missingness_regimes=_REGIMES,
    methods=_METHODS,
)


class ProtocolProfileConformance(NamedTuple):
    """The result of comparing one executed configuration against :data:`CONFIRMATORY_PROFILE`."""

    conforming: bool
    expected: dict[str, object]
    observed: dict[str, object]
    mismatches: list[str]


def protocol_profile_conformance(
    *,
    protocol_version: str,
    partitions: int,
    outer_folds: int,
    budgets: Sequence[int],
    alphabet: str,
    max_order: int,
    n_perturbations: int,
    random_seeds: Sequence[int],
    inner_folds: int,
    estimands: Sequence[str],
    missingness_regimes: Sequence[str],
    methods: Sequence[str],
    profile: ConfirmatoryProfile = CONFIRMATORY_PROFILE,
) -> ProtocolProfileConformance:
    """Compare one executed configuration against the frozen confirmatory profile.

    ``budgets`` is compared by exact ordered-list equality (never coerced/sorted): the AUC
    contrast trapezoidal-integrates over ``budgets`` in the given order, so a reordering that
    preserves the set would silently change the AUC's meaning without this check. ``random_seeds``,
    ``estimands``, ``missingness_regimes``, and ``methods`` are compared as sets — their frozen
    registered universe, not an emission order. No value is ever coerced toward the profile; a
    mismatch is always reported, never silently accepted.
    """
    observed: dict[str, object] = {
        "protocol_version": protocol_version,
        "partitions": partitions,
        "outer_folds": outer_folds,
        "budgets": list(budgets),
        "alphabet": alphabet,
        "max_order": max_order,
        "n_perturbations": n_perturbations,
        "random_seeds": sorted(set(random_seeds)),
        "inner_folds": inner_folds,
        "estimands": sorted(set(estimands)),
        "missingness_regimes": sorted(set(missingness_regimes)),
        "methods": sorted(set(methods)),
    }
    expected: dict[str, object] = {
        "protocol_version": profile.protocol_version,
        "partitions": profile.partitions,
        "outer_folds": profile.outer_folds,
        "budgets": list(profile.budgets),
        "alphabet": profile.alphabet,
        "max_order": profile.max_order,
        "n_perturbations": profile.n_perturbations,
        "random_seeds": sorted(profile.random_seeds),
        "inner_folds": profile.inner_folds,
        "estimands": sorted(profile.estimands),
        "missingness_regimes": sorted(profile.missingness_regimes),
        "methods": sorted(profile.methods),
    }
    mismatches = sorted(key for key in expected if observed[key] != expected[key])
    return ProtocolProfileConformance(not mismatches, expected, observed, mismatches)


@dataclass
class _FoldContext:
    """Everything about one held-out fold that is shared across methods, budgets, and estimands."""

    eval_variants: list[Variant]
    raw_fitness: FloatArray
    log_fitness: FloatArray
    relevance: FloatArray
    eval_ids: list[str]
    esm_score: FloatArray
    doubles_idx: list[int]
    triples_idx: list[int]
    active_full_idx: IntArray  # (n_eval, width) padded column gather, -1 = padding
    active_main_idx: IntArray
    triple_full_idx: IntArray
    fold_identity_hash: str


@dataclass
class _RawBundle:
    """Everything :func:`_evaluate_selection` computes for one (method/seed, fold, budget) cell."""

    selected_count: int
    selected_identity_hash: str
    selectable_pool_size: int
    revealed_count: int
    live_count: int
    dead_count: int
    missing_count: int
    unusable_count: int
    effective_train_size: int
    train_live_fraction: float | None
    selected_singles: int
    selected_doubles: int
    selected_triples: int
    train_singles: int
    train_doubles: int
    train_triples: int
    alpha_full: AlphaChoice
    alpha_main_only: AlphaChoice
    alpha_no_triples: AlphaChoice
    alpha_esm_offset: AlphaChoice
    n_eval: int
    s_macro: float | None
    rho_doubles: float | None
    rho_triples: float | None
    pooled_spearman: float | None
    pearson: float | None
    rmse: float | None
    ndcg: float | None
    hit_rate: float | None
    best_true_top_b: float | None
    regret: float | None
    live_fraction_top_b: float | None
    top_b_order_diversity: int | None
    top_b_identity_diversity: int | None
    uplift: float | None
    transfer_rho_triples: float | None
    transfer_train_singles: int
    transfer_train_doubles: int
    transfer_degenerate_double_coverage: bool
    esm_circular_s_macro: float | None
    esm_zero_shot_s_macro: float | None
    esm_offset_s_macro: float | None
    status: str
    warnings: list[str] = field(default_factory=list)


def _pad_active(active: Sequence[Sequence[int]], width: int) -> IntArray:
    """Pack variable-length active-column lists into an (n, width) int array (-1 = padding)."""
    idx = np.full((len(active), max(width, 1)), -1, dtype=np.int64)
    for row, cols in enumerate(active):
        if cols:
            idx[row, : len(cols)] = cols
    return idx


def _predict_padded(model: RidgeModel, idx: IntArray) -> FloatArray:
    """Vectorized sparse prediction: intercept + sum of coef over active columns (-1 gathers 0)."""
    coef_padded = np.append(model.coef, 0.0)
    gathered: FloatArray = np.asarray(coef_padded[idx], dtype=np.float64)
    out: FloatArray = model.intercept + gathered.sum(axis=1)
    return out


def _corr_subset(pred: FloatArray, true: FloatArray, idx: Sequence[int]) -> float | None:
    if len(idx) < _MIN_POINTS_FOR_CORR:
        return None
    return _corr(pred[list(idx)], true[list(idx)], "spearman")


def _order_counts(variants: Sequence[Variant]) -> tuple[int, int, int]:
    singles = sum(1 for v in variants if len(v) == 1)
    doubles = sum(1 for v in variants if len(v) == _PAIRWISE_ORDER)
    triples = sum(1 for v in variants if len(v) == _THIRD_ORDER)
    return singles, doubles, triples


def _ranked_top(pred: FloatArray, ids: Sequence[str], budget: int) -> list[int]:
    n = len(pred)
    order = sorted(range(n), key=lambda i: (-float(pred[i]), ids[i]))
    return order[: min(budget, n)]


def _hit_rate(pred: FloatArray, raw: FloatArray, ids: Sequence[str], budget: int) -> float:
    top_pred = set(_ranked_top(pred, ids, budget))
    top_true = set(_ranked_top(raw, ids, budget))
    return len(top_pred & top_true) / max(min(budget, len(pred)), 1)


def _best_true_top_b(pred: FloatArray, raw: FloatArray, ids: Sequence[str], budget: int) -> float:
    return max(float(raw[i]) for i in _ranked_top(pred, ids, budget))


def _regret(pred: FloatArray, raw: FloatArray, ids: Sequence[str], budget: int) -> float:
    return float(raw.max()) - _best_true_top_b(pred, raw, ids, budget)


def _top_live_fraction(pred: FloatArray, raw: FloatArray, ids: Sequence[str], budget: int) -> float:
    top = _ranked_top(pred, ids, budget)
    return float(np.mean([raw[i] > 0.0 for i in top]))


def _top_diversity(
    pred: FloatArray, ids: Sequence[str], variants: Sequence[Variant], budget: int
) -> tuple[int, int]:
    top = _ranked_top(pred, ids, budget)
    orders = {len(variants[i]) for i in top}
    identities = {ids[i] for i in top}
    return len(orders), len(identities)


def _evaluate_selection(
    space: FeatureSpace,
    ctx: _FoldContext,
    esm_of: Mapping[Variant, float],
    selected: Sequence[Variant],
    revealed: Mapping[Variant, float],
    budget: int,
    pool_size: int,
    grid_main: Sequence[float],
    grid_pair: Sequence[float],
    n_inner: int,
) -> _RawBundle:
    """Fit all four regime models on a selection's revealed labels; score them on the held-out fold.

    Returns the immutable bundle a raw fold record is built from.
    """
    live_ids = {v for v, f in revealed.items() if isfinite(f) and f > 0.0}
    dead_ids = {v for v, f in revealed.items() if isfinite(f) and f == 0.0}
    unusable_ids = {v for v, f in revealed.items() if not isfinite(f)}
    train_variants = sorted(live_ids | dead_ids, key=canonical_id)
    train_fit = np.array([revealed[v] for v in train_variants], dtype=np.float64)
    n_missing = len(selected) - len(revealed)
    s_singles, s_doubles, s_triples = _order_counts(selected)
    t_singles, t_doubles, t_triples = _order_counts(train_variants)
    live_fraction = float(len(live_ids)) / len(train_variants) if train_variants else None
    warns: list[str] = []

    empty_alpha = AlphaChoice(
        alpha_main=max(grid_main),
        alpha_pair=None,
        applicable=False,
        fell_back=True,
        fallback_reason="training_set_too_small",
        n_inner_folds_used=0,
    )
    empty = _RawBundle(
        selected_count=len(selected),
        selected_identity_hash=candidate_sha256(selected),
        selectable_pool_size=pool_size,
        revealed_count=len(revealed),
        live_count=len(live_ids),
        dead_count=len(dead_ids),
        missing_count=n_missing,
        unusable_count=len(unusable_ids),
        effective_train_size=len(train_variants),
        train_live_fraction=live_fraction,
        selected_singles=s_singles,
        selected_doubles=s_doubles,
        selected_triples=s_triples,
        train_singles=t_singles,
        train_doubles=t_doubles,
        train_triples=t_triples,
        alpha_full=empty_alpha,
        alpha_main_only=empty_alpha,
        alpha_no_triples=empty_alpha,
        alpha_esm_offset=empty_alpha,
        n_eval=len(ctx.eval_variants),
        s_macro=None,
        rho_doubles=None,
        rho_triples=None,
        pooled_spearman=None,
        pearson=None,
        rmse=None,
        ndcg=None,
        hit_rate=None,
        best_true_top_b=None,
        regret=None,
        live_fraction_top_b=None,
        top_b_order_diversity=None,
        top_b_identity_diversity=None,
        uplift=None,
        transfer_rho_triples=None,
        transfer_train_singles=0,
        transfer_train_doubles=0,
        transfer_degenerate_double_coverage=True,
        esm_circular_s_macro=_esm_circular(ctx, esm_of, revealed),
        esm_zero_shot_s_macro=_esm_zero_shot(ctx),
        esm_offset_s_macro=None,
        status="empty_training",
        warnings=["effective_train_size < 3: no model fit"],
    )
    if len(train_variants) < _MIN_POINTS_FOR_CORR:
        return empty

    y = np.log1p(train_fit)

    alpha_full = select_alpha(space, train_variants, y, grid_main, grid_pair, n_inner, _INNER_SALT)
    alpha_full_pair = alpha_full.alpha_pair if alpha_full.alpha_pair is not None else max(grid_pair)
    pen_full = space.penalties(alpha_full.alpha_main, alpha_full_pair)
    model_full = fit_ridge(space.design_matrix(train_variants), y, pen_full)
    pred = _predict_padded(model_full, ctx.active_full_idx)

    alpha_main_only = select_alpha_main_only(
        space, train_variants, y, grid_main, n_inner, _INNER_SALT
    )
    pen_main = space.penalties(alpha_main_only.alpha_main, max(grid_pair))
    model_main = fit_ridge(space.design_matrix(train_variants, include_pairs=False), y, pen_main)
    pred_main = _predict_padded(model_main, ctx.active_main_idx)

    rho_d = _corr_subset(pred, ctx.raw_fitness, ctx.doubles_idx)
    rho_t = _corr_subset(pred, ctx.raw_fitness, ctx.triples_idx)
    s_macro = macro_spearman(rho_d, rho_t)
    s_macro_main = macro_spearman(
        _corr_subset(pred_main, ctx.raw_fitness, ctx.doubles_idx),
        _corr_subset(pred_main, ctx.raw_fitness, ctx.triples_idx),
    )
    uplift = None if (s_macro is None or s_macro_main is None) else s_macro - s_macro_main

    alpha_no_triples, transfer_rho, tr_singles, tr_doubles, degenerate = _transfer_triples(
        space, ctx, revealed, grid_main, grid_pair, n_inner
    )
    if degenerate:
        warns.append("no_triples_transfer: degenerate_double_coverage")

    alpha_esm_offset, esm_offset_s_macro = _esm_offset_supervised(
        space, ctx, esm_of, train_variants, train_fit, grid_main, grid_pair, n_inner
    )
    esm_circular_s_macro = _esm_circular(ctx, esm_of, revealed)

    order_div, identity_div = _top_diversity(pred, ctx.eval_ids, ctx.eval_variants, budget)

    return _RawBundle(
        selected_count=len(selected),
        selected_identity_hash=candidate_sha256(selected),
        selectable_pool_size=pool_size,
        revealed_count=len(revealed),
        live_count=len(live_ids),
        dead_count=len(dead_ids),
        missing_count=n_missing,
        unusable_count=len(unusable_ids),
        effective_train_size=len(train_variants),
        train_live_fraction=live_fraction,
        selected_singles=s_singles,
        selected_doubles=s_doubles,
        selected_triples=s_triples,
        train_singles=t_singles,
        train_doubles=t_doubles,
        train_triples=t_triples,
        alpha_full=alpha_full,
        alpha_main_only=alpha_main_only,
        alpha_no_triples=alpha_no_triples,
        alpha_esm_offset=alpha_esm_offset,
        n_eval=len(ctx.eval_variants),
        s_macro=s_macro,
        rho_doubles=rho_d,
        rho_triples=rho_t,
        pooled_spearman=_corr(pred, ctx.raw_fitness, "spearman"),
        pearson=_corr(pred, ctx.log_fitness, "pearson"),
        rmse=float(np.sqrt(np.mean((pred - ctx.log_fitness) ** 2))),
        ndcg=ndcg_at_k(pred, ctx.relevance, budget, ctx.eval_ids),
        hit_rate=_hit_rate(pred, ctx.raw_fitness, ctx.eval_ids, budget),
        best_true_top_b=_best_true_top_b(pred, ctx.raw_fitness, ctx.eval_ids, budget),
        regret=_regret(pred, ctx.raw_fitness, ctx.eval_ids, budget),
        live_fraction_top_b=_top_live_fraction(pred, ctx.raw_fitness, ctx.eval_ids, budget),
        top_b_order_diversity=order_div,
        top_b_identity_diversity=identity_div,
        uplift=uplift,
        transfer_rho_triples=transfer_rho,
        transfer_train_singles=tr_singles,
        transfer_train_doubles=tr_doubles,
        transfer_degenerate_double_coverage=degenerate,
        esm_circular_s_macro=esm_circular_s_macro,
        esm_zero_shot_s_macro=_esm_zero_shot(ctx),
        esm_offset_s_macro=esm_offset_s_macro,
        status="ok",
        warnings=warns,
    )


def _transfer_triples(
    space: FeatureSpace,
    ctx: _FoldContext,
    revealed: Mapping[Variant, float],
    grid_main: Sequence[float],
    grid_pair: Sequence[float],
    n_inner: int,
) -> tuple[AlphaChoice, float | None, int, int, bool]:
    """No-triples training -> held-out-triples eval, with its own independent inner-CV alpha.

    Keeps the method's selected singles and doubles that were revealed; every selected triple is
    excluded from this sub-test's training rows (never used to predict held-out triples).
    """
    train = sorted(
        (v for v, f in revealed.items() if isfinite(f) and len(v) <= _PAIRWISE_ORDER),
        key=canonical_id,
    )
    n_singles, n_doubles, _n_triples = _order_counts(train)
    degenerate = n_doubles < _MIN_POINTS_FOR_CORR
    if len(train) < _MIN_POINTS_FOR_CORR or len(ctx.triples_idx) < _MIN_POINTS_FOR_CORR:
        fallback = AlphaChoice(
            alpha_main=max(grid_main),
            alpha_pair=max(grid_pair),
            fell_back=True,
            fallback_reason="training_set_too_small",
            n_inner_folds_used=0,
        )
        return fallback, None, n_singles, n_doubles, True
    y = np.log1p(np.array([revealed[v] for v in train], dtype=np.float64))
    alpha = select_alpha(space, train, y, grid_main, grid_pair, n_inner, _INNER_SALT)
    alpha_pair = alpha.alpha_pair if alpha.alpha_pair is not None else max(grid_pair)
    penalties = space.penalties(alpha.alpha_main, alpha_pair)
    model = fit_ridge(space.design_matrix(train), y, penalties)
    pred = _predict_padded(model, ctx.triple_full_idx)
    true = ctx.raw_fitness[ctx.triples_idx]
    return alpha, _corr(pred, true, "spearman"), n_singles, n_doubles, degenerate


def _esm_offset_supervised(
    space: FeatureSpace,
    ctx: _FoldContext,
    esm_of: Mapping[Variant, float],
    train_variants: Sequence[Variant],
    train_fit: FloatArray,
    grid_main: Sequence[float],
    grid_pair: Sequence[float],
    n_inner: int,
) -> tuple[AlphaChoice, float | None]:
    """ESM-offset supervised ridge: fit ``log1p(fit) - b*esm`` then add the offset back at predict.

    Uses the same method-selected revealed training labels as the clean full model. ``b`` and alpha
    are chosen by :func:`select_alpha_esm_offset`'s nested inner CV, which refits ``b_inner`` from
    each inner fold's training rows only — never leaking an inner-validation label into the offset
    it is later scored against; the final ``b`` and residual model are then refit once on all
    outer-training labels. Never decisional — reported alongside the clean ``s_macro`` as
    a diagnostic only. ``esm_of`` is the full-universe ESM map so every training variant's score is
    available, not only the fold's held-out set.
    """
    if len(train_variants) < _MIN_POINTS_FOR_CORR:
        fallback = AlphaChoice(
            alpha_main=max(grid_main),
            alpha_pair=max(grid_pair),
            fell_back=True,
            fallback_reason="training_set_too_small",
            n_inner_folds_used=0,
        )
        return fallback, None
    train_esm = np.array([esm_of[v] for v in train_variants], dtype=np.float64)
    y = np.log1p(train_fit)
    alpha = select_alpha_esm_offset(
        space, train_variants, y, train_esm, grid_main, grid_pair, n_inner, _INNER_SALT
    )
    b_final = alpha.b if alpha.b is not None else _through_origin_slope(train_esm, y)
    offset_y = y - b_final * train_esm
    alpha_pair = alpha.alpha_pair if alpha.alpha_pair is not None else max(grid_pair)
    penalties = space.penalties(alpha.alpha_main, alpha_pair)
    model = fit_ridge(space.design_matrix(train_variants), offset_y, penalties)
    pred_offset = _predict_padded(model, ctx.active_full_idx)
    pred = pred_offset + b_final * ctx.esm_score
    rho_d = _corr_subset(pred, ctx.raw_fitness, ctx.doubles_idx)
    rho_t = _corr_subset(pred, ctx.raw_fitness, ctx.triples_idx)
    return alpha, macro_spearman(rho_d, rho_t)


def _esm_circular(
    ctx: _FoldContext, esm_of: Mapping[Variant, float], revealed: Mapping[Variant, float]
) -> float | None:
    """ESM-circular diagnostic: the posterior-mean prior collapses to ``b*esm[v]`` for a held-out v.

    Reuses :func:`epibudget.validate.esm_prior_mu` — the exact mechanism ``infer_epistasis``
    conditions on — over the fold's revealed labels, demonstrating the tautology the primary
    predictor avoids.
    ``ctx.eval_variants`` and ``revealed`` are disjoint by construction (E_j vs pool_j), so every
    predicted value here is exactly ``b*esm[v]``, never a pinned measured value.
    """
    # TODO: use a downstream-specific calibration scale; log1p labels do not satisfy esm_prior_mu's WT-centered log-fitness contract.  # noqa: E501
    calibration = {v: float(np.log1p(f)) for v, f in revealed.items() if isfinite(f)}
    if not calibration:
        return None
    identities = set(ctx.eval_variants) | set(calibration)
    scored_for_mu = [
        ScoredVariant(variant=v, delta_g=esm_of[v], var_delta_g=0.0) for v in identities
    ]
    mu = esm_prior_mu(scored_for_mu, calibration)
    pred = np.array([mu[v] for v in ctx.eval_variants], dtype=np.float64)
    rho_d = _corr_subset(pred, ctx.raw_fitness, ctx.doubles_idx)
    rho_t = _corr_subset(pred, ctx.raw_fitness, ctx.triples_idx)
    return macro_spearman(rho_d, rho_t)


def _esm_zero_shot(ctx: _FoldContext) -> float | None:
    """No-budget control: raw zero-shot ESM ``delta_g`` ranking over the fold's held-out set."""
    rho_d = _corr_subset(ctx.esm_score, ctx.raw_fitness, ctx.doubles_idx)
    rho_t = _corr_subset(ctx.esm_score, ctx.raw_fitness, ctx.triples_idx)
    return macro_spearman(rho_d, rho_t)


def _build_fold_context(
    space: FeatureSpace,
    eval_measured: list[Variant],
    landscape: Mapping[Variant, float],
    active_full: Mapping[Variant, list[int]],
    active_main: Mapping[Variant, list[int]],
    esm_of: Mapping[Variant, float],
) -> _FoldContext:
    raw = np.array([landscape[v] for v in eval_measured], dtype=np.float64)
    doubles_idx = [i for i, v in enumerate(eval_measured) if len(v) == _PAIRWISE_ORDER]
    triples_idx = [i for i, v in enumerate(eval_measured) if len(v) == _THIRD_ORDER]
    full = [active_full[v] for v in eval_measured]
    main = [active_main[v] for v in eval_measured]
    ids = [canonical_id(v) for v in eval_measured]
    return _FoldContext(
        eval_variants=eval_measured,
        raw_fitness=raw,
        log_fitness=np.log1p(raw),
        relevance=percentile_relevance(raw),
        eval_ids=ids,
        esm_score=np.array([esm_of[v] for v in eval_measured], dtype=np.float64),
        doubles_idx=doubles_idx,
        triples_idx=triples_idx,
        active_full_idx=_pad_active(full, max((len(c) for c in full), default=1)),
        active_main_idx=_pad_active(main, max((len(c) for c in main), default=1)),
        triple_full_idx=_pad_active(
            [full[i] for i in triples_idx], max((len(c) for c in full), default=1)
        ),
        fold_identity_hash=candidate_sha256(eval_measured),
    )


# ------------------------------------------------------------------ report models


class MethodBudgetSummary(BaseModel):
    """Mean over fold-instances of one method's downstream metrics at one budget/estimand/regime.

    A pure function of the raw records (never a second, independently-computed aggregation path).
    """

    method: str
    estimand: str
    regime: str
    budget: int
    n_instances: int
    s_macro: float | None
    rho_doubles: float | None
    rho_triples: float | None
    pooled_spearman: float | None
    pearson: float | None
    rmse: float | None
    ndcg: float | None
    hit_rate: float | None
    regret: float | None
    live_fraction_top_b: float | None
    uplift: float | None
    transfer_rho_triples: float | None
    effective_train_size: float | None
    train_live_fraction: float | None
    esm_circular_s_macro: float | None
    esm_zero_shot_s_macro: float | None
    esm_offset_s_macro: float | None


class PartitionAggregate(BaseModel):
    """The mean paired fold-level delta of one contrast within one partition (pure over records)."""

    estimand: str
    regime: str
    method_a: str
    method_b: str
    statistic: str
    partition_index: int
    mean_delta: float | None
    n_valid_folds: int
    n_total_folds: int


class RobustnessGate(BaseModel):
    """The frozen 7-point partition-level robustness gate for one contrast (amendment 1)."""

    estimand: str
    regime: str
    method_a: str
    method_b: str
    statistic: str
    expected_partitions: int
    observed_valid_partitions: int
    complete_partition_coverage: bool
    sign_positive: int
    sign_threshold: int
    sign_pass: bool
    global_mean_delta: float | None
    global_mean_positive: bool
    median_partition_delta: float | None
    median_positive: bool
    min_effect_size: float
    effect_size_pass: bool
    decision_eligible: bool
    supported: bool | None
    status: str


class RawRecordCoverage(BaseModel):
    """Single source of truth for raw-record protocol coverage.

    Independently reconstructs the expected ``(protocol_version, partition, estimand, regime,
    method, budget, fold[, seed])`` cells from the declared protocol profile and the known
    production generation semantics (never from the observed records themselves, which would make
    coverage circular and unable to detect anything missing), then compares them against the raw
    ``deterministic_records``/``random_records`` with multiplicity (:class:`collections.Counter`,
    not a bare set) so a duplicated cell is detectable even when the unique-key set is unchanged. A
    record whose own ``protocol_version`` differs is automatically both a missing cell (its
    correct-version cell has zero observations) and an unexpected one (its own key is outside the
    expected set), since ``protocol_version`` is part of the identity key.
    """

    model_config = {"frozen": True}

    expected_deterministic_count: int
    observed_deterministic_count: int
    expected_random_count: int
    observed_random_count: int
    missing_deterministic_cell_count: int
    duplicate_deterministic_cell_count: int
    unexpected_deterministic_cell_count: int
    missing_random_cell_count: int
    duplicate_random_cell_count: int
    unexpected_random_cell_count: int
    # A duplicated key is exact (every copy byte-identical) or divergent (payloads
    # differ). ``duplicate_*_cell_count`` above is preserved for backward auditability and always
    # equals ``exact_* + divergent_*``. Only a divergent duplicate is scientifically ambiguous.
    exact_duplicate_deterministic_key_count: int
    divergent_duplicate_deterministic_key_count: int
    exact_duplicate_random_key_count: int
    divergent_duplicate_random_key_count: int
    missing_deterministic_cell_samples: list[str]
    duplicate_deterministic_cell_samples: list[str]
    unexpected_deterministic_cell_samples: list[str]
    missing_random_cell_samples: list[str]
    duplicate_random_cell_samples: list[str]
    unexpected_random_cell_samples: list[str]
    divergent_duplicate_deterministic_key_samples: list[str]
    divergent_duplicate_random_key_samples: list[str]
    observed_protocol_versions: list[str]
    observed_partition_indices: list[int]
    observed_random_seeds: list[int]
    has_divergent_duplicate: bool
    conforming: bool


class DecisionSummary(BaseModel):
    """The registered downstream decision, computed exclusively from the raw records."""

    protocol_version: str
    amendment_version: str
    primary_estimand: str
    primary_regime: str
    expected_protocol_profile: dict[str, object]
    observed_protocol_profile: dict[str, object]
    protocol_profile_mismatches: list[str]
    declared_protocol_profile_conforming: bool
    raw_record_coverage: RawRecordCoverage
    raw_record_coverage_conforming: bool
    protocol_profile_conforming: bool
    structural_gate: RobustnessGate
    esm_gate: RobustnessGate
    structural_downstream_supported: bool | None
    esm_uncertainty_supported: bool | None
    rule: str


class DownstreamReport(BaseModel):
    """The full downstream-impact result: raw records, pure aggregates, and the decision."""

    protocol_version: str
    amendment_version: str
    dataset: str
    model_id: str
    alphabet: str
    max_order: int
    budgets: list[int]
    seeds: int
    n_folds: int
    partitions: int
    n_candidates: int
    n_eval_universe: int
    grid_main: list[float]
    grid_pair: list[float]
    n_inner_folds: int
    note: str
    provenance: dict[str, object]
    deterministic_records: list[DeterministicFoldRecord]
    random_records: list[RandomFoldSeedRecord]
    # False iff a divergent duplicate raw record made every registered scientific
    # summary ambiguous. When false, ``method_budget``/``partition_aggregates``/
    # ``corrected_cv_companions`` are empty and both decision gates are descriptive-free; the raw
    # records and ``decision.raw_record_coverage`` diagnostics remain for forensic auditing.
    scientific_summaries_available: bool = True
    scientific_summaries_unavailable_reason: str | None = None
    method_budget: list[MethodBudgetSummary]
    partition_aggregates: list[PartitionAggregate]
    corrected_cv_companions: list[CorrectedCVCompanion]
    decision: DecisionSummary


# ------------------------------------------------------------------ pure aggregation over records


def _mean_opt(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else None


def method_budget_summaries(records: Sequence[FoldRecord]) -> list[MethodBudgetSummary]:
    """Per-(method, estimand, regime, budget) means, computed exclusively from raw records."""
    grouped: dict[tuple[str, str, str, int], list[FoldRecord]] = defaultdict(list)
    for r in records:
        grouped[(r.method, r.estimand, r.missingness_regime, r.budget)].append(r)
    out: list[MethodBudgetSummary] = []
    for (method, estimand, regime, budget), rs in sorted(grouped.items()):
        out.append(
            MethodBudgetSummary(
                method=method,
                estimand=estimand,
                regime=regime,
                budget=budget,
                n_instances=len(rs),
                s_macro=_mean_opt([r.s_macro for r in rs]),
                rho_doubles=_mean_opt([r.rho_doubles for r in rs]),
                rho_triples=_mean_opt([r.rho_triples for r in rs]),
                pooled_spearman=_mean_opt([r.pooled_spearman for r in rs]),
                pearson=_mean_opt([r.pearson for r in rs]),
                rmse=_mean_opt([r.rmse for r in rs]),
                ndcg=_mean_opt([r.ndcg for r in rs]),
                hit_rate=_mean_opt([r.hit_rate for r in rs]),
                regret=_mean_opt([r.regret for r in rs]),
                live_fraction_top_b=_mean_opt([r.live_fraction_top_b for r in rs]),
                uplift=_mean_opt([r.uplift for r in rs]),
                transfer_rho_triples=_mean_opt([r.transfer_rho_triples for r in rs]),
                effective_train_size=_mean_opt([float(r.effective_train_size) for r in rs]),
                train_live_fraction=_mean_opt([r.train_live_fraction for r in rs]),
                esm_circular_s_macro=_mean_opt([r.esm_circular_s_macro for r in rs]),
                esm_zero_shot_s_macro=_mean_opt([r.esm_zero_shot_s_macro for r in rs]),
                esm_offset_s_macro=_mean_opt([r.esm_offset_s_macro for r in rs]),
            )
        )
    return out


def _s_macro_cells(
    records: Sequence[FoldRecord], estimand: str, regime: str, method: str, budget: int
) -> dict[tuple[int, int], float | None]:
    """Mean ``s_macro`` per (partition, fold) for one method/budget (averages seeds if random)."""
    by_cell: dict[tuple[int, int], list[float | None]] = defaultdict(list)
    for r in records:
        if (r.estimand, r.missingness_regime, r.method, r.budget) == (
            estimand,
            regime,
            method,
            budget,
        ):
            by_cell[(r.partition_index, r.fold_index)].append(r.s_macro)
    return {cell: _mean_opt(values) for cell, values in by_cell.items()}


def _paired_cell_deltas(
    records: Sequence[FoldRecord],
    estimand: str,
    regime: str,
    method_a: str,
    method_b: str,
    mode: str,
    budgets: Sequence[int],
    max_budget: int,
) -> dict[tuple[int, int], float | None]:
    if mode == "auc":
        by_budget_a = {b: _s_macro_cells(records, estimand, regime, method_a, b) for b in budgets}
        by_budget_b = {b: _s_macro_cells(records, estimand, regime, method_b, b) for b in budgets}
        all_cells = set().union(*(set(c) for c in by_budget_a.values()))
        auc_out: dict[tuple[int, int], float | None] = {}
        for cell in all_cells:
            auc_a = learning_curve_auc([by_budget_a[b].get(cell) for b in budgets])
            auc_b = learning_curve_auc([by_budget_b[b].get(cell) for b in budgets])
            auc_out[cell] = None if (auc_a is None or auc_b is None) else auc_a - auc_b
        return auc_out
    at_max_a = _s_macro_cells(records, estimand, regime, method_a, max_budget)
    at_max_b = _s_macro_cells(records, estimand, regime, method_b, max_budget)
    at_max_out: dict[tuple[int, int], float | None] = {}
    for cell in set(at_max_a) | set(at_max_b):
        va, vb = at_max_a.get(cell), at_max_b.get(cell)
        at_max_out[cell] = None if (va is None or vb is None) else va - vb
    return at_max_out


def _global_deltas_within_expected_partitions(
    deltas_by_cell: Mapping[tuple[int, int], float | None], expected_partitions: int
) -> list[float | None]:
    """Fold-instance deltas restricted to the frozen ``{0, ..., expected_partitions-1}`` register.

    ``partition_aggregates_for`` already only ever builds a :class:`PartitionAggregate` for
    ``p in range(expected_partitions)``; a raw ``_paired_cell_deltas(...).values()`` list has no
    such bound. Without this filter, a run with ``partitions > expected_partitions`` would let the
    extra partitions' deltas move ``robustness_gate``'s global mean/median while being correctly
    excluded from its sign count — extra partitions must never influence any registered decision
    quantity, full stop.
    """
    return [
        v for (partition, _fold), v in deltas_by_cell.items() if partition < expected_partitions
    ]


def partition_aggregates_for(
    records: Sequence[FoldRecord],
    estimand: str,
    regime: str,
    method_a: str,
    method_b: str,
    statistic: str,
    mode: str,
    budgets: Sequence[int],
    n_folds: int,
    expected_partitions: int = EXPECTED_PARTITIONS,
) -> list[PartitionAggregate]:
    """One :class:`PartitionAggregate` per partition index in ``0..expected_partitions-1``."""
    deltas = _paired_cell_deltas(
        records, estimand, regime, method_a, method_b, mode, budgets, max(budgets)
    )
    by_partition: dict[int, list[float]] = defaultdict(list)
    for (partition, _fold), delta in deltas.items():
        if delta is not None:
            by_partition[partition].append(delta)
    out: list[PartitionAggregate] = []
    for p in range(expected_partitions):
        values = by_partition.get(p, [])
        out.append(
            PartitionAggregate(
                estimand=estimand,
                regime=regime,
                method_a=method_a,
                method_b=method_b,
                statistic=statistic,
                partition_index=p,
                mean_delta=float(np.mean(values)) if values else None,
                n_valid_folds=len(values),
                n_total_folds=n_folds,
            )
        )
    return out


def robustness_gate(
    aggregates: Sequence[PartitionAggregate],
    all_deltas: Sequence[float | None],
    *,
    expected_partitions: int = EXPECTED_PARTITIONS,
    sign_threshold: int = SIGN_THRESHOLD,
    min_effect_size: float = MIN_STRUCTURAL_EFFECT_SIZE,
) -> RobustnessGate:
    """The frozen 7-point partition-level robustness gate, computed purely from partition aggregates
    (the raw records already reduced to per-partition means).

    Fails closed: any of the ``expected_partitions`` missing or entirely degenerate (no valid fold)
    makes the gate ``decision_eligible=False`` with ``status="insufficient_valid_partitions"`` —
    never a reduced denominator.
    """
    valid = [a for a in aggregates if a.mean_delta is not None]
    observed = len(valid)
    complete = observed == expected_partitions and {a.partition_index for a in valid} == set(
        range(expected_partitions)
    )
    sign_positive = sum(1 for a in valid if a.mean_delta is not None and a.mean_delta > 0.0)
    sign_pass = sign_positive >= sign_threshold
    global_deltas = [float(d) for d in all_deltas if d is not None]
    global_mean = float(np.mean(global_deltas)) if global_deltas else None
    global_positive = global_mean is not None and global_mean > 0.0
    partition_means = [float(a.mean_delta) for a in valid if a.mean_delta is not None]
    median_delta = float(np.median(partition_means)) if partition_means else None
    median_positive = median_delta is not None and median_delta > 0.0
    effect_pass = global_mean is not None and global_mean > min_effect_size
    eligible = complete
    if not eligible:
        supported: bool | None = None
        status = "insufficient_valid_partitions"
    else:
        supported = bool(sign_pass and global_positive and median_positive and effect_pass)
        status = "ok"
    return RobustnessGate(
        estimand=aggregates[0].estimand if aggregates else "",
        regime=aggregates[0].regime if aggregates else "",
        method_a=aggregates[0].method_a if aggregates else "",
        method_b=aggregates[0].method_b if aggregates else "",
        statistic=aggregates[0].statistic if aggregates else "",
        expected_partitions=expected_partitions,
        observed_valid_partitions=observed,
        complete_partition_coverage=complete,
        sign_positive=sign_positive,
        sign_threshold=sign_threshold,
        sign_pass=sign_pass,
        global_mean_delta=global_mean,
        global_mean_positive=global_positive,
        median_partition_delta=median_delta,
        median_positive=median_positive,
        min_effect_size=min_effect_size,
        effect_size_pass=effect_pass,
        decision_eligible=eligible,
        supported=supported,
        status=status,
    )


def _corrected_cv_companion(
    records: Sequence[FoldRecord],
    estimand: str,
    regime: str,
    method_a: str,
    method_b: str,
    statistic: str,
    mode: str,
    budgets: Sequence[int],
) -> CorrectedCVCompanion:
    """Effective-size convention. The ``n_train``/``n_test`` size means feed
    only the SAME budget(s) the paired-delta statistic itself was computed over, never a silent
    "average every budget" default:

    - ``mode == "at_max"`` (currently only info-structural): the paired delta is ``s_macro`` at a
      single budget (``max(budgets)``), so its sizes come from that budget's records ONLY — mixing
      in the other budgets' sizes would report a ratio for training sizes the contrast never used.
    - ``mode == "auc"`` (structural-fitness, structural-random, practice-structural): the paired
      delta is the trapezoidal AUC of ``s_macro`` over EVERY budget in ``budgets``, so averaging
      sizes across all of those same budgets is the convention that matches what the AUC statistic
      itself aggregates — an explicit, intentional choice, not an accidental reuse of the
      ``at_max``-only logic (or vice versa).
    """
    deltas_by_cell = _paired_cell_deltas(
        records, estimand, regime, method_a, method_b, mode, budgets, max(budgets)
    )
    deltas = list(deltas_by_cell.values())
    contrast_budgets: tuple[int, ...] = (max(budgets),) if mode == "at_max" else tuple(budgets)

    def _mean_sizes(kind: str, size_budgets: Sequence[int]) -> tuple[float | None, float | None]:
        n_test_vals: list[float] = []
        n_train_vals: list[float] = []
        for r in records:
            if r.estimand != estimand or r.missingness_regime != regime:
                continue
            if r.method not in (method_a, method_b):
                continue
            if r.budget not in size_budgets:
                continue
            n_train = float(
                r.selectable_pool_size if kind == "pool_ratio" else r.effective_train_size
            )
            if n_train > 0.0:
                n_test_vals.append(float(r.n_eval))
                n_train_vals.append(n_train)
        if not n_test_vals:
            return None, None
        return float(np.mean(n_test_vals)), float(np.mean(n_train_vals))

    pool_n_test, pool_n_train = _mean_sizes("pool_ratio", contrast_budgets)
    pool_ratio = _corrected_cv_formula(deltas, pool_n_test, pool_n_train, "pool_ratio")
    label_n_test, label_n_train = _mean_sizes("effective_label_ratio", contrast_budgets)
    effective_label_ratio = _corrected_cv_formula(
        deltas, label_n_test, label_n_train, "effective_label_ratio"
    )
    return CorrectedCVCompanion(
        method_a=method_a,
        method_b=method_b,
        estimand=estimand,
        regime=regime,
        statistic=statistic,
        pool_ratio=pool_ratio,
        effective_label_ratio=effective_label_ratio,
    )


def _apply_protocol_profile_status(
    gate: RobustnessGate, conformance: ProtocolProfileConformance, coverage: RawRecordCoverage
) -> RobustnessGate:
    """Override a gate's decision fields when the executed run does not conform to the frozen
    confirmatory profile or when the raw records do not exactly, once each,
    cover the declared profile — checked here regardless of any upstream CLI check, and
    regardless of whatever ``robustness_gate`` independently computed from the (already
    registered-scope) partition aggregates.

    Precedence: ``nonconforming_protocol_profile`` > ``insufficient_valid_partitions`` >
    ``smoke_or_exploratory_profile`` > the gate's own status. An unexpected or duplicated raw
    record always reads as nonconforming, even under an otherwise-legitimate smaller-scale
    declaration — a corrupted "smoke" run is not a trustworthy exploratory execution, so it is
    never classified as one.

    Descriptive fields (``sign_positive``, ``global_mean_delta``, ...) are left untouched; only
    ``decision_eligible``/``supported``/``status`` are overridden, so a nonconforming or
    insufficient run may still report descriptive metrics but never a confirmatory decision. A
    profile or coverage defect always wins over whatever ``robustness_gate`` independently
    computed from partition coverage — even a coincidentally-complete 20/20 coverage under the
    wrong recipe (alphabet, budgets, ...) or over a corrupted record set must never read as
    confirmatory.
    """
    status: str | None
    if not conformance.conforming:
        status = (
            "smoke_or_exploratory_profile"
            if _is_smoke_mismatch(conformance) and coverage.conforming
            else "nonconforming_protocol_profile"
        )
    elif _coverage_has_unexpected_or_duplicate(coverage):
        status = "nonconforming_protocol_profile"
    elif _coverage_has_missing(coverage):
        status = "insufficient_valid_partitions"
    else:
        status = None
    if status is None:
        return gate
    return gate.model_copy(update={"decision_eligible": False, "supported": None, "status": status})


_SCALE_ONLY_PROFILE_FIELDS = frozenset({"partitions", "random_seeds"})


def _is_smoke_mismatch(conformance: ProtocolProfileConformance) -> bool:
    """True iff every mismatch is a *smaller* scale (fewer partitions and/or fewer random seeds
    than the frozen register), never a different identity (alphabet, budgets, ``K``, ``max_order``,
    ...) and never a *larger* scale — a deliberate exploratory/smoke execution (e.g. ``R=1``, one
    seed), distinct from a genuine protocol-profile mismatch.
    """
    mismatches = set(conformance.mismatches)
    if not mismatches or not mismatches <= _SCALE_ONLY_PROFILE_FIELDS:
        return False
    if "partitions" in mismatches:
        observed_p, expected_p = (
            conformance.observed["partitions"],
            conformance.expected["partitions"],
        )
        if not observed_p < expected_p:  # type: ignore[operator]
            return False
    if "random_seeds" in mismatches:
        observed_seeds = conformance.observed["random_seeds"]
        expected_seeds = conformance.expected["random_seeds"]
        if not len(observed_seeds) < len(expected_seeds):  # type: ignore[arg-type]
            return False
    return True


# ------------------------------------------------------------- raw-record coverage

_COVERAGE_SAMPLE_LIMIT = 20  # bounded forensic samples per corruption category, canonically sorted

_DeterministicKey = tuple[str, int, str, str, str, int, int]
_RandomKey = tuple[str, int, str, str, str, int, int, int]


def _det_key(r: DeterministicFoldRecord) -> _DeterministicKey:
    """The deterministic raw-record identity: one cell of partition x estimand x regime x method x
    budget x fold. ``protocol_version`` is part of the key, so a record stamped with the wrong
    version is automatically both a missing cell (its correct-version cell has zero observations)
    and an unexpected one (its own key is outside the expected set) -- no separate protocol-version
    check is needed.
    """
    return (
        r.protocol_version,
        r.partition_index,
        r.estimand,
        r.missingness_regime,
        r.method,
        r.budget,
        r.fold_index,
    )


def _rand_key(r: RandomFoldSeedRecord) -> _RandomKey:
    """The random raw-record identity: the deterministic key plus ``random_seed``."""
    return (
        r.protocol_version,
        r.partition_index,
        r.estimand,
        r.missingness_regime,
        r.method,
        r.budget,
        r.fold_index,
        r.random_seed,
    )


def _expected_deterministic_keys(profile: ConfirmatoryProfile) -> frozenset[_DeterministicKey]:
    """Expected deterministic cells, built purely from ``profile`` and the known production
    generation semantics (partitions x estimands x regimes x deterministic methods x budgets x
    outer folds) -- never from the observed records, which would make coverage circular.
    """
    return frozenset(
        (profile.protocol_version, partition, estimand, regime, method, budget, fold)
        for partition in range(profile.partitions)
        for estimand in profile.estimands
        for regime in profile.missingness_regimes
        for method in _DETERMINISTIC_METHODS
        for budget in profile.budgets
        for fold in range(profile.outer_folds)
    )


def _expected_random_keys(profile: ConfirmatoryProfile) -> frozenset[_RandomKey]:
    """Expected random cells (partitions x estimands x regimes x budgets x outer folds x seeds)."""
    return frozenset(
        (profile.protocol_version, partition, estimand, regime, "random", budget, fold, seed)
        for partition in range(profile.partitions)
        for estimand in profile.estimands
        for regime in profile.missingness_regimes
        for budget in profile.budgets
        for fold in range(profile.outer_folds)
        for seed in profile.random_seeds
    )


def _render_key(key: tuple[object, ...]) -> str:
    """Deterministic, canonically-ordered string form of a raw-record identity key, for bounded
    forensic samples in :class:`RawRecordCoverage` (never the full potentially-huge key set).
    """
    return "|".join(str(part) for part in key)


def _canonical_record_payload(record: BaseModel) -> str:
    """Complete canonical JSON serialization of a raw record: key-sorted, order-independent.

    Two records sharing a registered key are an *exact* duplicate iff their whole payloads compare
    equal here, and a *divergent* duplicate iff any field differs. The comparison is
    over the entire record, never a single metric such as ``s_macro``, so any difference in a
    scientific or provenance-bearing field makes the pair divergent. Deterministic across processes.
    """
    return json.dumps(record.model_dump(mode="json"), sort_keys=True, default=str)


def _canonical_record_order_key(record: FoldRecord) -> tuple[object, ...]:
    """Registered identity followed by the complete payload, solely for forensic serialization."""
    identity = _rand_key(record) if isinstance(record, RandomFoldSeedRecord) else _det_key(record)
    return (*identity, _canonical_record_payload(record))


def _group_is_exact(group: Sequence[BaseModel]) -> bool:
    """True iff every record sharing one registered key has an identical complete canonical payload.

    An exact-duplicate group collapses safely to one representative (byte-identical whichever is
    chosen); a group that fails this is a divergent duplicate and must never be resolved by an
    arbitrary pick.
    """
    return len({_canonical_record_payload(r) for r in group}) == 1


def raw_record_coverage(
    profile: ConfirmatoryProfile,
    deterministic_records: Sequence[DeterministicFoldRecord],
    random_records: Sequence[RandomFoldSeedRecord],
    *,
    sample_limit: int = _COVERAGE_SAMPLE_LIMIT,
) -> RawRecordCoverage:
    """The single source of truth for raw-record protocol coverage against ``profile``.

    Builds the expected deterministic/random key sets independently of the observed records
    (:func:`_expected_deterministic_keys`/:func:`_expected_random_keys`), counts the observed
    records per key with :class:`collections.Counter` (multiplicity-aware, so a duplicate is
    detectable even when the unique-key set is unchanged), and classifies every key as missing
    (count 0), duplicate (an expected key with count > 1), or unexpected (any observed key outside
    the expected set -- including a wrong ``protocol_version``, an out-of-register partition, an
    unexpected random seed, or an unexpected method/budget/fold/estimand/regime cell).
    """
    expected_det = _expected_deterministic_keys(profile)
    expected_rand = _expected_random_keys(profile)
    det_counter: Counter[_DeterministicKey] = Counter(_det_key(r) for r in deterministic_records)
    rand_counter: Counter[_RandomKey] = Counter(_rand_key(r) for r in random_records)

    missing_det = sorted(expected_det - det_counter.keys())
    duplicate_det = sorted(k for k, n in det_counter.items() if k in expected_det and n > 1)
    unexpected_det = sorted(k for k in det_counter if k not in expected_det)

    missing_rand = sorted(expected_rand - rand_counter.keys())
    duplicate_rand = sorted(k for k, n in rand_counter.items() if k in expected_rand and n > 1)
    unexpected_rand = sorted(k for k in rand_counter if k not in expected_rand)

    # Split each duplicated cell into exact (all copies byte-identical) vs divergent
    # (any field differs), comparing the WHOLE canonical payload, never a single metric. Only
    # divergent groups are scientifically ambiguous; the ordering of ``duplicate_*`` is preserved.
    det_groups: dict[_DeterministicKey, list[DeterministicFoldRecord]] = defaultdict(list)
    for det_r in deterministic_records:
        det_groups[_det_key(det_r)].append(det_r)
    rand_groups: dict[_RandomKey, list[RandomFoldSeedRecord]] = defaultdict(list)
    for rand_r in random_records:
        rand_groups[_rand_key(rand_r)].append(rand_r)
    exact_dup_det: list[_DeterministicKey] = []
    divergent_dup_det: list[_DeterministicKey] = []
    for det_k in duplicate_det:
        (exact_dup_det if _group_is_exact(det_groups[det_k]) else divergent_dup_det).append(det_k)
    exact_dup_rand: list[_RandomKey] = []
    divergent_dup_rand: list[_RandomKey] = []
    for rand_k in duplicate_rand:
        (exact_dup_rand if _group_is_exact(rand_groups[rand_k]) else divergent_dup_rand).append(
            rand_k
        )
    has_divergent_duplicate = bool(divergent_dup_det or divergent_dup_rand)

    conforming = not (
        missing_det
        or duplicate_det
        or unexpected_det
        or missing_rand
        or duplicate_rand
        or unexpected_rand
    )
    all_records: list[DeterministicFoldRecord | RandomFoldSeedRecord] = [
        *deterministic_records,
        *random_records,
    ]
    return RawRecordCoverage(
        expected_deterministic_count=len(expected_det),
        observed_deterministic_count=len(deterministic_records),
        expected_random_count=len(expected_rand),
        observed_random_count=len(random_records),
        missing_deterministic_cell_count=len(missing_det),
        duplicate_deterministic_cell_count=len(duplicate_det),
        unexpected_deterministic_cell_count=len(unexpected_det),
        missing_random_cell_count=len(missing_rand),
        duplicate_random_cell_count=len(duplicate_rand),
        unexpected_random_cell_count=len(unexpected_rand),
        exact_duplicate_deterministic_key_count=len(exact_dup_det),
        divergent_duplicate_deterministic_key_count=len(divergent_dup_det),
        exact_duplicate_random_key_count=len(exact_dup_rand),
        divergent_duplicate_random_key_count=len(divergent_dup_rand),
        missing_deterministic_cell_samples=[_render_key(k) for k in missing_det[:sample_limit]],
        duplicate_deterministic_cell_samples=[_render_key(k) for k in duplicate_det[:sample_limit]],
        unexpected_deterministic_cell_samples=[
            _render_key(k) for k in unexpected_det[:sample_limit]
        ],
        missing_random_cell_samples=[_render_key(k) for k in missing_rand[:sample_limit]],
        duplicate_random_cell_samples=[_render_key(k) for k in duplicate_rand[:sample_limit]],
        unexpected_random_cell_samples=[_render_key(k) for k in unexpected_rand[:sample_limit]],
        divergent_duplicate_deterministic_key_samples=[
            _render_key(k) for k in divergent_dup_det[:sample_limit]
        ],
        divergent_duplicate_random_key_samples=[
            _render_key(k) for k in divergent_dup_rand[:sample_limit]
        ],
        observed_protocol_versions=sorted({r.protocol_version for r in all_records}),
        observed_partition_indices=sorted({r.partition_index for r in all_records}),
        observed_random_seeds=sorted({r.random_seed for r in random_records}),
        has_divergent_duplicate=has_divergent_duplicate,
        conforming=conforming,
    )


def _coverage_has_missing(coverage: RawRecordCoverage) -> bool:
    return coverage.missing_deterministic_cell_count > 0 or coverage.missing_random_cell_count > 0


def _coverage_has_unexpected_or_duplicate(coverage: RawRecordCoverage) -> bool:
    return (
        coverage.duplicate_deterministic_cell_count > 0
        or coverage.unexpected_deterministic_cell_count > 0
        or coverage.duplicate_random_cell_count > 0
        or coverage.unexpected_random_cell_count > 0
    )


class DivergentDuplicateError(RuntimeError):
    """A registered raw-record cell holds >= 2 records with divergent payloads.

    A divergent duplicate is scientifically ambiguous: no independent trusted identity says which
    record is authoritative, so no arbitrary selection rule (lexical/first/last/min/max) may pick
    one for aggregation without either contaminating a summary or making it order-dependent. The
    downstream report detects divergence up front and fails closed (unavailable scientific
    summaries) before this is ever raised; the raise is the last-line guard that no aggregation path
    can silently resolve the ambiguity.
    """

    def __init__(self, key: tuple[object, ...]) -> None:
        self.key = key
        super().__init__(f"divergent duplicate raw records for registered cell {_render_key(key)}")


def _collapse_exact_duplicates(
    candidates: Sequence[FoldRecord], key: tuple[object, ...]
) -> FoldRecord:
    """One representative for a registered cell, collapsing ONLY records proven byte-identical.

    The common single-candidate case never pays the serialization cost. Two or more candidates are
    collapsed to ``candidates[0]`` only when every canonical payload matches, in which case the
    result is byte-identical whichever is returned, so the collapse is input-order-independent. A
    divergent group raises :class:`DivergentDuplicateError` — it never selects an arbitrary record
    among divergent payloads, which would silently alter a scientific summary and/or depend on input
    order. Callers fail the whole report closed before this raise can fire.
    """
    if len(candidates) == 1:
        return candidates[0]
    if not _group_is_exact(candidates):
        raise DivergentDuplicateError(key)
    return candidates[0]


def registered_records(
    profile: ConfirmatoryProfile,
    deterministic_records: Sequence[DeterministicFoldRecord],
    random_records: Sequence[RandomFoldSeedRecord],
) -> list[FoldRecord]:
    """The one canonical registered-record scope every scientific summary consumes.

    Exactly the records whose identity key (:func:`_det_key`/:func:`_rand_key`) is expected under
    ``profile``, deduplicated to one canonical record per expected cell when a cell is an *exact*
    duplicate (:func:`_collapse_exact_duplicates`). An unexpected record (wrong protocol version,
    out-of-register partition or seed, or any other cell outside the declared profile) is dropped
    here -- it never contributes to a summary, aggregate, or decision quantity -- but stays visible
    in the raw ``deterministic_records``/``random_records`` collections and in
    :func:`raw_record_coverage`'s diagnostics, so it is never silently discarded from the record.

    A *divergent* duplicate (an expected cell holding >= 2 records with differing payloads) is never
    resolved here: it raises :class:`DivergentDuplicateError`. Callers detect it via
    :attr:`RawRecordCoverage.has_divergent_duplicate` and fail the report closed with unavailable
    scientific summaries before ever calling this on a divergent record set; the raise is a
    defense-in-depth guard that no arbitrary record is ever chosen among divergent payloads.
    """
    expected_det = _expected_deterministic_keys(profile)
    expected_rand = _expected_random_keys(profile)
    det_by_key: dict[_DeterministicKey, list[DeterministicFoldRecord]] = defaultdict(list)
    for det_r in deterministic_records:
        det_key = _det_key(det_r)
        if det_key in expected_det:
            det_by_key[det_key].append(det_r)
    rand_by_key: dict[_RandomKey, list[RandomFoldSeedRecord]] = defaultdict(list)
    for rand_r in random_records:
        rand_key = _rand_key(rand_r)
        if rand_key in expected_rand:
            rand_by_key[rand_key].append(rand_r)
    out: list[FoldRecord] = [_collapse_exact_duplicates(v, k) for k, v in det_by_key.items()]
    out.extend(_collapse_exact_duplicates(v, k) for k, v in rand_by_key.items())
    return out


def _declared_profile(
    *,
    partitions: int,
    n_folds: int,
    budgets: Sequence[int],
    alphabet: str,
    max_order: int,
    n_perturbations: int,
    seeds: int,
    n_inner: int,
) -> ConfirmatoryProfile:
    """The run's own declared/requested profile. ``estimands``/``missingness_regimes``/``methods``
    are always the frozen universe (``downstream_report`` never makes them a per-run choice); only
    the scale/identity fields below vary by run, and feed both :func:`protocol_profile_conformance`
    (declared vs. the frozen confirmatory register) and :func:`raw_record_coverage`/
    :func:`registered_records` (raw records vs. this declared profile).
    """
    return ConfirmatoryProfile(
        protocol_version=PROTOCOL_VERSION,
        partitions=partitions,
        outer_folds=n_folds,
        budgets=tuple(budgets),
        alphabet=alphabet,
        max_order=max_order,
        n_perturbations=n_perturbations,
        random_seeds=tuple(range(seeds)),
        inner_folds=n_inner,
        estimands=_ESTIMANDS,
        missingness_regimes=_REGIMES,
        methods=_METHODS,
    )


def _unavailable_gate(
    estimand: str, regime: str, method_a: str, method_b: str, statistic: str
) -> RobustnessGate:
    """A robustness gate carrying no scientific descriptive: every partition-derived field is its
    empty/no-data value (null deltas, zero counts, no coverage) and the decision is fail-closed
    (``decision_eligible=False``, ``supported=None``, ``status="nonconforming_protocol_profile"``).

    Used when divergent-duplicate raw records make the entire registered scientific summary
    unavailable: the gate must never report a descriptive derived from an
    ambiguously-selected divergent record, so it reports none at all.
    """
    return RobustnessGate(
        estimand=estimand,
        regime=regime,
        method_a=method_a,
        method_b=method_b,
        statistic=statistic,
        expected_partitions=EXPECTED_PARTITIONS,
        observed_valid_partitions=0,
        complete_partition_coverage=False,
        sign_positive=0,
        sign_threshold=SIGN_THRESHOLD,
        sign_pass=False,
        global_mean_delta=None,
        global_mean_positive=False,
        median_partition_delta=None,
        median_positive=False,
        min_effect_size=MIN_STRUCTURAL_EFFECT_SIZE,
        effect_size_pass=False,
        decision_eligible=False,
        supported=None,
        status="nonconforming_protocol_profile",
    )


def _decision_summary(
    records: Sequence[FoldRecord],
    budgets: Sequence[int],
    n_folds: int,
    *,
    partitions: int,
    alphabet: str,
    max_order: int,
    n_perturbations: int,
    seeds: int,
    n_inner: int,
) -> tuple[DecisionSummary, list[PartitionAggregate], list[CorrectedCVCompanion]]:
    max_budget = max(budgets)
    deterministic_records = [r for r in records if isinstance(r, DeterministicFoldRecord)]
    random_records = [r for r in records if isinstance(r, RandomFoldSeedRecord)]
    profile = _declared_profile(
        partitions=partitions,
        n_folds=n_folds,
        budgets=budgets,
        alphabet=alphabet,
        max_order=max_order,
        n_perturbations=n_perturbations,
        seeds=seeds,
        n_inner=n_inner,
    )
    coverage = raw_record_coverage(profile, deterministic_records, random_records)
    conformance = protocol_profile_conformance(
        protocol_version=profile.protocol_version,
        partitions=profile.partitions,
        outer_folds=profile.outer_folds,
        budgets=profile.budgets,
        alphabet=profile.alphabet,
        max_order=profile.max_order,
        n_perturbations=profile.n_perturbations,
        random_seeds=profile.random_seeds,
        inner_folds=profile.inner_folds,
        estimands=profile.estimands,
        missingness_regimes=profile.missingness_regimes,
        methods=profile.methods,
    )

    aggregates: list[PartitionAggregate]
    companions: list[CorrectedCVCompanion]
    if coverage.has_divergent_duplicate:
        # Fail-closed: a divergent duplicate cell (same registered key, different
        # payload) is scientifically ambiguous -- no independent trusted identity says which record
        # is authoritative, so NO arbitrary record may be aggregated. Every registered scientific
        # summary is therefore unavailable: empty aggregates/companions and descriptive-free gates.
        # The forensic decision object (raw-record coverage, duplicate diagnostics, protocol status)
        # is still produced so the corruption stays fully auditable.
        struct_gate = _unavailable_gate(
            _PRIMARY_ESTIMAND, _PRIMARY_REGIME, "structural", "fitness", "s_macro_auc"
        )
        esm_gate = _unavailable_gate(
            _PRIMARY_ESTIMAND, _PRIMARY_REGIME, "info", "structural", f"s_macro_at_{max_budget}"
        )
        aggregates = []
        companions = []
    else:
        # The one canonical registered scope: every summary below is computed from this
        # list, never from the raw records directly, so an unexpected or exact-duplicate record can
        # never move a registered scientific quantity.
        registered = registered_records(profile, deterministic_records, random_records)

        struct_agg = partition_aggregates_for(
            registered,
            _PRIMARY_ESTIMAND,
            _PRIMARY_REGIME,
            "structural",
            "fitness",
            "s_macro_auc",
            "auc",
            budgets,
            n_folds,
        )
        struct_deltas_by_cell = _paired_cell_deltas(
            registered,
            _PRIMARY_ESTIMAND,
            _PRIMARY_REGIME,
            "structural",
            "fitness",
            "auc",
            budgets,
            max_budget,
        )
        struct_deltas = _global_deltas_within_expected_partitions(
            struct_deltas_by_cell, EXPECTED_PARTITIONS
        )
        struct_gate = _apply_protocol_profile_status(
            robustness_gate(struct_agg, struct_deltas), conformance, coverage
        )

        esm_agg = partition_aggregates_for(
            registered,
            _PRIMARY_ESTIMAND,
            _PRIMARY_REGIME,
            "info",
            "structural",
            f"s_macro_at_{max_budget}",
            "at_max",
            budgets,
            n_folds,
        )
        esm_deltas_by_cell = _paired_cell_deltas(
            registered,
            _PRIMARY_ESTIMAND,
            _PRIMARY_REGIME,
            "info",
            "structural",
            "at_max",
            budgets,
            max_budget,
        )
        esm_deltas = _global_deltas_within_expected_partitions(
            esm_deltas_by_cell, EXPECTED_PARTITIONS
        )
        esm_gate = _apply_protocol_profile_status(
            robustness_gate(esm_agg, esm_deltas), conformance, coverage
        )

        aggregates = []
        companions = []
        for estimand in _ESTIMANDS:
            for regime in _REGIMES:
                for method_a, method_b, mode in _PAIR_SPECS:
                    statistic = "s_macro_auc" if mode == "auc" else f"s_macro_at_{max_budget}"
                    aggregates.extend(
                        partition_aggregates_for(
                            registered,
                            estimand,
                            regime,
                            method_a,
                            method_b,
                            statistic,
                            mode,
                            budgets,
                            n_folds,
                        )
                    )
                    companions.append(
                        _corrected_cv_companion(
                            registered,
                            estimand,
                            regime,
                            method_a,
                            method_b,
                            statistic,
                            mode,
                            budgets,
                        )
                    )

    decision = DecisionSummary(
        protocol_version=PROTOCOL_VERSION,
        amendment_version=AMENDMENT_VERSION,
        primary_estimand=_PRIMARY_ESTIMAND,
        primary_regime=_PRIMARY_REGIME,
        expected_protocol_profile=conformance.expected,
        observed_protocol_profile=conformance.observed,
        protocol_profile_mismatches=conformance.mismatches,
        declared_protocol_profile_conforming=conformance.conforming,
        raw_record_coverage=coverage,
        raw_record_coverage_conforming=coverage.conforming,
        protocol_profile_conforming=conformance.conforming and coverage.conforming,
        structural_gate=struct_gate,
        esm_gate=esm_gate,
        structural_downstream_supported=struct_gate.supported,
        esm_uncertainty_supported=esm_gate.supported,
        rule=(
            "structural supported iff protocol_profile_conforming (the declared-profile check AND "
            "exact raw-record coverage) and complete_partition_coverage (20/20) and the 7-point "
            "partition-robustness gate all pass for structural-fitness S_macro-AUC; "
            "ESM-uncertainty supported iff the same gate passes for info-structural "
            f"S_macro at B={max_budget}; either is null with "
            "status=insufficient_valid_partitions if any of the 20 expected partitions or raw-"
            "record cells is missing or wholly degenerate, status=nonconforming_protocol_profile "
            "if the executed configuration or the raw records themselves do not match the frozen "
            "confirmatory profile exactly, or status=smoke_or_exploratory_profile if the only "
            "mismatch is a partition/seed count below the frozen register and the raw records "
            "exactly cover that smaller declared profile"
        ),
    )
    return decision, aggregates, companions


# ------------------------------------------------------------------ per-fold instance construction

_Evaluator = Callable[[Sequence[Variant], int], _RawBundle]


def _record_from_bundle(
    bundle: _RawBundle,
    estimand: str,
    regime: str,
    partition_index: int,
    salt: str,
    fold: int,
    fold_hash: str,
    method: str,
    budget: int,
    seed: int | None,
) -> FoldRecord:
    common = {
        "protocol_version": PROTOCOL_VERSION,
        "estimand": estimand,
        "missingness_regime": regime,
        "partition_index": partition_index,
        "partition_salt": salt,
        "fold_index": fold,
        "fold_identity_hash": fold_hash,
        "method": method,
        "budget": budget,
        "selected_count": bundle.selected_count,
        "selected_identity_hash": bundle.selected_identity_hash,
        "selectable_pool_size": bundle.selectable_pool_size,
        "revealed_count": bundle.revealed_count,
        "live_count": bundle.live_count,
        "dead_count": bundle.dead_count,
        "missing_count": bundle.missing_count,
        "unusable_count": bundle.unusable_count,
        "effective_train_size": bundle.effective_train_size,
        "train_live_fraction": bundle.train_live_fraction,
        "selected_singles": bundle.selected_singles,
        "selected_doubles": bundle.selected_doubles,
        "selected_triples": bundle.selected_triples,
        "train_singles": bundle.train_singles,
        "train_doubles": bundle.train_doubles,
        "train_triples": bundle.train_triples,
        "alpha_full": bundle.alpha_full,
        "alpha_main_only": bundle.alpha_main_only,
        "alpha_no_triples": bundle.alpha_no_triples,
        "alpha_esm_offset": bundle.alpha_esm_offset,
        "n_eval": bundle.n_eval,
        "s_macro": bundle.s_macro,
        "rho_doubles": bundle.rho_doubles,
        "rho_triples": bundle.rho_triples,
        "pooled_spearman": bundle.pooled_spearman,
        "pearson": bundle.pearson,
        "rmse": bundle.rmse,
        "ndcg": bundle.ndcg,
        "hit_rate": bundle.hit_rate,
        "best_true_top_b": bundle.best_true_top_b,
        "regret": bundle.regret,
        "live_fraction_top_b": bundle.live_fraction_top_b,
        "top_b_order_diversity": bundle.top_b_order_diversity,
        "top_b_identity_diversity": bundle.top_b_identity_diversity,
        "uplift": bundle.uplift,
        "transfer_rho_triples": bundle.transfer_rho_triples,
        "transfer_train_singles": bundle.transfer_train_singles,
        "transfer_train_doubles": bundle.transfer_train_doubles,
        "transfer_degenerate_double_coverage": bundle.transfer_degenerate_double_coverage,
        "esm_circular_s_macro": bundle.esm_circular_s_macro,
        "esm_zero_shot_s_macro": bundle.esm_zero_shot_s_macro,
        "esm_offset_s_macro": bundle.esm_offset_s_macro,
        "status": bundle.status,
        "warnings": bundle.warnings,
    }
    if seed is None:
        return DeterministicFoldRecord(**common)
    return RandomFoldSeedRecord(**common, random_seed=seed)


def _shared_method_records(
    ev: _Evaluator,
    pool: Sequence[ScoredVariant],
    budgets: Sequence[int],
    seeds: int,
    partition_index: int,
    salt: str,
    fold: int,
    fold_hash: str,
    regime: str,
) -> list[FoldRecord]:
    """fitness / practice / random are estimand-invariant: evaluate once, emit under both labels."""
    out: list[FoldRecord] = []
    for budget in budgets:
        det = {
            "fitness": ev(fitness_greedy(pool, budget), budget),
            "practice": ev(practice_heuristic(pool, budget), budget),
        }
        random_bundles = [ev(random_selection(pool, budget, s), budget) for s in range(seeds)]
        for estimand in _ESTIMANDS:
            for method, bundle in det.items():
                out.append(
                    _record_from_bundle(
                        bundle,
                        estimand,
                        regime,
                        partition_index,
                        salt,
                        fold,
                        fold_hash,
                        method,
                        budget,
                        None,
                    )
                )
            for seed, bundle in enumerate(random_bundles):
                out.append(
                    _record_from_bundle(
                        bundle,
                        estimand,
                        regime,
                        partition_index,
                        salt,
                        fold,
                        fold_hash,
                        "random",
                        budget,
                        seed,
                    )
                )
    return out


def _graph_method_records(
    ev: _Evaluator,
    pool: Sequence[ScoredVariant],
    graphs: Mapping[str, EpistasisFactorGraph],
    budgets: Sequence[int],
    partition_index: int,
    salt: str,
    fold: int,
    fold_hash: str,
    regime: str,
    estimand: str,
) -> list[FoldRecord]:
    """info / structural depend on the estimand's graph; rank once at max budget, then slice."""
    max_budget = max(budgets)
    ranked = {
        "info": allocate(graphs["info"], pool, max_budget, lambda_=0.0).selected,
        "structural": allocate(graphs["structural"], pool, max_budget, lambda_=0.0).selected,
    }
    out: list[FoldRecord] = []
    for budget in budgets:
        for method, selected in ranked.items():
            bundle = ev(selected[:budget], budget)
            out.append(
                _record_from_bundle(
                    bundle,
                    estimand,
                    regime,
                    partition_index,
                    salt,
                    fold,
                    fold_hash,
                    method,
                    budget,
                    None,
                )
            )
    return out


def _fold_records(
    space: FeatureSpace,
    land: Mapping[Variant, float],
    active_full: Mapping[Variant, list[int]],
    active_main: Mapping[Variant, list[int]],
    esm_of: Mapping[Variant, float],
    all_interactions: Sequence[Interaction],
    aware: Mapping[str, EpistasisFactorGraph],
    var_map: Mapping[Variant, float],
    unit_map: Mapping[Variant, float],
    full_by_regime: Mapping[str, Sequence[ScoredVariant]],
    budgets: Sequence[int],
    seeds: int,
    partition_index: int,
    salt: str,
    fold: int,
    e_j: set[Variant],
    grid_main: Sequence[float],
    grid_pair: Sequence[float],
    n_inner: int,
) -> list[FoldRecord]:
    """All records from one held-out fold (both estimands, both regimes, all methods)."""
    eval_measured = [v for v in sorted(e_j, key=canonical_id) if v in land]
    if len(eval_measured) < _MIN_POINTS_FOR_CORR:
        return []
    ctx = _build_fold_context(space, eval_measured, land, active_full, active_main, esm_of)

    blind = [i for i in all_interactions if frozenset(i.mutations) not in e_j]
    estimand_graphs = {
        "target_blind": {
            "info": EpistasisFactorGraph(blind, var_map),
            "structural": EpistasisFactorGraph(blind, unit_map),
        },
        "target_aware": dict(aware),
    }
    out: list[FoldRecord] = []
    max_budget = max(budgets)
    for regime, full in full_by_regime.items():
        pool = [sv for sv in full if sv.variant not in e_j]
        if len(pool) < max_budget:
            continue
        pool_size = len(pool)

        def ev(
            selected: Sequence[Variant], budget: int, *, _pool_size: int = pool_size
        ) -> _RawBundle:
            revealed = reveal_measured_fitness(dict(land), list(selected))
            return _evaluate_selection(
                space,
                ctx,
                esm_of,
                selected,
                revealed,
                budget,
                _pool_size,
                grid_main,
                grid_pair,
                n_inner,
            )

        out.extend(
            _shared_method_records(
                ev,
                pool,
                budgets,
                seeds,
                partition_index,
                salt,
                fold,
                ctx.fold_identity_hash,
                regime,
            )
        )
        for estimand, graphs in estimand_graphs.items():
            out.extend(
                _graph_method_records(
                    ev,
                    pool,
                    graphs,
                    budgets,
                    partition_index,
                    salt,
                    fold,
                    ctx.fold_identity_hash,
                    regime,
                    estimand,
                )
            )
    return out


def downstream_report(
    scored: Sequence[ScoredVariant],
    landscape: Mapping[Variant, float],
    budgets: Sequence[int],
    seeds: int,
    *,
    n_folds: int = 5,
    partitions: int = 20,
    max_order: int = 3,
    sites: Sequence[int] = GB1_SITES,
    wt_at_sites: Sequence[str] = GB1_WT_AT_SITES,
    alphabet: str = _AA20,
    grid_main: Sequence[float] = GRID_MAIN,
    grid_pair: Sequence[float] = GRID_PAIR,
    n_inner: int = N_INNER_FOLDS,
    n_perturbations: int = 16,
    dataset: str = "gb1_wu2016",
    model_id: str = "",
    provenance: Mapping[str, object] | None = None,
    out_dir: Path | None = None,
) -> DownstreamReport:
    """Run the downstream-impact benchmark over both estimands and both missingness regimes.

    ``scored`` is canonicalized by identity before anything else reads it: an arbitrary
    input order never changes a selection, a raw record, or the final report. Selection is
    zero-shot; measured fitness enters only via ``reveal_measured_fitness`` after each method has
    selected B from ``pool_j = universe \\ E_j``. Every summary/effect/decision field is a pure
    function of the raw per-fold records this function returns; none is computed on a
    second, parallel path.
    """
    scored_sorted = sorted(scored, key=lambda sv: canonical_id(sv.variant))
    space = FeatureSpace(sites, wt_at_sites, alphabet)
    land = dict(landscape)
    universe = [sv.variant for sv in scored_sorted]
    var_map = {sv.variant: sv.var_delta_g for sv in scored_sorted}
    unit_map: dict[Variant, float] = {sv.variant: 1.0 for sv in scored_sorted}
    esm_of = {sv.variant: sv.delta_g for sv in scored_sorted}
    all_interactions = predicted_epistasis(scored_sorted, max_order)
    aware = {
        "info": EpistasisFactorGraph(all_interactions, var_map),
        "structural": EpistasisFactorGraph(all_interactions, unit_map),
    }
    active_full = {v: space.active_columns(v) for v in universe}
    active_main = {v: space.active_columns(v, include_pairs=False) for v in universe}
    eval_universe = [v for v in universe if len(v) in (_PAIRWISE_ORDER, _THIRD_ORDER)]
    full_by_regime: dict[str, list[ScoredVariant]] = {
        "attempted_budget": list(scored_sorted),
        "measured_available": [sv for sv in scored_sorted if sv.variant in land],
    }

    records: list[FoldRecord] = []
    for partition_index in range(partitions):
        salt = partition_salt(partition_index)
        folds = assign_outer_folds(eval_universe, n_folds, salt)
        members: dict[int, set[Variant]] = defaultdict(set)
        for variant, fold in folds.items():
            members[fold].add(variant)
        for fold in range(n_folds):
            records.extend(
                _fold_records(
                    space,
                    land,
                    active_full,
                    active_main,
                    esm_of,
                    all_interactions,
                    aware,
                    var_map,
                    unit_map,
                    full_by_regime,
                    budgets,
                    seeds,
                    partition_index,
                    salt,
                    fold,
                    members[fold],
                    grid_main,
                    grid_pair,
                    n_inner,
                )
            )

    deterministic_records = [r for r in records if isinstance(r, DeterministicFoldRecord)]
    random_records = [r for r in records if isinstance(r, RandomFoldSeedRecord)]
    decision, aggregates, companions = _decision_summary(
        records,
        budgets,
        n_folds,
        partitions=partitions,
        alphabet=alphabet,
        max_order=max_order,
        n_perturbations=n_perturbations,
        seeds=seeds,
        n_inner=n_inner,
    )
    # When a divergent duplicate makes every registered summary ambiguous, method_budget
    # is unavailable (empty), exactly as ``aggregates``/``companions`` already are from
    # _decision_summary; calling registered_records on a divergent set would (correctly) raise.
    # Otherwise method_budget consumes the SAME canonical registered scope as every other summary
    # so an unexpected or exact-duplicate raw record can never move it either.
    if decision.raw_record_coverage.has_divergent_duplicate:
        method_budget: list[MethodBudgetSummary] = []
        summaries_available = False
        summaries_unavailable_reason: str | None = "divergent_duplicate_raw_record"
    else:
        report_profile = _declared_profile(
            partitions=partitions,
            n_folds=n_folds,
            budgets=budgets,
            alphabet=alphabet,
            max_order=max_order,
            n_perturbations=n_perturbations,
            seeds=seeds,
            n_inner=n_inner,
        )
        registered_for_report = registered_records(
            report_profile, deterministic_records, random_records
        )
        method_budget = method_budget_summaries(registered_for_report)
        summaries_available = True
        summaries_unavailable_reason = None

    forensic_deterministic_records = sorted(deterministic_records, key=_canonical_record_order_key)
    forensic_random_records = sorted(random_records, key=_canonical_record_order_key)
    report = DownstreamReport(
        protocol_version=PROTOCOL_VERSION,
        amendment_version=AMENDMENT_VERSION,
        dataset=dataset,
        model_id=model_id,
        alphabet=alphabet,
        max_order=max_order,
        budgets=list(budgets),
        seeds=seeds,
        n_folds=n_folds,
        partitions=partitions,
        n_candidates=len(scored_sorted),
        n_eval_universe=len(eval_universe),
        grid_main=list(grid_main),
        grid_pair=list(grid_pair),
        n_inner_folds=n_inner,
        note=_REPORT_NOTE,
        provenance=dict(provenance) if provenance is not None else {},
        deterministic_records=forensic_deterministic_records,
        random_records=forensic_random_records,
        scientific_summaries_available=summaries_available,
        scientific_summaries_unavailable_reason=summaries_unavailable_reason,
        method_budget=method_budget,
        partition_aggregates=aggregates,
        corrected_cv_companions=companions,
        decision=decision,
    )
    if out_dir is not None:
        write_json_atomic(out_dir / "downstream.json", report.model_dump(mode="json"))
    return report
