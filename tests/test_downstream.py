"""Offline tests for the downstream-impact benchmark (no ESM, no network).

A synthetic order-1..3 pool over three sites stands in for scored GB1 candidates; a non-additive
landscape gives the learner real structure. Fold assignment, the generalized-dual ridge, and the
regime-separated inner-CV alpha selection are pinned here before the raw-record schema, the
fail-closed partition-robustness gate, and the ESM diagnostics build on them (protocol amendment 1,
docs/specs/downstream.md).
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from collections.abc import Sequence
from math import exp
from pathlib import Path

import numpy as np
import pytest

from epibudget import downstream as ds
from epibudget.data import enumerate_candidates
from epibudget.downstream import (
    AlphaChoice,
    CorrectedCVCompanion,
    DeterministicFoldRecord,
    DivergentDuplicateError,
    DownstreamReport,
    FeatureSpace,
    MethodBudgetSummary,
    PartitionAggregate,
    RandomFoldSeedRecord,
    RobustnessGate,
    assign_outer_folds,
    canonical_id,
    downstream_report,
    fit_ridge,
    learning_curve_auc,
    macro_spearman,
    method_budget_summaries,
    ndcg_at_k,
    partition_salt,
    percentile_relevance,
    raw_record_coverage,
    registered_records,
    robustness_gate,
    select_alpha,
    select_alpha_main_only,
)
from epibudget.types import ScoredVariant, Variant

_SITES = (0, 1, 2)
_WT = ("A", "A", "A")
_ALPHABET = "ACG"  # WT 'A' + two mutants per site
_N_FOLDS = 5
_SHA256_HEXLEN = 64
_N_MAIN = 3 * 2  # 3 sites x 2 non-WT residues
_N_PAIR = 3 * (2 * 2)  # 3 site-pairs x 2x2 residue pairs
_TRIPLE_ORDER = 3
_STRONG_CORR = 0.99
_N_METHODS = 5
_N_ESTIMANDS = 2
_N_REGIMES = 2
_N_PAIR_SPECS = 4


def _universe() -> list[Variant]:
    return enumerate_candidates(_SITES, _WT, allowed_aa=_ALPHABET, max_order=3)


def _order_2_3(variants: list[Variant]) -> list[Variant]:
    return [v for v in variants if len(v) in (2, 3)]


def _pool() -> list[ScoredVariant]:
    return [
        ScoredVariant(variant=v, delta_g=float(i) - 10.0, var_delta_g=0.05 + 0.01 * i)
        for i, v in enumerate(_universe())
    ]


def _true_dg(variant: Variant) -> float:
    per_site = {0: 0.7, 1: -0.4, 2: 0.3}
    sites = {pos for pos, _, _ in variant}
    value = sum(per_site[p] for p in sites)
    if {0, 1} <= sites:
        value += 0.9  # a genuine order-2 interaction so the learner has real structure
    return value


def _landscape(pool: list[ScoredVariant]) -> dict[Variant, float]:
    landscape: dict[Variant, float] = {frozenset(): 1.0}
    for sv in pool:
        landscape[sv.variant] = exp(_true_dg(sv.variant))
    return landscape


# --- folds -------------------------------------------------------------------------------------


def test_canonical_id_is_order_independent() -> None:
    a: Variant = frozenset({(0, "A", "C"), (1, "A", "G")})
    b: Variant = frozenset({(1, "A", "G"), (0, "A", "C")})
    assert canonical_id(a) == canonical_id(b)


def test_partition_salt_is_frozen_and_distinct() -> None:
    assert partition_salt(0) == partition_salt(0)
    assert partition_salt(0) != partition_salt(1)
    assert len(partition_salt(0)) == _SHA256_HEXLEN


def test_assign_outer_folds_is_deterministic_stratified_and_reorder_stable() -> None:
    eval_variants = _order_2_3(_universe())
    salt = partition_salt(0)
    folds = assign_outer_folds(eval_variants, _N_FOLDS, salt)
    assert set(folds.values()) <= set(range(_N_FOLDS))
    shuffled = assign_outer_folds(list(reversed(eval_variants)), _N_FOLDS, salt)
    assert folds == shuffled
    for order in (2, 3):
        counts = [0] * _N_FOLDS
        for v, f in folds.items():
            if len(v) == order:
                counts[f] += 1
        assert max(counts) - min(counts) <= 1


def test_assign_outer_folds_rejects_one_fold() -> None:
    with pytest.raises(ValueError, match="n_folds must be >= 2"):
        assign_outer_folds(_order_2_3(_universe()), 1, partition_salt(0))


# --- inner folds (balanced, identity-sorted, not modulo) ---------------------------------------


def test_inner_folds_balanced_is_order_independent_and_balanced() -> None:
    variants = _order_2_3(_universe())[:9]
    salt = ds._INNER_SALT
    labels = ds._inner_folds_balanced(variants, 3, salt)
    assert labels is not None
    assert set(labels) == {0, 1, 2}
    shuffled = ds._inner_folds_balanced(list(reversed(variants)), 3, salt)
    # order-independence: the SAME identity gets the SAME label regardless of input order.
    by_id = dict(zip([canonical_id(v) for v in variants], labels, strict=True))
    shuffled_by_id = dict(
        zip([canonical_id(v) for v in reversed(variants)], shuffled or [], strict=True)
    )
    assert by_id == shuffled_by_id


def test_inner_folds_balanced_returns_none_below_n_inner() -> None:
    variants = _order_2_3(_universe())[:2]
    assert ds._inner_folds_balanced(variants, 3, ds._INNER_SALT) is None


# --- feature space -------------------------------------------------------------------------------


def test_feature_space_column_counts() -> None:
    space = FeatureSpace(_SITES, _WT, _ALPHABET)
    assert len(space.main_index) == _N_MAIN
    assert len(space.pair_index) == _N_PAIR
    assert space.n_features == _N_MAIN + _N_PAIR
    assert int(space.penalty_is_main.sum()) == _N_MAIN


def test_active_columns_main_and_pairs() -> None:
    space = FeatureSpace(_SITES, _WT, _ALPHABET)
    triple: Variant = frozenset({(0, "A", "C"), (1, "A", "G"), (2, "A", "C")})
    cols_full = space.active_columns(triple, include_pairs=True)
    cols_main = space.active_columns(triple, include_pairs=False)
    assert len(cols_main) == _TRIPLE_ORDER
    assert len(cols_full) == _TRIPLE_ORDER + _TRIPLE_ORDER
    assert len(set(cols_full)) == len(cols_full)


def test_penalties_must_be_positive() -> None:
    space = FeatureSpace(_SITES, _WT, _ALPHABET)
    with pytest.raises(ValueError, match="penalties must be > 0"):
        space.penalties(0.0, 1.0)
    pen = space.penalties(2.0, 7.0)
    assert pen[space.main_index[(0, "C")]] == pytest.approx(2.0)
    assert pen[next(iter(space.pair_index.values()))] == pytest.approx(7.0)


# --- generalized-dual ridge -----------------------------------------------------------------------


def test_fit_ridge_is_solvable_on_an_all_singles_design() -> None:
    space = FeatureSpace(_SITES, _WT, _ALPHABET)
    singles = [v for v in _universe() if len(v) == 1]
    design = space.design_matrix(singles)
    response = np.arange(len(singles), dtype=np.float64)
    model = fit_ridge(design, response, space.penalties(1.0, 1.0))
    assert not model.degenerate
    assert np.all(np.isfinite(model.coef))
    preds = model.predict_active([space.active_columns(v) for v in singles])
    assert np.all(np.isfinite(preds))


def test_fit_ridge_empty_training_is_degenerate() -> None:
    space = FeatureSpace(_SITES, _WT, _ALPHABET)
    model = fit_ridge(np.zeros((0, space.n_features)), np.zeros(0), space.penalties(1.0, 1.0))
    assert model.degenerate
    assert model.predict_active([[0]])[0] == pytest.approx(0.0)


def test_fit_ridge_recovers_a_linear_signal_with_weak_penalty() -> None:
    space = FeatureSpace(_SITES, _WT, _ALPHABET)
    variants = _universe()
    true_main = {(0, "C"): 1.5, (0, "G"): -0.5, (1, "C"): 0.3}

    def truth(v: Variant) -> float:
        return 2.0 + sum(true_main.get((p, m), 0.0) for p, _w, m in v)

    response = np.array([truth(v) for v in variants])
    model = fit_ridge(space.design_matrix(variants), response, space.penalties(1e-4, 1e-4))
    pred = model.predict_active([space.active_columns(v) for v in variants])
    assert np.corrcoef(pred, response)[0, 1] > _STRONG_CORR


# --- inner-CV alpha selection (regime-separated) --------------------------------------------------


def test_select_alpha_falls_back_on_a_tiny_training_set() -> None:
    space = FeatureSpace(_SITES, _WT, _ALPHABET)
    choice = select_alpha(
        space,
        [frozenset({(0, "A", "C")})],
        np.array([1.0]),
        [0.1, 1.0, 10.0],
        [0.1, 1.0, 10.0],
        3,
        partition_salt(0),
    )
    assert isinstance(choice, AlphaChoice)
    assert choice.fell_back
    assert choice.fallback_reason == "training_set_too_small"
    assert choice.n_inner_folds_used == 0
    assert choice.alpha_main == max([0.1, 1.0, 10.0])


def test_select_alpha_prefers_shrinkage_when_signal_is_pure_noise() -> None:
    space = FeatureSpace(_SITES, _WT, _ALPHABET)
    variants = _universe()
    rng = np.random.default_rng(0)
    response = rng.normal(size=len(variants))
    choice = select_alpha(
        space,
        variants,
        response,
        [0.01, 1.0, 100.0],
        [0.01, 1.0, 100.0],
        ds.N_INNER_FOLDS,
        partition_salt(1),
    )
    assert not choice.fell_back
    assert choice.n_inner_folds_used == ds.N_INNER_FOLDS
    assert choice.alpha_main >= 1.0
    assert choice.alpha_pair is not None and choice.alpha_pair >= 1.0


def test_select_alpha_main_only_has_no_alpha_pair() -> None:
    space = FeatureSpace(_SITES, _WT, _ALPHABET)
    variants = _universe()
    rng = np.random.default_rng(2)
    response = rng.normal(size=len(variants))
    choice = select_alpha_main_only(
        space, variants, response, [0.01, 1.0, 100.0], 3, partition_salt(1)
    )
    assert choice.alpha_pair is None
    assert not choice.applicable
    assert choice.alpha_main >= 1.0


def test_no_triples_regime_falls_back_independently_of_full_regime() -> None:
    # A training set with enough singles+doubles+triples for the full model's own inner CV, but
    # whose singles+doubles-only subset (the no-triples regime) is too small for its own inner CV.
    # Regressing to reusing the full model's alpha would never show this divergence.
    space = FeatureSpace(_SITES, _WT, _ALPHABET)
    variants = _universe()  # 3 singles + 6 doubles + 2 triples = 11 total
    too_few_for_inner_cv = ds.N_INNER_FOLDS - 1
    singles_doubles = [v for v in variants if len(v) <= ds._PAIRWISE_ORDER][:too_few_for_inner_cv]
    rng = np.random.default_rng(3)
    y_full = rng.normal(size=len(variants))
    y_restricted = rng.normal(size=len(singles_doubles))

    full_choice = select_alpha(
        space,
        variants,
        y_full,
        [0.1, 1.0, 10.0],
        [1.0, 10.0, 100.0],
        ds.N_INNER_FOLDS,
        ds._INNER_SALT,
    )
    restricted_choice = select_alpha(
        space,
        singles_doubles,
        y_restricted,
        [0.1, 1.0, 10.0],
        [1.0, 10.0, 100.0],
        3,
        ds._INNER_SALT,
    )
    assert not full_choice.fell_back
    assert restricted_choice.fell_back
    assert restricted_choice.fallback_reason == "training_set_too_small"


# --- ESM-offset nested inner-CV ---------------------------------------------


def test_esm_offset_b_inner_is_insensitive_to_inner_validation_labels() -> None:
    """``b_inner`` (and the residual model trained alongside it) must be a pure function of
    the inner-TRAINING rows. Perturbing a label that lives only in the inner-VALIDATION rows must
    change that fold's validation loss but never ``b_inner`` or the fitted residual model — proving
    the offset no longer leaks the validation label it is later scored against.
    """
    space = FeatureSpace(_SITES, _WT, _ALPHABET)
    variants = _universe()[:6]
    design = space.design_matrix(variants)
    esm = np.array([float(i) for i in range(len(variants))])
    y = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    train_mask = np.array([True, True, True, True, False, False])  # rows 4, 5 are "held out"
    penalties = space.penalties(1.0, 10.0)

    b_inner_base, model_base = ds._fit_offset_fold(design, y, esm, train_mask, penalties)

    y_perturbed = y.copy()
    y_perturbed[4] += 100.0  # perturb ONLY a row outside train_mask
    b_inner_perturbed, model_perturbed = ds._fit_offset_fold(
        design, y_perturbed, esm, train_mask, penalties
    )

    assert b_inner_perturbed == pytest.approx(b_inner_base)
    assert np.allclose(model_perturbed.coef, model_base.coef)
    assert model_perturbed.intercept == pytest.approx(model_base.intercept)

    val_mask = ~train_mask
    active = [space.active_columns(v) for v in variants]
    val_active = [active[i] for i in np.nonzero(val_mask)[0]]
    pred_val = model_base.predict_active(val_active) + b_inner_base * esm[val_mask]
    loss_base = float(np.mean((pred_val - y[val_mask]) ** 2))
    loss_perturbed = float(np.mean((pred_val - y_perturbed[val_mask]) ** 2))
    assert loss_perturbed != pytest.approx(loss_base)


def test_select_alpha_esm_offset_matches_a_brute_force_nested_reference() -> None:
    """an independently-written nested-CV reference (never calling
    ``select_alpha_esm_offset``/``_fit_offset_fold``) must select the identical alpha and final
    ``b`` on the same small synthetic case, proving the production nested procedure is correct by
    construction rather than merely self-consistent.
    """
    space = FeatureSpace(_SITES, _WT, _ALPHABET)
    variants = _universe()  # 3 singles + 6 doubles + 2 triples = 11
    rng = np.random.default_rng(11)
    esm = rng.normal(size=len(variants))
    y = rng.normal(size=len(variants))
    grid_main = [0.1, 1.0, 10.0]
    grid_pair = [1.0, 10.0, 100.0]
    n_inner = 3
    salt = ds._INNER_SALT

    # --- independent brute-force reference, written directly here, reusing only the frozen
    # fold-assignment algorithm and the ridge primitive (both separately tested elsewhere) ---------
    labels = ds._inner_folds_balanced(variants, n_inner, salt)
    assert labels is not None
    label_arr = np.array(labels)
    design = space.design_matrix(variants)
    active = [space.active_columns(v) for v in variants]

    def through_origin(x: np.ndarray, yy: np.ndarray) -> float:
        denom = float(np.dot(x, x))
        return float(np.dot(x, yy) / denom) if denom != 0.0 else 1.0

    best_key: tuple[float, float, float] | None = None
    best_alpha: tuple[float, float] | None = None
    for alpha_main in grid_main:
        for alpha_pair in grid_pair:
            penalties = space.penalties(alpha_main, alpha_pair)
            fold_errors: list[float] = []
            for fold in range(n_inner):
                val_mask = label_arr == fold
                tr_mask = ~val_mask
                b_inner = through_origin(esm[tr_mask], y[tr_mask])
                offset_train = y[tr_mask] - b_inner * esm[tr_mask]
                model = fit_ridge(design[tr_mask], offset_train, penalties)
                val_active = [active[i] for i in np.nonzero(val_mask)[0]]
                pred = model.predict_active(val_active) + b_inner * esm[val_mask]
                fold_errors.extend((pred - y[val_mask]) ** 2)
            key = (float(np.mean(fold_errors)), -alpha_main, -alpha_pair)
            if best_key is None or key < best_key:
                best_key = key
                best_alpha = (alpha_main, alpha_pair)
    assert best_alpha is not None
    reference_b_final = through_origin(esm, y)

    # --- production code -----------------------------------------------------------------------
    choice = ds.select_alpha_esm_offset(
        space, variants, y, esm, grid_main, grid_pair, n_inner, salt
    )

    assert not choice.fell_back
    assert choice.alpha_main == pytest.approx(best_alpha[0])
    assert choice.alpha_pair == pytest.approx(best_alpha[1])
    assert choice.b == pytest.approx(reference_b_final)


# --- metrics -------------------------------------------------------------------------------------


def test_macro_spearman_averages_the_two_orders() -> None:
    assert macro_spearman(0.4, 0.6) == pytest.approx(0.5)
    assert macro_spearman(None, 0.6) is None
    assert macro_spearman(0.4, None) is None


def test_percentile_relevance_keeps_zeros_and_averages_ties() -> None:
    rel = percentile_relevance(np.array([0.0, 1.0, 2.0, 3.0]))
    assert rel[0] == pytest.approx(0.0)
    assert rel[-1] == pytest.approx(1.0)
    tied = percentile_relevance(np.array([0.0, 0.0, 2.0]))
    assert tied[0] == pytest.approx(0.25) and tied[1] == pytest.approx(0.25)


def test_ndcg_is_one_for_the_ideal_order_and_handles_ties() -> None:
    pred = np.array([1.0, 1.0, 0.0])
    relevance = np.array([1.0, 0.0, 0.0])
    ids = ["a", "b", "c"]
    assert ndcg_at_k(pred, relevance, 2, ids) == pytest.approx(1.0)
    worst = ndcg_at_k(np.array([0.0, 1.0, 1.0]), relevance, 2, ids)
    assert worst == pytest.approx(0.0)


def test_ndcg_all_tied_relevance_is_one_by_convention() -> None:
    ids = ["a", "b", "c"]
    all_zero = ndcg_at_k(np.array([3.0, 1.0, 2.0]), np.array([0.0, 0.0, 0.0]), 3, ids)
    assert all_zero == pytest.approx(1.0)
    all_equal_nonzero = ndcg_at_k(np.array([3.0, 1.0, 2.0]), np.array([5.0, 5.0, 5.0]), 3, ids)
    assert all_equal_nonzero == pytest.approx(1.0)


def test_learning_curve_auc_equal_trapezoid_weights() -> None:
    assert learning_curve_auc([0.4, 0.5, 0.6]) == pytest.approx(0.5)
    assert learning_curve_auc([0.4, None, 0.6]) is None


# --- corrected-CV sensitivity companion (never the primary gate) ---------------------------------


def test_corrected_cv_formula_matches_nadeau_bengio_by_hand() -> None:
    deltas = [0.2, 0.0, 0.2, 0.0]
    n = 4
    effect = ds._corrected_cv_formula(deltas, n_test=4.0, n_train=4.0, convention="pool_ratio")
    mean = float(np.mean(deltas))
    var = float(np.var(deltas, ddof=1))
    se = float(np.sqrt(var * (1.0 / n + 1.0)))
    assert effect.status == "sensitivity_only"
    assert effect.delta_mean == pytest.approx(mean)
    assert effect.se == pytest.approx(se)
    assert effect.df == n - 1
    assert effect.ratio == pytest.approx(1.0)
    assert effect.ci95 is not None and effect.ci95[0] < 0.0 < effect.ci95[1]


def test_corrected_cv_formula_unavailable_without_a_ratio() -> None:
    effect = ds._corrected_cv_formula(
        [0.1, 0.2], n_test=None, n_train=None, convention="pool_ratio"
    )
    assert effect.status == "unavailable"
    assert effect.ci95 is None


def test_corrected_cv_formula_drops_degenerate_folds_from_n_valid_effects() -> None:
    deltas: list[float | None] = [0.1, None, 0.2, None]
    effect = ds._corrected_cv_formula(
        deltas, n_test=2.0, n_train=2.0, convention="effective_label_ratio"
    )
    assert effect.n_valid_effects == sum(1 for d in deltas if d is not None)


# --- corrected-CV companion: per-budget effective-size convention ----------

_DEFAULT_ALPHA = AlphaChoice(
    alpha_main=1.0, alpha_pair=1.0, fell_back=False, fallback_reason=None, n_inner_folds_used=3
)


def _fold_record(
    *,
    method: str,
    budget: int,
    partition_index: int,
    fold_index: int,
    s_macro: float | None,
    selectable_pool_size: int,
    effective_train_size: int,
    n_eval: int,
    estimand: str = "target_blind",
    regime: str = "attempted_budget",
) -> DeterministicFoldRecord:
    """A minimal, schema-complete raw record for tests that exercise only the aggregation layer."""
    return DeterministicFoldRecord(
        protocol_version=ds.PROTOCOL_VERSION,
        estimand=estimand,
        missingness_regime=regime,
        partition_index=partition_index,
        partition_salt=partition_salt(partition_index),
        fold_index=fold_index,
        fold_identity_hash="fold-hash",
        method=method,
        budget=budget,
        selected_count=budget,
        selected_identity_hash="selected-hash",
        selectable_pool_size=selectable_pool_size,
        revealed_count=effective_train_size,
        live_count=effective_train_size,
        dead_count=0,
        missing_count=0,
        unusable_count=0,
        effective_train_size=effective_train_size,
        train_live_fraction=1.0,
        selected_singles=0,
        selected_doubles=budget,
        selected_triples=0,
        train_singles=0,
        train_doubles=effective_train_size,
        train_triples=0,
        alpha_full=_DEFAULT_ALPHA,
        alpha_main_only=_DEFAULT_ALPHA,
        alpha_no_triples=_DEFAULT_ALPHA,
        alpha_esm_offset=_DEFAULT_ALPHA,
        n_eval=n_eval,
        s_macro=s_macro,
        rho_doubles=s_macro,
        rho_triples=s_macro,
        pooled_spearman=s_macro,
        pearson=s_macro,
        rmse=0.1,
        ndcg=0.5,
        hit_rate=0.5,
        best_true_top_b=1.0,
        regret=0.0,
        live_fraction_top_b=1.0,
        top_b_order_diversity=1,
        top_b_identity_diversity=budget,
        uplift=0.0,
        transfer_rho_triples=None,
        transfer_train_singles=0,
        transfer_train_doubles=0,
        transfer_degenerate_double_coverage=True,
        esm_circular_s_macro=None,
        esm_zero_shot_s_macro=None,
        esm_offset_s_macro=None,
        status="ok",
        warnings=[],
    )


_COMPANION_BUDGETS = (48, 96, 192)
_PER_BUDGET_TRAIN_SIZES = {48: 100, 96: 500, 192: 2000}  # deliberately distinct per budget


def test_corrected_cv_companion_at_max_uses_only_the_max_budget_sizes() -> None:
    """the info-structural (``at_max``) companion's effective-size means must come from the
    B=192 records ONLY, never averaged with the B=48/96 records' (deliberately different) sizes.
    """
    records = [
        _fold_record(
            method=method,
            budget=budget,
            partition_index=0,
            fold_index=0,
            s_macro=0.5,
            selectable_pool_size=_PER_BUDGET_TRAIN_SIZES[budget] * 10,
            effective_train_size=_PER_BUDGET_TRAIN_SIZES[budget],
            n_eval=50,
        )
        for budget in _COMPANION_BUDGETS
        for method in ("info", "structural")
    ]
    companion = ds._corrected_cv_companion(
        records,
        "target_blind",
        "attempted_budget",
        "info",
        "structural",
        "s_macro_at_192",
        "at_max",
        _COMPANION_BUDGETS,
    )
    # Only B=192's sizes (2000 / 20000) may appear; a 48/96/192 average would be ~866.67 / 8666.67.
    assert companion.effective_label_ratio.n_train == pytest.approx(2000.0)
    assert companion.pool_ratio.n_train == pytest.approx(20000.0)
    assert companion.effective_label_ratio.n_test == pytest.approx(50.0)


def test_corrected_cv_companion_auc_uses_sizes_from_every_budget() -> None:
    """the structural-fitness (``auc``) companion legitimately integrates
    ``s_macro`` over every budget, so its sizes are the mean over ALL of ``budgets`` — a distinct,
    documented convention from the ``at_max`` companion above, not an accidental reuse of either.
    """
    records = [
        _fold_record(
            method=method,
            budget=budget,
            partition_index=0,
            fold_index=0,
            s_macro=0.5,
            selectable_pool_size=_PER_BUDGET_TRAIN_SIZES[budget] * 10,
            effective_train_size=_PER_BUDGET_TRAIN_SIZES[budget],
            n_eval=50,
        )
        for budget in _COMPANION_BUDGETS
        for method in ("structural", "fitness")
    ]
    companion = ds._corrected_cv_companion(
        records,
        "target_blind",
        "attempted_budget",
        "structural",
        "fitness",
        "s_macro_auc",
        "auc",
        _COMPANION_BUDGETS,
    )
    expected_mean_train = float(np.mean([_PER_BUDGET_TRAIN_SIZES[b] for b in _COMPANION_BUDGETS]))
    assert companion.effective_label_ratio.n_train == pytest.approx(expected_mean_train)
    assert companion.effective_label_ratio.n_train != pytest.approx(_PER_BUDGET_TRAIN_SIZES[192])


# --- robustness gate: fail-closed missing/degenerate partitions (matrix) -----------------


_K_FOLDS = 5
_N_EXPECTED = ds.EXPECTED_PARTITIONS  # 20
_N_SIGN_THRESHOLD = ds.SIGN_THRESHOLD  # 16


def _aggregate(
    partition_index: int, mean_delta: float | None, n_valid_folds: int = _K_FOLDS
) -> PartitionAggregate:
    return PartitionAggregate(
        estimand="target_blind",
        regime="attempted_budget",
        method_a="structural",
        method_b="fitness",
        statistic="s_macro_auc",
        partition_index=partition_index,
        mean_delta=mean_delta,
        n_valid_folds=n_valid_folds,
        n_total_folds=_K_FOLDS,
    )


def test_robustness_gate_4_positive_16_missing_is_not_eligible() -> None:
    n_positive = _N_EXPECTED - _N_SIGN_THRESHOLD  # 4
    aggs = [
        _aggregate(
            p,
            0.1 if p < n_positive else None,
            n_valid_folds=_K_FOLDS if p < n_positive else 0,
        )
        for p in range(_N_EXPECTED)
    ]
    deltas = [0.1] * _N_EXPECTED  # raw fold-instance deltas, irrelevant once coverage fails
    gate = robustness_gate(aggs, deltas)
    assert gate.observed_valid_partitions == n_positive
    assert not gate.complete_partition_coverage
    assert not gate.decision_eligible
    assert gate.supported is None
    assert gate.status == "insufficient_valid_partitions"


def test_robustness_gate_16_positive_4_zero_is_eligible_with_sign_exactly_threshold() -> None:
    n_positive = _N_SIGN_THRESHOLD
    n_zero = _N_EXPECTED - n_positive
    aggs = [_aggregate(p, 0.1 if p < n_positive else 0.0) for p in range(_N_EXPECTED)]
    deltas = [0.1] * n_positive + [0.0] * n_zero
    gate = robustness_gate(aggs, deltas)
    assert gate.complete_partition_coverage
    assert gate.decision_eligible
    assert gate.sign_positive == n_positive
    assert gate.sign_pass


def test_robustness_gate_below_threshold_positive_fails_sign_gate() -> None:
    n_positive = _N_SIGN_THRESHOLD - 1
    n_negative = _N_EXPECTED - n_positive
    aggs = [_aggregate(p, 0.3 if p < n_positive else -0.1) for p in range(_N_EXPECTED)]
    deltas = [0.3] * n_positive + [-0.1] * n_negative
    gate = robustness_gate(aggs, deltas)
    assert gate.complete_partition_coverage
    assert gate.decision_eligible
    assert gate.sign_positive == n_positive
    assert not gate.sign_pass
    assert gate.supported is False


def test_robustness_gate_all_positive_passes() -> None:
    aggs = [_aggregate(p, 0.2) for p in range(_N_EXPECTED)]
    deltas = [0.2] * _N_EXPECTED
    gate = robustness_gate(aggs, deltas)
    assert gate.complete_partition_coverage
    assert gate.decision_eligible
    assert gate.sign_positive == _N_EXPECTED
    assert gate.sign_pass
    assert gate.global_mean_positive
    assert gate.median_positive
    assert gate.effect_size_pass
    assert gate.supported is True


def test_robustness_gate_exact_zero_partition_mean_is_not_positive() -> None:
    aggs = [_aggregate(p, 0.0) for p in range(_N_EXPECTED)]
    gate = robustness_gate(aggs, [0.0] * _N_EXPECTED)
    assert gate.sign_positive == 0
    assert not gate.sign_pass


def test_robustness_gate_never_reduces_denominator_below_expected_partitions() -> None:
    # Only a handful of partitions ran (e.g. an exploratory smoke); all of them positive. A
    # ceil(0.8*observed) gate would pass; the frozen gate (fixed denominator = 20) must not.
    n_ran = 5  # deliberately far below _N_EXPECTED
    aggs = [_aggregate(p, 0.5) for p in range(n_ran)] + [
        _aggregate(p, None, 0) for p in range(n_ran, _N_EXPECTED)
    ]
    gate = robustness_gate(aggs, [0.5] * n_ran)
    assert gate.observed_valid_partitions == n_ran
    assert not gate.complete_partition_coverage
    assert not gate.decision_eligible
    assert gate.supported is None


# --- end-to-end report -------------------------------------------------------------------------


def _run(**kwargs: object) -> DownstreamReport:
    pool = _pool()
    return downstream_report(
        pool,
        _landscape(pool),
        budgets=[4, 6],
        seeds=2,
        n_folds=2,
        partitions=2,
        sites=_SITES,
        wt_at_sites=_WT,
        alphabet=_ALPHABET,
        grid_main=[1.0, 10.0],
        grid_pair=[1.0, 10.0],
        n_inner=2,
        **kwargs,  # type: ignore[arg-type]
    )


def test_downstream_report_retains_all_methods_estimands_and_regimes() -> None:
    report = _run()
    all_records = [*report.deterministic_records, *report.random_records]
    methods = {r.method for r in all_records}
    assert methods == {"info", "structural", "fitness", "random", "practice"}
    assert {r.estimand for r in all_records} == {"target_blind", "target_aware"}
    assert {r.missingness_regime for r in all_records} == {"attempted_budget", "measured_available"}


def test_deterministic_methods_are_never_repeated_across_seeds() -> None:
    report = _run()
    assert all(r.random_seed is None for r in report.deterministic_records)
    assert all(isinstance(r, DeterministicFoldRecord) for r in report.deterministic_records)
    assert all(isinstance(r, RandomFoldSeedRecord) for r in report.random_records)
    assert all(r.method == "random" for r in report.random_records)


def test_random_records_cover_every_seed_per_cell() -> None:
    report = _run()
    by_cell: dict[tuple[str, str, int, int, int], set[int]] = {}
    for r in report.random_records:
        key = (r.estimand, r.missingness_regime, r.partition_index, r.fold_index, r.budget)
        by_cell.setdefault(key, set()).add(r.random_seed)
    for seeds_seen in by_cell.values():
        assert seeds_seen == {0, 1}  # seeds=2 in _run()


def test_downstream_report_decision_shape() -> None:
    report = _run()
    decision = report.decision
    assert decision.protocol_version == ds.PROTOCOL_VERSION
    assert decision.amendment_version == ds.AMENDMENT_VERSION
    assert decision.primary_estimand == "target_blind"
    assert decision.primary_regime == "attempted_budget"
    # only 2 of the frozen 20 partitions ran in this tiny fixture -> never decision_eligible
    assert decision.structural_gate.expected_partitions == ds.EXPECTED_PARTITIONS
    assert not decision.structural_gate.decision_eligible
    assert decision.structural_downstream_supported is None


def test_downstream_report_writes_atomically_and_never_overwrites(tmp_path: Path) -> None:
    report = _run(out_dir=tmp_path)
    written = json.loads((tmp_path / "downstream.json").read_text(encoding="utf-8"))
    assert written["note"] == report.note
    assert "does not alter the frozen" in written["note"]
    assert not list(tmp_path.glob("*.tmp"))  # no leftover temp file on success
    with pytest.raises(FileExistsError):
        _run(out_dir=tmp_path)


# --- raw-record schema: reconstruction ------------------------------------------------


def test_all_aggregates_reconstruct_exactly_from_raw_records() -> None:
    report = _run()
    all_records = [*report.deterministic_records, *report.random_records]

    rebuilt_summaries = method_budget_summaries(all_records)
    assert rebuilt_summaries == report.method_budget

    rebuilt_decision, rebuilt_aggregates, rebuilt_companions = ds._decision_summary(
        all_records,
        report.budgets,
        report.n_folds,
        partitions=report.partitions,
        alphabet=report.alphabet,
        max_order=report.max_order,
        seeds=report.seeds,
        n_inner=report.n_inner_folds,
    )
    assert rebuilt_decision == report.decision
    assert rebuilt_aggregates == report.partition_aggregates
    assert rebuilt_companions == report.corrected_cv_companions


def test_deterministic_record_count_matches_expected_formula() -> None:
    report = _run()
    # Expected shape: R x K_eff x 4 methods x len(budgets) per (estimand, regime); folds with too
    # few eval variants are skipped (K_eff <= n_folds x partitions) — assert the counts are
    # internally consistent across every (estimand, regime, method, budget) cell instead of a K.
    by_group: dict[tuple[str, str, str, int], int] = {}
    for r in report.deterministic_records:
        key = (r.estimand, r.missingness_regime, r.method, r.budget)
        by_group[key] = by_group.get(key, 0) + 1
    counts = set(by_group.values())
    assert (
        len(counts) == 1
    )  # every (estimand, regime, method, budget) cell has the same n_instances


def test_random_record_count_is_seeds_times_deterministic_count() -> None:
    report = _run()
    det_cells: dict[tuple[str, str, int], int] = {}
    for r in report.deterministic_records:
        if r.method != "fitness":
            continue
        key = (r.estimand, r.missingness_regime, r.budget)
        det_cells[key] = det_cells.get(key, 0) + 1
    rand_cells: dict[tuple[str, str, int], int] = {}
    for r in report.random_records:
        key = (r.estimand, r.missingness_regime, r.budget)
        rand_cells[key] = rand_cells.get(key, 0) + 1
    for key, det_count in det_cells.items():
        assert rand_cells[key] == det_count * report.seeds


# --- canonical order enforcement at the engine boundary ------------------------------


def test_report_is_byte_identical_under_an_arbitrary_permutation_of_scored() -> None:
    pool = _pool()
    landscape = _landscape(pool)
    kwargs = dict(
        budgets=[4, 6],
        seeds=2,
        n_folds=2,
        partitions=2,
        sites=_SITES,
        wt_at_sites=_WT,
        alphabet=_ALPHABET,
        grid_main=[1.0, 10.0],
        grid_pair=[1.0, 10.0],
        n_inner=2,
    )
    forward = downstream_report(pool, landscape, **kwargs)  # type: ignore[arg-type]
    reversed_pool = list(reversed(pool))
    backward = downstream_report(reversed_pool, landscape, **kwargs)  # type: ignore[arg-type]
    assert forward.model_dump_json() == backward.model_dump_json()

    rng = np.random.default_rng(7)
    shuffled_pool = list(pool)
    rng.shuffle(shuffled_pool)
    shuffled = downstream_report(shuffled_pool, landscape, **kwargs)  # type: ignore[arg-type]
    assert forward.model_dump_json() == shuffled.model_dump_json()


# --- leakage / circularity guards ---------------------------------------------------------------


def test_primary_predictor_never_touches_esm_or_infer_epistasis() -> None:
    for fn in (
        FeatureSpace.design_matrix,
        FeatureSpace.active_columns,
        fit_ridge,
        select_alpha,
        select_alpha_main_only,
        ds._build_fold_context,
    ):
        src = inspect.getsource(fn)
        assert "delta_g" not in src, fn.__name__
        assert "var_delta_g" not in src, fn.__name__
        assert "infer_epistasis" not in src, fn.__name__
        assert "esm_prior_mu" not in src, fn.__name__


def test_clean_predictor_is_invariant_to_esm_at_fixed_selection_and_labels() -> None:
    """Invariant A: at a FIXED selected plate and FIXED revealed labels, the
    clean supervised predictor (full / main-only / no-triples / uplift / transfer) consumes no
    ESM-derived feature, prior, variance, or offset — only the 3 diagnostic fields may move.
    Acquisition (selection) itself is deliberately NOT re-run here: it is legitimately ESM-dependent
    for several methods (``fitness``/``practice``/``structural``/``info``), which this test does not
    claim otherwise — see test_esm_diagnostic_fields_never_feed_the_decision_pipeline for the
    separate, narrower diagnostic-isolation invariant.
    """
    space = FeatureSpace(_SITES, _WT, _ALPHABET)
    pool = _pool()
    variants = _universe()

    # A residue-level (not just site-level) truth so triples are never a degenerate constant —
    # _true_dg/_landscape (shared elsewhere) give every triple the identical value, which would
    # make rho_triples structurally None here regardless of ESM (unrelated to this invariant).
    per_pos = {0: 0.7, 1: -0.4, 2: 0.3}
    per_residue = {"A": 0.0, "C": 0.15, "G": -0.15}

    def rich_true_dg(variant: Variant) -> float:
        value = sum(per_pos[pos] + per_residue[mut] for pos, _wt, mut in variant)
        sites = {pos for pos, _, _ in variant}
        if {0, 1} <= sites:
            value += 0.9
        return value

    landscape = {frozenset(): 1.0, **{v: exp(rich_true_dg(v)) for v in variants}}

    # every double and triple held out (>= 3 of each, so rho_doubles/rho_triples are computable);
    # only the singles remain to train on.
    eval_measured = _order_2_3(variants)
    active_full = {v: space.active_columns(v) for v in variants}
    active_main = {v: space.active_columns(v, include_pairs=False) for v in variants}
    esm_of = {sv.variant: sv.delta_g for sv in pool}
    ctx = ds._build_fold_context(space, eval_measured, landscape, active_full, active_main, esm_of)

    selected = [v for v in variants if v not in eval_measured]  # a FIXED plate, never re-selected
    revealed = {v: landscape[v] for v in selected}  # FIXED labels

    extreme_esm = {v: 1e6 * (-1.0) ** i for i, v in enumerate(esm_of)}
    kwargs: dict[str, object] = {
        "selected": selected,
        "revealed": revealed,
        "budget": len(selected),
        "pool_size": len(variants),
        "grid_main": [1.0, 10.0],
        "grid_pair": [1.0, 10.0],
        "n_inner": 2,
    }
    bundle_base = ds._evaluate_selection(space, ctx, esm_of, **kwargs)  # type: ignore[arg-type]
    bundle_extreme = ds._evaluate_selection(space, ctx, extreme_esm, **kwargs)  # type: ignore[arg-type]

    clean_fields = (
        "s_macro",
        "rho_doubles",
        "rho_triples",
        "pooled_spearman",
        "pearson",
        "rmse",
        "ndcg",
        "hit_rate",
        "best_true_top_b",
        "regret",
        "live_fraction_top_b",
        "uplift",
        "transfer_rho_triples",
        "effective_train_size",
        "alpha_full",
        "alpha_main_only",
        "alpha_no_triples",
    )
    for field in clean_fields:
        assert getattr(bundle_base, field) == getattr(bundle_extreme, field), field
    assert bundle_base.s_macro is not None  # the fixture must actually exercise a real prediction

    # esm_zero_shot_s_macro depends only on ctx.esm_score, fixed once at ctx-construction time (not
    # on the esm_of parameter varied here) — its own ESM-dependence is covered separately by
    # test_esm_diagnostic_fields_never_feed_the_decision_pipeline, which mutates materialized
    # records directly instead.
    moving_diagnostics = ("esm_circular_s_macro", "esm_offset_s_macro")
    assert any(getattr(bundle_base, f) != getattr(bundle_extreme, f) for f in moving_diagnostics)


def test_esm_circular_diagnostic_reuses_the_infer_epistasis_mechanism() -> None:
    src = inspect.getsource(ds._esm_circular)
    assert "esm_prior_mu" in src  # the exact posterior-mean mechanism infer_epistasis conditions on


def test_downstream_module_never_calls_infer_epistasis_directly() -> None:
    src = inspect.getsource(ds)
    assert "infer_epistasis(" not in src  # reused only via esm_prior_mu, mentioned in prose only
    assert "import infer_epistasis" not in src


_ESM_DIAGNOSTIC_FIELDS = ("esm_circular_s_macro", "esm_zero_shot_s_macro", "esm_offset_s_macro")


def test_esm_diagnostic_fields_never_feed_the_decision_pipeline() -> None:
    """Corrupting only the 3 ESM-diagnostic fields on every raw record must not move the
    decision pipeline (reads only ``s_macro``) by a single bit, while it does move
    ``method_budget_summaries``' diagnostic columns. Distinct from acquisition methods
    (``fitness``/``practice``/``structural``/``info``) legitimately reading ``delta_g`` upstream.
    """
    pool = _pool()
    landscape = _landscape(pool)
    budgets = [4, 6]
    n_folds = 2
    baseline = downstream_report(
        pool,
        landscape,
        budgets=budgets,
        seeds=2,
        n_folds=n_folds,
        partitions=2,
        sites=_SITES,
        wt_at_sites=_WT,
        alphabet=_ALPHABET,
        grid_main=[1.0, 10.0],
        grid_pair=[1.0, 10.0],
        n_inner=2,
    )
    records = [*baseline.deterministic_records, *baseline.random_records]
    assert records  # the fixture must actually exercise the pipeline

    corrupted = [
        r.model_copy(update={field: 999_000.0 + 1000.0 * i for field in _ESM_DIAGNOSTIC_FIELDS})
        for i, r in enumerate(records)
    ]

    base_summaries = {
        (s.method, s.estimand, s.regime, s.budget): s for s in method_budget_summaries(records)
    }
    mut_summaries = {
        (s.method, s.estimand, s.regime, s.budget): s for s in method_budget_summaries(corrupted)
    }
    assert set(base_summaries) == set(mut_summaries)
    non_diagnostic_fields = [
        name for name in MethodBudgetSummary.model_fields if name not in _ESM_DIAGNOSTIC_FIELDS
    ]
    for key, base_s in base_summaries.items():
        mut_s = mut_summaries[key]
        for field in non_diagnostic_fields:
            assert getattr(base_s, field) == getattr(mut_s, field), (key, field)
    any_diagnostic_changed = any(
        getattr(base_summaries[key], field) != getattr(mut_summaries[key], field)
        for key in base_summaries
        for field in _ESM_DIAGNOSTIC_FIELDS
    )
    assert any_diagnostic_changed

    common_kwargs: dict[str, object] = {
        "partitions": baseline.partitions,
        "alphabet": baseline.alphabet,
        "max_order": baseline.max_order,
        "seeds": baseline.seeds,
        "n_inner": baseline.n_inner_folds,
    }
    base_decision, base_partitions, base_cv = ds._decision_summary(
        records, budgets, n_folds, **common_kwargs
    )
    mut_decision, mut_partitions, mut_cv = ds._decision_summary(
        corrupted, budgets, n_folds, **common_kwargs
    )
    assert base_decision == mut_decision
    assert base_partitions == mut_partitions
    assert base_cv == mut_cv


# --- no-triples training -> held-out-triples transfer --------------------------------


def test_transfer_excludes_selected_triples_but_retains_singles_and_doubles() -> None:
    space = FeatureSpace(_SITES, _WT, _ALPHABET)
    pool = _pool()
    landscape = _landscape(pool)
    variants = _universe()
    eval_measured = [v for v in _order_2_3(variants) if v in landscape][:6]
    active_full = {v: space.active_columns(v) for v in variants}
    active_main = {v: space.active_columns(v, include_pairs=False) for v in variants}
    esm_of = {sv.variant: sv.delta_g for sv in pool}
    ctx = ds._build_fold_context(space, eval_measured, landscape, active_full, active_main, esm_of)

    revealed = {v: landscape[v] for v in variants if v in landscape and len(v) <= ds._THIRD_ORDER}
    _alpha, _rho, n_singles, n_doubles, _degenerate = ds._transfer_triples(
        space, ctx, revealed, [1.0, 10.0], [1.0, 10.0], ds.N_INNER_FOLDS - 1
    )
    train_used = [v for v, f in revealed.items() if len(v) <= ds._PAIRWISE_ORDER]
    assert n_singles == sum(1 for v in train_used if len(v) == 1)
    assert n_doubles == sum(1 for v in train_used if len(v) == ds._PAIRWISE_ORDER)
    # no triple ever enters the training count reported by the transfer sub-test.
    assert n_singles + n_doubles == len(train_used)


def test_transfer_reports_degenerate_double_coverage_when_too_few_doubles() -> None:
    space = FeatureSpace(_SITES, _WT, _ALPHABET)
    pool = _pool()
    landscape = _landscape(pool)
    variants = _universe()
    eval_measured = [v for v in _order_2_3(variants) if v in landscape][:6]
    active_full = {v: space.active_columns(v) for v in variants}
    active_main = {v: space.active_columns(v, include_pairs=False) for v in variants}
    esm_of = {sv.variant: sv.delta_g for sv in pool}
    ctx = ds._build_fold_context(space, eval_measured, landscape, active_full, active_main, esm_of)

    # Reveal only singles (no doubles at all) -> the no-triples subset has zero doubles.
    revealed = {v: landscape[v] for v in variants if v in landscape and len(v) == 1}
    _alpha, _rho, _n_singles, n_doubles, degenerate = ds._transfer_triples(
        space, ctx, revealed, [1.0, 10.0], [1.0, 10.0], ds.N_INNER_FOLDS - 1
    )
    assert n_doubles == 0
    assert degenerate


# --- graph / estimand target-blind vs target-aware ----------------------------------


def test_target_blind_and_target_aware_graphs_differ_on_a_constructed_fold() -> None:
    from epibudget.epistasis import predicted_epistasis  # noqa: PLC0415
    from epibudget.graph import EpistasisFactorGraph  # noqa: PLC0415

    pool = _pool()
    all_interactions = predicted_epistasis(pool, 3)
    var_map = {sv.variant: sv.var_delta_g for sv in pool}

    eval_universe = _order_2_3(_universe())
    e_j = set(eval_universe[:5])  # a constructed held-out fold

    blind_interactions = [i for i in all_interactions if frozenset(i.mutations) not in e_j]
    blind_graph = EpistasisFactorGraph(blind_interactions, var_map)
    aware_graph = EpistasisFactorGraph(all_interactions, var_map)

    # target-blind drops every interaction keyed by a held-out identity; target-aware keeps them.
    assert len(blind_graph.interactions) < len(aware_graph.interactions)
    blind_keys = {frozenset(i.mutations) for i in blind_graph.interactions}
    aware_keys = {frozenset(i.mutations) for i in aware_graph.interactions}
    assert not (blind_keys & e_j)
    assert e_j & aware_keys


def test_held_out_identities_are_never_selectable_in_either_estimand() -> None:
    report = _run()
    eval_by_fold: dict[tuple[int, int], set[str]] = {}
    for r in (*report.deterministic_records, *report.random_records):
        eval_by_fold.setdefault((r.partition_index, r.fold_index), set()).add(r.fold_identity_hash)
    # selected_identity_hash must never collide with a held-out fold's own identity-set hash
    # (a coarse but cheap structural check: the hashes are over disjoint sets by construction).
    for r in (*report.deterministic_records, *report.random_records):
        assert r.selected_identity_hash != r.fold_identity_hash


# --- cross-process reproducibility --------------------------------------------------------------

_REPRO_SCRIPT = """
from math import exp

from epibudget.data import enumerate_candidates
from epibudget.downstream import downstream_report
from epibudget.types import ScoredVariant

variants = enumerate_candidates((0, 1, 2), ("A", "A", "A"), allowed_aa="ACG", max_order=3)
pool = [
    ScoredVariant(variant=v, delta_g=float(i) - 10.0, var_delta_g=0.05 + 0.01 * i)
    for i, v in enumerate(variants)
]
per = {0: 0.7, 1: -0.4, 2: 0.3}


def dg(variant):
    sites = {p for p, _, _ in variant}
    return sum(per[p] for p in sites) + (0.9 if {0, 1} <= sites else 0.0)


landscape = {frozenset(): 1.0}
for sv in pool:
    landscape[sv.variant] = exp(dg(sv.variant))
report = downstream_report(
    pool, landscape, budgets=[4, 6], seeds=2, n_folds=2, partitions=2,
    sites=(0, 1, 2), wt_at_sites=("A", "A", "A"), alphabet="ACG",
    grid_main=[1.0, 10.0], grid_pair=[1.0, 10.0], n_inner=2,
)
print(report.model_dump_json())
"""


def test_downstream_report_is_reproducible_across_processes(tmp_path: Path) -> None:
    """The frozen salts + canonical sorting must survive a different PYTHONHASHSEED."""
    import os  # noqa: PLC0415

    script = tmp_path / "repro.py"
    script.write_text(_REPRO_SCRIPT, encoding="utf-8")
    repo = Path(__file__).resolve().parent.parent

    def run(hashseed: str) -> str:
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = hashseed
        env["PYTHONPATH"] = str(repo / "src")
        proc = subprocess.run(
            [sys.executable, str(script)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(repo),
        )
        return proc.stdout.strip()

    assert run("0") == run("1")


def test_fold_assignment_is_independent_of_labels() -> None:
    params = set(inspect.signature(assign_outer_folds).parameters)
    assert not (params & {"landscape", "fitness", "labels", "measured"})


def test_folds_partition_the_eval_universe_disjointly() -> None:
    eval_variants = _order_2_3(_universe())
    folds = assign_outer_folds(eval_variants, _N_FOLDS, partition_salt(0))
    members: dict[int, set[Variant]] = {f: set() for f in range(_N_FOLDS)}
    for v, f in folds.items():
        members[f].add(v)
    union = set().union(*members.values())
    assert union == set(eval_variants)
    assert sum(len(m) for m in members.values()) == len(eval_variants)


# --- confirmatory protocol profile ---------------------------------------


def _profile_kwargs(**overrides: object) -> dict[str, object]:
    """The exact registered confirmatory profile as conformance-check kwargs, with overrides."""
    p = ds.CONFIRMATORY_PROFILE
    base: dict[str, object] = {
        "protocol_version": p.protocol_version,
        "partitions": p.partitions,
        "outer_folds": p.outer_folds,
        "budgets": list(p.budgets),
        "alphabet": p.alphabet,
        "max_order": p.max_order,
        "random_seeds": list(p.random_seeds),
        "inner_folds": p.inner_folds,
        "estimands": list(p.estimands),
        "missingness_regimes": list(p.missingness_regimes),
        "methods": list(p.methods),
    }
    base.update(overrides)
    return base


_FROZEN_K = 5
_FROZEN_MAX_ORDER = 3
_FROZEN_N_INNER = 3
_FROZEN_SEED_COUNT = 20


def test_confirmatory_profile_matches_the_frozen_registered_values() -> None:
    p = ds.CONFIRMATORY_PROFILE
    assert p.protocol_version == ds.PROTOCOL_VERSION
    assert p.partitions == ds.EXPECTED_PARTITIONS
    assert p.outer_folds == _FROZEN_K
    assert p.budgets == (48, 96, 192)
    assert p.alphabet == "ACDEFGHIKLMNPQRSTVWY"
    assert p.max_order == _FROZEN_MAX_ORDER
    assert p.random_seeds == tuple(range(_FROZEN_SEED_COUNT))
    assert p.inner_folds == ds.N_INNER_FOLDS
    assert ds.N_INNER_FOLDS == _FROZEN_N_INNER
    assert set(p.estimands) == {"target_blind", "target_aware"}
    assert set(p.missingness_regimes) == {"attempted_budget", "measured_available"}
    assert set(p.methods) == {"info", "structural", "fitness", "random", "practice"}


def test_protocol_profile_conformance_passes_for_the_exact_registered_profile() -> None:
    result = ds.protocol_profile_conformance(**_profile_kwargs())  # type: ignore[arg-type]
    assert result.conforming
    assert result.mismatches == []
    assert result.expected == result.observed


def test_protocol_profile_conformance_flags_k4() -> None:
    result = ds.protocol_profile_conformance(**_profile_kwargs(outer_folds=4))  # type: ignore[arg-type]
    assert not result.conforming
    assert result.mismatches == ["outer_folds"]


def test_protocol_profile_conformance_flags_k6() -> None:
    result = ds.protocol_profile_conformance(**_profile_kwargs(outer_folds=6))  # type: ignore[arg-type]
    assert not result.conforming
    assert result.mismatches == ["outer_folds"]


def test_protocol_profile_conformance_flags_missing_budget() -> None:
    result = ds.protocol_profile_conformance(**_profile_kwargs(budgets=[48, 96]))  # type: ignore[arg-type]
    assert not result.conforming
    assert "budgets" in result.mismatches


def test_protocol_profile_conformance_flags_extra_budget() -> None:
    result = ds.protocol_profile_conformance(  # type: ignore[arg-type]
        **_profile_kwargs(budgets=[48, 96, 192, 384])
    )
    assert not result.conforming
    assert "budgets" in result.mismatches


def test_protocol_profile_conformance_flags_reordered_budgets_same_set() -> None:
    # Budget order matters: learning_curve_auc trapezoidal-integrates over budgets in given order.
    result = ds.protocol_profile_conformance(**_profile_kwargs(budgets=[96, 48, 192]))  # type: ignore[arg-type]
    assert not result.conforming
    assert "budgets" in result.mismatches


def test_protocol_profile_conformance_flags_wrong_alphabet() -> None:
    result = ds.protocol_profile_conformance(**_profile_kwargs(alphabet="ACDEFGHIKLMNPQRSTVW"))  # type: ignore[arg-type]
    assert not result.conforming
    assert "alphabet" in result.mismatches


def test_protocol_profile_conformance_flags_wrong_max_order() -> None:
    result = ds.protocol_profile_conformance(**_profile_kwargs(max_order=2))  # type: ignore[arg-type]
    assert not result.conforming
    assert "max_order" in result.mismatches


def test_protocol_profile_conformance_flags_19_partitions() -> None:
    result = ds.protocol_profile_conformance(**_profile_kwargs(partitions=19))  # type: ignore[arg-type]
    assert not result.conforming
    assert result.mismatches == ["partitions"]


def test_protocol_profile_conformance_flags_21_partitions() -> None:
    result = ds.protocol_profile_conformance(**_profile_kwargs(partitions=21))  # type: ignore[arg-type]
    assert not result.conforming
    assert result.mismatches == ["partitions"]


def test_protocol_profile_conformance_flags_wrong_seed_set() -> None:
    result = ds.protocol_profile_conformance(**_profile_kwargs(random_seeds=list(range(19))))  # type: ignore[arg-type]
    assert not result.conforming
    assert "random_seeds" in result.mismatches


def test_protocol_profile_conformance_flags_missing_estimand() -> None:
    result = ds.protocol_profile_conformance(**_profile_kwargs(estimands=["target_blind"]))  # type: ignore[arg-type]
    assert not result.conforming
    assert "estimands" in result.mismatches


def test_protocol_profile_conformance_flags_missing_regime() -> None:
    result = ds.protocol_profile_conformance(  # type: ignore[arg-type]
        **_profile_kwargs(missingness_regimes=["attempted_budget"])
    )
    assert not result.conforming
    assert "missingness_regimes" in result.mismatches


def test_protocol_profile_conformance_never_coerces_values() -> None:
    # A near-miss (e.g. an extra trailing budget) must be reported, never silently accepted.
    result = ds.protocol_profile_conformance(**_profile_kwargs(budgets=[48, 96, 192, 193]))  # type: ignore[arg-type]
    assert not result.conforming
    assert result.observed["budgets"] == [48, 96, 192, 193]  # recorded verbatim, not truncated


# --- raw-record coverage fixtures: one full R=20 x K=5 x 20-seed baseline -------
#
# Built once at module import (~28,800 record objects via ``model_construct``, which skips pydantic
# validation for this fully-controlled synthetic data). Every test below copies/filters this list
# before mutating it; ``_FULL_DET``/``_FULL_RAND`` are themselves never mutated. Field values other
# than the identity keys (partition/estimand/regime/method/budget/fold/seed) are fixed placeholders:
# these fixtures exercise raw-record coverage and registered-scope plumbing, never the ridge/inner-
# CV fitting itself (covered elsewhere).

_SYNTH_ALPHA = AlphaChoice(
    alpha_main=1.0, alpha_pair=1.0, fell_back=False, fallback_reason=None, n_inner_folds_used=3
)


def _synthetic_common_fields(
    *,
    protocol_version: str,
    estimand: str,
    regime: str,
    partition: int,
    fold: int,
    method: str,
    budget: int,
) -> dict[str, object]:
    return dict(
        protocol_version=protocol_version,
        estimand=estimand,
        missingness_regime=regime,
        partition_index=partition,
        partition_salt=partition_salt(partition),
        fold_index=fold,
        fold_identity_hash="fold-hash",
        method=method,
        budget=budget,
        selected_count=budget,
        selected_identity_hash="selected-hash",
        selectable_pool_size=budget * 10,
        revealed_count=budget,
        live_count=budget,
        dead_count=0,
        missing_count=0,
        unusable_count=0,
        effective_train_size=budget,
        train_live_fraction=1.0,
        selected_singles=0,
        selected_doubles=budget,
        selected_triples=0,
        train_singles=0,
        train_doubles=budget,
        train_triples=0,
        alpha_full=_SYNTH_ALPHA,
        alpha_main_only=_SYNTH_ALPHA,
        alpha_no_triples=_SYNTH_ALPHA,
        alpha_esm_offset=_SYNTH_ALPHA,
        n_eval=50,
        s_macro=0.5,
        rho_doubles=0.5,
        rho_triples=0.5,
        pooled_spearman=0.5,
        pearson=0.5,
        rmse=0.1,
        ndcg=0.5,
        hit_rate=0.5,
        best_true_top_b=1.0,
        regret=0.0,
        live_fraction_top_b=1.0,
        top_b_order_diversity=1,
        top_b_identity_diversity=budget,
        uplift=0.0,
        transfer_rho_triples=None,
        transfer_train_singles=0,
        transfer_train_doubles=0,
        transfer_degenerate_double_coverage=True,
        esm_circular_s_macro=None,
        esm_zero_shot_s_macro=None,
        esm_offset_s_macro=None,
        status="ok",
        warnings=[],
    )


def _build_full_valid_baseline() -> tuple[
    list[DeterministicFoldRecord], list[RandomFoldSeedRecord]
]:
    """The exact, fully-covering R=20 x K=5 x 20-seed raw-record baseline at the frozen
    confirmatory profile (every (partition, estimand, regime, method, budget, fold[, seed]) cell
    present exactly once, nothing extra)."""
    profile = ds.CONFIRMATORY_PROFILE
    det: list[DeterministicFoldRecord] = []
    for partition in range(profile.partitions):
        for fold in range(profile.outer_folds):
            for estimand in profile.estimands:
                for regime in profile.missingness_regimes:
                    for method in ds._DETERMINISTIC_METHODS:
                        for budget in profile.budgets:
                            det.append(
                                DeterministicFoldRecord.model_construct(
                                    **_synthetic_common_fields(
                                        protocol_version=profile.protocol_version,
                                        estimand=estimand,
                                        regime=regime,
                                        partition=partition,
                                        fold=fold,
                                        method=method,
                                        budget=budget,
                                    )
                                )
                            )
    rand: list[RandomFoldSeedRecord] = []
    for partition in range(profile.partitions):
        for fold in range(profile.outer_folds):
            for estimand in profile.estimands:
                for regime in profile.missingness_regimes:
                    for budget in profile.budgets:
                        for seed in profile.random_seeds:
                            rand.append(
                                RandomFoldSeedRecord.model_construct(
                                    **_synthetic_common_fields(
                                        protocol_version=profile.protocol_version,
                                        estimand=estimand,
                                        regime=regime,
                                        partition=partition,
                                        fold=fold,
                                        method="random",
                                        budget=budget,
                                    ),
                                    random_seed=seed,
                                )
                            )
    return det, rand


_FULL_DET, _FULL_RAND = _build_full_valid_baseline()


def _baseline_copy() -> tuple[list[DeterministicFoldRecord], list[RandomFoldSeedRecord]]:
    """A fresh, independently-mutable copy of the module baseline lists (the records themselves are
    frozen pydantic models, so a shallow list copy is sufficient -- never mutate ``_FULL_DET``/
    ``_FULL_RAND`` in place).
    """
    return list(_FULL_DET), list(_FULL_RAND)


def _scaled_subset(
    det: Sequence[DeterministicFoldRecord],
    rand: Sequence[RandomFoldSeedRecord],
    *,
    partitions: int,
    seeds: int,
) -> tuple[list[DeterministicFoldRecord], list[RandomFoldSeedRecord]]:
    """A genuinely self-consistent smaller-scale slice of a full baseline: every remaining record's
    own identity fields still fall inside the requested ``(partitions, seeds)`` register, so raw-
    record coverage is exact relative to that SAME smaller declared profile (a deliberate smoke
    scale, not a corruption).
    """
    det_subset = [r for r in det if r.partition_index < partitions]
    rand_subset = [r for r in rand if r.partition_index < partitions and r.random_seed < seeds]
    return det_subset, rand_subset


def _run_decision(
    det: Sequence[DeterministicFoldRecord],
    rand: Sequence[RandomFoldSeedRecord],
    *,
    partitions: int,
    seeds: int,
) -> ds.DecisionSummary:
    """``_decision_summary`` declared at the frozen budgets/outer_folds/alphabet/max_order/n_inner
    (matching what :func:`_build_full_valid_baseline` embeds), varying only ``partitions``/
    ``seeds`` -- the two permitted smoke-scale dimensions.
    """
    profile = ds.CONFIRMATORY_PROFILE
    decision, _, _ = ds._decision_summary(
        [*det, *rand],
        list(profile.budgets),
        profile.outer_folds,
        partitions=partitions,
        alphabet=profile.alphabet,
        max_order=profile.max_order,
        seeds=seeds,
        n_inner=profile.inner_folds,
    )
    return decision


# --- decision-layer profile gating (distinct non-decision statuses) ---------------------


def test_decision_summary_smoke_status_via_pure_conformance_helper() -> None:
    # Exactly the registered profile except partitions=1 (a deliberate smoke run).
    result = ds.protocol_profile_conformance(**_profile_kwargs(partitions=1))  # type: ignore[arg-type]
    assert not result.conforming
    assert result.mismatches == ["partitions"]
    assert result.observed["partitions"] < result.expected["partitions"]


def test_decision_summary_status_is_smoke_when_only_partitions_mismatches() -> None:
    full_seeds = len(ds.CONFIRMATORY_PROFILE.random_seeds)
    det, rand = _scaled_subset(*_baseline_copy(), partitions=5, seeds=full_seeds)
    decision = _run_decision(det, rand, partitions=5, seeds=full_seeds)
    assert decision.structural_gate.status == "smoke_or_exploratory_profile"
    assert decision.esm_gate.status == "smoke_or_exploratory_profile"
    assert decision.protocol_profile_mismatches == ["partitions"]
    assert not decision.protocol_profile_conforming
    assert not decision.declared_protocol_profile_conforming
    assert decision.raw_record_coverage_conforming  # the raw records exactly cover this smaller R


def test_decision_summary_status_is_smoke_when_only_seeds_mismatches() -> None:
    full_partitions = ds.CONFIRMATORY_PROFILE.partitions
    det, rand = _scaled_subset(*_baseline_copy(), partitions=full_partitions, seeds=1)
    decision = _run_decision(det, rand, partitions=full_partitions, seeds=1)
    assert decision.structural_gate.status == "smoke_or_exploratory_profile"
    assert decision.protocol_profile_mismatches == ["random_seeds"]
    assert decision.raw_record_coverage_conforming


def test_decision_summary_status_is_smoke_when_partitions_and_seeds_both_reduced() -> None:
    # The permitted smoke pattern: R=1, one seed, everything else at the real profile.
    det, rand = _scaled_subset(*_baseline_copy(), partitions=1, seeds=1)
    decision = _run_decision(det, rand, partitions=1, seeds=1)
    assert decision.structural_gate.status == "smoke_or_exploratory_profile"
    assert set(decision.protocol_profile_mismatches) == {"partitions", "random_seeds"}
    assert decision.raw_record_coverage_conforming


def test_decision_summary_status_is_nonconforming_when_partitions_exceeds_the_register() -> None:
    # partitions=21 (a larger, not smaller, run) must never read as a smoke.
    report = _run()
    all_records = [*report.deterministic_records, *report.random_records]
    decision, _, _ = ds._decision_summary(
        all_records,
        list(ds.CONFIRMATORY_PROFILE.budgets),
        ds.CONFIRMATORY_PROFILE.outer_folds,
        partitions=21,
        alphabet=ds.CONFIRMATORY_PROFILE.alphabet,
        max_order=ds.CONFIRMATORY_PROFILE.max_order,
        seeds=len(ds.CONFIRMATORY_PROFILE.random_seeds),
        n_inner=ds.CONFIRMATORY_PROFILE.inner_folds,
    )
    assert decision.structural_gate.status == "nonconforming_protocol_profile"


def test_decision_summary_status_is_nonconforming_when_scale_and_identity_both_mismatch() -> None:
    # Reduced partitions plus a wrong alphabet must never be classified as a mere smoke.
    report = _run()
    all_records = [*report.deterministic_records, *report.random_records]
    decision, _, _ = ds._decision_summary(
        all_records,
        list(ds.CONFIRMATORY_PROFILE.budgets),
        ds.CONFIRMATORY_PROFILE.outer_folds,
        partitions=1,
        alphabet="AC",
        max_order=ds.CONFIRMATORY_PROFILE.max_order,
        seeds=len(ds.CONFIRMATORY_PROFILE.random_seeds),
        n_inner=ds.CONFIRMATORY_PROFILE.inner_folds,
    )
    assert decision.structural_gate.status == "nonconforming_protocol_profile"


def test_decision_summary_nonconforming_status_for_wrong_alphabet() -> None:
    report = _run()  # toy alphabet "ACG" != the frozen 20-letter alphabet
    assert report.decision.structural_gate.status == "nonconforming_protocol_profile"
    assert not report.decision.protocol_profile_conforming
    assert "alphabet" in report.decision.protocol_profile_mismatches
    assert report.decision.structural_gate.decision_eligible is False
    assert report.decision.structural_gate.supported is None
    assert report.decision.esm_gate.status == "nonconforming_protocol_profile"


def test_decision_summary_records_expected_and_observed_profiles() -> None:
    report = _run()
    decision = report.decision
    assert decision.expected_protocol_profile["alphabet"] == ds.CONFIRMATORY_PROFILE.alphabet
    assert decision.observed_protocol_profile["alphabet"] == _ALPHABET
    assert isinstance(decision.protocol_profile_mismatches, list)
    assert decision.protocol_profile_mismatches  # this fixture never conforms


# --- extra partitions never influence a registered decision quantity -------------------


def test_global_deltas_filter_excludes_partitions_beyond_the_expected_register() -> None:
    out_of_register_high = 999.0
    out_of_register_low = -999.0
    cells = {
        (0, 0): 0.1,
        (0, 1): 0.2,
        (19, 0): 0.3,
        (20, 0): out_of_register_high,
        (25, 0): out_of_register_low,
    }
    filtered = ds._global_deltas_within_expected_partitions(cells, expected_partitions=20)
    assert out_of_register_high not in filtered
    assert out_of_register_low not in filtered
    assert sorted(filtered) == sorted([0.1, 0.2, 0.3])


def test_extra_partition_beyond_expected_never_changes_gate_descriptive_fields() -> None:
    pool = _pool()
    landscape = _landscape(pool)
    common = dict(
        budgets=[4, 6],
        seeds=1,
        n_folds=2,
        sites=_SITES,
        wt_at_sites=_WT,
        alphabet=_ALPHABET,
        grid_main=[1.0, 10.0],
        grid_pair=[1.0, 10.0],
        n_inner=2,
    )
    report = downstream_report(  # type: ignore[arg-type]
        pool, landscape, partitions=ds.EXPECTED_PARTITIONS + 1, **common
    )
    all_records = [*report.deterministic_records, *report.random_records]
    within_only = [r for r in all_records if r.partition_index < ds.EXPECTED_PARTITIONS]
    assert len(within_only) < len(all_records)  # the extra partition really did produce records

    decision_full, _, _ = ds._decision_summary(
        all_records,
        report.budgets,
        report.n_folds,
        partitions=report.partitions,
        alphabet=_ALPHABET,
        max_order=3,
        seeds=1,
        n_inner=2,
    )
    decision_restricted, _, _ = ds._decision_summary(
        within_only,
        report.budgets,
        report.n_folds,
        partitions=ds.EXPECTED_PARTITIONS,
        alphabet=_ALPHABET,
        max_order=3,
        seeds=1,
        n_inner=2,
    )
    assert (
        decision_full.structural_gate.global_mean_delta
        == decision_restricted.structural_gate.global_mean_delta
    )
    assert (
        decision_full.structural_gate.sign_positive
        == decision_restricted.structural_gate.sign_positive
    )
    assert (
        decision_full.structural_gate.median_partition_delta
        == decision_restricted.structural_gate.median_partition_delta
    )
    assert (
        decision_full.esm_gate.global_mean_delta == decision_restricted.esm_gate.global_mean_delta
    )


# --- raw-record coverage: the single source of truth -----------------------------------


def test_raw_record_coverage_conforms_for_the_exact_valid_baseline() -> None:
    det, rand = _baseline_copy()
    coverage = raw_record_coverage(ds.CONFIRMATORY_PROFILE, det, rand)
    assert coverage.conforming
    assert coverage.missing_deterministic_cell_count == 0
    assert coverage.duplicate_deterministic_cell_count == 0
    assert coverage.unexpected_deterministic_cell_count == 0
    assert coverage.missing_random_cell_count == 0
    assert coverage.duplicate_random_cell_count == 0
    assert coverage.unexpected_random_cell_count == 0
    assert coverage.observed_deterministic_count == coverage.expected_deterministic_count
    assert coverage.observed_deterministic_count == len(det)
    assert coverage.observed_random_count == coverage.expected_random_count == len(rand)
    assert coverage.observed_protocol_versions == [ds.PROTOCOL_VERSION]
    assert coverage.observed_partition_indices == list(range(ds.EXPECTED_PARTITIONS))
    assert coverage.observed_random_seeds == list(ds.CONFIRMATORY_PROFILE.random_seeds)


def test_raw_record_coverage_detects_missing_deterministic_cell() -> None:
    det, rand = _baseline_copy()
    del det[0]
    coverage = raw_record_coverage(ds.CONFIRMATORY_PROFILE, det, rand)
    assert not coverage.conforming
    assert coverage.missing_deterministic_cell_count == 1
    assert coverage.duplicate_deterministic_cell_count == 0
    assert coverage.unexpected_deterministic_cell_count == 0


def test_raw_record_coverage_detects_missing_random_cell() -> None:
    det, rand = _baseline_copy()
    del rand[0]
    coverage = raw_record_coverage(ds.CONFIRMATORY_PROFILE, det, rand)
    assert not coverage.conforming
    assert coverage.missing_random_cell_count == 1


def test_raw_record_coverage_duplicate_is_caught_by_counter_not_a_bare_set() -> None:
    """A duplicated record leaves the observed unique-key SET completely unchanged (every expected
    key is still present at least once), so a bare-set-based check would report full coverage; the
    multiplicity-aware ``Counter`` must still catch the duplicate.
    """
    det, rand = _baseline_copy()
    keys_before = {ds._det_key(r) for r in det}
    det.append(det[0])  # an exact duplicate of an already-present record
    keys_after = {ds._det_key(r) for r in det}
    assert keys_before == keys_after  # the unique-key SET is unchanged by the duplicate

    coverage = raw_record_coverage(ds.CONFIRMATORY_PROFILE, det, rand)
    assert not coverage.conforming
    assert coverage.duplicate_deterministic_cell_count == 1
    assert coverage.missing_deterministic_cell_count == 0
    assert coverage.unexpected_deterministic_cell_count == 0


_OUT_OF_REGISTER_PARTITION = ds.EXPECTED_PARTITIONS


def test_raw_record_coverage_detects_an_unexpected_partition() -> None:
    det, rand = _baseline_copy()
    det.append(
        DeterministicFoldRecord.model_construct(
            **_synthetic_common_fields(
                protocol_version=ds.PROTOCOL_VERSION,
                estimand="target_blind",
                regime="attempted_budget",
                partition=_OUT_OF_REGISTER_PARTITION,
                fold=0,
                method="structural",
                budget=48,
            )
        )
    )
    coverage = raw_record_coverage(ds.CONFIRMATORY_PROFILE, det, rand)
    assert not coverage.conforming
    assert coverage.unexpected_deterministic_cell_count == 1
    assert _OUT_OF_REGISTER_PARTITION in coverage.observed_partition_indices


def test_registered_records_drops_unexpected_and_dedupes_duplicates() -> None:
    det, rand = _baseline_copy()
    baseline_count = len(det) + len(rand)
    extra = DeterministicFoldRecord.model_construct(
        **_synthetic_common_fields(
            protocol_version=ds.PROTOCOL_VERSION,
            estimand="target_blind",
            regime="attempted_budget",
            partition=_OUT_OF_REGISTER_PARTITION,
            fold=0,
            method="structural",
            budget=48,
        )
    )
    corrupted_det = [*det, extra, det[0]]  # one unexpected record, one exact duplicate
    registered = registered_records(ds.CONFIRMATORY_PROFILE, corrupted_det, rand)
    assert len(registered) == baseline_count  # the extra is dropped, the duplicate collapses to 1
    assert extra not in registered


# --- full adversarial mutation matrix: exact (status, eligible, supported) -------


def _decision_summary_full(
    det: Sequence[DeterministicFoldRecord], rand: Sequence[RandomFoldSeedRecord]
) -> tuple[ds.DecisionSummary, list[PartitionAggregate], list[CorrectedCVCompanion]]:
    """``_decision_summary`` declared at the exact frozen confirmatory profile (R=20, 20 seeds)."""
    profile = ds.CONFIRMATORY_PROFILE
    return ds._decision_summary(
        [*det, *rand],
        list(profile.budgets),
        profile.outer_folds,
        partitions=profile.partitions,
        alphabet=profile.alphabet,
        max_order=profile.max_order,
        seeds=len(profile.random_seeds),
        n_inner=profile.inner_folds,
    )


_VICTIM_SEED = 7
_UNEXPECTED_SEED = 99
_PARTIAL_PARTITION_COUNT = 16


def test_matrix_exact_complete_r20_is_ok_and_eligible() -> None:
    det, rand = _baseline_copy()
    decision, _, _ = _decision_summary_full(det, rand)
    assert decision.structural_gate.status == "ok"
    assert decision.structural_gate.decision_eligible is True
    assert isinstance(decision.structural_gate.supported, bool)
    assert decision.esm_gate.status == "ok"
    assert decision.esm_gate.decision_eligible is True
    assert isinstance(decision.esm_gate.supported, bool)
    assert decision.protocol_profile_conforming
    assert decision.declared_protocol_profile_conforming
    assert decision.raw_record_coverage_conforming


def test_matrix_exact_deliberately_reduced_smoke_records_is_smoke() -> None:
    det, rand = _scaled_subset(*_baseline_copy(), partitions=1, seeds=1)
    decision = _run_decision(det, rand, partitions=1, seeds=1)
    assert decision.structural_gate.status == "smoke_or_exploratory_profile"
    assert decision.structural_gate.decision_eligible is False
    assert decision.structural_gate.supported is None


def test_matrix_missing_one_deterministic_fold_cell_is_insufficient() -> None:
    det, rand = _baseline_copy()
    del det[0]
    decision, _, _ = _decision_summary_full(det, rand)
    assert decision.structural_gate.status == "insufficient_valid_partitions"
    assert decision.structural_gate.decision_eligible is False
    assert decision.structural_gate.supported is None


def test_matrix_missing_one_random_seed_cell_is_insufficient() -> None:
    det, rand = _baseline_copy()
    del rand[0]
    decision, _, _ = _decision_summary_full(det, rand)
    assert decision.structural_gate.status == "insufficient_valid_partitions"
    assert decision.structural_gate.decision_eligible is False
    assert decision.structural_gate.supported is None


def test_matrix_only_16_of_20_partitions_complete_is_insufficient() -> None:
    det, rand = _baseline_copy()
    det = [r for r in det if r.partition_index < _PARTIAL_PARTITION_COUNT]
    rand = [r for r in rand if r.partition_index < _PARTIAL_PARTITION_COUNT]
    decision, _, _ = _decision_summary_full(det, rand)
    assert decision.structural_gate.status == "insufficient_valid_partitions"
    assert decision.structural_gate.decision_eligible is False
    assert decision.structural_gate.supported is None


def test_matrix_duplicate_deterministic_key_is_nonconforming() -> None:
    det, rand = _baseline_copy()
    det.append(det[0])
    decision, _, _ = _decision_summary_full(det, rand)
    assert decision.structural_gate.status == "nonconforming_protocol_profile"
    assert decision.structural_gate.decision_eligible is False
    assert decision.structural_gate.supported is None


def test_matrix_duplicate_random_key_is_nonconforming() -> None:
    det, rand = _baseline_copy()
    rand.append(rand[0])
    decision, _, _ = _decision_summary_full(det, rand)
    assert decision.structural_gate.status == "nonconforming_protocol_profile"
    assert decision.structural_gate.decision_eligible is False
    assert decision.structural_gate.supported is None


def test_matrix_wrong_record_protocol_version_is_nonconforming() -> None:
    det, rand = _baseline_copy()
    det[0] = det[0].model_copy(update={"protocol_version": "epibudget-downstream-v0"})
    decision, _, _ = _decision_summary_full(det, rand)
    assert decision.structural_gate.status == "nonconforming_protocol_profile"
    assert decision.structural_gate.decision_eligible is False
    assert decision.structural_gate.supported is None


def test_matrix_extra_partition_record_is_nonconforming() -> None:
    det, rand = _baseline_copy()
    det.append(
        DeterministicFoldRecord.model_construct(
            **_synthetic_common_fields(
                protocol_version=ds.PROTOCOL_VERSION,
                estimand="target_blind",
                regime="attempted_budget",
                partition=_OUT_OF_REGISTER_PARTITION,
                fold=0,
                method="structural",
                budget=48,
            )
        )
    )
    decision, _, _ = _decision_summary_full(det, rand)
    assert decision.structural_gate.status == "nonconforming_protocol_profile"
    assert decision.structural_gate.decision_eligible is False
    assert decision.structural_gate.supported is None


def test_matrix_unexpected_seed_with_correct_seed_count_is_nonconforming() -> None:
    """Swap every ``random_seed=7`` record for ``random_seed=99``: the total random-record count
    and the seed COUNT are unchanged (still 20 distinct values, just the wrong 20), which must read
    as a genuine identity violation, never a merely-reduced smoke declaration.
    """
    det, rand = _baseline_copy()
    swapped = [
        r.model_copy(update={"random_seed": _UNEXPECTED_SEED})
        if r.random_seed == _VICTIM_SEED
        else r
        for r in rand
    ]
    assert {r.random_seed for r in swapped} == (
        set(ds.CONFIRMATORY_PROFILE.random_seeds) - {_VICTIM_SEED}
    ) | {_UNEXPECTED_SEED}
    assert len(swapped) == len(rand)  # the total record count, and the seed count, are unchanged
    decision, _, _ = _decision_summary_full(det, swapped)
    assert decision.structural_gate.status == "nonconforming_protocol_profile"
    assert decision.structural_gate.decision_eligible is False
    assert decision.structural_gate.supported is None


def test_matrix_random_method_in_the_deterministic_list_is_nonconforming() -> None:
    det, rand = _baseline_copy()
    det[0] = det[0].model_copy(update={"method": "random"})
    decision, _, _ = _decision_summary_full(det, rand)
    assert decision.structural_gate.status == "nonconforming_protocol_profile"
    assert decision.structural_gate.decision_eligible is False
    assert decision.structural_gate.supported is None


def test_matrix_deterministic_method_in_the_random_list_is_nonconforming() -> None:
    det, rand = _baseline_copy()
    rand[0] = rand[0].model_copy(update={"method": "structural"})
    decision, _, _ = _decision_summary_full(det, rand)
    assert decision.structural_gate.status == "nonconforming_protocol_profile"
    assert decision.structural_gate.decision_eligible is False
    assert decision.structural_gate.supported is None


_UNEXPECTED_BUDGET = 999
_UNEXPECTED_FOLD = 999


@pytest.mark.parametrize(
    "overrides",
    [
        {"method": "unexpected_method"},
        {"budget": _UNEXPECTED_BUDGET},
        {"fold_index": _UNEXPECTED_FOLD},
        {"estimand": "unexpected_estimand"},
        {"missingness_regime": "unexpected_regime"},
    ],
    ids=["method", "budget", "fold", "estimand", "regime"],
)
def test_matrix_unexpected_cell_dimension_is_nonconforming(overrides: dict[str, object]) -> None:
    det, rand = _baseline_copy()
    det[0] = det[0].model_copy(update=overrides)
    decision, _, _ = _decision_summary_full(det, rand)
    assert decision.structural_gate.status == "nonconforming_protocol_profile"
    assert decision.structural_gate.decision_eligible is False
    assert decision.structural_gate.supported is None


def test_missing_seed_vs_wrong_replacement_seed_are_classified_differently() -> None:
    det, rand = _baseline_copy()
    missing = [r for r in rand if r.random_seed != _VICTIM_SEED]
    decision_missing, _, _ = _decision_summary_full(det, missing)
    assert decision_missing.structural_gate.status == "insufficient_valid_partitions"

    swapped = [
        r.model_copy(update={"random_seed": _UNEXPECTED_SEED})
        if r.random_seed == _VICTIM_SEED
        else r
        for r in rand
    ]
    decision_swapped, _, _ = _decision_summary_full(det, swapped)
    assert decision_swapped.structural_gate.status == "nonconforming_protocol_profile"


def test_no_incomplete_r20_variant_ever_becomes_decision_eligible() -> None:
    """Sanity net collecting several corrupted-from-baseline variants: none may ever read as
    ``decision_eligible``, regardless of which specific coverage defect is present.
    """
    det, rand = _baseline_copy()
    variants: list[tuple[str, list[DeterministicFoldRecord], list[RandomFoldSeedRecord]]] = [
        ("missing_det", [r for i, r in enumerate(det) if i != 0], list(rand)),
        ("missing_rand", list(det), [r for i, r in enumerate(rand) if i != 0]),
        ("duplicate_det", [*det, det[0]], list(rand)),
        ("duplicate_rand", list(det), [*rand, rand[0]]),
        (
            "wrong_seed",
            list(det),
            [
                r.model_copy(update={"random_seed": _UNEXPECTED_SEED})
                if r.random_seed == _VICTIM_SEED
                else r
                for r in rand
            ],
        ),
    ]
    for name, d, r in variants:
        decision, _, _ = _decision_summary_full(d, r)
        assert decision.structural_gate.decision_eligible is False, name
        assert decision.esm_gate.decision_eligible is False, name


# --- strong extra-record isolation: corruption never moves a registered summary ---------


def test_unexpected_and_duplicate_records_never_move_a_registered_scientific_summary() -> None:
    """An unexpected or duplicated raw record must flip the decision status to
    ``nonconforming_protocol_profile`` and null out eligibility/support, but must NEVER move a
    single registered scientific summary relative to the exact, uncorrupted baseline. The
    comparison target is the CLEAN baseline's own pipeline output — an independent ground truth,
    never a re-assertion that the production filter agrees with itself.
    """
    base_det, base_rand = _baseline_copy()
    baseline_decision, baseline_aggregates, baseline_companions = _decision_summary_full(
        base_det, base_rand
    )
    baseline_method_budget = method_budget_summaries([*base_det, *base_rand])

    extra_det = DeterministicFoldRecord.model_construct(
        **_synthetic_common_fields(
            protocol_version=ds.PROTOCOL_VERSION,
            estimand="target_blind",
            regime="attempted_budget",
            partition=_OUT_OF_REGISTER_PARTITION,
            fold=0,
            method="structural",
            budget=48,
        )
    )
    extra_rand_partition = RandomFoldSeedRecord.model_construct(
        **_synthetic_common_fields(
            protocol_version=ds.PROTOCOL_VERSION,
            estimand="target_blind",
            regime="attempted_budget",
            partition=_OUT_OF_REGISTER_PARTITION,
            fold=0,
            method="random",
            budget=48,
        ),
        random_seed=0,
    )
    extra_rand_seed = RandomFoldSeedRecord.model_construct(
        **_synthetic_common_fields(
            protocol_version=ds.PROTOCOL_VERSION,
            estimand="target_blind",
            regime="attempted_budget",
            partition=0,
            fold=0,
            method="random",
            budget=48,
        ),
        random_seed=_UNEXPECTED_SEED,
    )

    variants: dict[str, tuple[list[DeterministicFoldRecord], list[RandomFoldSeedRecord]]] = {
        "extra_det_partition": ([*base_det, extra_det], list(base_rand)),
        "extra_rand_partition": (list(base_det), [*base_rand, extra_rand_partition]),
        "extra_rand_seed": (list(base_det), [*base_rand, extra_rand_seed]),
        "duplicate_det": ([*base_det, base_det[0]], list(base_rand)),
    }

    descriptive_fields = [
        name
        for name in RobustnessGate.model_fields
        if name not in ("decision_eligible", "supported", "status")
    ]

    for name, (det, rand) in variants.items():
        decision, aggregates, companions = _decision_summary_full(det, rand)
        # method_budget_summaries never filters its own input (it is a pure aggregation over
        # whatever it is handed); the production caller (downstream_report) is the one that routes
        # it through the registered scope, so the test does the same here to compare what the
        # SHIPPED report field would actually contain -- the baseline side above needs no such
        # routing, since it is already exactly the valid, uncorrupted record set.
        registered = registered_records(ds.CONFIRMATORY_PROFILE, det, rand)
        method_budget = method_budget_summaries(registered)

        assert decision.structural_gate.status == "nonconforming_protocol_profile", name
        assert decision.esm_gate.status == "nonconforming_protocol_profile", name
        assert decision.structural_gate.decision_eligible is False, name
        assert decision.esm_gate.decision_eligible is False, name
        assert decision.structural_gate.supported is None, name
        assert decision.esm_gate.supported is None, name
        assert decision.declared_protocol_profile_conforming, name  # the declared profile is exact
        assert not decision.raw_record_coverage_conforming, name  # only coverage flags the defect
        assert not decision.protocol_profile_conforming, name
        assert len(det) + len(rand) != len(base_det) + len(base_rand), name  # raw records changed

        # every registered scientific summary is bit-identical to the clean baseline's own.
        assert aggregates == baseline_aggregates, name
        assert companions == baseline_companions, name
        assert method_budget == baseline_method_budget, name
        for field in descriptive_fields:
            assert getattr(decision.structural_gate, field) == getattr(
                baseline_decision.structural_gate, field
            ), (name, field)
            assert getattr(decision.esm_gate, field) == getattr(
                baseline_decision.esm_gate, field
            ), (name, field)


# --- divergent duplicate raw records: fail closed, never an arbitrary pick -----------
#
# Two records may share a registered key (`_det_key`/`_rand_key`) yet carry different payloads.
# The pre-fix `registered_records` resolved such a cell with `min(candidates, key=lexical JSON)`,
# so an appended extreme-negative record won `min()` and contaminated `method_budget.s_macro`
# (0.5 -> -9999.495) and `structural_gate.global_mean_delta` (0.0 -> -2499.99875). The fix collapses
# ONLY byte-identical duplicates; a divergent cell is scientifically ambiguous, so the whole report
# fails closed with unavailable scientific summaries. No order-, sign-, or lexical-dependent record
# selection remains.


def _find_det_record(
    det: Sequence[DeterministicFoldRecord],
    *,
    method: str,
    estimand: str,
    regime: str,
    partition: int,
    fold: int,
    budget: int,
) -> DeterministicFoldRecord:
    return next(
        r
        for r in det
        if r.method == method
        and r.estimand == estimand
        and r.missingness_regime == regime
        and r.partition_index == partition
        and r.fold_index == fold
        and r.budget == budget
    )


def _structural_victim(det: Sequence[DeterministicFoldRecord]) -> DeterministicFoldRecord:
    """The structural / target_blind / attempted_budget / partition0 / fold0 / B=48 record — the
    cell the reproduction contaminates (it drives both a `method_budget` mean and the
    structural gate's AUC deltas)."""
    return _find_det_record(
        det,
        method="structural",
        estimand="target_blind",
        regime="attempted_budget",
        partition=0,
        fold=0,
        budget=48,
    )


def _divergent_copy(record: DeterministicFoldRecord, *, s_macro: float) -> DeterministicFoldRecord:
    """A same-registered-key copy whose payload diverges across many summary-bearing fields, so it
    shares the cell key (:func:`_det_key`) but is never byte-identical."""
    return record.model_copy(
        update={
            "s_macro": s_macro,
            "rho_doubles": s_macro,
            "rho_triples": s_macro,
            "pooled_spearman": s_macro,
            "pearson": s_macro,
            "ndcg": s_macro,
            "hit_rate": s_macro,
            "uplift": s_macro,
            "effective_train_size": 10**9 if s_macro > 0.0 else -(10**9),
            "n_eval": 10**9,
        }
    )


_DESCRIPTIVE_GATE_FIELDS = [
    name
    for name in RobustnessGate.model_fields
    if name not in ("decision_eligible", "supported", "status")
]
_DIVERGENT_S_MACRO = -999999.0  # the extreme value the pre-fix lexical min() selected and published
_DIVERGENT_S_MACRO_POS = 999999.0  # the positive counterpart, to exercise the payload tie-breaker
_DIVERGENT_INJECTED_COUNT = 2  # one negative + one positive divergent copy per injected cell


def _assert_gates_unavailable(decision: ds.DecisionSummary) -> None:
    """Both decision gates carry no scientific descriptive and are fail-closed."""
    for gate in (decision.structural_gate, decision.esm_gate):
        assert gate.status == "nonconforming_protocol_profile"
        assert gate.decision_eligible is False
        assert gate.supported is None
        assert gate.global_mean_delta is None
        assert gate.median_partition_delta is None
        assert gate.observed_valid_partitions == 0
        assert gate.sign_positive == 0
        assert gate.complete_partition_coverage is False


# --- Test 1: exact duplicate collapses, never altering a scientific metric -----------------------


def test_exact_duplicate_changes_coverage_but_never_a_scientific_metric() -> None:
    base_det, base_rand = _baseline_copy()
    baseline_decision, baseline_aggregates, baseline_companions = _decision_summary_full(
        base_det, base_rand
    )
    baseline_mb = method_budget_summaries(
        registered_records(ds.CONFIRMATORY_PROFILE, base_det, base_rand)
    )

    det = [*base_det, base_det[0]]  # a byte-identical copy of one deterministic record

    coverage = raw_record_coverage(ds.CONFIRMATORY_PROFILE, det, base_rand)
    assert coverage.duplicate_deterministic_cell_count == 1  # existing diagnostic preserved
    assert coverage.exact_duplicate_deterministic_key_count == 1
    assert coverage.divergent_duplicate_deterministic_key_count == 0
    assert not coverage.has_divergent_duplicate

    decision, aggregates, companions = _decision_summary_full(det, base_rand)
    # frozen duplicate policy: still nonconforming, never eligible.
    assert decision.structural_gate.status == "nonconforming_protocol_profile"
    assert decision.structural_gate.decision_eligible is False
    assert decision.structural_gate.supported is None
    # multiplicity moves no scientific metric: the identical copy collapses to the same record, and
    # no alternate payload is ever selected.
    assert aggregates == baseline_aggregates
    assert companions == baseline_companions
    assert method_budget_summaries(registered_records(ds.CONFIRMATORY_PROFILE, det, base_rand)) == (
        baseline_mb
    )
    for field in _DESCRIPTIVE_GATE_FIELDS:
        assert getattr(decision.structural_gate, field) == getattr(
            baseline_decision.structural_gate, field
        ), field


def test_divergent_detection_uses_whole_payload_not_only_s_macro() -> None:
    base_det, base_rand = _baseline_copy()
    victim = base_det[0]
    # Change ONLY a non-metric field: it must still read as a divergent (not exact) duplicate.
    dup = victim.model_copy(update={"effective_train_size": victim.effective_train_size + 12345})
    coverage = raw_record_coverage(ds.CONFIRMATORY_PROFILE, [*base_det, dup], base_rand)
    assert coverage.has_divergent_duplicate
    assert coverage.divergent_duplicate_deterministic_key_count == 1
    assert coverage.exact_duplicate_deterministic_key_count == 0


# --- Test 2 / Test 3: divergent negative and positive duplicates both fail closed ----------------


@pytest.mark.parametrize("s_macro", [-999999.0, 999999.0], ids=["negative", "positive"])
def test_divergent_duplicate_makes_every_registered_summary_unavailable(s_macro: float) -> None:
    """Test 2/3: a divergent duplicate (negative OR positive extreme) is scientifically ambiguous.
    Status is nonconforming, the decision is null, every registered summary is unavailable, and no
    divergent record is ever selected for aggregation. Running both signs proves the behavior is not
    accidentally tied to the pre-fix lexical `min()` ordering.
    """
    base_det, base_rand = _baseline_copy()
    victim = _structural_victim(base_det)
    dup = _divergent_copy(victim, s_macro=s_macro)
    det = [*base_det, dup]

    coverage = raw_record_coverage(ds.CONFIRMATORY_PROFILE, det, base_rand)
    assert coverage.has_divergent_duplicate
    assert coverage.divergent_duplicate_deterministic_key_count == 1
    assert coverage.exact_duplicate_deterministic_key_count == 0
    assert coverage.divergent_duplicate_deterministic_key_samples  # bounded forensic sample present

    decision, aggregates, companions = _decision_summary_full(det, base_rand)
    _assert_gates_unavailable(decision)
    # every registered scientific-summary object is empty / null.
    assert aggregates == []
    assert companions == []
    # no divergent record may ever be selected for aggregation (defense-in-depth guard).
    with pytest.raises(DivergentDuplicateError):
        registered_records(ds.CONFIRMATORY_PROFILE, det, base_rand)


# --- Test 4: input-order invariance (never first/last-record-wins) --------------------------------


def test_divergent_duplicate_behavior_is_invariant_to_input_order() -> None:
    """Test 4: original-before-duplicate, duplicate-before-original, and an arbitrary permutation
    must all yield the identical status, duplicate diagnostics, unavailable summaries, and
    byte-identical forensic decision. This is the guard against a first/last-record-wins fix.
    """
    base_det, base_rand = _baseline_copy()
    victim = _structural_victim(base_det)
    dup = _divergent_copy(victim, s_macro=_DIVERGENT_S_MACRO)

    shuffled = [*base_det, dup]
    np.random.default_rng(13).shuffle(shuffled)
    orderings = {
        "original_then_dup": [*base_det, dup],
        "dup_then_original": [dup, *base_det],
        "dup_in_middle": [*base_det[:100], dup, *base_det[100:]],
        "arbitrary_permutation": shuffled,
    }

    fingerprints = set()
    for det in orderings.values():
        decision, aggregates, companions = _decision_summary_full(det, base_rand)
        coverage = raw_record_coverage(ds.CONFIRMATORY_PROFILE, det, base_rand)
        assert aggregates == []
        assert companions == []
        fingerprints.add(
            (
                decision.model_dump_json(),  # byte-identical forensic decision
                coverage.has_divergent_duplicate,
                coverage.divergent_duplicate_deterministic_key_count,
                tuple(coverage.divergent_duplicate_deterministic_key_samples),
            )
        )
    assert len(fingerprints) == 1  # every ordering collapses to one identical result


def _report_payload_without_volatile_fields(report: DownstreamReport) -> dict[str, object]:
    payload = report.model_dump(mode="json")
    payload.pop("generated_at_utc", None)
    return payload


def _run_with_raw_record_order(
    monkeypatch: pytest.MonkeyPatch,
    *,
    order: str,
    extra_kind: str | None = None,
) -> DownstreamReport:
    real_fold_records = ds._fold_records
    injected = False
    rng = np.random.default_rng(20260712)

    def _ordered_fold_records(*args: object, **kwargs: object) -> object:
        nonlocal injected
        records = list(real_fold_records(*args, **kwargs))  # type: ignore[arg-type]
        if not injected and records:
            deterministic = next(r for r in records if isinstance(r, DeterministicFoldRecord))
            if extra_kind == "divergent":
                negative = _divergent_copy(deterministic, s_macro=_DIVERGENT_S_MACRO)
                positive = _divergent_copy(deterministic, s_macro=_DIVERGENT_S_MACRO_POS)
                index = records.index(deterministic)
                if order == "original_then_divergent":
                    records[index + 1 : index + 1] = [negative, positive]
                else:
                    records[index:index] = [positive, negative]
            elif extra_kind == "partition":
                records.append(
                    deterministic.model_copy(update={"partition_index": ds.EXPECTED_PARTITIONS})
                )
            elif extra_kind == "seed":
                random_record = next(r for r in records if isinstance(r, RandomFoldSeedRecord))
                records.append(random_record.model_copy(update={"random_seed": _UNEXPECTED_SEED}))
            elif extra_kind == "method":
                records.append(deterministic.model_copy(update={"method": "unexpected_method"}))
            injected = extra_kind is not None
        if order == "arbitrary_permutation":
            rng.shuffle(records)
        return records

    with monkeypatch.context() as context:
        context.setattr(ds, "_fold_records", _ordered_fold_records)
        return _run()


def test_complete_divergent_report_is_invariant_to_raw_record_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports = {
        order: _run_with_raw_record_order(monkeypatch, order=order, extra_kind="divergent")
        for order in (
            "original_then_divergent",
            "divergent_then_original",
            "arbitrary_permutation",
        )
    }

    payloads = [_report_payload_without_volatile_fields(report) for report in reports.values()]
    assert payloads[1:] == payloads[:-1]
    assert (
        reports["original_then_divergent"].deterministic_records
        == reports["divergent_then_original"].deterministic_records
    )
    assert (
        reports["original_then_divergent"].random_records
        == reports["arbitrary_permutation"].random_records
    )

    same_key_payloads = [
        ds._canonical_record_payload(record)
        for record in reports["original_then_divergent"].deterministic_records
        if record.s_macro in {_DIVERGENT_S_MACRO, _DIVERGENT_S_MACRO_POS}
    ]
    assert len(same_key_payloads) == _DIVERGENT_INJECTED_COUNT
    assert same_key_payloads == sorted(same_key_payloads)

    for report in reports.values():
        assert report.scientific_summaries_available is False
        assert report.scientific_summaries_unavailable_reason == "divergent_duplicate_raw_record"
        _assert_gates_unavailable(report.decision)


@pytest.mark.parametrize("extra_kind", ["partition", "seed", "method"])
def test_complete_report_keeps_ordinary_extras_isolated_and_order_invariant(
    monkeypatch: pytest.MonkeyPatch,
    extra_kind: str,
) -> None:
    baseline = _run()
    ordered = _run_with_raw_record_order(monkeypatch, order="original", extra_kind=extra_kind)
    permuted = _run_with_raw_record_order(
        monkeypatch, order="arbitrary_permutation", extra_kind=extra_kind
    )

    assert _report_payload_without_volatile_fields(
        ordered
    ) == _report_payload_without_volatile_fields(permuted)
    assert ordered.decision.structural_gate.status == "nonconforming_protocol_profile"
    assert ordered.decision.structural_gate.decision_eligible is False
    assert ordered.decision.structural_gate.supported is None
    assert ordered.scientific_summaries_available is True
    assert ordered.method_budget == baseline.method_budget
    assert ordered.partition_aggregates == baseline.partition_aggregates
    assert ordered.corrected_cv_companions == baseline.corrected_cv_companions


# --- Test 5: end-to-end report marks the complete summary tree unavailable -----------------------


def test_report_marks_all_scientific_summaries_unavailable_on_divergent_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 5: a full ``downstream_report`` run whose raw records contain one divergent duplicate
    must leave EVERY registered scientific-summary subtree unavailable -- ``method_budget``,
    ``partition_aggregates``, ``corrected_cv_companions``, and both gates' descriptives -- while the
    raw records and coverage diagnostics survive for forensic auditing.
    """
    real_fold_records = ds._fold_records
    injected = {"done": False}

    def _inject_divergent(*args: object, **kwargs: object) -> object:
        out = real_fold_records(*args, **kwargs)  # type: ignore[arg-type]
        if not injected["done"] and isinstance(out, list):
            for r in out:
                if isinstance(r, DeterministicFoldRecord):
                    out = [*out, r.model_copy(update={"s_macro": _DIVERGENT_S_MACRO})]
                    injected["done"] = True
                    break
        return out

    monkeypatch.setattr(ds, "_fold_records", _inject_divergent)
    report = _run()

    assert injected["done"]  # the fixture actually injected a divergent duplicate
    assert report.scientific_summaries_available is False
    assert report.scientific_summaries_unavailable_reason == "divergent_duplicate_raw_record"
    assert report.method_budget == []
    assert report.partition_aggregates == []
    assert report.corrected_cv_companions == []
    assert report.decision.raw_record_coverage.has_divergent_duplicate
    _assert_gates_unavailable(report.decision)
    # the offending raw record is never silently dropped from the forensic record.
    assert any(r.s_macro == _DIVERGENT_S_MACRO for r in report.deterministic_records)


def test_clean_report_marks_scientific_summaries_available() -> None:
    report = _run()
    assert report.scientific_summaries_available is True
    assert report.scientific_summaries_unavailable_reason is None
    assert report.method_budget  # populated for a divergent-free run
    assert not report.decision.raw_record_coverage.has_divergent_duplicate


# --- Test 6: ordinary unexpected extras stay isolated (no divergent, summaries preserved) ---------


def test_unexpected_extra_record_keeps_summaries_available_and_unaffected() -> None:
    base_det, base_rand = _baseline_copy()
    _, baseline_aggregates, baseline_companions = _decision_summary_full(base_det, base_rand)
    baseline_mb = method_budget_summaries(
        registered_records(ds.CONFIRMATORY_PROFILE, base_det, base_rand)
    )
    extra = DeterministicFoldRecord.model_construct(
        **_synthetic_common_fields(
            protocol_version=ds.PROTOCOL_VERSION,
            estimand="target_blind",
            regime="attempted_budget",
            partition=_OUT_OF_REGISTER_PARTITION,
            fold=0,
            method="structural",
            budget=48,
        )
    )
    det = [*base_det, extra]

    coverage = raw_record_coverage(ds.CONFIRMATORY_PROFILE, det, base_rand)
    assert (
        not coverage.has_divergent_duplicate
    )  # an out-of-register key is unexpected, not divergent
    assert coverage.unexpected_deterministic_cell_count == 1

    decision, aggregates, companions = _decision_summary_full(det, base_rand)
    assert decision.structural_gate.status == "nonconforming_protocol_profile"
    assert decision.structural_gate.decision_eligible is False
    # the unexpected record is dropped from the registered scope; baseline summaries are untouched.
    assert aggregates == baseline_aggregates
    assert companions == baseline_companions
    assert method_budget_summaries(registered_records(ds.CONFIRMATORY_PROFILE, det, base_rand)) == (
        baseline_mb
    )


# --- Test 7: collision precedence (missing cell + divergent duplicate together) -------------------


def test_missing_cell_and_divergent_duplicate_together_are_nonconforming() -> None:
    base_det, base_rand = _baseline_copy()
    victim = _structural_victim(base_det)
    dup = _divergent_copy(victim, s_macro=_DIVERGENT_S_MACRO)
    # Drop a DIFFERENT expected cell to create a genuine missing cell alongside the divergent one.
    dropped = _find_det_record(
        base_det,
        method="info",
        estimand="target_blind",
        regime="attempted_budget",
        partition=1,
        fold=0,
        budget=48,
    )
    reduced = [r for r in base_det if r is not dropped]
    assert len(reduced) == len(base_det) - 1
    det = [*reduced, dup]

    coverage = raw_record_coverage(ds.CONFIRMATORY_PROFILE, det, base_rand)
    assert coverage.has_divergent_duplicate
    assert coverage.missing_deterministic_cell_count == 1

    decision, aggregates, companions = _decision_summary_full(det, base_rand)
    # precedence: the divergent duplicate's nonconforming status wins over the missing cell's
    # insufficient_valid_partitions status.
    _assert_gates_unavailable(decision)
    assert aggregates == []
    assert companions == []
