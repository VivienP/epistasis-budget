"""Tests for the GB1 validation harness: inference, recovery, baselines, isolation.

All offline: a synthetic candidate pool + a toy landscape stand in for ESM scoring and real GB1. The
inference round-trips (empty reveal ⇒ ESM prior; full-loop reveal ⇒ true ε) pin the estimator; the
end-to-end run proves the harness produces a real metrics.json; the signature guards keep every
selector structurally blind to measured fitness.
"""

from __future__ import annotations

import inspect
import json
from itertools import pairwise
from math import exp
from pathlib import Path

import numpy as np
import pytest

from epibudget.acquisition import allocate
from epibudget.data import enumerate_candidates
from epibudget.epistasis import epsilon_pairwise, predicted_epistasis
from epibudget.graph import EpistasisFactorGraph
from epibudget.types import Mutation, ScoredVariant, Variant
from epibudget.validate import (
    OrderMetric,
    _calibrate_slope,
    hit_rate,
    infer_epistasis,
    map_recovery,
    practice_heuristic,
    random_selection,
    run_validation,
    structural_graph,
)

_SITES = (0, 1, 2)
_ALPHABET = "ACG"  # WT 'A' + two mutants per site
_N_METHODS = 5  # info, fitness, structural, practice, random
_RANDOM_B = 5
_PRACTICE_B = 6
_PAIR_ORDER = 2
_TRIPLE_ORDER = 3


def _pool() -> list[ScoredVariant]:
    """A complete order-1..3 pool over three sites with distinct ESM-style scores."""
    variants = enumerate_candidates(_SITES, ("A", "A", "A"), allowed_aa=_ALPHABET, max_order=3)
    return [
        ScoredVariant(variant=v, delta_g=float(i) - 10.0, var_delta_g=0.05 + 0.01 * i)
        for i, v in enumerate(variants)
    ]


def _true_dg(variant: Variant) -> float:
    """A non-additive ground-truth ΔG: additive main effects + a pairwise coupling on sites 0,1."""
    per_site = {0: 0.7, 1: -0.4, 2: 0.3}
    sites = {pos for pos, _, _ in variant}
    value = sum(per_site[p] for p in sites)
    if {0, 1} <= sites:
        value += 0.9  # a genuine order-2 interaction, so Var[ε] > 0
    return value


def _landscape(pool: list[ScoredVariant]) -> dict[Variant, float]:
    """Fitness = exp(ΔG_true) for every candidate and the WT (all live, so nothing is dropped)."""
    landscape = {frozenset(): 1.0}
    for sv in pool:
        landscape[sv.variant] = exp(_true_dg(sv.variant))
    return landscape


# --- inference round-trips ---------------------------------------------------------------------


def test_infer_empty_reveal_reproduces_the_esm_prior() -> None:
    pool = _pool()
    inferred = {i.mutations: i.epsilon_hat for i in infer_epistasis({}, pool)}
    predicted = {i.mutations: i.epsilon_hat for i in predicted_epistasis(pool)}
    assert inferred.keys() == predicted.keys()
    for term, value in predicted.items():
        assert inferred[term] == pytest.approx(value)


def test_infer_full_loop_reveal_reproduces_ground_truth() -> None:
    pool = _pool()
    mi: Mutation = (0, "A", "C")
    mj: Mutation = (1, "A", "C")
    loop = [frozenset({mi}), frozenset({mj}), frozenset({mi, mj})]
    revealed = {v: _true_dg(v) for v in loop}  # measure the full loop with true ΔG
    inferred = {i.mutations: i.epsilon_hat for i in infer_epistasis(revealed, pool)}
    true_eps = epsilon_pairwise({v: _true_dg(v) for v in loop}, mi, mj)
    assert inferred[(mi, mj)] == pytest.approx(true_eps)


def test_calibrate_slope_ignores_the_best_fit_intercept() -> None:
    # Points sit on y = 1 + 2x (affine slope 2), but the through-origin slope is Σxy/Σx² = 34/14.
    assert _calibrate_slope([1.0, 2.0, 3.0], [3.0, 5.0, 7.0]) == pytest.approx(34.0 / 14.0)
    assert _calibrate_slope([], []) == pytest.approx(1.0)  # degenerate → identity


def test_infer_partial_loop_prior_has_no_intercept_offset() -> None:
    # Reveal two pairs (to fit the slope) but not the singles of the target pair, on a measured
    # scale unrelated to ESM. Unmeasured loop members must be b·ESM (through origin), not a + b·ESM;
    # a stray intercept would bias this partially-measured ε.
    pool = _pool()
    esm = {sv.variant: sv.delta_g for sv in pool}
    mi, mj, mk, ml = (0, "A", "C"), (1, "A", "C"), (0, "A", "G"), (2, "A", "C")
    pair1, pair2 = frozenset({mi, mj}), frozenset({mk, ml})
    revealed = {pair1: 3.0, pair2: 5.0}

    inferred = {i.mutations: i.epsilon_hat for i in infer_epistasis(revealed, pool)}
    b = _calibrate_slope([esm[pair1], esm[pair2]], [3.0, 5.0])
    expected = revealed[pair1] - b * esm[frozenset({mi})] - b * esm[frozenset({mj})]
    assert inferred[(mi, mj)] == pytest.approx(expected)


# --- recovery ----------------------------------------------------------------------------------


def test_map_recovery_is_perfect_when_inferred_equals_truth() -> None:
    pool = _pool()
    inferred = predicted_epistasis(pool)
    truth = {i.mutations: i.epsilon_hat for i in inferred}  # inferred == truth by construction
    metrics = map_recovery(inferred, truth, frozenset(), with_ci=False)
    pooled = next(m for m in metrics if m.order == "pooled")
    assert pooled.pearson == pytest.approx(1.0)
    assert pooled.spearman == pytest.approx(1.0)


# --- baselines ---------------------------------------------------------------------------------


def test_random_selection_is_deterministic_and_distinct() -> None:
    pool = _pool()
    a = random_selection(pool, _RANDOM_B, seed=0)
    b = random_selection(pool, _RANDOM_B, seed=0)
    assert a == b  # same seed
    assert len(set(a)) == _RANDOM_B  # distinct
    assert random_selection(pool, _RANDOM_B, seed=1) != a  # different seed


def test_practice_heuristic_returns_valid_cross_site_pairs() -> None:
    pool = _pool()
    selected = practice_heuristic(pool, _PRACTICE_B)
    assert len(selected) == _PRACTICE_B
    for variant in selected:
        assert len(variant) == _PAIR_ORDER  # only pairs spend budget
        positions = [pos for pos, _, _ in variant]
        assert positions[0] != positions[1]  # cross-site


def test_hit_rate_is_a_fraction_of_true_top_budget() -> None:
    fitness = {frozenset({(0, "A", "C")}): 5.0, frozenset({(1, "A", "C")}): 3.0, frozenset(): 1.0}
    # true top-1 is the first variant; selecting it scores 1.0, selecting another scores 0.0
    assert hit_rate([frozenset({(0, "A", "C")})], fitness, 1) == pytest.approx(1.0)
    assert hit_rate([frozenset()], fitness, 1) == pytest.approx(0.0)


# --- end-to-end harness (offline) --------------------------------------------------------------


def test_run_validation_end_to_end_writes_a_real_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("epibudget.validate._N_BOOTSTRAP", 40)  # keep the offline run snappy
    pool = _pool()
    landscape = _landscape(pool)
    budgets = [4, 6]
    report = run_validation(
        pool, landscape, budgets=budgets, seeds=2, model_id="toy", out_dir=tmp_path
    )

    assert report.var_epsilon > 0.0  # invariant #1 sanity: the true map is non-additive
    assert len(report.results) == _N_METHODS * len(budgets)
    methods = {r.method for r in report.results}
    assert methods == {"info", "fitness", "structural", "random", "practice"}
    for r in report.results:
        assert {m.order for m in r.metrics} == {"pairwise", "third", "pooled"}
        assert 0.0 <= r.hit_rate <= 1.0
    assert {r.ci_method for r in report.results if r.method == "random"} == {"bootstrap-over-seeds"}
    assert {r.ci_method for r in report.results if r.method == "info"} == {"bootstrap-over-terms"}

    written = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert written["dataset"] == "gb1_wu2016"
    assert written["n_truth_terms"] > 0


def test_run_validation_never_overwrites_an_existing_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("epibudget.validate._N_BOOTSTRAP", 10)
    pool = _pool()
    run_validation(pool, _landscape(pool), [4], seeds=2, model_id="toy", out_dir=tmp_path)

    with pytest.raises(FileExistsError):
        run_validation(pool, _landscape(pool), [4], seeds=2, model_id="toy", out_dir=tmp_path)


def _pooled(report: object, method: str, budget: int) -> OrderMetric:
    result = next(r for r in report.results if r.method == method and r.budget == budget)  # type: ignore[attr-defined]
    return next(m for m in result.metrics if m.order == "pooled")


def test_run_validation_selection_is_blind_to_the_landscape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Permute fitness VALUES across the (non-WT) keys — all stay live, so which variants are dead is
    # unchanged. Selection ignores the landscape, so each method informs the SAME terms: per-method
    # coverage (n_informed) is invariant. A harness-level label leak in selection would break this.
    monkeypatch.setattr("epibudget.validate._N_BOOTSTRAP", 20)
    pool = _pool()
    landscape = _landscape(pool)
    non_wt = [v for v in landscape if v]
    values = [landscape[v] for v in non_wt]
    perm = list(reversed(values))
    permuted = {frozenset(): landscape[frozenset()], **dict(zip(non_wt, perm, strict=True))}

    base = run_validation(pool, landscape, [4], seeds=2, model_id="t", out_dir=tmp_path / "a")
    other = run_validation(pool, permuted, [4], seeds=2, model_id="t", out_dir=tmp_path / "b")
    for method in ("info", "fitness", "practice"):
        assert _pooled(base, method, 4).n_informed == _pooled(other, method, 4).n_informed


def test_random_baseline_reports_a_seed_bootstrap_ci_with_enough_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("epibudget.validate._N_BOOTSTRAP", 20)
    pool = _pool()
    report = run_validation(pool, _landscape(pool), [4], seeds=5, model_id="t", out_dir=tmp_path)
    ci = _pooled(report, "random", 4).spearman_ci95
    assert ci is not None and ci[0] <= ci[1]  # the over-seeds CI is actually computed


def test_blend_shifts_selection_toward_fitness_as_lambda_grows() -> None:
    pool = _pool()
    graph = EpistasisFactorGraph(
        predicted_epistasis(pool), {sv.variant: sv.var_delta_g for sv in pool}
    )
    dg = {sv.variant: sv.delta_g for sv in pool}
    mean_dg = [
        float(np.mean([dg[v] for v in allocate(graph, pool, 8, lambda_=lam).selected]))
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert all(a <= b + 1e-9 for a, b in pairwise(mean_dg))  # monotone in λ
    assert mean_dg[-1] > mean_dg[0]  # fitness end genuinely fitter than info end


# --- isolation / no-label-leakage boundary -----------------------------------------------------


def test_selectors_expose_no_measured_fitness_parameter() -> None:
    # Structural guard mirroring the no-label-leakage skill: no selector may take a landscape/label.
    for fn in (random_selection, practice_heuristic):
        params = set(inspect.signature(fn).parameters)
        assert not (params & {"landscape", "fitness", "labels", "measured"})


def test_order_metric_reports_breadth_and_precision() -> None:
    # The breadth (coverage / n_pinned) and precision (predicted-term correlation) split must exist.
    fields = set(OrderMetric.model_fields)
    assert {
        "coverage_fraction",
        "n_informed",
        "n_pinned",
        "pearson_predicted",
        "spearman_predicted",
    } <= fields


def test_structural_baseline_ranks_by_loop_count_ignoring_var_delta_g() -> None:
    # With τ²≡const the structural info-gain is n(v): a single (in ~all loops) outranks a triple.
    pool = _pool()
    graph = structural_graph(pool)
    single = next(sv.variant for sv in pool if len(sv.variant) == 1)
    triple = next(sv.variant for sv in pool if len(sv.variant) == _TRIPLE_ORDER)
    assert graph.info_gain(frozenset(), single) > graph.info_gain(frozenset(), triple)


def test_breadth_and_precision_split_counts_pinned_vs_predicted() -> None:
    # A fully-measured pairwise loop pins that term (exact); a partially-touched term is predicted.
    pool = _pool()
    mi, mj, mk = (0, "A", "C"), (1, "A", "C"), (2, "A", "C")
    measured = frozenset({frozenset({mi}), frozenset({mj}), frozenset({mi, mj}), frozenset({mk})})
    inferred = infer_epistasis({v: 0.5 for v in measured}, pool)
    truth = {i.mutations: i.epsilon_hat for i in inferred}
    pairwise = next(m for m in map_recovery(inferred, truth, measured) if m.order == "pairwise")
    assert pairwise.n_pinned >= 1  # (mi,mj) is fully pinned
    assert pairwise.n_informed > pairwise.n_pinned  # (mi,mk)/(mj,mk) informed but not pinned
