"""Tests for the GB1 validation harness: inference, recovery, baselines, isolation.

All offline: a synthetic candidate pool + a toy landscape stand in for ESM scoring and real GB1. The
inference round-trips (empty reveal ⇒ ESM prior; full-loop reveal ⇒ true ε) pin the estimator; the
end-to-end run proves the harness produces a real metrics.json; the signature guards keep every
selector structurally blind to measured fitness.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Sequence
from itertools import combinations, pairwise
from math import exp, log
from pathlib import Path

import numpy as np
import pytest

from epibudget import validate as validate_module
from epibudget.acquisition import allocate
from epibudget.data import enumerate_candidates, load_gb1, reveal_measured_fitness
from epibudget.epistasis import (
    epsilon_pairwise,
    ground_truth_epistasis,
    predicted_epistasis,
    wt_centered_log_fitness,
)
from epibudget.graph import EpistasisFactorGraph
from epibudget.types import Interaction, Mutation, ScoredVariant, Variant
from epibudget.validate import (
    OrderMetric,
    Report,
    _calibrate_slope,
    _measured_dg,
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
    landscape: dict[Variant, float] = {frozenset(): 1.0}
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


def test_measured_dg_reveals_wt_with_selection_but_returns_candidates_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wt: Variant = frozenset()
    selected = [frozenset({(0, "A", "C")}), frozenset({(1, "A", "C")})]
    landscape: dict[Variant, float] = {wt: 2.0, selected[0]: 8.0, selected[1]: 0.0}
    requests: list[list[Variant]] = []
    real_reveal = reveal_measured_fitness

    def _spy(source: dict[Variant, float], variants: list[Variant]) -> dict[Variant, float]:
        requests.append(list(variants))
        return real_reveal(source, variants)

    monkeypatch.setattr(validate_module, "reveal_measured_fitness", _spy)
    measured = _measured_dg(landscape, selected)

    assert requests == [[wt, *selected]]
    assert measured == {selected[0]: pytest.approx(log(4.0))}
    assert wt not in measured


def test_f0_not_one_changes_within_order_spearman_when_not_centered() -> None:
    a = -0.5
    mut_a, mut_b, mut_c = (0, "A", "C"), (1, "A", "C"), (2, "A", "C")
    va, vb, vc = frozenset({mut_a}), frozenset({mut_b}), frozenset({mut_c})
    vab, vac, vbc = (
        frozenset({mut_a, mut_b}),
        frozenset({mut_a, mut_c}),
        frozenset({mut_b, mut_c}),
    )
    x = {va: 1.0, vb: 1.0, vc: 0.0, vab: 2.0, vac: 0.8, vbc: 1.4}
    scored = [ScoredVariant(variant=v, delta_g=value, var_delta_g=0.1) for v, value in x.items()]
    centered_dg = {
        frozenset(): 0.0,
        va: 1.0,
        vb: 2.0,
        vc: 0.0,
        vab: 3.0,
        vac: 3.0,
        vbc: 3.0,
    }
    landscape = {variant: exp(a + dg) for variant, dg in centered_dg.items()}
    truth = {
        interaction.mutations: interaction.epsilon_hat
        for interaction in ground_truth_epistasis(wt_centered_log_fitness(landscape), max_order=2)
    }
    selected = [va, vb]

    centered_revealed = _measured_dg(landscape, selected)
    legacy_revealed = {variant: log(landscape[variant]) for variant in selected}
    esm = {sv.variant: sv.delta_g for sv in scored}
    assert _calibrate_slope(
        [esm[variant] for variant in centered_revealed],
        [centered_revealed[variant] for variant in centered_revealed],
    ) == pytest.approx(1.5)
    assert _calibrate_slope(
        [esm[variant] for variant in legacy_revealed],
        [legacy_revealed[variant] for variant in legacy_revealed],
    ) == pytest.approx(1.0)

    centered_inferred = infer_epistasis(centered_revealed, scored, max_order=2)
    legacy_inferred = infer_epistasis(legacy_revealed, scored, max_order=2)
    centered_metric = next(
        metric
        for metric in map_recovery(centered_inferred, truth, frozenset(selected), with_ci=False)
        if metric.order == "pairwise"
    )
    legacy_metric = next(
        metric
        for metric in map_recovery(legacy_inferred, truth, frozenset(selected), with_ci=False)
        if metric.order == "pairwise"
    )

    terms = [(mut_a, mut_b), (mut_a, mut_c), (mut_b, mut_c)]
    assert [truth[term] for term in terms] == pytest.approx([0.0, 2.0, 1.0])
    centered_hat = {
        interaction.mutations: interaction.epsilon_hat for interaction in centered_inferred
    }
    legacy_hat = {interaction.mutations: interaction.epsilon_hat for interaction in legacy_inferred}
    assert [centered_hat[term] for term in terms] == pytest.approx([0.0, 0.2, 0.1])
    assert [legacy_hat[term] for term in terms] == pytest.approx([0.0, 0.3, -0.1])
    assert centered_metric.spearman == pytest.approx(1.0)
    assert legacy_metric.spearman == pytest.approx(0.5)


def test_ref_one_centering_is_bit_exact_for_slope_truth_inference_and_recovery() -> None:
    pool = _pool()
    landscape = _landscape(pool)
    selected = [sv.variant for sv in pool[:8]]
    legacy_revealed = {
        variant: log(landscape[variant]) for variant in selected if landscape[variant] > 0.0
    }
    centered_revealed = _measured_dg(landscape, selected)
    esm = {sv.variant: sv.delta_g for sv in pool}

    assert centered_revealed == legacy_revealed
    assert _calibrate_slope(
        [esm[v] for v in centered_revealed], [centered_revealed[v] for v in centered_revealed]
    ) == _calibrate_slope(
        [esm[v] for v in legacy_revealed], [legacy_revealed[v] for v in legacy_revealed]
    )
    legacy_truth = ground_truth_epistasis(
        {variant: log(value) for variant, value in landscape.items() if value > 0.0}
    )
    centered_truth = ground_truth_epistasis(wt_centered_log_fitness(landscape))
    assert centered_truth == legacy_truth
    centered_inferred = infer_epistasis(centered_revealed, pool)
    legacy_inferred = infer_epistasis(legacy_revealed, pool)
    assert centered_inferred == legacy_inferred
    truth_by_term = {interaction.mutations: interaction.epsilon_hat for interaction in legacy_truth}
    assert map_recovery(
        centered_inferred, truth_by_term, frozenset(centered_revealed), with_ci=False
    ) == map_recovery(legacy_inferred, truth_by_term, frozenset(legacy_revealed), with_ci=False)


@pytest.mark.data
def test_real_gb1_ref_one_centering_is_bit_exact_for_truth() -> None:
    csv_path = Path(__file__).resolve().parent.parent / "data" / "proteingym" / "gb1_wu2016.csv"
    landscape = load_gb1(csv_path)
    wt: Variant = frozenset()
    assert landscape[wt] == 1.0

    legacy = {variant: log(fitness) for variant, fitness in landscape.items() if fitness > 0.0}
    centered = wt_centered_log_fitness(landscape)
    assert centered == legacy
    assert ground_truth_epistasis(centered) == ground_truth_epistasis(legacy)


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

    assert report.var_epsilon > 0.0
    assert report.var_predicted_epsilon > 0.0  # invariant #1 sanity: ESM is non-additive
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
    assert written["var_predicted_epsilon"] == report.var_predicted_epsilon


def test_report_rejects_additive_decimal_roundoff_as_epistasis_signal(tmp_path: Path) -> None:
    base = _pool()
    mutations = sorted(next(iter(sv.variant)) for sv in base if len(sv.variant) == 1)
    effects = {mutation: 0.1 * (index + 1) for index, mutation in enumerate(mutations)}
    pool = [
        ScoredVariant(
            variant=sv.variant,
            delta_g=sum(effects[mutation] for mutation in sorted(sv.variant)),
            var_delta_g=sv.var_delta_g,
        )
        for sv in base
    ]
    interactions = predicted_epistasis(pool)
    residuals = [abs(interaction.epsilon_hat) for interaction in interactions]
    expected_tolerance = (
        np.finfo(np.float64).eps * max(abs(sv.delta_g) for sv in pool) * ((1 << _TRIPLE_ORDER) - 1)
    )
    assert 0.0 < max(residuals) <= expected_tolerance

    report = run_validation(pool, _landscape(pool), [], seeds=1, model_id="toy", out_dir=tmp_path)
    assert report.var_predicted_epsilon > 0.0
    assert report.predicted_epistasis_tolerance == expected_tolerance
    assert report.predicted_epistasis_signal is False


def test_report_accepts_constant_nonzero_epistasis_when_variance_is_zero(tmp_path: Path) -> None:
    base = _pool()
    mutations = sorted(next(iter(sv.variant)) for sv in base if len(sv.variant) == 1)
    effects = {mutation: float(index + 1) for index, mutation in enumerate(mutations)}
    pool = [
        ScoredVariant(
            variant=sv.variant,
            delta_g=sum(effects[mutation] for mutation in sv.variant)
            + (1.0 if len(sv.variant) == _PAIR_ORDER else 0.0),
            var_delta_g=sv.var_delta_g,
        )
        for sv in base
    ]
    interactions = predicted_epistasis(pool, max_order=_PAIR_ORDER)
    assert {interaction.epsilon_hat for interaction in interactions} == {1.0}

    report = run_validation(
        pool,
        _landscape(pool),
        [],
        seeds=1,
        model_id="toy",
        out_dir=tmp_path,
        max_order=_PAIR_ORDER,
    )
    assert report.var_predicted_epsilon == 0.0
    assert report.predicted_epistasis_signal is True
    assert report.predicted_epistasis_tolerance > 0.0


def test_run_validation_builds_predicted_epistasis_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validate_module, "_N_BOOTSTRAP", 10)
    real_predicted = predicted_epistasis
    calls = 0

    def _counted(scored: Sequence[ScoredVariant], max_order: int = 3) -> list[Interaction]:
        nonlocal calls
        calls += 1
        return real_predicted(scored, max_order)

    monkeypatch.setattr(validate_module, "predicted_epistasis", _counted)
    pool = _pool()
    run_validation(pool, _landscape(pool), [4], seeds=2, model_id="toy", out_dir=tmp_path)
    assert calls == 1


def test_run_validation_is_invariant_to_multiplying_all_fitness_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validate_module, "_N_BOOTSTRAP", 20)
    pool = _pool()
    mutations = sorted(next(iter(sv.variant)) for sv in pool if len(sv.variant) == 1)
    weights = dict(zip(mutations, (2.0, 3.0, 5.0, 7.0, 11.0, 13.0), strict=True))

    def _nondegenerate_esm(variant: Variant) -> float:
        present = [weights[mutation] for mutation in sorted(variant)]
        pair_effect = sum(0.013 * left * right for left, right in combinations(present, 2))
        third_effect = 0.0009 * float(np.prod(present)) if len(present) == _TRIPLE_ORDER else 0.0
        return sum(0.08 * value for value in present) + pair_effect + third_effect

    pool = [
        ScoredVariant(
            variant=sv.variant,
            delta_g=_nondegenerate_esm(sv.variant),
            var_delta_g=sv.var_delta_g,
        )
        for sv in pool
    ]

    def _nondegenerate_dg(variant: Variant) -> float:
        present = [weights[mutation] for mutation in sorted(variant)]
        pair_effect = sum(0.02 * left * right for left, right in combinations(present, 2))
        third_effect = 0.0005 * float(np.prod(present)) if len(present) == _TRIPLE_ORDER else 0.0
        return sum(0.1 * value for value in present) + pair_effect + third_effect

    reference = 2.5
    landscape: dict[Variant, float] = {frozenset(): reference}
    landscape.update({sv.variant: reference * exp(_nondegenerate_dg(sv.variant)) for sv in pool})
    scaled = {variant: 7.0 * fitness for variant, fitness in landscape.items()}
    base = run_validation(pool, landscape, [4], seeds=3, model_id="toy", out_dir=tmp_path / "base")
    other = run_validation(pool, scaled, [4], seeds=3, model_id="toy", out_dir=tmp_path / "scaled")

    assert other.var_epsilon == pytest.approx(base.var_epsilon)
    assert other.var_predicted_epsilon == base.var_predicted_epsilon
    base_results = {(result.method, result.budget): result for result in base.results}
    other_results = {(result.method, result.budget): result for result in other.results}
    assert other_results.keys() == base_results.keys()
    for key, base_result in base_results.items():
        other_result = other_results[key]
        assert other_result.hit_rate == base_result.hit_rate
        for base_metric, other_metric in zip(
            base_result.metrics, other_result.metrics, strict=True
        ):
            assert other_metric.order == base_metric.order
            assert other_metric.n_terms == base_metric.n_terms
            assert other_metric.n_informed == base_metric.n_informed
            assert other_metric.n_pinned == base_metric.n_pinned
            for field in ("pearson", "spearman", "pearson_predicted", "spearman_predicted"):
                expected = getattr(base_metric, field)
                observed = getattr(other_metric, field)
                if expected is None:
                    assert observed is None
                else:
                    assert observed == pytest.approx(expected), (key, base_metric.order, field)


def test_run_validation_never_overwrites_an_existing_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("epibudget.validate._N_BOOTSTRAP", 10)
    pool = _pool()
    run_validation(pool, _landscape(pool), [4], seeds=2, model_id="toy", out_dir=tmp_path)

    with pytest.raises(FileExistsError):
        run_validation(pool, _landscape(pool), [4], seeds=2, model_id="toy", out_dir=tmp_path)


def _pooled(report: Report, method: str, budget: int) -> OrderMetric:
    result = next(r for r in report.results if r.method == method and r.budget == budget)
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
