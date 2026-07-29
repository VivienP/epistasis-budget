"""Offline tests for the TrpB pairwise Fourier recovery diagnostic."""

# ruff: noqa: PLR2004

from __future__ import annotations

import inspect
from itertools import product

import numpy as np
import pytest

from epibudget.coeff_recovery import _build_fourier_config, _design_matrix, _site_indices
from epibudget.fourier_recovery import (
    RecoveryCell,
    benchmark_doptimal_prefix,
    benchmark_synthetic_fit,
    build_selection_plan,
    coefficient_metrics,
    decide_recovery,
    doptimal_workspace_bytes,
    evaluate_plate,
    fit_pairwise_lasso,
    pairwise_truth,
    reduced_doptimal_order,
    registered_fit_count,
    validate_recovery_dataset,
    validate_runtime_preflight,
)
from epibudget.types import ScoredVariant, Variant

_Q3 = "ACD"
_SITES = (0, 1, 2, 3)
_WT = ("A", "A", "A", "A")


def _all_genotypes() -> list[Variant]:
    return [
        frozenset(
            (site, _WT[index], aa)
            for index, (site, aa) in enumerate(zip(_SITES, residues, strict=True))
            if aa != _WT[index]
        )
        for residues in product(_Q3, repeat=len(_SITES))
    ]


def _sparse_pairwise_fixture() -> tuple[
    list[Variant], np.ndarray, np.ndarray, tuple[tuple[int, ...], ...]
]:
    genotypes = _all_genotypes()
    config = _build_fourier_config(_SITES, _WT, _Q3, max_order=2)
    design = np.sqrt(len(genotypes)) * _design_matrix(config, _site_indices(config, genotypes))
    pairwise_columns = np.array(
        [index for index, mode in enumerate(config.modes) if np.count_nonzero(mode) == 2],
        dtype=np.int64,
    )
    truth = np.zeros(len(pairwise_columns), dtype=np.float64)
    truth[[1, 5, 13]] = np.array([1.5, -0.8, 0.45])
    beta = np.zeros(design.shape[1], dtype=np.float64)
    beta[pairwise_columns] = truth
    response = 0.2 + design @ beta
    return genotypes, response, truth, config.modes


def test_pairwise_truth_uses_registered_population_normalization() -> None:
    genotypes, response, expected, modes = _sparse_pairwise_fixture()
    landscape = {
        genotype: float(value) for genotype, value in zip(genotypes, response, strict=True)
    }

    truth = pairwise_truth(landscape, _SITES)

    assert truth.modes == tuple(mode for mode in modes if np.count_nonzero(mode) == 2)
    assert truth.coefficients.shape == (24,)
    assert np.allclose(truth.coefficients, expected, atol=1e-12)


def test_coefficient_metrics_keep_constant_correlation_unavailable() -> None:
    truth = np.array([2.0, -1.0, 0.5], dtype=np.float64)
    predicted = np.zeros_like(truth)

    metrics = coefficient_metrics(predicted, truth)

    assert metrics.spearman is None
    assert metrics.relative_sse_gain == pytest.approx(0.0)
    assert metrics.support_size == 0
    assert metrics.coefficient_count == 3


def test_coefficient_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        coefficient_metrics(np.zeros(2), np.zeros(3))


def test_pairwise_lasso_recovers_sparse_synthetic_coefficients() -> None:
    genotypes, response, truth, _modes = _sparse_pairwise_fixture()
    rng = np.random.default_rng(7)
    measured_indices = np.sort(rng.choice(len(genotypes), size=70, replace=False))
    measured = [genotypes[int(index)] for index in measured_indices]
    measured_response = response[measured_indices]
    config = _build_fourier_config(_SITES, _WT, _Q3, max_order=2)

    fit = fit_pairwise_lasso(config, measured, measured_response, n_folds=5)

    metrics = coefficient_metrics(fit.pairwise_coefficients, truth)
    assert fit.converged is True
    assert fit.lambda_ratio in tuple(float(value) for value in np.geomspace(1.0, 1e-3, 20))
    assert metrics.spearman is not None
    assert metrics.spearman > 0.85
    assert metrics.relative_sse_gain is not None
    assert metrics.relative_sse_gain > 0.80


def test_pairwise_lasso_optimizes_a_free_intercept_on_an_unbalanced_plate() -> None:
    genotypes, response, truth, _modes = _sparse_pairwise_fixture()
    measured_indices = np.arange(70, dtype=np.int64)
    measured = [genotypes[int(index)] for index in measured_indices]
    measured_response = response[measured_indices]
    config = _build_fourier_config(_SITES, _WT, _Q3, max_order=2)

    fit = fit_pairwise_lasso(
        config,
        measured,
        measured_response,
        n_folds=5,
        lambda_ratios=(1e-3,),
    )

    metrics = coefficient_metrics(fit.pairwise_coefficients, truth)
    assert fit.converged is True
    assert metrics.relative_sse_gain is not None
    assert metrics.relative_sse_gain > 0.999


def test_pairwise_lasso_rejects_constant_response() -> None:
    genotypes = _all_genotypes()[:30]
    config = _build_fourier_config(_SITES, _WT, _Q3, max_order=2)

    with pytest.raises(ValueError, match="effectively constant"):
        fit_pairwise_lasso(config, genotypes, np.ones(len(genotypes)), n_folds=5)


def test_selection_plan_is_label_free_prefix_consistent_and_order_invariant() -> None:
    candidates = [variant for variant in _all_genotypes() if 1 <= len(variant) <= 3]
    scored = [
        ScoredVariant(
            variant=variant,
            delta_g=float(index),
            var_delta_g=float(index + 1) / len(candidates),
        )
        for index, variant in enumerate(candidates)
    ]

    assert "landscape" not in inspect.signature(build_selection_plan).parameters
    plan = build_selection_plan(scored, budgets=(8, 20), seeds=(0, 1), max_order=3)
    reversed_plan = build_selection_plan(
        list(reversed(scored)), budgets=(8, 20), seeds=(0, 1), max_order=3
    )

    assert plan == reversed_plan
    assert len(plan.sequences) == 7  # three deterministic and two seeds each for random/structural
    for sequence in plan.sequences:
        assert len(sequence.selected) == 20
        assert len(set(sequence.selected)) == 20
        assert plan.plate(sequence.method, sequence.seed, 8) == sequence.selected[:8]


def test_reduced_doptimal_is_prefix_consistent_and_input_order_invariant() -> None:
    candidates = [variant for variant in _all_genotypes() if variant]
    config = _build_fourier_config(_SITES, _WT, _Q3, max_order=2)

    long = reduced_doptimal_order(config, candidates, budget=20)
    short = reduced_doptimal_order(config, list(reversed(candidates)), budget=8)

    assert len(long) == 20
    assert len(set(long)) == 20
    assert short == long[:8]


def test_registered_runtime_dimensions_are_explicit() -> None:
    budgets = (48, 96, 192, 384, 768, 1536, 2242, 3072)
    seeds = tuple(range(20))

    assert registered_fit_count(budgets, seeds) == 344
    assert doptimal_workspace_bytes(29_678, 3_072) == 729_366_528


def test_synthetic_runtime_benchmark_never_needs_landscape_labels() -> None:
    candidates = [variant for variant in _all_genotypes() if variant]
    config = _build_fourier_config(_SITES, _WT, _Q3, max_order=2)

    assert "landscape" not in inspect.signature(benchmark_synthetic_fit).parameters
    result = benchmark_synthetic_fit(config, candidates, budget=30, seed=0, n_folds=5)

    assert result.budget == 30
    assert result.design_shape == (30, 32)
    assert result.design_bytes == 30 * 32 * 8
    assert result.doptimal_update_bytes == len(candidates) * 30 * 8
    assert result.design_seconds >= 0.0
    assert result.fit_seconds >= 0.0
    assert result.converged is True


def test_doptimal_runtime_pilot_projects_registered_maximum_without_labels() -> None:
    candidates = [variant for variant in _all_genotypes() if variant]
    config = _build_fourier_config(_SITES, _WT, _Q3, max_order=2)

    assert "landscape" not in inspect.signature(benchmark_doptimal_prefix).parameters
    result = benchmark_doptimal_prefix(config, candidates, pilot_budget=8, maximum_budget=20)

    assert result.pilot_budget == 8
    assert result.maximum_budget == 20
    assert result.pilot_seconds >= 0.0
    assert result.projected_maximum_seconds == pytest.approx(result.pilot_seconds * (20 / 8) ** 2)
    assert result.pilot_update_bytes == len(candidates) * 8 * 8
    assert result.maximum_update_bytes == len(candidates) * 20 * 8


def test_recovery_gate_requires_every_registered_stochastic_seed() -> None:
    cells = [
        RecoveryCell(
            method="random",
            budget=192,
            seed=seed,
            spearman=0.4,
            relative_sse_gain=0.2,
            support_size=5,
            coefficient_count=24,
        )
        for seed in range(19)
    ]

    decision = decide_recovery(
        cells,
        stochastic_seeds=tuple(range(20)),
        expected_methods=("random",),
        expected_budgets=(192,),
    )

    assert decision.status == "invalid_coverage"
    assert "missing seeds" in decision.reasons[0]


def test_recovery_gate_accepts_registered_stochastic_thresholds() -> None:
    cells = [
        RecoveryCell(
            method="structural",
            budget=768,
            seed=seed,
            spearman=0.35,
            relative_sse_gain=0.15 if seed < 16 else -0.01,
            support_size=10,
            coefficient_count=24,
        )
        for seed in range(20)
    ]

    decision = decide_recovery(
        cells,
        stochastic_seeds=tuple(range(20)),
        expected_methods=("structural",),
        expected_budgets=(768,),
    )

    assert decision.status == "stage_b_justified"
    assert decision.passing_cells == (("structural", 768),)
    aggregate = decision.aggregates[0]
    assert aggregate.minimum_relative_sse_gain == pytest.approx(-0.01)
    assert aggregate.maximum_relative_sse_gain == pytest.approx(0.15)
    assert aggregate.positive_sse_fraction == pytest.approx(0.8)


def test_recovery_gate_preserves_an_honest_null() -> None:
    cell = RecoveryCell(
        method="info",
        budget=3072,
        seed=None,
        spearman=0.29,
        relative_sse_gain=0.2,
        support_size=10,
        coefficient_count=24,
    )

    decision = decide_recovery(
        [cell],
        stochastic_seeds=tuple(range(20)),
        expected_methods=("info",),
        expected_budgets=(3072,),
    )

    assert decision.status == "estimator_family_stopped"
    assert decision.passing_cells == ()


def test_recovery_gate_rejects_a_nonconverged_cell() -> None:
    cell = RecoveryCell(
        method="info",
        budget=192,
        seed=None,
        spearman=None,
        relative_sse_gain=None,
        support_size=0,
        coefficient_count=24,
        converged=False,
        error="coordinate descent did not converge",
    )

    decision = decide_recovery(
        [cell],
        stochastic_seeds=tuple(range(20)),
        expected_methods=("info",),
        expected_budgets=(192,),
    )

    assert decision.status == "invalid_coverage"
    assert "did not converge" in decision.reasons[0]


def test_recovery_gate_retains_valid_aggregates_when_one_seed_fails() -> None:
    cells = [
        RecoveryCell(
            method="random",
            budget=192,
            seed=seed,
            spearman=0.4,
            relative_sse_gain=0.2,
            support_size=5,
            coefficient_count=24,
        )
        for seed in range(19)
    ]
    cells.append(
        RecoveryCell(
            method="random",
            budget=192,
            seed=19,
            spearman=None,
            relative_sse_gain=None,
            support_size=0,
            coefficient_count=24,
            converged=False,
            error="coordinate descent did not converge",
        )
    )

    decision = decide_recovery(
        cells,
        stochastic_seeds=tuple(range(20)),
        expected_methods=("random",),
        expected_budgets=(192,),
    )

    assert decision.status == "invalid_coverage"
    assert decision.aggregates[0].n_records == 20
    assert decision.aggregates[0].n_valid_sse == 19
    assert decision.aggregates[0].positive_sse_fraction == pytest.approx(1.0)


def test_recovery_gate_rejects_missing_method_budget_cells() -> None:
    decision = decide_recovery(
        [],
        stochastic_seeds=tuple(range(20)),
        expected_methods=("info", "fitness"),
        expected_budgets=(48, 96),
    )

    assert decision.status == "invalid_coverage"
    assert decision.reasons == (
        "missing method-budget cells: fitness/48, fitness/96, info/48, info/96",
    )


def test_recovery_dataset_is_trpb_only() -> None:
    validate_recovery_dataset("trpb_johnston2024")

    with pytest.raises(ValueError, match="TrpB-only"):
        validate_recovery_dataset("gb1_wu2016")


def test_runtime_preflight_ignores_projected_duration_and_rejects_a_stale_commit() -> None:
    budgets = (48, 96, 192, 384, 768, 1536, 2242, 3072)
    candidate_count = 29_678
    feature_count = 2_242
    measurements = [
        {
            "budget": budget,
            "design_shape": [budget, feature_count],
            "design_bytes": budget * feature_count * 8,
            "doptimal_update_bytes": candidate_count * budget * 8,
            "design_seconds": 1.0,
            "fit_seconds": 1.0,
            "support_size": 8,
            "converged": True,
        }
        for budget in budgets
    ]
    doptimal_pilot_seconds = 10.0
    projected_doptimal_seconds = doptimal_pilot_seconds * (budgets[-1] / budgets[0]) ** 2
    payload = {
        "schema_version": "epibudget-fourier-runtime-v4",
        "uses_measured_labels": False,
        "candidate_count": candidate_count,
        "candidate_sha256": "a" * 64,
        "registered_fit_count": 344,
        "measured_budgets": list(budgets),
        "measurements": measurements,
        "doptimal_pilot": {
            "pilot_budget": budgets[0],
            "maximum_budget": budgets[-1],
            "pilot_seconds": doptimal_pilot_seconds,
            "projected_maximum_seconds": projected_doptimal_seconds,
            "pilot_update_bytes": candidate_count * budgets[0] * 8,
            "maximum_update_bytes": candidate_count * budgets[-1] * 8,
        },
        "projected_lasso_seconds": 344.0,
        "projected_seconds": 344.0 + projected_doptimal_seconds,
        "maximum_doptimal_bytes": candidate_count * budgets[-1] * 8,
        "provenance": {
            "workspace_start": {"execution_commit": "old", "code_state": "clean"},
            "workspace_end": {"execution_commit": "old", "code_state": "clean"},
            "workspace_state_matches": True,
        },
    }

    assert payload["projected_seconds"] > 8.0 * 3600.0

    validate_runtime_preflight(
        payload,
        expected_commit="old",
        expected_candidate_count=29_678,
        expected_candidate_sha256="a" * 64,
        expected_budgets=budgets,
        expected_fit_count=344,
        expected_feature_count=feature_count,
    )

    incomplete = dict(payload)
    incomplete.pop("measurements")
    with pytest.raises(ValueError, match="incomplete budget measurements"):
        validate_runtime_preflight(
            incomplete,
            expected_commit="old",
            expected_candidate_count=candidate_count,
            expected_candidate_sha256="a" * 64,
            expected_budgets=budgets,
            expected_fit_count=344,
            expected_feature_count=feature_count,
        )

    with pytest.raises(ValueError, match="execution commit"):
        validate_runtime_preflight(
            payload,
            expected_commit="new",
            expected_candidate_count=candidate_count,
            expected_candidate_sha256="a" * 64,
            expected_budgets=budgets,
            expected_fit_count=344,
            expected_feature_count=feature_count,
        )


def test_plate_evaluation_reveals_only_the_frozen_selection() -> None:
    genotypes, response, truth, _modes = _sparse_pairwise_fixture()
    landscape = {
        genotype: float(np.expm1(value))
        for genotype, value in zip(genotypes, response, strict=True)
    }
    rng = np.random.default_rng(11)
    measured_indices = np.sort(rng.choice(len(genotypes), size=70, replace=False))
    selected = tuple(genotypes[int(index)] for index in measured_indices)
    config = _build_fourier_config(_SITES, _WT, _Q3, max_order=2)

    cell = evaluate_plate(
        config,
        selected,
        landscape,
        truth,
        method="random",
        seed=11,
        budget=70,
        n_folds=5,
    )

    assert cell.method == "random"
    assert cell.seed == 11
    assert cell.coefficient_count == 24
    assert len(cell.selected_sha256) == 64
    assert len(cell.fold_sha256) == 64
    assert cell.lambda_ratio is not None
    assert cell.lambda_value is not None
    assert cell.converged is True
    assert cell.spearman is not None and cell.spearman > 0.85
    assert cell.relative_sse_gain is not None and cell.relative_sse_gain > 0.80
