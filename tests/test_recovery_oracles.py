"""Frozen oracles for the recovery numerics on a small synthetic landscape.

These values were produced by the current selection and estimation code. They exist so that a later
restructuring into resumable steppers can be proved to change nothing: every rewrite must reproduce
the same D-optimal order, the same selection sequences, the same cross-validated lambda, the same
support, and the same metrics. Discrete outcomes are asserted exactly; float metrics are asserted to
a tolerance far tighter than any plausible drift and far looser than BLAS-level noise, and bitwise
stability is asserted separately by re-running each computation in the same process.
"""

# ruff: noqa: PLR2004

from __future__ import annotations

import inspect
from itertools import product

import numpy as np
import pytest

from epibudget.coeff_recovery import _build_fourier_config, _design_matrix, _site_indices
from epibudget.fourier_recovery import (
    ReducedDOptimalState,
    _sequence_sha256,
    advance_reduced_doptimal,
    build_selection_plan,
    coefficient_metrics,
    fit_pairwise_lasso,
    initialise_reduced_doptimal,
    reduced_doptimal_order,
    selected_reduced_doptimal,
    validate_completed_reduced_doptimal_state,
)
from epibudget.recovery_protocol import (
    REGISTERED_EXECUTION_POLICY,
    REGISTERED_RECOVERY_PROTOCOL,
)
from epibudget.tie_break import canonical_id
from epibudget.types import ScoredVariant, Variant

_Q3 = "ACD"
_SITES = (0, 1, 2, 3)
_WT = ("A", "A", "A", "A")

_DOPTIMAL_BUDGET = 72
_DOPTIMAL_SHA256 = "fec1a04174952f0141b18de1df208fb9c10a14501ff925328b592ff0c5d269f1"
_DOPTIMAL_HEAD = (
    '[[1,"A","D"],[2,"A","D"],[3,"A","D"]]',
    '[[3,"A","D"]]',
    '[[1,"A","D"],[3,"A","C"]]',
    '[[0,"A","C"],[1,"A","D"],[3,"A","D"]]',
)

_PLAN_BUDGETS = (8, 20, 40)
_PLAN_SEEDS = (0, 1)
_PLAN_SEQUENCE_SHA256 = {
    ("info", None): "cc68e4c527620561bf50480db9863568488eae543d59341472139f91f79332eb",
    ("fitness", None): "c7581dcf81831971cc9929e6ee121dda613e2c9dc16e82547046b0f7d1dad525",
    ("doptimal_reduced_pairwise", None): (
        "55711e08f740b2c632e6f0cb6ea1861556ec7342f2ccf04bc6d7e9a0b39920f0"
    ),
    ("random", 0): "591fc55682b8246be4a8d5f5b46d7687bae6c1b827ba6b586cc8c71fc86ec37e",
    ("structural", 0): "8be912776e8387a751036af41d257cc8ba106357e49b9224ce753d8071263723",
    ("random", 1): "328620e44c492b3e8bd41d12343ea92a133323869095249c31c90e4352b00a88",
    ("structural", 1): "7474890aa6157978373c5532297e23c44d4dcd974a5c691518804ae60b0791f7",
}

_LASSO_MEASURED_ROWS = 70
_LASSO_LAMBDA_RATIO = 0.012742749857031341
_LASSO_LAMBDA_VALUE = 0.034368466203307205
_LASSO_SUPPORT_SIZE = 11
_LASSO_SPEARMAN = 0.9147550817044595
_LASSO_SSE_GAIN = 0.9996598204883093


def _all_genotypes() -> list[Variant]:
    return [
        frozenset(
            (site, _WT[index], aa)
            for index, (site, aa) in enumerate(zip(_SITES, residues, strict=True))
            if aa != _WT[index]
        )
        for residues in product(_Q3, repeat=len(_SITES))
    ]


def _scored_candidates() -> list[ScoredVariant]:
    candidates = [variant for variant in _all_genotypes() if 1 <= len(variant) <= 3]
    rng = np.random.default_rng(20260810)
    return [
        ScoredVariant(variant=variant, delta_g=float(value), var_delta_g=float(spread))
        for variant, value, spread in zip(
            candidates,
            rng.normal(size=len(candidates)),
            rng.uniform(0.05, 1.5, size=len(candidates)),
            strict=True,
        )
    ]


def _noisy_pairwise_fixture() -> tuple[list[Variant], np.ndarray, np.ndarray]:
    genotypes = _all_genotypes()
    config = _build_fourier_config(_SITES, _WT, _Q3, max_order=2)
    design = np.sqrt(len(genotypes)) * _design_matrix(config, _site_indices(config, genotypes))
    pairwise_columns = np.array(
        [index for index, mode in enumerate(config.modes) if np.count_nonzero(mode) == 2],
        dtype=np.int64,
    )
    truth = np.zeros(len(pairwise_columns), dtype=np.float64)
    truth[[0, 1, 4, 5, 9, 13, 17, 21]] = np.array([1.5, -0.8, 0.45, 0.9, -1.2, 0.3, -0.55, 0.7])
    beta = np.zeros(design.shape[1], dtype=np.float64)
    beta[pairwise_columns] = truth
    noise = np.random.default_rng(101).normal(0.0, 0.15, size=design.shape[0])
    return genotypes, 0.2 + design @ beta + noise, truth


def _measured_plate(genotypes: list[Variant]) -> np.ndarray:
    rng = np.random.default_rng(7)
    return np.sort(rng.choice(len(genotypes), size=_LASSO_MEASURED_ROWS, replace=False))


def test_reduced_doptimal_order_matches_the_frozen_oracle() -> None:
    pool = [variant for variant in _all_genotypes() if variant]
    config = _build_fourier_config(_SITES, _WT, _Q3, max_order=2)

    order = reduced_doptimal_order(config, pool, budget=_DOPTIMAL_BUDGET)

    assert len(pool) == 80
    assert len(order) == _DOPTIMAL_BUDGET
    assert len(set(order)) == _DOPTIMAL_BUDGET
    assert tuple(canonical_id(variant) for variant in order[:4]) == _DOPTIMAL_HEAD
    assert _sequence_sha256(order) == _DOPTIMAL_SHA256


def test_reduced_doptimal_order_is_prefix_stable_across_a_block_boundary() -> None:
    pool = [variant for variant in _all_genotypes() if variant]
    config = _build_fourier_config(_SITES, _WT, _Q3, max_order=2)
    boundary = REGISTERED_EXECUTION_POLICY.doptimal_block_size

    full = reduced_doptimal_order(config, pool, budget=_DOPTIMAL_BUDGET)
    at_boundary = reduced_doptimal_order(config, list(reversed(pool)), budget=boundary)

    assert boundary < _DOPTIMAL_BUDGET
    assert at_boundary == full[:boundary]


def test_reduced_doptimal_order_is_bitwise_repeatable() -> None:
    pool = [variant for variant in _all_genotypes() if variant]
    config = _build_fourier_config(_SITES, _WT, _Q3, max_order=2)

    first = reduced_doptimal_order(config, pool, budget=_DOPTIMAL_BUDGET)
    second = reduced_doptimal_order(config, pool, budget=_DOPTIMAL_BUDGET)

    assert first == second


def test_reduced_doptimal_state_resumes_bitwise_at_the_frozen_boundary() -> None:
    pool = [variant for variant in _all_genotypes() if variant]
    config = _build_fourier_config(_SITES, _WT, _Q3, max_order=2)
    direct = initialise_reduced_doptimal(config, pool, target_budget=_DOPTIMAL_BUDGET)
    advance_reduced_doptimal(direct, _DOPTIMAL_BUDGET)

    checkpointed = initialise_reduced_doptimal(config, pool, target_budget=_DOPTIMAL_BUDGET)
    advance_reduced_doptimal(checkpointed, REGISTERED_EXECUTION_POLICY.doptimal_block_size)
    restored = ReducedDOptimalState(
        candidates=checkpointed.candidates,
        site_indices=checkpointed.site_indices.copy(order="C"),
        q=checkpointed.q,
        population_size=checkpointed.population_size,
        target_budget=checkpointed.target_budget,
        selected_indices=list(checkpointed.selected_indices),
        posterior_variance=checkpointed.posterior_variance.copy(order="C"),
        updates=checkpointed.updates.copy(order="C"),
    )
    advance_reduced_doptimal(restored, _DOPTIMAL_BUDGET)

    assert _sequence_sha256(selected_reduced_doptimal(restored)) == _DOPTIMAL_SHA256
    assert selected_reduced_doptimal(restored) == selected_reduced_doptimal(direct)
    assert restored.posterior_variance.tobytes() == direct.posterior_variance.tobytes()
    assert restored.updates.tobytes() == direct.updates.tobytes()


def test_selection_plan_accepts_the_resumed_doptimal_order_without_numeric_drift() -> None:
    scored = _scored_candidates()
    variants = [item.variant for item in scored]
    site_positions = sorted({mutation[0] for variant in variants for mutation in variant})
    config = _build_fourier_config(site_positions, _WT, _Q3, max_order=2)
    state = initialise_reduced_doptimal(config, variants, target_budget=_PLAN_BUDGETS[-1])
    advance_reduced_doptimal(state, _PLAN_BUDGETS[-1])

    plan = build_selection_plan(
        scored,
        budgets=_PLAN_BUDGETS,
        seeds=_PLAN_SEEDS,
        max_order=REGISTERED_RECOVERY_PROTOCOL.selection_max_order,
        doptimal_state=state,
    )

    observed = {
        (sequence.method, sequence.seed): sequence.selected_sha256 for sequence in plan.sequences
    }
    assert observed == _PLAN_SEQUENCE_SHA256


def test_completed_doptimal_state_rejects_a_mutated_pivot_order() -> None:
    scored = _scored_candidates()
    variants = [item.variant for item in scored]
    state = initialise_reduced_doptimal(
        _build_fourier_config(_SITES, _WT, _Q3, max_order=2),
        variants,
        target_budget=_PLAN_BUDGETS[-1],
    )
    advance_reduced_doptimal(state, _PLAN_BUDGETS[-1])
    validate_completed_reduced_doptimal_state(state)
    state.selected_indices[0], state.selected_indices[1] = (
        state.selected_indices[1],
        state.selected_indices[0],
    )

    with pytest.raises(ValueError, match="variance argmax"):
        validate_completed_reduced_doptimal_state(state)
    with pytest.raises(ValueError, match="variance argmax"):
        build_selection_plan(
            scored,
            budgets=_PLAN_BUDGETS,
            seeds=_PLAN_SEEDS,
            max_order=REGISTERED_RECOVERY_PROTOCOL.selection_max_order,
            doptimal_state=state,
        )


def test_selection_plan_rejects_a_raw_doptimal_permutation() -> None:
    scored = _scored_candidates()
    raw_order = tuple(item.variant for item in scored[: _PLAN_BUDGETS[-1]])

    assert "doptimal_order" not in inspect.signature(build_selection_plan).parameters
    with pytest.raises(TypeError, match="D-optimal state"):
        build_selection_plan(
            scored,
            budgets=_PLAN_BUDGETS,
            seeds=_PLAN_SEEDS,
            max_order=REGISTERED_RECOVERY_PROTOCOL.selection_max_order,
            doptimal_state=raw_order,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("damage", ["incomplete", "candidate_pool"])
def test_selection_plan_rejects_an_incompatible_doptimal_state(damage: str) -> None:
    scored = _scored_candidates()
    variants = [item.variant for item in scored]
    candidates = variants if damage == "incomplete" else variants[:-1]
    state = initialise_reduced_doptimal(
        _build_fourier_config(_SITES, _WT, _Q3, max_order=2),
        candidates,
        target_budget=_PLAN_BUDGETS[-1],
    )
    advance_reduced_doptimal(
        state,
        _PLAN_BUDGETS[-1] - 1 if damage == "incomplete" else _PLAN_BUDGETS[-1],
    )

    with pytest.raises(ValueError, match=r"complete|candidate pool"):
        build_selection_plan(
            scored,
            budgets=_PLAN_BUDGETS,
            seeds=_PLAN_SEEDS,
            max_order=REGISTERED_RECOVERY_PROTOCOL.selection_max_order,
            doptimal_state=state,
        )


def test_selection_plan_matches_the_frozen_oracle() -> None:
    scored = _scored_candidates()

    plan = build_selection_plan(
        scored,
        budgets=_PLAN_BUDGETS,
        seeds=_PLAN_SEEDS,
        max_order=REGISTERED_RECOVERY_PROTOCOL.selection_max_order,
    )

    assert len(scored) == 64
    assert plan.budgets == _PLAN_BUDGETS
    observed = {
        (sequence.method, sequence.seed): sequence.selected_sha256 for sequence in plan.sequences
    }
    assert observed == _PLAN_SEQUENCE_SHA256
    assert [(sequence.method, sequence.seed) for sequence in plan.sequences] == list(
        _PLAN_SEQUENCE_SHA256
    )
    for sequence in plan.sequences:
        assert len(sequence.selected) == _PLAN_BUDGETS[-1]
        assert _sequence_sha256(sequence.selected) == sequence.selected_sha256


def test_pairwise_lasso_matches_the_frozen_oracle() -> None:
    genotypes, response, truth = _noisy_pairwise_fixture()
    indices = _measured_plate(genotypes)
    measured = [genotypes[int(index)] for index in indices]
    config = _build_fourier_config(_SITES, _WT, _Q3, max_order=2)

    fit = fit_pairwise_lasso(
        config, measured, response[indices], n_folds=REGISTERED_RECOVERY_PROTOCOL.n_folds
    )
    metrics = coefficient_metrics(fit.pairwise_coefficients, truth)

    assert fit.converged is True
    assert fit.lambda_ratio == _LASSO_LAMBDA_RATIO
    assert fit.lambda_value == pytest.approx(_LASSO_LAMBDA_VALUE, rel=1e-9)
    assert fit.support_size == _LASSO_SUPPORT_SIZE
    assert metrics.support_size == _LASSO_SUPPORT_SIZE
    assert metrics.coefficient_count == 24
    assert metrics.spearman == pytest.approx(_LASSO_SPEARMAN, rel=1e-9)
    assert metrics.relative_sse_gain == pytest.approx(_LASSO_SSE_GAIN, rel=1e-9)


def test_pairwise_lasso_is_bitwise_repeatable() -> None:
    genotypes, response, _truth = _noisy_pairwise_fixture()
    indices = _measured_plate(genotypes)
    measured = [genotypes[int(index)] for index in indices]
    config = _build_fourier_config(_SITES, _WT, _Q3, max_order=2)

    first = fit_pairwise_lasso(config, measured, response[indices], n_folds=5)
    second = fit_pairwise_lasso(config, measured, response[indices], n_folds=5)

    assert first.lambda_ratio == second.lambda_ratio
    assert first.lambda_value == second.lambda_value
    assert first.pairwise_coefficients.tobytes() == second.pairwise_coefficients.tobytes()
