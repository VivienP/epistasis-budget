"""Private numerics for the TrpB pairwise Fourier recovery diagnostic."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from time import perf_counter

import numpy as np
import numpy.typing as npt
from scipy.stats import spearmanr

from epibudget.coeff_recovery import (
    _build_fourier_config,
    _design_matrix,
    _fista_lasso_path_with_status,
    _FourierConfig,
    _full_modes,
    _order_symmetric_kernel,
    _site_indices,
)
from epibudget.data import reveal_measured_fitness
from epibudget.epistasis import _landscape_tensor, _wht_forward
from epibudget.graph import selection_graph
from epibudget.labels import training_target
from epibudget.recovery_protocol import REGISTERED_EXECUTION_POLICY, REGISTERED_RECOVERY_PROTOCOL
from epibudget.robustness import variant_fold
from epibudget.tie_break import (
    TIE_BREAK_VERSION,
    canonical_id,
    loop_counts_over_universe,
    seeded_order,
    stratum_crosses_budget,
)
from epibudget.types import ScoredVariant, Variant

FloatArray = npt.NDArray[np.float64]

_PAIRWISE_ORDER = 2
_CONSTANT_RTOL = 1e-12
_SUPPORT_THRESHOLD = 1e-12
_DEFAULT_LAMBDA_RATIOS: tuple[float, ...] = tuple(
    float(value) for value in np.geomspace(1.0, 1e-3, 20)
)
_FLOAT64_BYTES = np.dtype(np.float64).itemsize
_SHA256_LENGTH = 64
_MIN_FOLDS = 2
_MIN_GATE_SPEARMAN = 0.30
_MIN_GATE_SSE_GAIN = 0.10
_MIN_POSITIVE_STOCHASTIC_SEEDS = 16


def doptimal_workspace_bytes(n_candidates: int, budget: int) -> int:
    """Bytes in the load-bearing dense D-optimal rank-one update matrix."""
    if n_candidates < 1 or budget < 1:
        raise ValueError("candidate count and budget must be positive")
    return n_candidates * budget * _FLOAT64_BYTES


def doptimal_resumable_peak_bytes(n_candidates: int, target_budget: int, block_size: int) -> int:
    """Account for the full state and durable serialization round-trip peak."""
    if n_candidates < 1 or target_budget < 1 or not 1 <= block_size <= target_budget:
        raise ValueError(
            "candidate count, target budget, and compatible block size must be positive"
        )
    update_buffer = doptimal_workspace_bytes(n_candidates, target_budget)
    serialized_chunk = doptimal_workspace_bytes(n_candidates, block_size)
    posterior_snapshot = n_candidates * _FLOAT64_BYTES
    selected_indices = block_size * np.dtype(np.int64).itemsize
    largest_serialized_array = max(serialized_chunk, posterior_snapshot, selected_indices)
    return update_buffer + 2 * largest_serialized_array + posterior_snapshot + selected_indices


def registered_fit_count(budgets: Sequence[int], seeds: Sequence[int]) -> int:
    """Number of method-budget LASSO fits in the frozen A1 protocol."""
    return REGISTERED_RECOVERY_PROTOCOL.fit_count(budgets, seeds)


@dataclass(frozen=True)
class PairwiseTruth:
    """Canonical order-2 modes and their population-normalized coefficients."""

    modes: tuple[tuple[int, ...], ...]
    coefficients: FloatArray


@dataclass(frozen=True)
class CoefficientMetrics:
    """Direct recovery metrics on one fixed coefficient population."""

    spearman: float | None
    relative_sse_gain: float | None
    support_size: int
    coefficient_count: int


@dataclass(frozen=True)
class PairwiseLassoFit:
    """One registered LASSO fit, restricted to its order-2 coefficient output."""

    pairwise_coefficients: FloatArray
    lambda_ratio: float
    lambda_value: float
    support_size: int
    converged: bool


@dataclass(frozen=True)
class PairwiseLassoCVState:
    """Immutable cumulative cross-validation state at a completed fold boundary."""

    problem_sha256: str
    completed_folds: int
    n_folds: int
    lambda_ratios: tuple[float, ...]
    cv_sse: FloatArray
    converged: bool

    def __post_init__(self) -> None:
        """Own and validate the exact numeric snapshot used for deterministic resume."""
        if not _is_sha256(self.problem_sha256):
            raise ValueError("LASSO CV problem SHA must be a lowercase SHA-256")
        if type(self.n_folds) is not int or self.n_folds < _MIN_FOLDS:
            raise ValueError("LASSO CV fold count must be at least 2")
        if type(self.completed_folds) is not int or not 0 <= self.completed_folds <= self.n_folds:
            raise ValueError("LASSO CV completed folds must be in 0..n_folds")
        if type(self.lambda_ratios) is not tuple or not self.lambda_ratios:
            raise ValueError("LASSO CV lambda ratios must be a non-empty tuple")
        if any(
            type(value) is not float or not np.isfinite(value) or not 0.0 < value <= 1.0
            for value in self.lambda_ratios
        ):
            raise ValueError("LASSO CV lambda ratios must contain positive finite floats")
        if any(first < second for first, second in pairwise(self.lambda_ratios)):
            raise ValueError("LASSO CV lambda ratios must be in descending order")
        if not isinstance(self.cv_sse, np.ndarray) or self.cv_sse.dtype != np.dtype(np.float64):
            raise TypeError("LASSO CV SSE must be a float64 array")
        if self.cv_sse.shape != (len(self.lambda_ratios),):
            raise ValueError("LASSO CV SSE shape must match the lambda ratios")
        if not np.all(np.isfinite(self.cv_sse)):
            raise ValueError("LASSO CV SSE must be finite")
        if np.any(self.cv_sse < 0.0):
            raise ValueError("LASSO CV SSE must be nonnegative")
        if type(self.converged) is not bool:
            raise TypeError("LASSO CV converged flag must be boolean")
        content = np.asarray(self.cv_sse, dtype=np.float64, order="C").tobytes(order="C")
        snapshot = np.frombuffer(content, dtype=np.float64)
        object.__setattr__(self, "cv_sse", snapshot)


@dataclass(frozen=True)
class RecoveryCell:
    """One method-budget-seed coefficient-recovery result."""

    method: str
    budget: int
    seed: int | None
    spearman: float | None
    relative_sse_gain: float | None
    support_size: int
    coefficient_count: int
    selected_sha256: str = ""
    fold_sha256: str = ""
    lambda_ratio: float | None = None
    lambda_value: float | None = None
    converged: bool = True
    error: str | None = None


@dataclass(frozen=True)
class RecoveryAggregate:
    """Fail-closed aggregate for one method and budget."""

    method: str
    budget: int
    n_records: int
    n_valid_spearman: int
    n_valid_sse: int
    median_spearman: float | None
    median_relative_sse_gain: float | None
    minimum_relative_sse_gain: float | None
    maximum_relative_sse_gain: float | None
    positive_sse_count: int
    positive_sse_fraction: float | None
    passes_gate: bool


@dataclass(frozen=True)
class RecoveryDecision:
    """Internal resource decision; never a public scientific claim."""

    status: str
    passing_cells: tuple[tuple[str, int], ...]
    aggregates: tuple[RecoveryAggregate, ...]
    reasons: tuple[str, ...]


def decide_recovery(  # noqa: PLR0912, PLR0915
    cells: Sequence[RecoveryCell],
    *,
    stochastic_seeds: Sequence[int],
    expected_methods: Sequence[str],
    expected_budgets: Sequence[int],
) -> RecoveryDecision:
    """Apply the frozen Stage-B gate and fail closed on incomplete seed coverage."""
    expected_seeds = tuple(int(seed) for seed in stochastic_seeds)
    if len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("stochastic seeds must be unique")
    grouped: dict[tuple[str, int], list[RecoveryCell]] = {}
    for cell in cells:
        grouped.setdefault((cell.method, cell.budget), []).append(cell)

    stochastic_methods = {"random", "structural"}
    reasons: list[str] = []
    expected_keys = {
        (str(method), int(budget)) for method in expected_methods for budget in expected_budgets
    }
    observed_keys = set(grouped)
    missing_keys = sorted(expected_keys - observed_keys)
    unexpected_keys = sorted(observed_keys - expected_keys)
    if missing_keys:
        rendered = ", ".join(f"{method}/{budget}" for method, budget in missing_keys)
        reasons.append(f"missing method-budget cells: {rendered}")
    if unexpected_keys:
        rendered = ", ".join(f"{method}/{budget}" for method, budget in unexpected_keys)
        reasons.append(f"unexpected method-budget cells: {rendered}")
    aggregates: list[RecoveryAggregate] = []
    passing: list[tuple[str, int]] = []
    for (method, budget), group in sorted(grouped.items()):
        failed_cells = [
            cell.error or "registered estimator did not converge"
            for cell in group
            if cell.error is not None or not cell.converged
        ]
        if failed_cells:
            reasons.append(f"{method} budget {budget}: {failed_cells[0]}")
        observed_seeds = [cell.seed for cell in group]
        if len(set(observed_seeds)) != len(observed_seeds):
            reasons.append(f"{method} budget {budget} has duplicate seed records")
            continue
        stochastic = method in stochastic_methods
        if stochastic:
            missing = sorted(
                set(expected_seeds) - {seed for seed in observed_seeds if seed is not None}
            )
            unexpected = sorted(
                {seed for seed in observed_seeds if seed is not None} - set(expected_seeds)
            )
            if None in observed_seeds or missing or unexpected:
                reasons.append(
                    f"{method} budget {budget} has missing seeds {missing} and unexpected seeds "
                    f"{unexpected}"
                )
                continue
        elif len(group) != 1 or observed_seeds != [None]:
            reasons.append(
                f"deterministic method {method} budget {budget} must have one seed=None record"
            )
            continue

        spearman_values = [cell.spearman for cell in group if cell.spearman is not None]
        sse_values = [
            cell.relative_sse_gain for cell in group if cell.relative_sse_gain is not None
        ]
        median_spearman = float(np.median(spearman_values)) if spearman_values else None
        median_sse = float(np.median(sse_values)) if sse_values else None
        minimum_sse = float(np.min(sse_values)) if sse_values else None
        maximum_sse = float(np.max(sse_values)) if sse_values else None
        positive_sse = sum(value > 0.0 for value in sse_values)
        positive_sse_fraction = positive_sse / len(sse_values) if sse_values else None
        complete = (
            len(spearman_values) == len(group)
            and len(sse_values) == len(group)
            and (not stochastic or len(group) == len(expected_seeds))
        )
        passes = bool(
            complete
            and median_spearman is not None
            and median_spearman >= _MIN_GATE_SPEARMAN
            and median_sse is not None
            and median_sse >= _MIN_GATE_SSE_GAIN
            and (positive_sse >= _MIN_POSITIVE_STOCHASTIC_SEEDS if stochastic else median_sse > 0.0)
        )
        aggregate = RecoveryAggregate(
            method=method,
            budget=budget,
            n_records=len(group),
            n_valid_spearman=len(spearman_values),
            n_valid_sse=len(sse_values),
            median_spearman=median_spearman,
            median_relative_sse_gain=median_sse,
            minimum_relative_sse_gain=minimum_sse,
            maximum_relative_sse_gain=maximum_sse,
            positive_sse_count=positive_sse,
            positive_sse_fraction=positive_sse_fraction,
            passes_gate=passes,
        )
        aggregates.append(aggregate)
        if passes:
            passing.append((method, budget))

    if reasons:
        status = "invalid_coverage"
    elif passing:
        status = "stage_b_justified"
    else:
        status = "estimator_family_stopped"
    return RecoveryDecision(
        status=status,
        passing_cells=tuple(passing),
        aggregates=tuple(aggregates),
        reasons=tuple(reasons),
    )


def validate_recovery_dataset(dataset: str) -> None:
    """Reject any dataset outside the frozen TrpB-only A1 diagnostic."""
    if dataset != REGISTERED_RECOVERY_PROTOCOL.dataset:
        raise ValueError("the Fourier recovery diagnostic is TrpB-only")


def _finite_runtime_number(value: object, field: str, *, positive: bool = False) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not np.isfinite(value)
        or (value <= 0.0 if positive else value < 0.0)
    ):
        raise ValueError(f"runtime preflight has invalid {field}")
    return float(value)


def _validate_runtime_measurements(
    payload: Mapping[str, object],
    *,
    expected_budgets: Sequence[int],
    expected_candidate_count: int,
    expected_feature_count: int,
) -> list[float]:
    measurements = payload.get("measurements")
    if not isinstance(measurements, list) or len(measurements) != len(expected_budgets):
        raise ValueError("runtime preflight has incomplete budget measurements")
    fit_seconds: list[float] = []
    for expected_budget, measurement in zip(expected_budgets, measurements, strict=True):
        if not isinstance(measurement, Mapping):
            raise ValueError("runtime preflight contains a malformed budget measurement")
        expected_shape = [expected_budget, expected_feature_count]
        expected_design_bytes = expected_budget * expected_feature_count * _FLOAT64_BYTES
        expected_update_bytes = expected_candidate_count * expected_budget * _FLOAT64_BYTES
        if measurement.get("budget") != expected_budget:
            raise ValueError("runtime preflight budget measurement order does not match")
        if measurement.get("design_shape") != expected_shape:
            raise ValueError("runtime preflight design shape does not match")
        if measurement.get("design_bytes") != expected_design_bytes:
            raise ValueError("runtime preflight design byte count does not match")
        if measurement.get("doptimal_update_bytes") != expected_update_bytes:
            raise ValueError("runtime preflight D-optimal byte count does not match")
        if measurement.get("converged") is not True:
            raise ValueError("runtime preflight contains a non-converged fit")
        _finite_runtime_number(measurement.get("design_seconds"), "design timing")
        fit_seconds.append(_finite_runtime_number(measurement.get("fit_seconds"), "fit timing"))
    return fit_seconds


def _validate_runtime_doptimal(
    payload: Mapping[str, object],
    *,
    expected_budgets: Sequence[int],
    expected_candidate_count: int,
) -> float:
    doptimal = payload.get("doptimal_pilot")
    if not isinstance(doptimal, Mapping):
        raise ValueError("runtime preflight has no D-optimal pilot")
    pilot_budget = int(expected_budgets[0])
    maximum_budget = int(expected_budgets[-1])
    if (
        doptimal.get("pilot_budget") != pilot_budget
        or doptimal.get("maximum_budget") != maximum_budget
        or doptimal.get("pilot_update_bytes")
        != expected_candidate_count * pilot_budget * _FLOAT64_BYTES
        or doptimal.get("maximum_update_bytes")
        != expected_candidate_count * maximum_budget * _FLOAT64_BYTES
    ):
        raise ValueError("runtime preflight D-optimal pilot dimensions do not match")
    resumable_block_size = doptimal.get("resumable_block_size")
    resumable_peak = doptimal.get("maximum_resumable_peak_bytes")
    if resumable_block_size is not None or resumable_peak is not None:
        expected_block_size = REGISTERED_EXECUTION_POLICY.doptimal_block_size
        expected_peak = doptimal_resumable_peak_bytes(
            expected_candidate_count, maximum_budget, expected_block_size
        )
        if resumable_block_size != expected_block_size or resumable_peak != expected_peak:
            raise ValueError("runtime preflight resumable D-optimal memory does not match")
    pilot_seconds = _finite_runtime_number(doptimal.get("pilot_seconds"), "D-optimal timing")
    projected_seconds = _finite_runtime_number(
        doptimal.get("projected_maximum_seconds"), "D-optimal projection"
    )
    expected_seconds = pilot_seconds * (maximum_budget / pilot_budget) ** 2
    if not np.isclose(projected_seconds, expected_seconds):
        raise ValueError("runtime preflight D-optimal projection does not match")
    return projected_seconds


def _validate_runtime_projection(
    payload: Mapping[str, object],
    *,
    fit_seconds: Sequence[float],
    projected_doptimal_seconds: float,
    expected_budgets: Sequence[int],
    expected_fit_count: int,
    expected_candidate_count: int,
) -> None:
    fits_per_budget, remainder = divmod(expected_fit_count, len(expected_budgets))
    if remainder:
        raise ValueError("registered fit count is not divisible by the budget count")
    expected_lasso_seconds = sum(fit_seconds) * fits_per_budget
    projected_lasso_seconds = _finite_runtime_number(
        payload.get("projected_lasso_seconds"), "LASSO projection"
    )
    projected_seconds = _finite_runtime_number(payload.get("projected_seconds"), "total projection")
    if not np.isclose(projected_lasso_seconds, expected_lasso_seconds) or not np.isclose(
        projected_seconds, expected_lasso_seconds + projected_doptimal_seconds
    ):
        raise ValueError("runtime preflight total projection does not match")
    maximum_doptimal_bytes = expected_candidate_count * expected_budgets[-1] * _FLOAT64_BYTES
    if payload.get("maximum_doptimal_bytes") != maximum_doptimal_bytes:
        raise ValueError("runtime preflight maximum D-optimal byte count does not match")


def _validate_runtime_provenance(payload: Mapping[str, object], expected_commit: str) -> None:
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("runtime preflight has no provenance")
    start = provenance.get("workspace_start")
    end = provenance.get("workspace_end")
    if not isinstance(start, Mapping) or not isinstance(end, Mapping):
        raise ValueError("runtime preflight has incomplete workspace provenance")
    if provenance.get("workspace_state_matches") is not True or start != end:
        raise ValueError("runtime preflight workspace state changed during execution")
    if start.get("code_state") != "clean" or end.get("code_state") != "clean":
        raise ValueError("runtime preflight must run from a clean workspace")
    if (
        start.get("execution_commit") != expected_commit
        or end.get("execution_commit") != expected_commit
    ):
        raise ValueError("runtime preflight execution commit does not match the recovery run")


def validate_runtime_preflight(
    payload: Mapping[str, object],
    *,
    expected_commit: str,
    expected_candidate_count: int,
    expected_candidate_sha256: str,
    expected_budgets: Sequence[int],
    expected_fit_count: int,
    expected_feature_count: int,
) -> None:
    """Validate the complete, clean, label-free v4 resource preflight."""
    if payload.get("schema_version") != "epibudget-fourier-runtime-v4":
        raise ValueError("runtime preflight must use epibudget-fourier-runtime-v4")
    if payload.get("uses_measured_labels") is not False:
        raise ValueError("runtime preflight must be label-independent")
    if payload.get("candidate_count") != expected_candidate_count:
        raise ValueError("runtime preflight candidate count does not match")
    if payload.get("candidate_sha256") != expected_candidate_sha256:
        raise ValueError("runtime preflight candidate hash does not match")
    if payload.get("registered_fit_count") != expected_fit_count:
        raise ValueError("runtime preflight fit count does not match")
    if payload.get("measured_budgets") != list(expected_budgets):
        raise ValueError("runtime preflight did not measure every registered budget")
    fit_seconds = _validate_runtime_measurements(
        payload,
        expected_budgets=expected_budgets,
        expected_candidate_count=expected_candidate_count,
        expected_feature_count=expected_feature_count,
    )
    projected_doptimal_seconds = _validate_runtime_doptimal(
        payload,
        expected_budgets=expected_budgets,
        expected_candidate_count=expected_candidate_count,
    )
    _validate_runtime_projection(
        payload,
        fit_seconds=fit_seconds,
        projected_doptimal_seconds=projected_doptimal_seconds,
        expected_budgets=expected_budgets,
        expected_fit_count=expected_fit_count,
        expected_candidate_count=expected_candidate_count,
    )
    _validate_runtime_provenance(payload, expected_commit)


@dataclass(frozen=True)
class SelectionSequence:
    """One label-free acquisition sequence built once at the maximum budget."""

    method: str
    seed: int | None
    selected: tuple[Variant, ...]
    selected_sha256: str
    tie_break_version: str


@dataclass(frozen=True)
class SelectionPlan:
    """Registered budgets and every acquisition sequence needed by the diagnostic."""

    budgets: tuple[int, ...]
    sequences: tuple[SelectionSequence, ...]

    def plate(self, method: str, seed: int | None, budget: int) -> tuple[Variant, ...]:
        """Return one exact registered prefix, rejecting an unknown cell."""
        if budget not in self.budgets:
            raise ValueError(f"budget {budget} is not registered")
        matches = [
            sequence
            for sequence in self.sequences
            if sequence.method == method and sequence.seed == seed
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one sequence for method={method!r}, seed={seed!r}")
        return matches[0].selected[:budget]


@dataclass(frozen=True)
class SyntheticFitBenchmark:
    """Label-independent resource measurement for one registered budget."""

    budget: int
    design_shape: tuple[int, int]
    design_bytes: int
    doptimal_update_bytes: int
    design_seconds: float
    fit_seconds: float
    support_size: int
    converged: bool


@dataclass(frozen=True)
class DOptimalBenchmark:
    """Measured D-optimal pilot and its registered quadratic time projection."""

    pilot_budget: int
    maximum_budget: int
    pilot_seconds: float
    projected_maximum_seconds: float
    pilot_update_bytes: int
    maximum_update_bytes: int
    resumable_block_size: int
    maximum_resumable_peak_bytes: int


def benchmark_synthetic_fit(
    config: _FourierConfig,
    candidates: Sequence[Variant],
    *,
    budget: int,
    seed: int,
    n_folds: int,
) -> SyntheticFitBenchmark:
    """Time design construction and one sparse synthetic fit without accepting real labels."""
    canonical = tuple(sorted(candidates, key=canonical_id))
    if budget > len(canonical):
        raise ValueError(f"budget {budget} exceeds candidate count {len(canonical)}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(canonical))
    measured = [canonical[int(index)] for index in order[:budget]]

    started = perf_counter()
    population_size = config.q ** len(config.sites)
    design = np.sqrt(population_size) * _design_matrix(config, _site_indices(config, measured))
    design_seconds = perf_counter() - started
    beta = np.zeros(design.shape[1], dtype=np.float64)
    nonzero = min(8, design.shape[1])
    if nonzero:
        support = np.linspace(0, design.shape[1] - 1, nonzero, dtype=np.int64)
        beta[support] = np.linspace(0.25, 1.0, nonzero, dtype=np.float64)
    response = 0.1 + design @ beta

    started = perf_counter()
    fit = fit_pairwise_lasso(config, measured, response, n_folds=n_folds)
    fit_seconds = perf_counter() - started
    return SyntheticFitBenchmark(
        budget=budget,
        design_shape=design.shape,
        design_bytes=design.nbytes,
        doptimal_update_bytes=doptimal_workspace_bytes(len(canonical), budget),
        design_seconds=design_seconds,
        fit_seconds=fit_seconds,
        support_size=fit.support_size,
        converged=fit.converged,
    )


def _sequence_sha256(selected: Sequence[Variant]) -> str:
    payload = json.dumps(
        [canonical_id(variant) for variant in selected], separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _fold_sha256(selected: Sequence[Variant], n_folds: int) -> str:
    rows = sorted((canonical_id(variant), variant_fold(variant, n_folds)) for variant in selected)
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _selection_sequence(
    method: str, seed: int | None, selected: Sequence[Variant], tie_break_version: str
) -> SelectionSequence:
    selected_tuple = tuple(selected)
    return SelectionSequence(
        method=method,
        seed=seed,
        selected=selected_tuple,
        selected_sha256=_sequence_sha256(selected_tuple),
        tie_break_version=tie_break_version,
    )


def _deterministic_order(
    candidates: Sequence[ScoredVariant], scores: Mapping[Variant, float]
) -> list[ScoredVariant]:
    return sorted(candidates, key=lambda item: (-scores[item.variant], canonical_id(item.variant)))


def _canonical_identity_sha256(variant: Variant) -> str:
    return hashlib.sha256(canonical_id(variant).encode("ascii")).hexdigest()


@dataclass
class ReducedDOptimalState:
    """Mutable state for the reduced pairwise D-optimal pivot sequence."""

    candidates: tuple[Variant, ...]
    site_indices: npt.NDArray[np.int64]
    q: int
    population_size: int
    target_budget: int
    selected_indices: list[int]
    posterior_variance: FloatArray
    updates: FloatArray


def initialise_reduced_doptimal(
    config: _FourierConfig,
    candidates: Sequence[Variant],
    *,
    target_budget: int,
) -> ReducedDOptimalState:
    """Allocate the full deterministic D-optimal state before the first pivot."""
    if config.max_order != _PAIRWISE_ORDER:
        raise ValueError(f"reduced D-optimal requires max_order=2, got {config.max_order}")
    canonical = tuple(sorted(candidates, key=canonical_id))
    if len(set(canonical)) != len(canonical):
        raise ValueError("D-optimal candidates contain duplicate identities")
    if not 1 <= target_budget <= len(canonical):
        raise ValueError(f"budget {target_budget} must be in 1..{len(canonical)}")

    site_indices = np.asarray(_site_indices(config, canonical), dtype=np.int64, order="C")
    population_size = config.q ** len(config.sites)
    diagonal = float(
        population_size
        * _order_symmetric_kernel(site_indices[[0]], site_indices[[0]], config.q, (1, 2))[0, 0]
    )
    return ReducedDOptimalState(
        candidates=canonical,
        site_indices=site_indices,
        q=config.q,
        population_size=population_size,
        target_budget=target_budget,
        selected_indices=[],
        posterior_variance=np.full(len(canonical), diagonal, dtype=np.float64),
        updates=np.zeros((len(canonical), target_budget), dtype=np.float64, order="C"),
    )


def advance_reduced_doptimal(state: ReducedDOptimalState, stop: int) -> None:
    """Advance one state through the absolute, exclusive pivot ``stop``."""
    start = len(state.selected_indices)
    if not start <= stop <= state.target_budget:
        raise ValueError(f"D-optimal stop {stop} must be in {start}..{state.target_budget}")
    for step in range(start, stop):
        available = state.posterior_variance.copy()
        available[state.selected_indices] = -np.inf
        maximum = float(np.max(available))
        tied = np.flatnonzero(available == maximum)
        pick = min(
            (int(index) for index in tied),
            key=lambda index: _canonical_identity_sha256(state.candidates[index]),
        )
        state.selected_indices.append(pick)
        prior_covariance = (
            state.population_size
            * _order_symmetric_kernel(
                state.site_indices, state.site_indices[[pick]], state.q, (1, 2)
            )[:, 0]
        )
        covariance = prior_covariance - state.updates[:, :step] @ state.updates[pick, :step]
        denominator = 1.0 + max(float(covariance[pick]), 0.0)
        state.updates[:, step] = covariance / np.sqrt(denominator)
        state.posterior_variance = np.maximum(
            state.posterior_variance - np.square(state.updates[:, step]), 0.0
        )


def selected_reduced_doptimal(state: ReducedDOptimalState) -> tuple[Variant, ...]:
    """Return the selected candidate prefix represented by one state."""
    return tuple(state.candidates[index] for index in state.selected_indices)


def _initial_reduced_doptimal_posterior(state: ReducedDOptimalState) -> FloatArray:
    diagonal = float(
        state.population_size
        * _order_symmetric_kernel(
            state.site_indices[[0]], state.site_indices[[0]], state.q, (1, 2)
        )[0, 0]
    )
    return np.full(len(state.candidates), diagonal, dtype=np.float64)


def validate_completed_reduced_doptimal_state(state: ReducedDOptimalState) -> None:
    """Validate a complete state by replaying pivots and posterior updates in O(NB)."""
    candidate_count = len(state.candidates)
    if len(state.selected_indices) != state.target_budget:
        raise ValueError("D-optimal state is not complete at its target budget")
    if (
        state.updates.dtype != np.dtype(np.float64)
        or not state.updates.flags.c_contiguous
        or state.updates.shape != (candidate_count, state.target_budget)
        or state.posterior_variance.dtype != np.dtype(np.float64)
        or not state.posterior_variance.flags.c_contiguous
        or state.posterior_variance.shape != (candidate_count,)
    ):
        raise ValueError("D-optimal state has invalid completed-state binary layout")
    if len(set(state.selected_indices)) != state.target_budget or any(
        not 0 <= index < candidate_count for index in state.selected_indices
    ):
        raise ValueError("D-optimal state has invalid completed-state indices")
    if not np.all(np.isfinite(state.updates)) or not np.all(np.isfinite(state.posterior_variance)):
        raise ValueError("D-optimal state contains non-finite numeric values")
    if np.any(state.posterior_variance < 0.0):
        raise ValueError("D-optimal state posterior variance must be nonnegative")

    posterior = _initial_reduced_doptimal_posterior(state)
    selected: list[int] = []
    for step, observed in enumerate(state.selected_indices):
        available = posterior.copy()
        available[selected] = -np.inf
        maximum = float(np.max(available))
        tied = np.flatnonzero(available == maximum)
        expected = min(
            (int(index) for index in tied),
            key=lambda index: _canonical_identity_sha256(state.candidates[index]),
        )
        if observed != expected:
            raise ValueError("D-optimal state pivot does not match the variance argmax")
        selected.append(observed)
        posterior = np.maximum(posterior - np.square(state.updates[:, step]), 0.0)
    if posterior.tobytes(order="C") != state.posterior_variance.tobytes(order="C"):
        raise ValueError("D-optimal state posterior recurrence does not match its final variance")


def reduced_doptimal_order(
    config: _FourierConfig, candidates: Sequence[Variant], budget: int
) -> tuple[Variant, ...]:
    """Build the deterministic reduced order-1-plus-order-2 Bayesian D-optimal sequence."""
    state = initialise_reduced_doptimal(config, candidates, target_budget=budget)
    advance_reduced_doptimal(state, budget)
    return selected_reduced_doptimal(state)


def benchmark_doptimal_prefix(
    config: _FourierConfig,
    candidates: Sequence[Variant],
    *,
    pilot_budget: int,
    maximum_budget: int,
    block_size: int = REGISTERED_EXECUTION_POLICY.doptimal_block_size,
) -> DOptimalBenchmark:
    """Time a label-free D-optimal prefix and project its O(N*B^2) maximum-budget cost."""
    if not 1 <= pilot_budget <= maximum_budget <= len(candidates):
        raise ValueError("require 1 <= pilot_budget <= maximum_budget <= candidate count")
    if not 1 <= block_size <= maximum_budget:
        raise ValueError("D-optimal checkpoint block size must fit the maximum budget")
    started = perf_counter()
    reduced_doptimal_order(config, candidates, pilot_budget)
    pilot_seconds = perf_counter() - started
    scale = (maximum_budget / pilot_budget) ** 2
    return DOptimalBenchmark(
        pilot_budget=pilot_budget,
        maximum_budget=maximum_budget,
        pilot_seconds=pilot_seconds,
        projected_maximum_seconds=pilot_seconds * scale,
        pilot_update_bytes=doptimal_workspace_bytes(len(candidates), pilot_budget),
        maximum_update_bytes=doptimal_workspace_bytes(len(candidates), maximum_budget),
        resumable_block_size=block_size,
        maximum_resumable_peak_bytes=doptimal_resumable_peak_bytes(
            len(candidates), maximum_budget, block_size
        ),
    )


def _require_no_boundary_ties(
    method: str,
    candidates: Sequence[ScoredVariant],
    scores: Mapping[Variant, float],
    budgets: Sequence[int],
) -> None:
    for budget in budgets:
        if stratum_crosses_budget(
            candidates,
            lambda item: scores[item.variant],
            lambda item: canonical_id(item.variant),
            budget,
        ):
            raise ValueError(f"{method} has an exact score tie crossing registered budget {budget}")


def _validated_doptimal_state(
    state: ReducedDOptimalState, candidates: Sequence[Variant], budget: int
) -> tuple[Variant, ...]:
    if not isinstance(state, ReducedDOptimalState):
        raise TypeError("injected D-optimal state must be a ReducedDOptimalState")
    canonical = tuple(sorted(candidates, key=canonical_id))
    if state.target_budget != budget or len(state.selected_indices) != budget:
        raise ValueError("injected D-optimal state is not complete at the requested target budget")
    if state.candidates != canonical:
        raise ValueError("injected D-optimal state candidate pool does not match scored candidates")
    validate_completed_reduced_doptimal_state(state)
    return selected_reduced_doptimal(state)


def build_selection_plan(
    scored: Sequence[ScoredVariant],
    *,
    budgets: Sequence[int],
    seeds: Sequence[int],
    max_order: int,
    doptimal_state: ReducedDOptimalState | None = None,
) -> SelectionPlan:
    """Build label-free info, fitness, random, and structural sequences from scored candidates."""
    registered_budgets = tuple(int(budget) for budget in budgets)
    if not registered_budgets or any(budget < 1 for budget in registered_budgets):
        raise ValueError("budgets must be non-empty positive integers")
    if any(first >= second for first, second in pairwise(registered_budgets)):
        raise ValueError("budgets must be strictly increasing")
    registered_seeds = tuple(int(seed) for seed in seeds)
    if len(set(registered_seeds)) != len(registered_seeds):
        raise ValueError("seeds must be unique")

    canonical = tuple(sorted(scored, key=lambda item: canonical_id(item.variant)))
    variants = [item.variant for item in canonical]
    if len(set(variants)) != len(variants):
        raise ValueError("scored candidates contain duplicate identities")
    maximum_budget = registered_budgets[-1]
    if maximum_budget > len(canonical):
        raise ValueError(
            f"maximum budget {maximum_budget} exceeds candidate count {len(canonical)}"
        )

    info_graph = selection_graph(canonical, max_order=max_order, method="info")
    info_scores = {
        item.variant: info_graph.info_gain(frozenset(), item.variant) for item in canonical
    }
    fitness_scores = {item.variant: item.delta_g for item in canonical}
    _require_no_boundary_ties("info", canonical, info_scores, registered_budgets)
    _require_no_boundary_ties("fitness", canonical, fitness_scores, registered_budgets)

    sequences = [
        _selection_sequence(
            "info",
            None,
            [
                item.variant
                for item in _deterministic_order(canonical, info_scores)[:maximum_budget]
            ],
            "exact-score-canonical-v1",
        ),
        _selection_sequence(
            "fitness",
            None,
            [
                item.variant
                for item in _deterministic_order(canonical, fitness_scores)[:maximum_budget]
            ],
            "exact-score-canonical-v1",
        ),
    ]

    site_positions = sorted({mutation[0] for variant in variants for mutation in variant})
    wt_by_site: dict[int, str] = {}
    alphabet: set[str] = set()
    for variant in variants:
        for site, wt_aa, mutant_aa in variant:
            if wt_by_site.setdefault(site, wt_aa) != wt_aa:
                raise ValueError(f"inconsistent WT residue at site {site}")
            alphabet.update((wt_aa, mutant_aa))
    if site_positions:
        doptimal_config = _build_fourier_config(
            site_positions,
            [wt_by_site[site] for site in site_positions],
            "".join(sorted(alphabet)),
            max_order=_PAIRWISE_ORDER,
        )
        if doptimal_state is None:
            doptimal = reduced_doptimal_order(doptimal_config, variants, maximum_budget)
        else:
            doptimal = _validated_doptimal_state(doptimal_state, variants, maximum_budget)
        sequences.append(
            _selection_sequence(
                "doptimal_reduced_pairwise",
                None,
                doptimal,
                "canonical-identity-sha256-v1",
            )
        )

    structural_scores = loop_counts_over_universe(variants, max_order=max_order)
    for seed in registered_seeds:
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(canonical))
        sequences.append(
            _selection_sequence(
                "random",
                seed,
                [canonical[int(index)].variant for index in permutation[:maximum_budget]],
                "canonical-pcg64-permutation-v1",
            )
        )
        structural = seeded_order(
            canonical,
            lambda item: float(structural_scores[item.variant]),
            lambda item: canonical_id(item.variant),
            seed,
        )
        sequences.append(
            _selection_sequence(
                "structural",
                seed,
                [item.variant for item in structural[:maximum_budget]],
                TIE_BREAK_VERSION,
            )
        )
    return SelectionPlan(budgets=registered_budgets, sequences=tuple(sequences))


def pairwise_truth(values: Mapping[Variant, float], sites: Sequence[int]) -> PairwiseTruth:
    """Extract the fixed order-2 Fourier estimand with ``psi = sqrt(N) * chi`` normalization."""
    if any(not np.isfinite(value) for value in values.values()):
        raise ValueError("Fourier truth values must all be finite")
    tensor, bases = _landscape_tensor(values, sites)
    coefficients = _wht_forward(tensor, bases)
    modes = tuple(
        mode
        for mode in _full_modes(len(bases), tensor.shape[0], _PAIRWISE_ORDER)
        if np.count_nonzero(mode) == _PAIRWISE_ORDER
    )
    scale = np.sqrt(tensor.size)
    pairwise = np.array([coefficients[mode] / scale for mode in modes], dtype=np.float64)
    return PairwiseTruth(modes=modes, coefficients=pairwise)


def _effectively_constant(values: FloatArray) -> bool:
    if values.size == 0:
        return True
    scale = max(1.0, float(np.max(np.abs(values))))
    return bool(np.ptp(values) <= _CONSTANT_RTOL * scale)


def coefficient_metrics(predicted: FloatArray, truth: FloatArray) -> CoefficientMetrics:
    """Score one estimate against the fixed coefficient vector without coercing undefined values."""
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if predicted.shape != truth.shape:
        raise ValueError(
            f"predicted and truth must have the same shape, got {predicted.shape} and {truth.shape}"
        )
    if predicted.ndim != 1:
        raise ValueError("predicted and truth coefficients must be one-dimensional")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(truth)):
        raise ValueError("predicted and truth coefficients must all be finite")

    spearman: float | None = None
    if not _effectively_constant(predicted) and not _effectively_constant(truth):
        statistic = float(spearmanr(predicted, truth).statistic)
        spearman = statistic if np.isfinite(statistic) else None

    prior_sse = float(np.sum(np.square(truth)))
    fitted_sse = float(np.sum(np.square(predicted - truth)))
    relative_sse_gain = None if prior_sse == 0.0 else 1.0 - fitted_sse / prior_sse
    return CoefficientMetrics(
        spearman=spearman,
        relative_sse_gain=relative_sse_gain,
        support_size=int(np.count_nonzero(np.abs(predicted) > _SUPPORT_THRESHOLD)),
        coefficient_count=truth.size,
    )


def evaluate_plate(
    config: _FourierConfig,
    selected: Sequence[Variant],
    landscape: Mapping[Variant, float],
    truth: FloatArray,
    *,
    method: str,
    seed: int | None,
    budget: int,
    n_folds: int,
    resume_cv: PairwiseLassoCVState | None = None,
    on_fold_completed: Callable[[PairwiseLassoCVState], None] | None = None,
) -> RecoveryCell:
    """Reveal exactly one frozen plate, fit it, and score the fixed pairwise estimand."""
    if len(selected) != budget:
        raise ValueError(f"selected plate has {len(selected)} rows but budget is {budget}")
    if len(set(selected)) != len(selected):
        raise ValueError("selected plate contains duplicate variants")
    revealed = reveal_measured_fitness(landscape, selected)
    if len(revealed) != len(selected):
        missing = [variant for variant in selected if variant not in revealed]
        raise ValueError(f"selected plate has {len(missing)} missing measured labels")
    response = np.array([training_target(revealed[variant]) for variant in selected])
    fit = fit_pairwise_lasso(
        config,
        selected,
        response,
        n_folds=n_folds,
        resume_cv=resume_cv,
        on_fold_completed=on_fold_completed,
    )
    if not fit.converged:
        raise RuntimeError("FISTA did not converge")
    metrics = coefficient_metrics(fit.pairwise_coefficients, truth)
    return RecoveryCell(
        method=method,
        budget=budget,
        seed=seed,
        spearman=metrics.spearman,
        relative_sse_gain=metrics.relative_sse_gain,
        support_size=metrics.support_size,
        coefficient_count=metrics.coefficient_count,
        selected_sha256=_sequence_sha256(selected),
        fold_sha256=_fold_sha256(selected, n_folds),
        lambda_ratio=fit.lambda_ratio,
        lambda_value=fit.lambda_value,
        converged=fit.converged,
    )


def _lambda_max(design: FloatArray, centered_response: FloatArray) -> float:
    if design.shape[0] == 0:
        raise ValueError("cannot fit an empty training fold")
    return 2.0 * float(np.max(np.abs(design.T @ centered_response))) / float(design.shape[0])


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def pairwise_lasso_problem_sha256(
    config: _FourierConfig,
    measured: Sequence[Variant],
    response: FloatArray,
    *,
    n_folds: int,
    lambda_ratios: Sequence[float],
) -> str:
    """Hash the exact ordered numeric problem that makes a CV snapshot reusable."""
    response_array = np.asarray(response, dtype=np.float64, order="C")
    if response_array.shape != (len(measured),):
        raise ValueError(f"response must have shape ({len(measured)},), got {response_array.shape}")
    if not np.all(np.isfinite(response_array)):
        raise ValueError("response must be finite")
    if type(n_folds) is not int or n_folds < _MIN_FOLDS:
        raise ValueError("n_folds must be at least 2")
    ratios = tuple(float(value) for value in lambda_ratios)
    if not ratios or any(not np.isfinite(value) or not 0.0 < value <= 1.0 for value in ratios):
        raise ValueError("lambda ratios must be non-empty and in (0, 1]")
    if any(first < second for first, second in pairwise(ratios)):
        raise ValueError("lambda ratios must be in descending order")
    folds = np.asarray(
        [variant_fold(variant, n_folds) for variant in measured], dtype=np.int64, order="C"
    )
    return _pairwise_lasso_problem_sha256_from_arrays(
        config,
        measured,
        response_array,
        folds,
        n_folds=n_folds,
        lambda_ratios=ratios,
    )


def _pairwise_lasso_problem_sha256_from_arrays(
    config: _FourierConfig,
    measured: Sequence[Variant],
    response: FloatArray,
    folds: npt.NDArray[np.int64],
    *,
    n_folds: int,
    lambda_ratios: tuple[float, ...],
) -> str:
    """Hash already-normalized arrays that are used directly by the CV solve."""

    digest = hashlib.sha256()

    def update_segment(name: str, content: bytes) -> None:
        encoded_name = name.encode("ascii")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)

    def update_array(name: str, value: npt.NDArray[np.generic]) -> None:
        array = np.asarray(value)
        if array.dtype.hasobject or array.dtype.kind in {"O", "V", "U", "S"}:
            raise ValueError(f"{name} must use a plain numeric dtype")
        little_dtype = array.dtype.newbyteorder("<")
        canonical = np.asarray(array, dtype=little_dtype, order="C")
        descriptor = {
            "dtype": array.dtype.name,
            "byteorder": "little",
            "shape": [int(extent) for extent in array.shape],
            "order": "C",
        }
        update_segment(
            f"{name}.descriptor",
            json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
                "ascii"
            ),
        )
        update_segment(f"{name}.bytes", canonical.tobytes(order="C"))

    config_payload = {
        "sites": list(config.sites),
        "q": config.q,
        "max_order": config.max_order,
        "alphabet_index": [
            [[residue, index] for residue, index in sorted(mapping.items())]
            for mapping in config.alphabet_index
        ],
        "modes": [list(mode) for mode in config.modes],
    }
    update_segment(
        "config",
        json.dumps(config_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "ascii"
        ),
    )
    update_array("basis", np.asarray(config.basis))
    update_segment("measured.count", len(measured).to_bytes(8, "big"))
    for index, variant in enumerate(measured):
        update_segment(f"measured.{index}", canonical_id(variant).encode("ascii"))
    update_array("response", response)
    update_array("folds", folds)
    update_segment("n_folds", n_folds.to_bytes(8, "big"))
    update_array("lambda_ratios", np.asarray(lambda_ratios, dtype=np.float64, order="C"))
    return digest.hexdigest()


def fit_pairwise_lasso(  # noqa: PLR0912, PLR0915
    config: _FourierConfig,
    measured: Sequence[Variant],
    response: FloatArray,
    *,
    n_folds: int = 5,
    lambda_ratios: Sequence[float] = _DEFAULT_LAMBDA_RATIOS,
    resume_cv: PairwiseLassoCVState | None = None,
    on_fold_completed: Callable[[PairwiseLassoCVState], None] | None = None,
) -> PairwiseLassoFit:
    """Fit the registered order-1-plus-order-2 LASSO with fold-local response preprocessing."""
    response = np.asarray(response, dtype=np.float64, order="C")
    if config.max_order != _PAIRWISE_ORDER:
        raise ValueError(f"pairwise LASSO requires max_order=2, got {config.max_order}")
    if response.shape != (len(measured),):
        raise ValueError(f"response must have shape ({len(measured)},), got {response.shape}")
    if not np.all(np.isfinite(response)):
        raise ValueError("response must be finite")
    if _effectively_constant(response):
        raise ValueError("response is effectively constant")
    if n_folds < _MIN_FOLDS:
        raise ValueError("n_folds must be at least 2")
    ratios = tuple(float(value) for value in lambda_ratios)
    if not ratios or any(not 0.0 < value <= 1.0 for value in ratios):
        raise ValueError("lambda ratios must be non-empty and in (0, 1]")
    if any(first < second for first, second in pairwise(ratios)):
        raise ValueError("lambda ratios must be in descending order")
    folds = np.asarray(
        [variant_fold(variant, n_folds) for variant in measured], dtype=np.int64, order="C"
    )
    problem_sha256 = _pairwise_lasso_problem_sha256_from_arrays(
        config,
        measured,
        response,
        folds,
        n_folds=n_folds,
        lambda_ratios=ratios,
    )
    if resume_cv is not None:
        if resume_cv.n_folds != n_folds:
            raise ValueError("resume CV fold count does not match n_folds")
        if resume_cv.lambda_ratios != ratios:
            raise ValueError("resume CV lambda ratios do not match")
        if resume_cv.problem_sha256 != problem_sha256:
            raise ValueError("resume CV problem SHA does not match the current fit")
    if on_fold_completed is not None and not callable(on_fold_completed):
        raise TypeError("on_fold_completed must be callable")

    population_size = config.q ** len(config.sites)
    design = np.sqrt(population_size) * _design_matrix(config, _site_indices(config, measured))
    start_fold = 0 if resume_cv is None else resume_cv.completed_folds
    cv_sse = (
        np.zeros(len(ratios), dtype=np.float64)
        if resume_cv is None
        else np.array(resume_cv.cv_sse, dtype=np.float64, order="C", copy=True)
    )
    converged = True if resume_cv is None else resume_cv.converged
    for fold in range(start_fold, n_folds):
        train = folds != fold
        test = folds == fold
        if not np.any(train) or not np.any(test):
            raise ValueError(f"fold {fold} has no train or validation rows")
        y_train = response[train]
        mean_train = float(np.mean(y_train))
        centered_train = y_train - mean_train
        design_mean = np.mean(design[train], axis=0)
        centered_design_train = design[train] - design_mean
        centered_design_test = design[test] - design_mean
        fold_lambda_max = _lambda_max(centered_design_train, centered_train)
        if fold_lambda_max == 0.0:
            path = [np.zeros(design.shape[1], dtype=np.float64) for _ in ratios]
        else:
            unscaled_path = [
                centered_design_train.shape[0] * ratio * fold_lambda_max for ratio in ratios
            ]
            path, fold_converged = _fista_lasso_path_with_status(
                centered_design_train, centered_train, unscaled_path
            )
            converged = converged and fold_converged
        for index, beta in enumerate(path):
            predicted = mean_train + centered_design_test @ beta
            cv_sse[index] += float(np.sum(np.square(predicted - response[test])))
        if on_fold_completed is not None:
            on_fold_completed(
                PairwiseLassoCVState(
                    problem_sha256=problem_sha256,
                    completed_folds=fold + 1,
                    n_folds=n_folds,
                    lambda_ratios=ratios,
                    cv_sse=cv_sse,
                    converged=converged,
                )
            )

    best = int(np.argmin(cv_sse))
    mean_all = float(np.mean(response))
    centered_all = response - mean_all
    centered_design = design - np.mean(design, axis=0)
    full_lambda_max = _lambda_max(centered_design, centered_all)
    selected_lambda = ratios[best] * full_lambda_max
    if full_lambda_max == 0.0:
        beta = np.zeros(design.shape[1], dtype=np.float64)
    else:
        unscaled_path = [len(measured) * ratio * full_lambda_max for ratio in ratios[: best + 1]]
        full_path, full_converged = _fista_lasso_path_with_status(
            centered_design, centered_all, unscaled_path
        )
        converged = converged and full_converged
        beta = full_path[-1]

    pairwise_mask = np.array(
        [np.count_nonzero(mode) == _PAIRWISE_ORDER for mode in config.modes], dtype=np.bool_
    )
    pairwise_coefficients = np.asarray(beta[pairwise_mask], dtype=np.float64)
    return PairwiseLassoFit(
        pairwise_coefficients=pairwise_coefficients,
        lambda_ratio=ratios[best],
        lambda_value=selected_lambda,
        support_size=int(np.count_nonzero(np.abs(pairwise_coefficients) > _SUPPORT_THRESHOLD)),
        converged=converged,
    )
