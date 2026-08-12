"""Durable orchestration for the registered Fourier recovery diagnostic."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

import numpy as np

from epibudget.coeff_recovery import AA20, _build_fourier_config, _FourierConfig
from epibudget.data import (
    DatasetSpec,
    enumerate_candidates,
    resolve_dataset,
    reveal_measured_fitness,
)
from epibudget.fourier_recovery import (
    _DEFAULT_LAMBDA_RATIOS,
    PairwiseLassoCVState,
    PairwiseTruth,
    RecoveryCell,
    ReducedDOptimalState,
    SelectionPlan,
    SelectionSequence,
    _fold_sha256,
    _sequence_sha256,
    advance_reduced_doptimal,
    build_selection_plan,
    decide_recovery,
    evaluate_plate,
    initialise_reduced_doptimal,
    pairwise_lasso_problem_sha256,
    pairwise_truth,
    registered_fit_count,
    validate_deterministic_selection_boundaries,
    validate_runtime_preflight,
)
from epibudget.labels import training_target
from epibudget.recovery_doptimal import (
    DOptimalCheckpointCursor,
    DOptimalCheckpointIdentity,
    doptimal_geometry_sha256,
)
from epibudget.recovery_lasso import (
    LassoCheckpointCursor,
    LassoCheckpointIdentity,
)
from epibudget.recovery_protocol import (
    REGISTERED_EXECUTION_POLICY,
    REGISTERED_RECOVERY_PROTOCOL,
    validate_recovery_configuration,
)
from epibudget.recovery_runtime import (
    RecoveryInputBundle,
    RecoveryProvenance,
    RecoveryRuntimeRecord,
    archive_recovery_inputs,
    capture_numeric_compatibility,
    materialize_archived_recovery_input,
    require_numeric_compatibility,
    scientific_identity_from_inputs,
)
from epibudget.recovery_state import (
    ExecutionAttemptCompletion,
    ExecutionAttemptStart,
    PreparedRecoveryRun,
    PublishedSelectionPlan,
    RecoveryCellKey,
    RecoveryRunState,
    RecoveryStateCursor,
    publish_execution_attempt_completed_at,
    publish_execution_attempt_started_at,
    publish_prepared_run_at,
    publish_recovery_cell_at,
    publish_recovery_report_at,
    publish_selection_plan_at,
    registered_cell_keys,
    replay_recovery_state,
)
from epibudget.run_store import (
    BlobRef,
    ContentAddressedRunStore,
    RunStoreError,
    RunStoreSession,
    canonical_json_bytes,
)
from epibudget.scored_cache import CacheIdentity, candidate_sha256, validate_cache_against_universe
from epibudget.spectrum_diagnostic import _capture_workspace_snapshot
from epibudget.tie_break import canonical_id
from epibudget.types import ScoredVariant, Variant

_MODEL_ID = "facebook/esm2_t33_650M_UR50D"
_SCORER_SEED = 0
_N_PERTURBATIONS = 16
_GIT_COMMIT_LENGTH = 40
_SHA256_LENGTH = 64
_REGISTERED_INPUT_COUNT = 4
_PLAN_SCHEMA = "epibudget-recovery-selection-plan-payload-v1"
_REPORT_SCHEMA = "epibudget-fourier-recovery-v3"
_IMPUTATION_NOTE = (
    "The redistributed target contains 159,129 measured values and 871 source-imputed "
    "values whose identities are not exposed by the mirror."
)


class RecoveryEngineError(Exception):
    """A recovery run cannot be prepared, resumed, verified, or exported safely."""


@dataclass(frozen=True)
class RecoveryProgress:
    """One progress event emitted only after its corresponding state is durable."""

    stage: str
    completed: int
    total: int
    method: str | None = None
    seed: int | None = None
    budget: int | None = None


@dataclass(frozen=True)
class RecoveryStatus:
    """Read-only summary of one verified durable recovery run."""

    prepared: bool
    selection_complete: bool
    doptimal_completed: int
    completed_cells: int
    total_cells: int
    active_cell: str | None
    completed_folds: int
    report_available: bool
    latest_sequence: int | None


@dataclass(frozen=True)
class _RegisteredInputs:
    specification: DatasetSpec
    candidates: tuple[Variant, ...]
    config: _FourierConfig
    scored: tuple[ScoredVariant, ...]
    expected_cache_identity: CacheIdentity
    observed_cache_identity: CacheIdentity


def _repo_path(repo: Path | None) -> Path:
    return Path(__file__).resolve().parents[2] if repo is None else repo.resolve()


def _clean_commit(repo: Path) -> str:
    snapshot = _capture_workspace_snapshot(repo)
    if snapshot.code_state != "clean" or len(snapshot.execution_commit) != _GIT_COMMIT_LENGTH:
        raise RecoveryEngineError("recovery execution requires an exact clean Git commit")
    return snapshot.execution_commit


def _file_digest(path: Path) -> tuple[str, int]:
    if not path.is_file():
        raise RecoveryEngineError(f"recovery input is not an existing file: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _require_registered_input_digests(digests: Sequence[tuple[str, int]]) -> None:
    protocol = REGISTERED_RECOVERY_PROTOCOL
    expected = (
        ("dataset", protocol.dataset_sha256),
        ("cache", protocol.cache_sha256),
        ("sidecar", protocol.sidecar_sha256),
    )
    if len(digests) != _REGISTERED_INPUT_COUNT:
        raise RecoveryEngineError("registered recovery requires four input digests")
    for (name, expected_sha256), (observed_sha256, _size) in zip(
        expected, digests[:3], strict=True
    ):
        if observed_sha256 != expected_sha256:
            raise RecoveryEngineError(f"registered {name} SHA-256 does not match")


def _require_registered_configuration() -> None:
    try:
        validate_recovery_configuration(
            REGISTERED_RECOVERY_PROTOCOL,
            REGISTERED_EXECUTION_POLICY,
        )
    except ValueError as error:
        raise RecoveryEngineError("registered recovery configuration is incompatible") from error


def _validate_registered_inputs(
    cache: Path,
    sidecar: Path,
    runtime_preflight: Path,
    *,
    execution_commit: str,
) -> _RegisteredInputs:
    protocol = REGISTERED_RECOVERY_PROTOCOL
    specification = resolve_dataset(protocol.dataset)
    candidates = tuple(
        enumerate_candidates(
            specification.sites,
            specification.wt_at_sites,
            AA20,
            max_order=protocol.selection_max_order,
        )
    )
    config = _build_fourier_config(
        specification.sites,
        specification.wt_at_sites,
        AA20,
        max_order=protocol.estimation_max_order,
    )
    if len(config.modes) != protocol.feature_count:
        raise RecoveryEngineError("registered Fourier feature count does not match the universe")
    pairwise_count = sum(
        np.count_nonzero(mode) == protocol.estimation_max_order for mode in config.modes
    )
    if pairwise_count != protocol.coefficient_count:
        raise RecoveryEngineError("registered coefficient count does not match the universe")
    try:
        preflight_value = json.loads(runtime_preflight.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecoveryEngineError("runtime preflight is not valid JSON") from error
    if not isinstance(preflight_value, dict):
        raise RecoveryEngineError("runtime preflight must be a JSON object")
    validate_runtime_preflight(
        preflight_value,
        expected_commit=execution_commit,
        expected_candidate_count=len(candidates),
        expected_candidate_sha256=candidate_sha256(candidates),
        expected_budgets=protocol.budgets,
        expected_fit_count=registered_fit_count(protocol.budgets, protocol.seeds),
        expected_feature_count=len(config.modes),
    )
    loaded, metadata, expected_identity = validate_cache_against_universe(
        cache,
        candidates,
        candidate_alphabet=AA20,
        max_order=protocol.selection_max_order,
        model_id=_MODEL_ID,
        scorer_seed=_SCORER_SEED,
        n_perturbations=_N_PERTURBATIONS,
        wt_sequence=specification.wt_sequence,
        sidecar_path=sidecar,
    )
    scored = tuple(loaded[variant] for variant in candidates)
    validate_deterministic_selection_boundaries(
        scored,
        budgets=protocol.budgets,
        max_order=protocol.selection_max_order,
    )
    return _RegisteredInputs(
        specification=specification,
        candidates=candidates,
        config=config,
        scored=scored,
        expected_cache_identity=expected_identity,
        observed_cache_identity=CacheIdentity.from_metadata(metadata),
    )


def _open_initialised_store(run_dir: Path) -> ContentAddressedRunStore:
    if not run_dir.is_dir():
        raise RecoveryEngineError(f"recovery run directory does not exist: {run_dir}")
    store = ContentAddressedRunStore(run_dir)
    if not store.has_valid_header():
        raise RecoveryEngineError(f"recovery run is not initialised: {run_dir}")
    return store


def _open_journal(store: ContentAddressedRunStore) -> RecoveryStateCursor:
    """Open one verified in-memory cursor over the immutable recovery journal."""
    return RecoveryStateCursor.open(RunStoreSession.open(store))


def _runtime_and_bundle(
    store: ContentAddressedRunStore, state: RecoveryRunState
) -> tuple[RecoveryRuntimeRecord, RecoveryInputBundle]:
    if state.prepared is None:
        raise RecoveryEngineError("recovery run has no prepared root")
    try:
        runtime = RecoveryRuntimeRecord.from_payload(
            store.get_json(state.prepared.runtime_record_ref)
        )
        bundle = RecoveryInputBundle.from_payload(store.get_json(state.prepared.input_bundle_ref))
    except (RunStoreError, ValueError) as error:
        raise RecoveryEngineError("prepared runtime or input bundle is invalid") from error
    if runtime.scientific_identity.sha256 != state.prepared.scientific_identity_sha256:
        raise RecoveryEngineError("prepared runtime scientific identity does not match the root")
    expected_inputs = (
        runtime.scientific_identity.dataset_ref,
        runtime.scientific_identity.cache_ref,
        runtime.scientific_identity.sidecar_ref,
        runtime.scientific_identity.runtime_preflight_ref,
    )
    observed_inputs = tuple(archived.blob for archived in bundle.inputs())
    if observed_inputs != expected_inputs:
        raise RecoveryEngineError("prepared input bundle differs from its scientific identity")
    numeric_sha = hashlib.sha256(
        canonical_json_bytes(runtime.numeric_compatibility.payload())
    ).hexdigest()
    if numeric_sha != state.prepared.numerical_compatibility_sha256:
        raise RecoveryEngineError("prepared numeric compatibility does not match the root")
    return runtime, bundle


def _status(state: RecoveryRunState) -> RecoveryStatus:
    active = state.active_cell
    active_text = (
        None
        if active is None
        else f"{active.method}:{active.seed if active.seed is not None else 'none'}:{active.budget}"
    )
    return RecoveryStatus(
        prepared=state.prepared is not None,
        selection_complete=state.selection_plan is not None,
        doptimal_completed=state.doptimal_completed,
        completed_cells=len(state.completed_cells),
        total_cells=state.expected_cell_count,
        active_cell=active_text,
        completed_folds=state.completed_folds,
        report_available=state.report_ref is not None,
        latest_sequence=None if state.latest_manifest is None else state.latest_manifest.sequence,
    )


def _same_archived_input(path: Path, reference: BlobRef) -> bool:
    digest, size = _file_digest(path)
    return digest == reference.sha256 and size == reference.size


def prepare_recovery_run(
    run_dir: Path,
    dataset: Path,
    cache: Path,
    sidecar: Path,
    runtime_preflight: Path,
    repo: Path | None = None,
) -> RecoveryStatus:
    """Prepare a registered recovery run without loading measured labels."""
    _require_registered_configuration()
    started_utc = datetime.now(UTC).isoformat()
    preparation_argv = tuple(sys.argv) or ("python",)
    if not run_dir.is_dir():
        raise RecoveryEngineError(f"recovery run directory does not exist: {run_dir}")
    store = ContentAddressedRunStore(run_dir)
    if not store.has_valid_header() and any(run_dir.iterdir()):
        raise RecoveryEngineError(
            "legacy or unrelated state is present; use a new empty recovery run directory"
        )
    repository = _repo_path(repo)
    commit = _clean_commit(repository)
    initialised = store.has_valid_header()
    if initialised:
        audit = store.verify()
        if audit.has_errors:
            raise RecoveryEngineError(f"recovery run store verification failed: {audit.problems()}")
        state = _open_journal(store).snapshot()
        if state.prepared is not None:
            runtime, bundle = _runtime_and_bundle(store, state)
            supplied = (dataset, cache, sidecar, runtime_preflight)
            if runtime.scientific_identity.execution_commit != commit or not all(
                _same_archived_input(path, archived.blob)
                for path, archived in zip(supplied, bundle.inputs(), strict=True)
            ):
                raise RecoveryEngineError("prepared recovery inputs or execution commit differ")
            return _status(state)
        if state.manifest_count:
            raise RecoveryEngineError("initialised recovery run has no valid prepared root")
    input_paths = (dataset, cache, sidecar, runtime_preflight)
    before = tuple(_file_digest(path) for path in input_paths)
    _require_registered_input_digests(before)
    registered = _validate_registered_inputs(
        cache, sidecar, runtime_preflight, execution_commit=commit
    )
    numeric = capture_numeric_compatibility()
    if (
        _clean_commit(repository) != commit
        or tuple(_file_digest(path) for path in input_paths) != before
    ):
        raise RecoveryEngineError("recovery inputs or repository drifted during preparation")
    if not initialised:
        store.initialise()
    bundle = archive_recovery_inputs(
        store,
        dataset=dataset,
        cache=cache,
        sidecar=sidecar,
        runtime_preflight=runtime_preflight,
    )
    archived_identity = tuple((item.blob.sha256, item.blob.size) for item in bundle.inputs())
    if archived_identity != before or _clean_commit(repository) != commit:
        raise RecoveryEngineError("recovery inputs or repository drifted during archival")
    identity = scientific_identity_from_inputs(
        execution_commit=commit,
        protocol=REGISTERED_RECOVERY_PROTOCOL,
        candidate_sha256=candidate_sha256(registered.candidates),
        inputs=bundle,
    )
    runtime = RecoveryRuntimeRecord(
        scientific_identity=identity,
        numeric_compatibility=numeric,
        provenance=RecoveryProvenance(
            platform=sys.platform,
            machine=os.environ.get("PROCESSOR_ARCHITECTURE", "unknown"),
            argv=preparation_argv,
            started_utc=started_utc,
            completed_utc=datetime.now(UTC).isoformat(),
        ),
    )
    runtime_ref = store.put_json(runtime.payload())
    bundle_ref = store.put_json(bundle.payload())
    numeric_sha = hashlib.sha256(canonical_json_bytes(numeric.payload())).hexdigest()
    prepared = PreparedRecoveryRun(
        scientific_identity_sha256=identity.sha256,
        protocol_semantic_sha256=REGISTERED_RECOVERY_PROTOCOL.semantic_sha256,
        execution_policy_sha256=REGISTERED_EXECUTION_POLICY.policy_sha256,
        numerical_compatibility_sha256=numeric_sha,
        candidate_sha256=identity.candidate_sha256,
        runtime_record_ref=runtime_ref,
        input_bundle_ref=bundle_ref,
    )
    journal = _open_journal(store)
    publish_prepared_run_at(journal, prepared)
    state = journal.snapshot()
    if state.prepared != prepared:
        raise RecoveryEngineError("prepared recovery root did not round-trip")
    return _status(state)


def _selection_payload(plan: SelectionPlan) -> dict[str, object]:
    return {
        "schema_version": _PLAN_SCHEMA,
        "budgets": list(plan.budgets),
        "sequences": [
            {
                "method": sequence.method,
                "seed": sequence.seed,
                "selected_sha256": sequence.selected_sha256,
                "tie_break_version": sequence.tie_break_version,
                "selected_ids": [canonical_id(variant) for variant in sequence.selected],
            }
            for sequence in plan.sequences
        ],
    }


def _load_selection_plan(
    store: ContentAddressedRunStore,
    published: PublishedSelectionPlan,
    candidates: Sequence[Variant],
) -> SelectionPlan:
    value = store.get_json(published.plan_ref)
    if not isinstance(value, dict) or set(value) != {"schema_version", "budgets", "sequences"}:
        raise RecoveryEngineError("selection-plan payload fields do not match")
    if value["schema_version"] != _PLAN_SCHEMA:
        raise RecoveryEngineError("selection-plan payload schema does not match")
    budgets_value = value["budgets"]
    rows = value["sequences"]
    if budgets_value != list(REGISTERED_RECOVERY_PROTOCOL.budgets) or not isinstance(rows, list):
        raise RecoveryEngineError("selection-plan registered grid does not match")
    candidate_by_id = {canonical_id(variant): variant for variant in candidates}
    sequences: list[SelectionSequence] = []
    for row_value in rows:
        if not isinstance(row_value, dict) or set(row_value) != {
            "method",
            "seed",
            "selected_sha256",
            "tie_break_version",
            "selected_ids",
        }:
            raise RecoveryEngineError("selection-plan sequence fields do not match")
        selected_ids = row_value["selected_ids"]
        method = row_value["method"]
        seed = row_value["seed"]
        selected_sha256 = row_value["selected_sha256"]
        tie_break_version = row_value["tie_break_version"]
        if (
            not isinstance(method, str)
            or (seed is not None and type(seed) is not int)
            or not isinstance(selected_sha256, str)
            or len(selected_sha256) != _SHA256_LENGTH
            or not isinstance(tie_break_version, str)
            or not tie_break_version
        ):
            raise RecoveryEngineError("selection-plan sequence identity is invalid")
        if not isinstance(selected_ids, list) or not all(
            isinstance(item, str) for item in selected_ids
        ):
            raise RecoveryEngineError("selection-plan selected identities are invalid")
        try:
            selected = tuple(candidate_by_id[item] for item in selected_ids)
        except KeyError as error:
            raise RecoveryEngineError("selection plan references an unknown candidate") from error
        sequence = SelectionSequence(
            method=method,
            seed=seed,
            selected=selected,
            selected_sha256=selected_sha256,
            tie_break_version=tie_break_version,
        )
        if len(selected) != REGISTERED_RECOVERY_PROTOCOL.budgets[-1]:
            raise RecoveryEngineError("selection-plan sequence length does not match")
        if len(set(selected)) != len(selected) or _sequence_sha256(selected) != (
            sequence.selected_sha256
        ):
            raise RecoveryEngineError("selection-plan sequence identity does not match")
        sequences.append(sequence)
    if tuple((item.method, item.seed) for item in sequences) != (
        REGISTERED_RECOVERY_PROTOCOL.sequence_keys
    ):
        raise RecoveryEngineError("selection-plan sequence order does not match")
    return SelectionPlan(
        budgets=REGISTERED_RECOVERY_PROTOCOL.budgets,
        sequences=tuple(sequences),
    )


def _notify(callback: Callable[[RecoveryProgress], None] | None, event: RecoveryProgress) -> None:
    if callback is not None:
        callback(event)


def _restore_doptimal_cursor(
    journal: RecoveryStateCursor,
    runtime: RecoveryRuntimeRecord,
    registered: _RegisteredInputs,
) -> DOptimalCheckpointCursor:
    protocol = REGISTERED_RECOVERY_PROTOCOL
    initial = initialise_reduced_doptimal(
        registered.config,
        registered.candidates,
        target_budget=protocol.budgets[-1],
    )
    numerical = runtime.numeric_compatibility.payload()
    identity = DOptimalCheckpointIdentity(
        scientific_identity_sha256=runtime.scientific_identity.sha256,
        execution_policy=REGISTERED_EXECUTION_POLICY,
        numerical_compatibility=numerical,
        numerical_compatibility_sha256=hashlib.sha256(canonical_json_bytes(numerical)).hexdigest(),
        candidate_universe_sha256=candidate_sha256(registered.candidates),
        candidate_sequence_sha256=_sequence_sha256(initial.candidates),
        candidate_count=len(registered.candidates),
        target_budget=protocol.budgets[-1],
        geometry_sha256=doptimal_geometry_sha256(initial),
    )
    return DOptimalCheckpointCursor.restore(journal, initial, identity)


def _require_selection_matches_doptimal(
    plan: SelectionPlan,
    doptimal: ReducedDOptimalState,
) -> None:
    expected = tuple(doptimal.candidates[index] for index in doptimal.selected_indices)
    observed = plan.plate(
        "doptimal_reduced_pairwise",
        None,
        REGISTERED_RECOVERY_PROTOCOL.budgets[-1],
    )
    if observed != expected:
        raise RecoveryEngineError(
            "durable D-optimal state does not match the published selection plan"
        )


def _complete_selection(
    journal: RecoveryStateCursor,
    state: RecoveryRunState,
    runtime: RecoveryRuntimeRecord,
    registered: _RegisteredInputs,
    on_progress: Callable[[RecoveryProgress], None] | None,
    require_stable_workspace: Callable[[], None],
) -> tuple[RecoveryRunState, SelectionPlan]:
    store = journal.store
    protocol = REGISTERED_RECOVERY_PROTOCOL
    policy = REGISTERED_EXECUTION_POLICY
    doptimal_cursor = _restore_doptimal_cursor(journal, runtime, registered)
    doptimal = doptimal_cursor.state
    if state.selection_plan is not None:
        plan = _load_selection_plan(store, state.selection_plan, registered.candidates)
        _require_selection_matches_doptimal(plan, doptimal)
        return state, plan
    while len(doptimal.selected_indices) < doptimal.target_budget:
        stop = min(
            len(doptimal.selected_indices) + policy.doptimal_block_size,
            doptimal.target_budget,
        )
        advance_reduced_doptimal(doptimal, stop)
        require_stable_workspace()
        doptimal_cursor.publish(doptimal)
        _notify(
            on_progress,
            RecoveryProgress("doptimal", stop, doptimal.target_budget),
        )
    generated = build_selection_plan(
        registered.scored,
        budgets=protocol.budgets,
        seeds=protocol.seeds,
        max_order=protocol.selection_max_order,
        doptimal_state=doptimal,
    )
    plan_ref = store.put_json(_selection_payload(generated))
    published = PublishedSelectionPlan(
        scientific_identity_sha256=runtime.scientific_identity.sha256,
        selection_plan_sha256=plan_ref.sha256,
        plan_ref=plan_ref,
    )
    require_stable_workspace()
    publish_selection_plan_at(journal, published)
    state = journal.snapshot()
    if state.selection_plan != published:
        raise RecoveryEngineError("selection-plan publication did not round-trip")
    _notify(on_progress, RecoveryProgress("selection", 1, 1))
    return state, _load_selection_plan(store, published, registered.candidates)


def _cell_metrics(cell: RecoveryCell) -> dict[str, object]:
    return cast("dict[str, object]", asdict(cell))


def _scientific_failure(error: Exception) -> str | None:
    if isinstance(error, FloatingPointError):
        return f"{type(error).__name__}: {error}"
    if isinstance(error, RuntimeError) and str(error) == "FISTA did not converge":
        return f"{type(error).__name__}: {error}"
    if isinstance(error, ValueError) and str(error) == "response is effectively constant":
        return f"{type(error).__name__}: {error}"
    return None


def _lasso_identity(
    runtime: RecoveryRuntimeRecord,
    state: RecoveryRunState,
    registered: _RegisteredInputs,
    plan: SelectionPlan,
    landscape: Mapping[Variant, float],
    key: RecoveryCellKey,
) -> tuple[LassoCheckpointIdentity, tuple[Variant, ...]]:
    protocol = REGISTERED_RECOVERY_PROTOCOL
    if state.selection_plan is None:
        raise RecoveryEngineError("LASSO identity requires a durable selection plan")
    selected = plan.plate(key.method, key.seed, key.budget)
    revealed = reveal_measured_fitness(landscape, selected)
    if len(revealed) != len(selected):
        raise RecoveryEngineError("selected plate is missing measured labels")
    response = np.asarray(
        [training_target(revealed[variant]) for variant in selected],
        dtype=np.float64,
        order="C",
    )
    numerical = runtime.numeric_compatibility.payload()
    identity = LassoCheckpointIdentity(
        scientific_identity_sha256=runtime.scientific_identity.sha256,
        execution_policy=REGISTERED_EXECUTION_POLICY,
        numerical_compatibility=numerical,
        numerical_compatibility_sha256=hashlib.sha256(canonical_json_bytes(numerical)).hexdigest(),
        selection_plan_sha256=state.selection_plan.selection_plan_sha256,
        method=key.method,
        seed=key.seed,
        budget=key.budget,
        selected_sha256=_sequence_sha256(selected),
        fold_sha256=_fold_sha256(selected, protocol.n_folds),
        problem_sha256=pairwise_lasso_problem_sha256(
            registered.config,
            selected,
            response,
            n_folds=protocol.n_folds,
            lambda_ratios=_DEFAULT_LAMBDA_RATIOS,
        ),
        n_folds=protocol.n_folds,
        lambda_ratios=_DEFAULT_LAMBDA_RATIOS,
    )
    return identity, selected


def _run_cells(
    journal: RecoveryStateCursor,
    state: RecoveryRunState,
    runtime: RecoveryRuntimeRecord,
    registered: _RegisteredInputs,
    plan: SelectionPlan,
    landscape: Mapping[Variant, float],
    truth: PairwiseTruth,
    on_progress: Callable[[RecoveryProgress], None] | None,
    require_stable_workspace: Callable[[], None],
) -> RecoveryRunState:
    protocol = REGISTERED_RECOVERY_PROTOCOL
    keys = registered_cell_keys()
    completed_count = len(state.completed_cells)
    for key in keys[completed_count:]:
        identity, selected = _lasso_identity(runtime, state, registered, plan, landscape, key)
        checkpoint = LassoCheckpointCursor.restore(journal, identity)
        resume = checkpoint.state

        def on_fold_completed(
            cv_state: PairwiseLassoCVState,
            current_key: RecoveryCellKey = key,
            current_checkpoint: LassoCheckpointCursor = checkpoint,
        ) -> None:
            require_stable_workspace()
            current_checkpoint.publish(cv_state)
            completed = cv_state.completed_folds
            _notify(
                on_progress,
                RecoveryProgress(
                    "lasso_fold",
                    completed,
                    protocol.n_folds,
                    current_key.method,
                    current_key.seed,
                    current_key.budget,
                ),
            )

        try:
            cell = evaluate_plate(
                registered.config,
                selected,
                landscape,
                truth.coefficients,
                method=key.method,
                seed=key.seed,
                budget=key.budget,
                n_folds=protocol.n_folds,
                resume_cv=resume,
                on_fold_completed=on_fold_completed,
            )
        except (FloatingPointError, RuntimeError, ValueError) as error:
            scientific_error = _scientific_failure(error)
            if scientific_error is None:
                raise
            require_stable_workspace()
            publish_recovery_cell_at(
                journal, key, valid=False, metrics=None, error=scientific_error
            )
        else:
            require_stable_workspace()
            publish_recovery_cell_at(
                journal, key, valid=True, metrics=_cell_metrics(cell), error=None
            )
        completed_count += 1
        _notify(
            on_progress,
            RecoveryProgress(
                "cell",
                completed_count,
                protocol.cell_count,
                key.method,
                key.seed,
                key.budget,
            ),
        )
    replayed = journal.snapshot()
    if len(replayed.completed_cells) != completed_count:
        raise RecoveryEngineError("recovery cell journal did not round-trip")
    return replayed


def _stored_cells(
    store: ContentAddressedRunStore, state: RecoveryRunState
) -> tuple[RecoveryCell, ...]:
    cells: list[RecoveryCell] = []
    if len(state.completed_cells) != len(state.cell_result_refs):
        raise RecoveryEngineError("recovery-cell keys and result references disagree")
    for expected_key, reference in zip(state.completed_cells, state.cell_result_refs, strict=True):
        value = store.get_json(reference)
        if not isinstance(value, dict):
            raise RecoveryEngineError("stored recovery-cell result is not an object")
        key_value = value.get("cell")
        key = RecoveryCellKey.from_payload(key_value)
        if key != expected_key:
            raise RecoveryEngineError("stored recovery-cell result key does not match the journal")
        if value.get("valid") is True:
            metrics = value.get("metrics")
            if not isinstance(metrics, dict):
                raise RecoveryEngineError("valid stored recovery cell has no metrics")
            try:
                cells.append(
                    RecoveryCell(
                        method=cast("str", metrics["method"]),
                        budget=cast("int", metrics["budget"]),
                        seed=cast("int | None", metrics["seed"]),
                        spearman=cast("float | None", metrics["spearman"]),
                        relative_sse_gain=cast("float | None", metrics["relative_sse_gain"]),
                        support_size=cast("int", metrics["support_size"]),
                        coefficient_count=cast("int", metrics["coefficient_count"]),
                        selected_sha256=cast("str", metrics["selected_sha256"]),
                        fold_sha256=cast("str", metrics["fold_sha256"]),
                        lambda_ratio=cast("float | None", metrics["lambda_ratio"]),
                        lambda_value=cast("float | None", metrics["lambda_value"]),
                        converged=cast("bool", metrics["converged"]),
                        error=cast("str | None", metrics["error"]),
                    )
                )
            except KeyError as missing_error:
                raise RecoveryEngineError(
                    "valid stored recovery cell has incomplete metrics"
                ) from missing_error
        else:
            cell_error = value.get("error")
            if not isinstance(cell_error, str):
                raise RecoveryEngineError("invalid stored recovery cell has no error")
            cells.append(
                RecoveryCell(
                    method=key.method,
                    budget=key.budget,
                    seed=key.seed,
                    spearman=None,
                    relative_sse_gain=None,
                    support_size=0,
                    coefficient_count=REGISTERED_RECOVERY_PROTOCOL.coefficient_count,
                    converged=False,
                    error=cell_error,
                )
            )
    return tuple(cells)


def _report_payload(
    store: ContentAddressedRunStore,
    state: RecoveryRunState,
    runtime: RecoveryRuntimeRecord,
    registered: _RegisteredInputs,
    plan: SelectionPlan,
    truth: PairwiseTruth,
    execution_provenance: Mapping[str, object],
) -> dict[str, object]:
    protocol = REGISTERED_RECOVERY_PROTOCOL
    cells = _stored_cells(store, state)
    decision = decide_recovery(
        cells,
        stochastic_seeds=protocol.seeds,
        expected_methods=protocol.methods,
        expected_budgets=protocol.budgets,
    )
    modes_payload = json.dumps(truth.modes, separators=(",", ":")).encode("ascii")
    return {
        "schema_version": _REPORT_SCHEMA,
        "public_claim_eligible": False,
        "architecture_decision_eligible": decision.status != "invalid_coverage",
        "dataset": protocol.dataset,
        "label_transform": protocol.label_transform,
        "candidate_count": len(registered.candidates),
        "candidate_composition": {"1": 76, "2": 2166, "3": 27436},
        "budgets": list(protocol.budgets),
        "stochastic_seeds": list(protocol.seeds),
        "pairwise_coefficient_count": protocol.coefficient_count,
        "pairwise_truth_sha256": hashlib.sha256(truth.coefficients.tobytes(order="C")).hexdigest(),
        "pairwise_modes_sha256": hashlib.sha256(modes_payload).hexdigest(),
        "imputation_note": _IMPUTATION_NOTE,
        "cache_identity_expected": registered.expected_cache_identity.model_dump(mode="json"),
        "cache_identity_observed": registered.observed_cache_identity.model_dump(mode="json"),
        "selection_sequences": [
            {
                "method": sequence.method,
                "seed": sequence.seed,
                "selected_sha256": sequence.selected_sha256,
                "tie_break_version": sequence.tie_break_version,
                "selected_ids": [canonical_id(variant) for variant in sequence.selected],
            }
            for sequence in plan.sequences
        ],
        "cells": [asdict(cell) for cell in cells],
        "aggregates": [asdict(item) for item in decision.aggregates],
        "decision": {
            "status": decision.status,
            "passing_cells": [list(item) for item in decision.passing_cells],
            "reasons": list(decision.reasons),
        },
        "provenance": {
            "scientific_identity": runtime.scientific_identity.payload(),
            "numeric_compatibility": runtime.numeric_compatibility.payload(),
            "execution_policy": REGISTERED_EXECUTION_POLICY.identity_payload(),
            "execution": dict(execution_provenance),
        },
    }


def _execution_history_provenance(
    state: RecoveryRunState,
    runtime: RecoveryRuntimeRecord,
    bundle: RecoveryInputBundle,
) -> dict[str, object]:
    commit = runtime.scientific_identity.execution_commit
    prepared = state.prepared
    if prepared is None:
        raise RecoveryEngineError("execution history requires a prepared run")
    attempts: list[dict[str, object]] = []
    for record in state.execution_attempts:
        if (
            record.start.scientific_identity_sha256 != runtime.scientific_identity.sha256
            or record.start.runtime_record_ref != prepared.runtime_record_ref
            or record.start.input_bundle_ref != prepared.input_bundle_ref
            or record.start.commit_sha != commit
        ):
            raise RecoveryEngineError("durable execution-attempt identity drifted")
        if record.completion is not None and record.completion.commit_sha != commit:
            raise RecoveryEngineError("durable execution-attempt completion commit drifted")
        attempts.append(
            {
                "start": record.start.payload(),
                "start_ref": record.start_ref.payload(),
                "completion": (None if record.completion is None else record.completion.payload()),
                "completion_ref": (
                    None if record.completion_ref is None else record.completion_ref.payload()
                ),
                "abandoned": record.abandoned,
            }
        )
    return {
        "input_sha256": {archived.name: archived.blob.sha256 for archived in bundle.inputs()},
        "attempts": attempts,
    }


def _require_execution_history(
    value: object,
    state: RecoveryRunState,
    runtime: RecoveryRuntimeRecord,
    bundle: RecoveryInputBundle,
) -> dict[str, object]:
    expected = _execution_history_provenance(state, runtime, bundle)
    try:
        observed = canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise RecoveryEngineError("recovery execution-attempt history is invalid") from error
    if observed != canonical_json_bytes(expected):
        raise RecoveryEngineError("recovery execution-attempt history does not match")
    return expected


def run_recovery(  # noqa: PLR0912, PLR0915 - one linear resume sequence that must stay auditable
    run_dir: Path,
    repo: Path | None = None,
    on_progress: Callable[[RecoveryProgress], None] | None = None,
) -> RecoveryStatus:
    """Resume the registered recovery computation from verified durable state."""
    _require_registered_configuration()
    started_utc = datetime.now(UTC).isoformat()
    process_argv = tuple(sys.argv) or ("python",)
    store = _open_initialised_store(run_dir)
    audit = store.verify()
    if audit.has_errors:
        raise RecoveryEngineError(f"recovery run store verification failed: {audit.problems()}")
    journal = _open_journal(store)
    state = journal.snapshot()
    runtime, bundle = _runtime_and_bundle(store, state)
    repository = _repo_path(repo)
    commit = _clean_commit(repository)

    def require_stable_workspace() -> None:
        if _clean_commit(repository) != commit:
            raise RecoveryEngineError("repository drifted during recovery execution")

    if commit != runtime.scientific_identity.execution_commit:
        raise RecoveryEngineError("execution commit differs from the prepared run")
    require_numeric_compatibility(runtime.numeric_compatibility, capture_numeric_compatibility())
    if state.report_ref is not None:
        _verify_selection_journal(store, journal, state)
        return _status(state)
    prepared = state.prepared
    if prepared is None:
        raise RecoveryEngineError("recovery run has no prepared root")
    attempt: ExecutionAttemptStart | None = None
    if state.finalized_execution_attempt is None:
        attempt = ExecutionAttemptStart(
            attempt_id=uuid4().hex,
            scientific_identity_sha256=runtime.scientific_identity.sha256,
            runtime_record_ref=prepared.runtime_record_ref,
            input_bundle_ref=prepared.input_bundle_ref,
            commit_sha=commit,
            workspace_clean=True,
            scientific_diff_sha256=None,
            argv=process_argv,
            started_utc=started_utc,
        )
        publish_execution_attempt_started_at(journal, attempt)
        state = journal.snapshot()
    with tempfile.TemporaryDirectory(prefix="epibudget-recovery-") as temporary:
        materialized = Path(temporary)
        cache_path = materialize_archived_recovery_input(store, bundle.cache, materialized)
        sidecar_path = materialize_archived_recovery_input(store, bundle.sidecar, materialized)
        preflight_path = materialize_archived_recovery_input(
            store, bundle.runtime_preflight, materialized
        )
        registered = _validate_registered_inputs(
            cache_path,
            sidecar_path,
            preflight_path,
            execution_commit=commit,
        )
        if candidate_sha256(registered.candidates) != prepared.candidate_sha256:
            raise RecoveryEngineError("registered candidate universe differs from preparation")
        state, plan = _complete_selection(
            journal,
            state,
            runtime,
            registered,
            on_progress,
            require_stable_workspace,
        )
        dataset_path = materialize_archived_recovery_input(store, bundle.dataset, materialized)
        landscape = registered.specification.loader(dataset_path)
        transformed = {variant: training_target(value) for variant, value in landscape.items()}
        truth = pairwise_truth(transformed, registered.specification.sites)
        if len(truth.coefficients) != REGISTERED_RECOVERY_PROTOCOL.coefficient_count:
            raise RecoveryEngineError("pairwise truth coefficient count does not match")
        state = _run_cells(
            journal,
            state,
            runtime,
            registered,
            plan,
            landscape,
            truth,
            on_progress,
            require_stable_workspace,
        )
        if state.report_ref is None:
            require_stable_workspace()
            completed_utc = datetime.now(UTC).isoformat()
            if state.finalized_execution_attempt is None:
                open_attempt = state.open_execution_attempt
                if attempt is None or open_attempt is None or open_attempt.start != attempt:
                    raise RecoveryEngineError("recovery execution attempt is not open")
                publish_execution_attempt_completed_at(
                    journal,
                    ExecutionAttemptCompletion(
                        attempt_id=attempt.attempt_id,
                        start_ref=open_attempt.start_ref,
                        commit_sha=commit,
                        workspace_clean=True,
                        scientific_diff_sha256=None,
                        completed_utc=completed_utc,
                    ),
                )
                state = journal.snapshot()
            execution = _execution_history_provenance(state, runtime, bundle)
            report = _report_payload(
                store,
                state,
                runtime,
                registered,
                plan,
                truth,
                execution,
            )
            require_stable_workspace()
            publish_recovery_report_at(journal, report)
            state = journal.snapshot()
            expected_report = dict(report)
            expected_report["cell_results_sha256"] = state.cell_results_sha256
            if state.report_ref is None or store.get_json(state.report_ref) != expected_report:
                raise RecoveryEngineError("recovery report did not round-trip")
            require_stable_workspace()
            _notify(on_progress, RecoveryProgress("report", 1, 1))
    require_stable_workspace()
    final_audit = store.verify()
    if final_audit.has_errors:
        raise RecoveryEngineError(
            f"final recovery run store verification failed: {final_audit.problems()}"
        )
    reopened_journal = _open_journal(store)
    reopened = reopened_journal.snapshot()
    if reopened != state:
        raise RecoveryEngineError("incremental recovery state differs from full replay")
    _verify_selection_journal(store, reopened_journal, reopened)
    return _status(reopened)


def recovery_status(run_dir: Path) -> RecoveryStatus:
    """Return durable progress without changing the run store."""
    store = _open_initialised_store(run_dir)
    return _status(replay_recovery_state(store))


def _verify_selection_journal(
    store: ContentAddressedRunStore,
    journal: RecoveryStateCursor,
    state: RecoveryRunState,
) -> None:
    runtime, bundle = _runtime_and_bundle(store, state)
    with tempfile.TemporaryDirectory(prefix="epibudget-recovery-verify-") as temporary:
        materialized = Path(temporary)
        cache_path = materialize_archived_recovery_input(store, bundle.cache, materialized)
        sidecar_path = materialize_archived_recovery_input(store, bundle.sidecar, materialized)
        preflight_path = materialize_archived_recovery_input(
            store, bundle.runtime_preflight, materialized
        )
        registered = _validate_registered_inputs(
            cache_path,
            sidecar_path,
            preflight_path,
            execution_commit=runtime.scientific_identity.execution_commit,
        )
        doptimal = _restore_doptimal_cursor(journal, runtime, registered).state
        plan: SelectionPlan | None = None
        if state.selection_plan is not None:
            plan = _load_selection_plan(store, state.selection_plan, registered.candidates)
            _require_selection_matches_doptimal(plan, doptimal)
        landscape: Mapping[Variant, float] | None = None
        if state.active_cell is not None or state.report_ref is not None:
            if plan is None:
                raise RecoveryEngineError("recovery estimation state has no selection plan")
            dataset_path = materialize_archived_recovery_input(store, bundle.dataset, materialized)
            landscape = registered.specification.loader(dataset_path)
        if state.active_cell is not None:
            if plan is None or landscape is None:
                raise RecoveryEngineError("active LASSO state cannot be reconstructed")
            identity, _selected = _lasso_identity(
                runtime,
                state,
                registered,
                plan,
                landscape,
                state.active_cell,
            )
            LassoCheckpointCursor.restore(journal, identity)
        if state.report_ref is not None:
            if plan is None or landscape is None:
                raise RecoveryEngineError("published recovery report cannot be reconstructed")
            transformed = {variant: training_target(value) for variant, value in landscape.items()}
            truth = pairwise_truth(transformed, registered.specification.sites)
            if len(truth.coefficients) != REGISTERED_RECOVERY_PROTOCOL.coefficient_count:
                raise RecoveryEngineError(
                    "pairwise truth coefficient count does not match during verification"
                )
            observed = store.get_json(state.report_ref)
            if not isinstance(observed, dict):
                raise RecoveryEngineError("published recovery report is not an object")
            provenance = observed.get("provenance")
            if not isinstance(provenance, dict):
                raise RecoveryEngineError("published recovery report provenance is invalid")
            execution = _require_execution_history(
                provenance.get("execution"), state, runtime, bundle
            )
            expected = _report_payload(store, state, runtime, registered, plan, truth, execution)
            expected["cell_results_sha256"] = state.cell_results_sha256
            if canonical_json_bytes(observed) != canonical_json_bytes(expected):
                raise RecoveryEngineError(
                    "published recovery report does not match recomputed scientific results"
                )


def verify_recovery_run(run_dir: Path) -> RecoveryStatus:
    """Verify the complete store and domain state without mutating either."""
    store = _open_initialised_store(run_dir)
    report = store.verify()
    if report.has_errors:
        raise RecoveryEngineError(f"recovery run store verification failed: {report.problems()}")
    journal = _open_journal(store)
    state = journal.snapshot()
    _require_verification_identity(store, state)
    _verify_selection_journal(store, journal, state)
    return _status(state)


def _require_verification_identity(
    store: ContentAddressedRunStore, state: RecoveryRunState
) -> None:
    runtime, _bundle = _runtime_and_bundle(store, state)
    commit = _clean_commit(_repo_path(None))
    if commit != runtime.scientific_identity.execution_commit:
        raise RecoveryEngineError("verification checkout differs from the prepared run")
    require_numeric_compatibility(runtime.numeric_compatibility, capture_numeric_compatibility())


def export_recovery_report(run_dir: Path, out: Path) -> Path:
    """Publish the exact durable report bytes outside the run directory, exclusively."""
    store = _open_initialised_store(run_dir)
    audit = store.verify()
    if audit.has_errors:
        raise RecoveryEngineError(f"recovery run store verification failed: {audit.problems()}")
    journal = _open_journal(store)
    state = journal.snapshot()
    if state.report_ref is None:
        raise RecoveryEngineError("recovery run has no published report")
    _require_verification_identity(store, state)
    _verify_selection_journal(store, journal, state)
    run_root = run_dir.resolve()
    target = out.resolve()
    if target == run_root or target.is_relative_to(run_root):
        raise RecoveryEngineError("recovery report export must be outside the run directory")
    if not target.parent.is_dir():
        raise RecoveryEngineError(
            f"recovery report export directory does not exist: {target.parent}"
        )
    content = store.get_bytes(state.report_ref)
    try:
        with target.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if target.read_bytes() != content:
            raise RecoveryEngineError("exported recovery report did not verify")
    except FileExistsError as error:
        raise RecoveryEngineError(f"recovery report export already exists: {target}") from error
    except (OSError, RecoveryEngineError):
        if target.exists():
            target.unlink()
        raise
    return target
