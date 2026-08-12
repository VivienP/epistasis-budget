"""Strict domain replay for durable Fourier-recovery runs."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from itertools import pairwise
from typing import Final, cast

import numpy as np

from epibudget.recovery_protocol import (
    REGISTERED_EXECUTION_POLICY,
    REGISTERED_RECOVERY_PROTOCOL,
    RecoveryExecutionPolicy,
    RecoveryScientificProtocol,
)
from epibudget.run_store import (
    ArrayRef,
    BlobRef,
    ContentAddressedRunStore,
    Manifest,
    ManifestDraft,
    RunStoreError,
    RunStoreSession,
    canonical_json_bytes,
)

_PREPARED = "recovery_prepared"
_DOPTIMAL = "reduced_doptimal"
_SELECTION = "selection_plan"
_LASSO = "pairwise_lasso_cv"
_CELL = "recovery_cell"
_REPORT = "recovery_report"
_ATTEMPT_STARTED = "execution_attempt_started"
_ATTEMPT_COMPLETED = "execution_attempt_completed"
_KNOWN_KINDS: Final = frozenset(
    {
        _PREPARED,
        _DOPTIMAL,
        _SELECTION,
        _LASSO,
        _CELL,
        _REPORT,
        _ATTEMPT_STARTED,
        _ATTEMPT_COMPLETED,
    }
)
_PREPARED_SCHEMA = "epibudget-recovery-prepared-v1"
_SELECTION_SCHEMA = "epibudget-recovery-selection-plan-v1"
_CELL_SCHEMA = "epibudget-recovery-cell-v1"
_CELL_RESULT_SCHEMA = "epibudget-recovery-cell-result-v1"
_REPORT_SCHEMA = "epibudget-recovery-report-v1"
_DOPTIMAL_SCHEMA = "epibudget-reduced-doptimal-delta-v2"
_LASSO_SCHEMA = "epibudget-pairwise-lasso-fold-v1"
_ATTEMPT_STARTED_SCHEMA = "epibudget-execution-attempt-started-v1"
_ATTEMPT_COMPLETED_SCHEMA = "epibudget-execution-attempt-completed-v1"
_SHA256_WIDTH = 64
_MATRIX_DIMENSIONS = 2
_VALID_METRIC_FIELDS: Final = frozenset(
    {
        "method",
        "budget",
        "seed",
        "spearman",
        "relative_sse_gain",
        "support_size",
        "coefficient_count",
        "selected_sha256",
        "fold_sha256",
        "lambda_ratio",
        "lambda_value",
        "converged",
        "error",
    }
)


class RecoveryStateError(RunStoreError):
    """The durable manifest chain violates the registered recovery state machine."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_WIDTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, field: str) -> str:
    if not _is_sha256(value):
        raise RecoveryStateError(f"{field} must be a lowercase SHA-256 digest")
    return cast("str", value)


def _validate_sha256(value: str, field: str) -> None:
    if not _is_sha256(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _require_object(value: object, fields: frozenset[str], description: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise RecoveryStateError(f"{description} must be a JSON object")
    if set(value) != fields:
        raise RecoveryStateError(f"{description} has unexpected or missing fields")
    return dict(value)


def _require_integer(value: object, field: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise RecoveryStateError(f"{field} must be an integer")
    result = value
    if minimum is not None and result < minimum:
        raise RecoveryStateError(f"{field} must be at least {minimum}")
    return result


def _require_json_value(value: object, field: str) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise RecoveryStateError(f"{field} contains a non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _require_json_value(item, field)
        return
    if isinstance(value, dict) and all(type(key) is str for key in value):
        for item in value.values():
            _require_json_value(item, field)
        return
    raise RecoveryStateError(f"{field} is not a canonical JSON value")


def _validate_public_json(value: object, field: str) -> None:
    try:
        _require_json_value(value, field)
    except RecoveryStateError as error:
        raise ValueError(str(error)) from error


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _parse_utc_timestamp(value: str, field: str) -> datetime:
    if type(value) is not str:
        raise ValueError(f"{field} must be an ISO-8601 UTC timestamp")
    if value.endswith("Z"):
        normalized = f"{value[:-1]}+00:00"
    elif value.endswith("+00:00"):
        normalized = value
    else:
        raise ValueError(f"{field} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{field} must be a valid ISO-8601 UTC timestamp") from error
    if parsed.tzinfo != UTC:
        raise ValueError(f"{field} must use UTC")
    return parsed


def _require_commit_sha(value: str) -> None:
    if (
        type(value) is not str
        or len(value) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("commit_sha must be a lowercase 40- or 64-character Git object ID")


@dataclass(frozen=True)
class RecoveryCellKey:
    """One method-seed-budget cell in the registered breadth-first schedule."""

    method: str
    seed: int | None
    budget: int

    def __post_init__(self) -> None:
        if type(self.method) is not str or not self.method:
            raise ValueError("recovery cell method must be a non-empty string")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("recovery cell seed must be an integer or None")
        if type(self.budget) is not int or self.budget < 1:
            raise ValueError("recovery cell budget must be a positive integer")

    def payload(self) -> dict[str, object]:
        """Return the canonical cell address."""
        return {"method": self.method, "seed": self.seed, "budget": self.budget}

    @classmethod
    def from_payload(cls, value: object) -> RecoveryCellKey:
        """Decode a cell address without accepting booleans as integers."""
        row = _require_object(value, frozenset({"method", "seed", "budget"}), "recovery cell key")
        method = row["method"]
        seed = row["seed"]
        budget = row["budget"]
        if type(method) is not str or not method:
            raise RecoveryStateError("recovery cell method must be a non-empty string")
        if seed is not None and type(seed) is not int:
            raise RecoveryStateError("recovery cell seed must be an integer or null")
        return cls(
            method=method,
            seed=seed,
            budget=_require_integer(budget, "recovery cell budget", minimum=1),
        )


@dataclass(frozen=True)
class PreparedRecoveryRun:
    """Immutable identities and archived inputs fixed before scientific execution."""

    scientific_identity_sha256: str
    protocol_semantic_sha256: str
    execution_policy_sha256: str
    numerical_compatibility_sha256: str
    candidate_sha256: str
    runtime_record_ref: BlobRef
    input_bundle_ref: BlobRef

    def __post_init__(self) -> None:
        for name, value in (
            ("scientific_identity_sha256", self.scientific_identity_sha256),
            ("protocol_semantic_sha256", self.protocol_semantic_sha256),
            ("execution_policy_sha256", self.execution_policy_sha256),
            ("numerical_compatibility_sha256", self.numerical_compatibility_sha256),
            ("candidate_sha256", self.candidate_sha256),
        ):
            _validate_sha256(value, name)
        for name, reference in (
            ("runtime_record_ref", self.runtime_record_ref),
            ("input_bundle_ref", self.input_bundle_ref),
        ):
            try:
                BlobRef.from_payload(reference.payload())
            except RunStoreError as error:
                raise ValueError(f"{name} must be a valid blob reference") from error

    def payload(self) -> dict[str, object]:
        """Return the exact root-run payload."""
        return {
            "scientific_identity_sha256": self.scientific_identity_sha256,
            "protocol_semantic_sha256": self.protocol_semantic_sha256,
            "execution_policy_sha256": self.execution_policy_sha256,
            "numerical_compatibility_sha256": self.numerical_compatibility_sha256,
            "candidate_sha256": self.candidate_sha256,
            "runtime_record_ref": self.runtime_record_ref.payload(),
            "input_bundle_ref": self.input_bundle_ref.payload(),
        }


@dataclass(frozen=True)
class ExecutionAttemptStart:
    """Clean, commit-fixed invocation provenance captured before execution."""

    attempt_id: str
    scientific_identity_sha256: str
    runtime_record_ref: BlobRef
    input_bundle_ref: BlobRef
    commit_sha: str
    workspace_clean: bool
    scientific_diff_sha256: None
    argv: tuple[str, ...]
    started_utc: str

    def __post_init__(self) -> None:
        if type(self.attempt_id) is not str or not self.attempt_id:
            raise ValueError("execution attempt ID must be a non-empty string")
        _validate_sha256(self.scientific_identity_sha256, "scientific_identity_sha256")
        for name, reference in (
            ("runtime_record_ref", self.runtime_record_ref),
            ("input_bundle_ref", self.input_bundle_ref),
        ):
            try:
                BlobRef.from_payload(reference.payload())
            except RunStoreError as error:
                raise ValueError(f"{name} must be a valid blob reference") from error
        _require_commit_sha(self.commit_sha)
        if self.workspace_clean is not True or self.scientific_diff_sha256 is not None:
            raise ValueError(
                "execution attempts require a clean workspace and null scientific diff"
            )
        if not self.argv or any(
            type(argument) is not str or not argument for argument in self.argv
        ):
            raise ValueError("execution argv must contain non-empty strings")
        _parse_utc_timestamp(self.started_utc, "started_utc")

    def payload(self) -> dict[str, object]:
        """Return the canonical invocation-start record."""
        return {
            "attempt_id": self.attempt_id,
            "scientific_identity_sha256": self.scientific_identity_sha256,
            "runtime_record_ref": self.runtime_record_ref.payload(),
            "input_bundle_ref": self.input_bundle_ref.payload(),
            "commit_sha": self.commit_sha,
            "workspace_clean": self.workspace_clean,
            "scientific_diff_sha256": self.scientific_diff_sha256,
            "argv": list(self.argv),
            "started_utc": self.started_utc,
        }

    @classmethod
    def from_payload(cls, value: object) -> ExecutionAttemptStart:
        """Decode and validate one invocation-start record."""
        row = _require_object(
            value,
            frozenset(
                {
                    "attempt_id",
                    "scientific_identity_sha256",
                    "runtime_record_ref",
                    "input_bundle_ref",
                    "commit_sha",
                    "workspace_clean",
                    "scientific_diff_sha256",
                    "argv",
                    "started_utc",
                }
            ),
            "execution-attempt start",
        )
        argv = row["argv"]
        if not isinstance(argv, list):
            raise RecoveryStateError("execution-attempt argv must be an array")
        if row["scientific_diff_sha256"] is not None:
            raise RecoveryStateError("execution-attempt scientific diff must be null")
        try:
            return cls(
                attempt_id=cast("str", row["attempt_id"]),
                scientific_identity_sha256=cast("str", row["scientific_identity_sha256"]),
                runtime_record_ref=BlobRef.from_payload(row["runtime_record_ref"]),
                input_bundle_ref=BlobRef.from_payload(row["input_bundle_ref"]),
                commit_sha=cast("str", row["commit_sha"]),
                workspace_clean=cast("bool", row["workspace_clean"]),
                scientific_diff_sha256=None,
                argv=tuple(cast("list[str]", argv)),
                started_utc=cast("str", row["started_utc"]),
            )
        except (RunStoreError, TypeError, ValueError) as error:
            raise RecoveryStateError("execution-attempt start record is invalid") from error


@dataclass(frozen=True)
class ExecutionAttemptCompletion:
    """Durable completion bound to one exact invocation-start blob."""

    attempt_id: str
    start_ref: BlobRef
    commit_sha: str
    workspace_clean: bool
    scientific_diff_sha256: None
    completed_utc: str

    def __post_init__(self) -> None:
        if type(self.attempt_id) is not str or not self.attempt_id:
            raise ValueError("execution attempt ID must be a non-empty string")
        try:
            BlobRef.from_payload(self.start_ref.payload())
        except RunStoreError as error:
            raise ValueError("start_ref must be a valid blob reference") from error
        _require_commit_sha(self.commit_sha)
        if self.workspace_clean is not True or self.scientific_diff_sha256 is not None:
            raise ValueError("execution completion requires a clean workspace and null diff")
        _parse_utc_timestamp(self.completed_utc, "completed_utc")

    def payload(self) -> dict[str, object]:
        """Return the canonical invocation-completion record."""
        return {
            "attempt_id": self.attempt_id,
            "start_ref": self.start_ref.payload(),
            "commit_sha": self.commit_sha,
            "workspace_clean": self.workspace_clean,
            "scientific_diff_sha256": self.scientific_diff_sha256,
            "completed_utc": self.completed_utc,
        }

    @classmethod
    def from_payload(cls, value: object) -> ExecutionAttemptCompletion:
        """Decode and validate one invocation-completion record."""
        row = _require_object(
            value,
            frozenset(
                {
                    "attempt_id",
                    "start_ref",
                    "commit_sha",
                    "workspace_clean",
                    "scientific_diff_sha256",
                    "completed_utc",
                }
            ),
            "execution-attempt completion",
        )
        if row["scientific_diff_sha256"] is not None:
            raise RecoveryStateError("execution-attempt completion diff must be null")
        try:
            return cls(
                attempt_id=cast("str", row["attempt_id"]),
                start_ref=BlobRef.from_payload(row["start_ref"]),
                commit_sha=cast("str", row["commit_sha"]),
                workspace_clean=cast("bool", row["workspace_clean"]),
                scientific_diff_sha256=None,
                completed_utc=cast("str", row["completed_utc"]),
            )
        except (RunStoreError, TypeError, ValueError) as error:
            raise RecoveryStateError("execution-attempt completion record is invalid") from error


@dataclass(frozen=True)
class ExecutionAttemptRecord:
    """One invocation and its derived abandoned or finalized state."""

    start: ExecutionAttemptStart
    start_ref: BlobRef
    completion: ExecutionAttemptCompletion | None = None
    completion_ref: BlobRef | None = None
    abandoned: bool = False


@dataclass(frozen=True)
class PublishedSelectionPlan:
    """The label-independent acquisition plan published after D-optimal completion."""

    scientific_identity_sha256: str
    selection_plan_sha256: str
    plan_ref: BlobRef

    def __post_init__(self) -> None:
        _validate_sha256(self.scientific_identity_sha256, "scientific_identity_sha256")
        _validate_sha256(self.selection_plan_sha256, "selection_plan_sha256")
        if self.plan_ref.sha256 != self.selection_plan_sha256:
            raise ValueError("selection plan SHA-256 must equal the archived plan digest")

    def payload(self) -> dict[str, object]:
        """Return the exact published selection-plan payload."""
        return {
            "scientific_identity_sha256": self.scientific_identity_sha256,
            "selection_plan_sha256": self.selection_plan_sha256,
            "plan_ref": self.plan_ref.payload(),
        }


@dataclass(frozen=True)
class RecoveryVerification:
    """Compact verified progress summary derived without mutating the store."""

    manifest_count: int
    completed_cell_count: int
    is_complete: bool
    latest_sequence: int | None
    report_sha256: str | None


@dataclass(frozen=True)
class RecoveryRunState:
    """State reconstructed from the complete immutable manifest chain."""

    prepared: PreparedRecoveryRun | None
    doptimal_completed: int
    selection_plan: PublishedSelectionPlan | None
    completed_cells: tuple[RecoveryCellKey, ...]
    active_cell: RecoveryCellKey | None
    completed_folds: int
    lasso_converged: bool
    cell_result_refs: tuple[BlobRef, ...]
    report_ref: BlobRef | None
    latest_manifest: Manifest | None
    manifest_count: int
    expected_cell_count: int
    lasso_selected_sha256: str | None = None
    lasso_fold_sha256: str | None = None
    lasso_lambda_ratios: tuple[float, ...] = ()
    execution_attempts: tuple[ExecutionAttemptRecord, ...] = ()

    @property
    def is_complete(self) -> bool:
        """True only after a report follows every registered cell."""
        return self.report_ref is not None and len(self.completed_cells) == self.expected_cell_count

    @property
    def cell_results_sha256(self) -> str:
        """Return the canonical identity of every completed cell result in journal order."""
        return _cell_results_sha256(self.completed_cells, self.cell_result_refs)

    @property
    def abandoned_execution_attempts(self) -> tuple[ExecutionAttemptRecord, ...]:
        """Return every superseded invocation in durable start order."""
        return tuple(attempt for attempt in self.execution_attempts if attempt.abandoned)

    @property
    def finalized_execution_attempt(self) -> ExecutionAttemptRecord | None:
        """Return the unique completed invocation, if execution has finalized."""
        return next(
            (attempt for attempt in reversed(self.execution_attempts) if attempt.completion),
            None,
        )

    @property
    def open_execution_attempt(self) -> ExecutionAttemptRecord | None:
        """Return the latest unfinished invocation, if one remains open."""
        if not self.execution_attempts:
            return None
        latest = self.execution_attempts[-1]
        return latest if latest.completion is None and not latest.abandoned else None

    def verification(self) -> RecoveryVerification:
        """Return a non-mutating summary of the verified replay."""
        return RecoveryVerification(
            manifest_count=self.manifest_count,
            completed_cell_count=len(self.completed_cells),
            is_complete=self.is_complete,
            latest_sequence=None if self.latest_manifest is None else self.latest_manifest.sequence,
            report_sha256=None if self.report_ref is None else self.report_ref.sha256,
        )


@dataclass(frozen=True)
class RecoveryManifestIndex:
    """Manifest addresses retained by the in-memory recovery cursor."""

    doptimal_manifests: tuple[Manifest, ...]
    selection_manifest: Manifest | None
    active_lasso_manifests: tuple[Manifest, ...]
    cell_manifests: tuple[Manifest, ...]
    report_manifest: Manifest | None
    execution_attempt_manifests: tuple[Manifest, ...] = ()


@dataclass(frozen=True)
class RecoveryLassoView:
    """Constant-size view of the active LASSO checkpoint chain."""

    active_cell: RecoveryCellKey | None
    completed_folds: int
    converged: bool
    selected_sha256: str | None
    fold_sha256: str | None
    lambda_ratios: tuple[float, ...]
    manifests: tuple[Manifest, ...]


def registered_cell_keys(
    protocol: RecoveryScientificProtocol = REGISTERED_RECOVERY_PROTOCOL,
) -> tuple[RecoveryCellKey, ...]:
    """Return the exact budget-major cell schedule fixed by a scientific protocol."""
    return tuple(
        RecoveryCellKey(method=method, seed=seed, budget=budget)
        for budget in protocol.budgets
        for method, seed in protocol.sequence_keys
    )


def _cell_results_sha256(keys: tuple[RecoveryCellKey, ...], references: tuple[BlobRef, ...]) -> str:
    if len(keys) != len(references):
        raise RecoveryStateError("recovery cell keys and result references have different lengths")
    payload = [
        {"cell": key.payload(), "result_ref": reference.payload()}
        for key, reference in zip(keys, references, strict=True)
    ]
    return _payload_sha256(payload)


def _report_cells_from_results(
    store: ContentAddressedRunStore,
    keys: tuple[RecoveryCellKey, ...],
    references: tuple[BlobRef, ...],
    protocol: RecoveryScientificProtocol,
) -> list[dict[str, object]]:
    if len(keys) != len(references):
        raise RecoveryStateError("recovery report cell sources have different lengths")
    cells: list[dict[str, object]] = []
    result_fields = frozenset({"schema_version", "cell", "valid", "metrics", "error"})
    for key, reference in zip(keys, references, strict=True):
        try:
            value = store.get_json(reference)
        except (RunStoreError, ValueError) as error:
            raise RecoveryStateError("recovery report cell source cannot be verified") from error
        result = _require_object(value, result_fields, "recovery report cell source")
        if result["schema_version"] != _CELL_RESULT_SCHEMA:
            raise RecoveryStateError("recovery report cell source schema does not match")
        if RecoveryCellKey.from_payload(result["cell"]) != key:
            raise RecoveryStateError("recovery report cell source key does not match")
        if result["valid"] is True:
            metrics = _require_object(
                result["metrics"], _VALID_METRIC_FIELDS, "recovery report cell metrics"
            )
            cells.append(dict(metrics))
            continue
        if result["valid"] is not False or type(result["error"]) is not str:
            raise RecoveryStateError("invalid recovery report cell source has no error")
        cells.append(
            {
                "method": key.method,
                "budget": key.budget,
                "seed": key.seed,
                "spearman": None,
                "relative_sse_gain": None,
                "support_size": 0,
                "coefficient_count": protocol.coefficient_count,
                "selected_sha256": "",
                "fold_sha256": "",
                "lambda_ratio": None,
                "lambda_value": None,
                "converged": False,
                "error": result["error"],
            }
        )
    return cells


def _blob_ref_from_payload(value: object, field: str) -> BlobRef:
    try:
        return BlobRef.from_payload(value)
    except RunStoreError as error:
        raise RecoveryStateError(f"{field} contains a malformed blob reference") from error


def _verify_blob(store: ContentAddressedRunStore, reference: BlobRef, field: str) -> None:
    try:
        store.get_bytes(reference)
    except (RunStoreError, ValueError) as error:
        raise RecoveryStateError(f"{field} does not resolve to a verified blob") from error


def _require_entry_reference(
    store: ContentAddressedRunStore,
    manifest: Manifest,
    *,
    entry_name: str,
    payload: object,
) -> BlobRef:
    reference = _blob_ref_from_payload(payload, entry_name)
    try:
        entry = manifest.entry(entry_name)
    except KeyError as error:
        raise RecoveryStateError(f"manifest is missing the {entry_name} entry") from error
    if reference != entry:
        raise RecoveryStateError(f"{entry_name} reference does not match its manifest entry")
    _verify_blob(store, reference, entry_name)
    return reference


def _prepared_from_manifest(
    store: ContentAddressedRunStore,
    manifest: Manifest,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> PreparedRecoveryRun:
    if set(manifest.entries) != {"runtime_record", "input_bundle"}:
        raise RecoveryStateError("prepared manifest entries do not match")
    meta = _require_object(
        manifest.meta,
        frozenset({"schema_version", "state_kind", "prepared"}),
        "prepared manifest metadata",
    )
    if meta["schema_version"] != _PREPARED_SCHEMA:
        raise RecoveryStateError("prepared manifest schema does not match")
    payload = _require_object(
        meta["prepared"],
        frozenset(
            {
                "scientific_identity_sha256",
                "protocol_semantic_sha256",
                "execution_policy_sha256",
                "numerical_compatibility_sha256",
                "candidate_sha256",
                "runtime_record_ref",
                "input_bundle_ref",
            }
        ),
        "prepared run",
    )
    runtime = _require_entry_reference(
        store, manifest, entry_name="runtime_record", payload=payload["runtime_record_ref"]
    )
    bundle = _require_entry_reference(
        store, manifest, entry_name="input_bundle", payload=payload["input_bundle_ref"]
    )
    try:
        prepared = PreparedRecoveryRun(
            scientific_identity_sha256=_require_sha256(
                payload["scientific_identity_sha256"], "scientific identity"
            ),
            protocol_semantic_sha256=_require_sha256(
                payload["protocol_semantic_sha256"], "protocol semantic identity"
            ),
            execution_policy_sha256=_require_sha256(
                payload["execution_policy_sha256"], "execution policy identity"
            ),
            numerical_compatibility_sha256=_require_sha256(
                payload["numerical_compatibility_sha256"], "numerical compatibility identity"
            ),
            candidate_sha256=_require_sha256(payload["candidate_sha256"], "candidate identity"),
            runtime_record_ref=runtime,
            input_bundle_ref=bundle,
        )
    except ValueError as error:
        raise RecoveryStateError(f"prepared run is invalid: {error}") from error
    if prepared.protocol_semantic_sha256 != protocol.semantic_sha256:
        raise RecoveryStateError("prepared protocol semantic identity does not match")
    if prepared.execution_policy_sha256 != policy.policy_sha256:
        raise RecoveryStateError("prepared execution policy identity does not match")
    return prepared


def _attempt_record_ref(
    store: ContentAddressedRunStore,
    manifest: Manifest,
    *,
    entry_name: str,
    schema: str,
    kind: str,
) -> BlobRef:
    if set(manifest.entries) != {entry_name}:
        raise RecoveryStateError("execution-attempt manifest entries do not match")
    meta = _require_object(
        manifest.meta,
        frozenset({"schema_version", "state_kind", "attempt_id", "record_ref"}),
        "execution-attempt manifest metadata",
    )
    if meta["schema_version"] != schema or meta["state_kind"] != kind:
        raise RecoveryStateError("execution-attempt manifest schema does not match")
    if type(meta["attempt_id"]) is not str or not meta["attempt_id"]:
        raise RecoveryStateError("execution-attempt manifest ID is invalid")
    return _require_entry_reference(
        store, manifest, entry_name=entry_name, payload=meta["record_ref"]
    )


def _attempt_start_from_manifest(
    store: ContentAddressedRunStore,
    manifest: Manifest,
    prepared: PreparedRecoveryRun,
) -> ExecutionAttemptRecord:
    reference = _attempt_record_ref(
        store,
        manifest,
        entry_name="attempt_start",
        schema=_ATTEMPT_STARTED_SCHEMA,
        kind=_ATTEMPT_STARTED,
    )
    try:
        value = store.get_json(reference)
    except (RunStoreError, ValueError) as error:
        raise RecoveryStateError("execution-attempt start is not verified JSON") from error
    start = ExecutionAttemptStart.from_payload(value)
    if manifest.meta["attempt_id"] != start.attempt_id:
        raise RecoveryStateError("execution-attempt start ID does not match its manifest")
    if start.scientific_identity_sha256 != prepared.scientific_identity_sha256:
        raise RecoveryStateError("execution-attempt scientific identity drifted")
    if (
        start.runtime_record_ref != prepared.runtime_record_ref
        or start.input_bundle_ref != prepared.input_bundle_ref
    ):
        raise RecoveryStateError("execution-attempt inputs do not match the prepared run")
    return ExecutionAttemptRecord(start=start, start_ref=reference)


def _attempt_completion_from_manifest(
    store: ContentAddressedRunStore,
    manifest: Manifest,
    open_attempt: ExecutionAttemptRecord,
) -> tuple[ExecutionAttemptCompletion, BlobRef]:
    reference = _attempt_record_ref(
        store,
        manifest,
        entry_name="attempt_completion",
        schema=_ATTEMPT_COMPLETED_SCHEMA,
        kind=_ATTEMPT_COMPLETED,
    )
    try:
        value = store.get_json(reference)
    except (RunStoreError, ValueError) as error:
        raise RecoveryStateError("execution-attempt completion is not verified JSON") from error
    completion = ExecutionAttemptCompletion.from_payload(value)
    if manifest.meta["attempt_id"] != completion.attempt_id:
        raise RecoveryStateError("execution-attempt completion ID does not match its manifest")
    if (
        completion.attempt_id != open_attempt.start.attempt_id
        or completion.start_ref != open_attempt.start_ref
    ):
        raise RecoveryStateError("execution-attempt completion does not match latest open start")
    if completion.commit_sha != open_attempt.start.commit_sha:
        raise RecoveryStateError("execution-attempt commit changed between start and completion")
    if _parse_utc_timestamp(completion.completed_utc, "completed_utc") < _parse_utc_timestamp(
        open_attempt.start.started_utc, "started_utc"
    ):
        raise RecoveryStateError("execution-attempt completion precedes its start")
    return completion, reference


def _require_exact_policy(value: object, policy: RecoveryExecutionPolicy) -> None:
    if canonical_json_bytes(value) != canonical_json_bytes(policy.identity_payload()):
        raise RecoveryStateError("checkpoint execution policy identity drifted")


def _require_identity_digest(value: object, declared: str, field: str) -> None:
    if _payload_sha256(value) != declared:
        raise RecoveryStateError(f"{field} payload does not match its SHA-256")


def _array_reference(
    store: ContentAddressedRunStore,
    manifest: Manifest,
    arrays: Mapping[str, object],
    name: str,
) -> tuple[ArrayRef, np.ndarray]:
    try:
        reference = ArrayRef.from_payload(arrays[name])
    except (KeyError, RunStoreError, TypeError, ValueError) as error:
        raise RecoveryStateError(f"{name} contains an invalid array reference") from error
    try:
        entry = manifest.entry(name)
    except KeyError as error:
        raise RecoveryStateError(f"manifest is missing the {name} array entry") from error
    if reference.blob != entry:
        raise RecoveryStateError(f"{name} array reference does not match its manifest entry")
    try:
        array = np.asarray(store.get_array(reference))
    except (RunStoreError, TypeError, ValueError) as error:
        raise RecoveryStateError(f"{name} array cannot be verified") from error
    return reference, array


def _replay_doptimal(  # noqa: PLR0912, PLR0915
    store: ContentAddressedRunStore,
    manifest: Manifest,
    prepared: PreparedRecoveryRun,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
    completed: int,
    prior_identity: bytes | None,
    prior_indices: set[int],
    prior_prefixes: set[str],
) -> tuple[int, bytes, tuple[int, ...], str]:
    if set(manifest.entries) != {"selected_indices", "updates", "posterior_variance"}:
        raise RecoveryStateError("D-optimal manifest entries do not match")
    meta = _require_object(
        manifest.meta,
        frozenset(
            {"schema_version", "state_kind", "identity", "start", "stop", "prefix_sha256", "arrays"}
        ),
        "D-optimal manifest metadata",
    )
    if meta["schema_version"] != _DOPTIMAL_SCHEMA:
        raise RecoveryStateError("D-optimal manifest schema does not match")
    start = _require_integer(meta["start"], "D-optimal start", minimum=0)
    stop = _require_integer(meta["stop"], "D-optimal stop", minimum=1)
    if start != completed or stop - start != policy.doptimal_block_size:
        raise RecoveryStateError("D-optimal checkpoint blocks are not contiguous")
    if stop > protocol.budgets[-1]:
        raise RecoveryStateError("D-optimal checkpoint exceeds the registered target budget")
    prefix_sha256 = _require_sha256(meta["prefix_sha256"], "D-optimal prefix")
    identity = _require_object(
        meta["identity"],
        frozenset(
            {
                "scientific_identity_sha256",
                "execution_policy",
                "numerical_compatibility",
                "numerical_compatibility_sha256",
                "candidate_universe_sha256",
                "candidate_sequence_sha256",
                "candidate_count",
                "target_budget",
                "geometry_sha256",
            }
        ),
        "D-optimal identity",
    )
    if _require_sha256(identity["scientific_identity_sha256"], "scientific identity") != (
        prepared.scientific_identity_sha256
    ):
        raise RecoveryStateError("D-optimal scientific identity drifted")
    _require_exact_policy(identity["execution_policy"], policy)
    numerical_sha = _require_sha256(
        identity["numerical_compatibility_sha256"], "numerical compatibility"
    )
    if numerical_sha != prepared.numerical_compatibility_sha256:
        raise RecoveryStateError("D-optimal numerical compatibility identity drifted")
    _require_identity_digest(identity["numerical_compatibility"], numerical_sha, "numerical")
    if _require_sha256(identity["candidate_universe_sha256"], "candidate universe identity") != (
        prepared.candidate_sha256
    ):
        raise RecoveryStateError("D-optimal candidate universe identity drifted")
    _require_sha256(identity["candidate_sequence_sha256"], "candidate sequence identity")
    candidate_count = _require_integer(identity["candidate_count"], "candidate count", minimum=1)
    target = _require_integer(identity["target_budget"], "D-optimal target budget", minimum=1)
    if target != protocol.budgets[-1] or target > candidate_count:
        raise RecoveryStateError("D-optimal target budget does not match the protocol")
    _require_sha256(identity["geometry_sha256"], "D-optimal geometry")
    identity_bytes = canonical_json_bytes(identity)
    if prior_identity is not None and identity_bytes != prior_identity:
        raise RecoveryStateError("D-optimal checkpoint identity drifted between blocks")
    arrays = _require_object(
        meta["arrays"],
        frozenset({"selected_indices", "updates", "posterior_variance"}),
        "D-optimal arrays",
    )
    selected_ref, selected = _array_reference(store, manifest, arrays, "selected_indices")
    updates_ref, updates = _array_reference(store, manifest, arrays, "updates")
    posterior_ref, posterior = _array_reference(store, manifest, arrays, "posterior_variance")
    if selected_ref.dtype != "int64" or selected_ref.shape != (policy.doptimal_block_size,):
        raise RecoveryStateError("D-optimal selected indices have an incompatible layout")
    if (
        updates_ref.dtype != "float64"
        or updates.ndim != _MATRIX_DIMENSIONS
        or updates.shape[0] != candidate_count
        or updates.shape[1] != stop - start
    ):
        raise RecoveryStateError(
            "D-optimal updates do not match the declared candidate count or block layout"
        )
    if posterior_ref.dtype != "float64" or posterior.shape != (candidate_count,):
        raise RecoveryStateError("D-optimal posterior does not match the declared candidate count")
    if np.any(selected < 0) or np.any(selected >= candidate_count):
        raise RecoveryStateError("D-optimal selected index is out of range")
    selected_values = tuple(int(value) for value in selected)
    if len(set(selected_values)) != len(selected_values) or prior_indices.intersection(
        selected_values
    ):
        raise RecoveryStateError("D-optimal selected indices must be globally unique")
    if prefix_sha256 in prior_prefixes:
        raise RecoveryStateError("D-optimal prefix digest must advance with every block")
    if not np.all(np.isfinite(updates)) or not np.all(np.isfinite(posterior)):
        raise RecoveryStateError("D-optimal arrays contain a non-finite number")
    if np.any(posterior < 0.0):
        raise RecoveryStateError("D-optimal posterior must be nonnegative")
    prior_indices.update(selected_values)
    prior_prefixes.add(prefix_sha256)
    return stop, identity_bytes, selected_values, prefix_sha256


def _selection_from_manifest(
    store: ContentAddressedRunStore,
    manifest: Manifest,
    prepared: PreparedRecoveryRun,
) -> PublishedSelectionPlan:
    if set(manifest.entries) != {"selection_plan"}:
        raise RecoveryStateError("selection-plan manifest entries do not match")
    meta = _require_object(
        manifest.meta,
        frozenset({"schema_version", "state_kind", "selection"}),
        "selection-plan manifest metadata",
    )
    if meta["schema_version"] != _SELECTION_SCHEMA:
        raise RecoveryStateError("selection-plan manifest schema does not match")
    payload = _require_object(
        meta["selection"],
        frozenset({"scientific_identity_sha256", "selection_plan_sha256", "plan_ref"}),
        "selection plan",
    )
    reference = _require_entry_reference(
        store, manifest, entry_name="selection_plan", payload=payload["plan_ref"]
    )
    scientific = _require_sha256(payload["scientific_identity_sha256"], "scientific identity")
    selection_sha = _require_sha256(payload["selection_plan_sha256"], "selection plan")
    if scientific != prepared.scientific_identity_sha256:
        raise RecoveryStateError("selection-plan scientific identity drifted")
    if selection_sha != reference.sha256:
        raise RecoveryStateError("selection-plan SHA-256 does not match its blob")
    try:
        return PublishedSelectionPlan(scientific, selection_sha, reference)
    except ValueError as error:
        raise RecoveryStateError(f"selection plan is invalid: {error}") from error


@dataclass
class _LassoProgress:
    key: RecoveryCellKey | None = None
    completed_folds: int = 0
    identity: bytes | None = None
    cv_sse: np.ndarray | None = None
    converged: bool = True
    selected_sha256: str | None = None
    fold_sha256: str | None = None
    lambda_ratios: tuple[float, ...] = ()


def _replay_lasso(  # noqa: PLR0912, PLR0915
    store: ContentAddressedRunStore,
    manifest: Manifest,
    prepared: PreparedRecoveryRun,
    selection: PublishedSelectionPlan,
    expected_key: RecoveryCellKey,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
    progress: _LassoProgress,
) -> None:
    if set(manifest.entries) != {"cv_sse"}:
        raise RecoveryStateError("LASSO manifest entries do not match")
    meta = _require_object(
        manifest.meta,
        frozenset(
            {
                "schema_version",
                "state_kind",
                "cell",
                "identity",
                "completed_folds",
                "converged",
                "cv_sse",
            }
        ),
        "LASSO manifest metadata",
    )
    if meta["schema_version"] != _LASSO_SCHEMA:
        raise RecoveryStateError("LASSO manifest schema does not match")
    key = RecoveryCellKey.from_payload(meta["cell"])
    if key != expected_key or (progress.key is not None and key != progress.key):
        raise RecoveryStateError("LASSO cell does not follow the registered order")
    identity = _require_object(
        meta["identity"],
        frozenset(
            {
                "scientific_identity_sha256",
                "execution_policy",
                "numerical_compatibility",
                "numerical_compatibility_sha256",
                "selection_plan_sha256",
                "cell",
                "selected_sha256",
                "fold_sha256",
                "problem_sha256",
                "n_folds",
                "lambda_ratios",
            }
        ),
        "LASSO identity",
    )
    if RecoveryCellKey.from_payload(identity["cell"]) != key:
        raise RecoveryStateError("LASSO cell identity does not match its manifest")
    if _require_sha256(identity["scientific_identity_sha256"], "scientific identity") != (
        prepared.scientific_identity_sha256
    ):
        raise RecoveryStateError("LASSO scientific identity drifted")
    _require_exact_policy(identity["execution_policy"], policy)
    numerical_sha = _require_sha256(
        identity["numerical_compatibility_sha256"], "numerical compatibility"
    )
    if numerical_sha != prepared.numerical_compatibility_sha256:
        raise RecoveryStateError("LASSO numerical compatibility identity drifted")
    _require_identity_digest(identity["numerical_compatibility"], numerical_sha, "numerical")
    if _require_sha256(identity["selection_plan_sha256"], "selection plan") != (
        selection.selection_plan_sha256
    ):
        raise RecoveryStateError("LASSO selection-plan identity drifted")
    selected_sha256 = _require_sha256(identity["selected_sha256"], "LASSO selected_sha256")
    fold_sha256 = _require_sha256(identity["fold_sha256"], "LASSO fold_sha256")
    _require_sha256(identity["problem_sha256"], "LASSO problem_sha256")
    n_folds = _require_integer(identity["n_folds"], "LASSO fold count", minimum=2)
    if n_folds != protocol.n_folds:
        raise RecoveryStateError("LASSO fold count does not match the protocol")
    ratios_value = identity["lambda_ratios"]
    if not isinstance(ratios_value, list) or not ratios_value:
        raise RecoveryStateError("LASSO lambda ratios must be a non-empty array")
    ratios: list[float] = []
    for ratio in ratios_value:
        if type(ratio) not in {int, float} or isinstance(ratio, bool):
            raise RecoveryStateError("LASSO lambda ratio must be numeric")
        converted = float(ratio)
        if not math.isfinite(converted) or not 0.0 < converted <= 1.0:
            raise RecoveryStateError("LASSO lambda ratio is invalid")
        ratios.append(converted)
    if any(first < second for first, second in pairwise(ratios)):
        raise RecoveryStateError("LASSO lambda ratios must be descending")
    identity_bytes = canonical_json_bytes(identity)
    if progress.identity is not None and identity_bytes != progress.identity:
        raise RecoveryStateError("LASSO checkpoint identity drifted between folds")
    completed = _require_integer(meta["completed_folds"], "LASSO completed fold", minimum=1)
    if completed != progress.completed_folds + 1 or completed > n_folds:
        raise RecoveryStateError("LASSO fold chain has a gap or duplicate")
    converged = meta["converged"]
    if type(converged) is not bool:
        raise RecoveryStateError("LASSO converged flag must be boolean")
    if not progress.converged and converged:
        raise RecoveryStateError("LASSO convergence flag is not cumulative")
    try:
        reference = ArrayRef.from_payload(meta["cv_sse"])
    except (RunStoreError, TypeError, ValueError) as error:
        raise RecoveryStateError("LASSO cv_sse contains an invalid array reference") from error
    if reference.blob != manifest.entry("cv_sse"):
        raise RecoveryStateError("LASSO cv_sse reference does not match its manifest entry")
    try:
        cv_sse = np.asarray(store.get_array(reference))
    except (RunStoreError, TypeError, ValueError) as error:
        raise RecoveryStateError("LASSO cv_sse cannot be verified") from error
    if reference.dtype != "float64" or cv_sse.shape != (len(ratios),):
        raise RecoveryStateError("LASSO cv_sse has an incompatible layout")
    if not np.all(np.isfinite(cv_sse)) or np.any(cv_sse < 0.0):
        raise RecoveryStateError("LASSO cv_sse must be finite and nonnegative")
    if progress.cv_sse is not None and np.any(cv_sse < progress.cv_sse):
        raise RecoveryStateError("LASSO cv_sse is not cumulative")
    progress.key = key
    progress.completed_folds = completed
    progress.identity = identity_bytes
    progress.cv_sse = cv_sse
    progress.converged = converged
    progress.selected_sha256 = selected_sha256
    progress.fold_sha256 = fold_sha256
    progress.lambda_ratios = tuple(ratios)


def _require_optional_finite_float(value: object, field: str) -> float | None:
    if value is None:
        return None
    if type(value) is not float or not math.isfinite(value):
        raise RecoveryStateError(f"recovery-cell metrics {field} must be a finite float or null")
    return value


def _require_valid_metrics(
    value: object,
    key: RecoveryCellKey,
    protocol: RecoveryScientificProtocol,
    *,
    selected_sha256: str | None,
    fold_sha256: str | None,
    lambda_ratios: tuple[float, ...],
) -> None:
    metrics = _require_object(value, _VALID_METRIC_FIELDS, "valid recovery-cell metrics")
    if metrics["method"] != key.method or metrics["budget"] != key.budget:
        raise RecoveryStateError("recovery-cell metrics do not match the cell key")
    if metrics["seed"] != key.seed or (
        metrics["seed"] is not None and type(metrics["seed"]) is not int
    ):
        raise RecoveryStateError("recovery-cell metrics seed does not match the cell key")
    _require_optional_finite_float(metrics["spearman"], "spearman")
    _require_optional_finite_float(metrics["relative_sse_gain"], "relative_sse_gain")
    support_size = _require_integer(metrics["support_size"], "support size", minimum=0)
    coefficient_count = _require_integer(
        metrics["coefficient_count"], "coefficient count", minimum=1
    )
    if coefficient_count != protocol.coefficient_count or support_size > coefficient_count:
        raise RecoveryStateError("recovery-cell metrics coefficient dimensions do not match")
    if metrics["selected_sha256"] != selected_sha256:
        raise RecoveryStateError("recovery-cell metrics selected SHA does not match LASSO")
    if metrics["fold_sha256"] != fold_sha256:
        raise RecoveryStateError("recovery-cell metrics fold SHA does not match LASSO")
    _require_sha256(metrics["selected_sha256"], "recovery-cell selected identity")
    _require_sha256(metrics["fold_sha256"], "recovery-cell fold identity")
    lambda_ratio = _require_optional_finite_float(metrics["lambda_ratio"], "lambda_ratio")
    lambda_value = _require_optional_finite_float(metrics["lambda_value"], "lambda_value")
    if (
        lambda_ratio is None
        or lambda_value is None
        or lambda_ratio not in lambda_ratios
        or lambda_value <= 0.0
    ):
        raise RecoveryStateError("recovery-cell metrics lambda identity does not match LASSO")
    if metrics["converged"] is not True or metrics["error"] is not None:
        raise RecoveryStateError("valid recovery-cell metrics require convergence and no error")


def _replay_cell(  # noqa: PLR0912
    store: ContentAddressedRunStore,
    manifest: Manifest,
    prepared: PreparedRecoveryRun,
    selection: PublishedSelectionPlan,
    expected_key: RecoveryCellKey,
    protocol: RecoveryScientificProtocol,
    progress: _LassoProgress,
) -> BlobRef:
    if set(manifest.entries) != {"result"}:
        raise RecoveryStateError("recovery-cell manifest entries do not match")
    meta = _require_object(
        manifest.meta,
        frozenset(
            {
                "schema_version",
                "state_kind",
                "scientific_identity_sha256",
                "selection_plan_sha256",
                "cell",
                "valid",
                "error",
                "result_ref",
            }
        ),
        "recovery-cell manifest metadata",
    )
    if meta["schema_version"] != _CELL_SCHEMA:
        raise RecoveryStateError("recovery-cell manifest schema does not match")
    if _require_sha256(meta["scientific_identity_sha256"], "scientific identity") != (
        prepared.scientific_identity_sha256
    ):
        raise RecoveryStateError("recovery-cell scientific identity drifted")
    if _require_sha256(meta["selection_plan_sha256"], "selection plan") != (
        selection.selection_plan_sha256
    ):
        raise RecoveryStateError("recovery-cell selection-plan identity drifted")
    key = RecoveryCellKey.from_payload(meta["cell"])
    if key != expected_key or (progress.key is not None and progress.key != key):
        raise RecoveryStateError("recovery cell does not follow the registered order")
    valid = meta["valid"]
    if type(valid) is not bool:
        raise RecoveryStateError("recovery-cell valid flag must be boolean")
    error_value = meta["error"]
    if valid:
        if progress.completed_folds != protocol.n_folds:
            raise RecoveryStateError("valid recovery cell requires every LASSO fold")
        if not progress.converged:
            raise RecoveryStateError("valid recovery cell requires a converged LASSO fit")
        if error_value is not None:
            raise RecoveryStateError("valid recovery cell cannot carry an error")
    elif type(error_value) is not str or not error_value:
        raise RecoveryStateError("invalid recovery cell requires a non-empty error")
    result_ref = _require_entry_reference(
        store, manifest, entry_name="result", payload=meta["result_ref"]
    )
    try:
        result_value = store.get_json(result_ref)
    except (RunStoreError, ValueError) as error:
        raise RecoveryStateError("recovery-cell result is not verified canonical JSON") from error
    result = _require_object(
        result_value,
        frozenset({"schema_version", "cell", "valid", "metrics", "error"}),
        "recovery-cell result",
    )
    if result["schema_version"] != _CELL_RESULT_SCHEMA:
        raise RecoveryStateError("recovery-cell result schema does not match")
    if RecoveryCellKey.from_payload(result["cell"]) != key:
        raise RecoveryStateError("recovery-cell result key does not match")
    if result["valid"] is not valid or result["error"] != error_value:
        raise RecoveryStateError("recovery-cell result status does not match its manifest")
    metrics = result["metrics"]
    if valid:
        _require_valid_metrics(
            metrics,
            key,
            protocol,
            selected_sha256=progress.selected_sha256,
            fold_sha256=progress.fold_sha256,
            lambda_ratios=progress.lambda_ratios,
        )
    elif metrics is not None:
        raise RecoveryStateError("invalid recovery cell metrics must be null")
    _require_json_value(metrics, "recovery-cell metrics")
    return result_ref


def _report_from_manifest(
    store: ContentAddressedRunStore,
    manifest: Manifest,
    prepared: PreparedRecoveryRun,
    selection: PublishedSelectionPlan,
    cell_keys: tuple[RecoveryCellKey, ...],
    cell_result_refs: tuple[BlobRef, ...],
    protocol: RecoveryScientificProtocol,
) -> BlobRef:
    if set(manifest.entries) != {"report"}:
        raise RecoveryStateError("recovery-report manifest entries do not match")
    meta = _require_object(
        manifest.meta,
        frozenset(
            {
                "schema_version",
                "state_kind",
                "scientific_identity_sha256",
                "selection_plan_sha256",
                "cell_results_sha256",
                "report_ref",
            }
        ),
        "recovery-report manifest metadata",
    )
    if meta["schema_version"] != _REPORT_SCHEMA:
        raise RecoveryStateError("recovery-report manifest schema does not match")
    if _require_sha256(meta["scientific_identity_sha256"], "scientific identity") != (
        prepared.scientific_identity_sha256
    ):
        raise RecoveryStateError("recovery-report scientific identity drifted")
    if _require_sha256(meta["selection_plan_sha256"], "selection plan") != (
        selection.selection_plan_sha256
    ):
        raise RecoveryStateError("recovery-report selection-plan identity drifted")
    observed_cell_results = _require_sha256(meta["cell_results_sha256"], "recovery cell results")
    expected_cell_results = _cell_results_sha256(cell_keys, cell_result_refs)
    if observed_cell_results != expected_cell_results:
        raise RecoveryStateError("recovery report does not bind the verified cell results")
    reference = _require_entry_reference(
        store, manifest, entry_name="report", payload=meta["report_ref"]
    )
    try:
        value = store.get_json(reference)
    except (RunStoreError, ValueError) as error:
        raise RecoveryStateError("recovery report is not verified canonical JSON") from error
    if not isinstance(value, dict):
        raise RecoveryStateError("recovery report must be a JSON object")
    if value.get("cell_results_sha256") != expected_cell_results:
        raise RecoveryStateError("recovery report payload does not bind the verified cell results")
    expected_cells = _report_cells_from_results(store, cell_keys, cell_result_refs, protocol)
    if canonical_json_bytes(value.get("cells")) != canonical_json_bytes(expected_cells):
        raise RecoveryStateError("recovery report cells do not match the verified cell results")
    _require_json_value(value, "recovery report")
    return reference


@dataclass(frozen=True)
class _RecoveryEventDelta:
    doptimal_indices: tuple[int, ...] = ()
    doptimal_prefix: str | None = None


@dataclass(frozen=True)
class _RecoveryUndo:
    prepared: PreparedRecoveryRun | None
    doptimal_completed: int
    doptimal_identity: bytes | None
    selection: PublishedSelectionPlan | None
    lasso: _LassoProgress
    report_ref: BlobRef | None
    selection_manifest: Manifest | None
    report_manifest: Manifest | None
    manifests_length: int
    doptimal_manifests_length: int
    completed_cells_length: int
    cell_result_refs_length: int
    active_lasso_manifests: tuple[Manifest, ...]
    cell_manifests_length: int
    execution_attempts_length: int
    execution_attempt_previous_last: ExecutionAttemptRecord | None
    execution_attempt_manifests_length: int


def _copy_lasso(progress: _LassoProgress) -> _LassoProgress:
    return _LassoProgress(
        key=progress.key,
        completed_folds=progress.completed_folds,
        identity=progress.identity,
        cv_sse=None if progress.cv_sse is None else progress.cv_sse.copy(),
        converged=progress.converged,
        selected_sha256=progress.selected_sha256,
        fold_sha256=progress.fold_sha256,
        lambda_ratios=progress.lambda_ratios,
    )


class _RecoveryAccumulator:
    def __init__(
        self,
        store: ContentAddressedRunStore,
        protocol: RecoveryScientificProtocol,
        execution_policy: RecoveryExecutionPolicy,
    ) -> None:
        self.store = store
        self.protocol = protocol
        self.execution_policy = execution_policy
        self.prepared: PreparedRecoveryRun | None = None
        self.doptimal_completed = 0
        self.doptimal_identity: bytes | None = None
        self.doptimal_indices: set[int] = set()
        self.doptimal_prefixes: set[str] = set()
        self.selection: PublishedSelectionPlan | None = None
        self.completed_cells: list[RecoveryCellKey] = []
        self.cell_result_refs: list[BlobRef] = []
        self.lasso = _LassoProgress()
        self.report_ref: BlobRef | None = None
        self.cell_keys = registered_cell_keys(protocol)
        self.manifests: list[Manifest] = []
        self.doptimal_manifests: list[Manifest] = []
        self.selection_manifest: Manifest | None = None
        self.active_lasso_manifests: list[Manifest] = []
        self.cell_manifests: list[Manifest] = []
        self.report_manifest: Manifest | None = None
        self.execution_attempts: list[ExecutionAttemptRecord] = []
        self.execution_attempt_manifests: list[Manifest] = []

    def checkpoint(self) -> _RecoveryUndo:
        """Capture a bounded undo record for one proposed event."""
        return _RecoveryUndo(
            prepared=self.prepared,
            doptimal_completed=self.doptimal_completed,
            doptimal_identity=self.doptimal_identity,
            selection=self.selection,
            lasso=_copy_lasso(self.lasso),
            report_ref=self.report_ref,
            selection_manifest=self.selection_manifest,
            report_manifest=self.report_manifest,
            manifests_length=len(self.manifests),
            doptimal_manifests_length=len(self.doptimal_manifests),
            completed_cells_length=len(self.completed_cells),
            cell_result_refs_length=len(self.cell_result_refs),
            active_lasso_manifests=tuple(self.active_lasso_manifests),
            cell_manifests_length=len(self.cell_manifests),
            execution_attempts_length=len(self.execution_attempts),
            execution_attempt_previous_last=(
                self.execution_attempts[-1] if self.execution_attempts else None
            ),
            execution_attempt_manifests_length=len(self.execution_attempt_manifests),
        )

    def rollback(self, undo: _RecoveryUndo, delta: _RecoveryEventDelta) -> None:
        """Undo one validated but unpublished event without copying history."""
        self.prepared = undo.prepared
        self.doptimal_completed = undo.doptimal_completed
        self.doptimal_identity = undo.doptimal_identity
        self.selection = undo.selection
        self.lasso = undo.lasso
        self.report_ref = undo.report_ref
        self.selection_manifest = undo.selection_manifest
        self.report_manifest = undo.report_manifest
        del self.manifests[undo.manifests_length :]
        del self.doptimal_manifests[undo.doptimal_manifests_length :]
        del self.completed_cells[undo.completed_cells_length :]
        del self.cell_result_refs[undo.cell_result_refs_length :]
        self.active_lasso_manifests[:] = undo.active_lasso_manifests
        del self.cell_manifests[undo.cell_manifests_length :]
        del self.execution_attempts[undo.execution_attempts_length :]
        if undo.execution_attempt_previous_last is not None:
            self.execution_attempts[-1] = undo.execution_attempt_previous_last
        del self.execution_attempt_manifests[undo.execution_attempt_manifests_length :]
        self.doptimal_indices.difference_update(delta.doptimal_indices)
        if delta.doptimal_prefix is not None:
            self.doptimal_prefixes.discard(delta.doptimal_prefix)

    def _require_next_address(self, manifest: Manifest) -> None:
        expected_sequence = len(self.manifests)
        expected_parent = None if not self.manifests else self.manifests[-1].sha256
        if manifest.sequence != expected_sequence or manifest.parent_sha256 != expected_parent:
            raise RecoveryStateError("recovery manifest does not extend the verified cursor tip")

    def _adopt(self, manifest: Manifest) -> None:
        self.manifests.append(manifest)

    def apply(self, manifest: Manifest) -> _RecoveryEventDelta:  # noqa: PLR0911, PLR0912, PLR0915
        """Validate and apply one next manifest through the shared state reducer."""
        self._require_next_address(manifest)
        kind = manifest.meta.get("state_kind")
        if kind not in _KNOWN_KINDS:
            raise RecoveryStateError(f"manifest {manifest.sequence} has an unknown state kind")
        if self.report_ref is not None:
            raise RecoveryStateError("no manifest may follow the recovery report")
        if kind == _PREPARED:
            if self.prepared is not None or manifest.sequence != 0:
                raise RecoveryStateError("the prepared run must be the unique root manifest")
            self.prepared = _prepared_from_manifest(
                self.store, manifest, self.protocol, self.execution_policy
            )
            self._adopt(manifest)
            return _RecoveryEventDelta()
        if self.prepared is None:
            raise RecoveryStateError(
                "the recovery run must be prepared before any scientific state"
            )
        if kind == _ATTEMPT_STARTED:
            if self.execution_attempts and self.execution_attempts[-1].completion is not None:
                raise RecoveryStateError("no execution attempt may start after finalization")
            record = _attempt_start_from_manifest(self.store, manifest, self.prepared)
            if any(
                attempt.start.attempt_id == record.start.attempt_id
                for attempt in self.execution_attempts
            ):
                raise RecoveryStateError("execution attempt IDs must be unique")
            if self.execution_attempts and not self.execution_attempts[-1].abandoned:
                self.execution_attempts[-1] = replace(self.execution_attempts[-1], abandoned=True)
            self.execution_attempts.append(record)
            self.execution_attempt_manifests.append(manifest)
            self._adopt(manifest)
            return _RecoveryEventDelta()
        if kind == _ATTEMPT_COMPLETED:
            if (
                not self.execution_attempts
                or self.execution_attempts[-1].abandoned
                or self.execution_attempts[-1].completion is not None
            ):
                raise RecoveryStateError("execution completion requires the latest open start")
            completion, reference = _attempt_completion_from_manifest(
                self.store, manifest, self.execution_attempts[-1]
            )
            if len(self.completed_cells) != len(self.cell_keys) or self.lasso.key is not None:
                raise RecoveryStateError(
                    "execution completion requires every cell and no active LASSO fit"
                )
            self.execution_attempts[-1] = replace(
                self.execution_attempts[-1],
                completion=completion,
                completion_ref=reference,
            )
            self.execution_attempt_manifests.append(manifest)
            self._adopt(manifest)
            return _RecoveryEventDelta()
        if (
            self.execution_attempts
            and self.execution_attempts[-1].completion is not None
            and kind != _REPORT
        ):
            raise RecoveryStateError("only the recovery report may follow attempt finalization")
        if kind == _DOPTIMAL:
            if self.selection is not None or self.completed_cells or self.lasso.key is not None:
                raise RecoveryStateError("D-optimal checkpoints must precede the selection plan")
            (
                self.doptimal_completed,
                self.doptimal_identity,
                selected_indices,
                prefix_sha256,
            ) = _replay_doptimal(
                self.store,
                manifest,
                self.prepared,
                self.protocol,
                self.execution_policy,
                self.doptimal_completed,
                self.doptimal_identity,
                self.doptimal_indices,
                self.doptimal_prefixes,
            )
            self.doptimal_manifests.append(manifest)
            self._adopt(manifest)
            return _RecoveryEventDelta(selected_indices, prefix_sha256)
        if kind == _SELECTION:
            if self.selection is not None:
                raise RecoveryStateError("the selection plan may be published only once")
            if self.doptimal_completed != self.protocol.budgets[-1]:
                raise RecoveryStateError("selection plan requires complete D-optimal selection")
            self.selection = _selection_from_manifest(self.store, manifest, self.prepared)
            self.selection_manifest = manifest
            self._adopt(manifest)
            return _RecoveryEventDelta()
        if self.selection is None:
            raise RecoveryStateError("LASSO and recovery cells require a published selection plan")
        if len(self.completed_cells) >= len(self.cell_keys) and kind != _REPORT:
            raise RecoveryStateError("all recovery cells are complete; only the report may follow")
        if kind == _LASSO:
            if self.lasso.completed_folds == self.protocol.n_folds:
                raise RecoveryStateError(
                    "a completed LASSO fit must be closed by its recovery cell"
                )
            _replay_lasso(
                self.store,
                manifest,
                self.prepared,
                self.selection,
                self.cell_keys[len(self.completed_cells)],
                self.protocol,
                self.execution_policy,
                self.lasso,
            )
            self.active_lasso_manifests.append(manifest)
            self._adopt(manifest)
            return _RecoveryEventDelta()
        if kind == _CELL:
            if len(self.completed_cells) >= len(self.cell_keys):
                raise RecoveryStateError("recovery cell count exceeds the registered protocol")
            expected_key = self.cell_keys[len(self.completed_cells)]
            self.cell_result_refs.append(
                _replay_cell(
                    self.store,
                    manifest,
                    self.prepared,
                    self.selection,
                    expected_key,
                    self.protocol,
                    self.lasso,
                )
            )
            self.completed_cells.append(expected_key)
            self.cell_manifests.append(manifest)
            self.lasso = _LassoProgress()
            self.active_lasso_manifests.clear()
            self._adopt(manifest)
            return _RecoveryEventDelta()
        if kind == _REPORT:
            if len(self.completed_cells) != self.protocol.cell_count:
                raise RecoveryStateError("recovery report requires every registered cell")
            if self.lasso.key is not None:
                raise RecoveryStateError("recovery report cannot follow an open LASSO cell")
            if (
                not self.execution_attempts
                or self.execution_attempts[-1].completion is None
                or self.execution_attempts[-1].abandoned
            ):
                raise RecoveryStateError(
                    "recovery report requires a completed latest execution attempt"
                )
            self.report_ref = _report_from_manifest(
                self.store,
                manifest,
                self.prepared,
                self.selection,
                tuple(self.completed_cells),
                tuple(self.cell_result_refs),
                self.protocol,
            )
            self.report_manifest = manifest
            self._adopt(manifest)
            return _RecoveryEventDelta()
        raise RecoveryStateError(f"manifest {manifest.sequence} violates the state machine")

    def replace_latest(self, manifest: Manifest) -> None:
        """Replace a validated draft view with the byte-identical durable manifest."""
        if not self.manifests or self.manifests[-1].sha256 != manifest.sha256:
            raise RecoveryStateError("published manifest does not match the validated draft")
        self.manifests[-1] = manifest
        kind = manifest.meta.get("state_kind")
        if kind == _DOPTIMAL:
            self.doptimal_manifests[-1] = manifest
        elif kind == _SELECTION:
            self.selection_manifest = manifest
        elif kind == _LASSO:
            self.active_lasso_manifests[-1] = manifest
        elif kind == _CELL:
            self.cell_manifests[-1] = manifest
        elif kind == _REPORT:
            self.report_manifest = manifest
        elif kind in {_ATTEMPT_STARTED, _ATTEMPT_COMPLETED}:
            self.execution_attempt_manifests[-1] = manifest

    def snapshot(self) -> RecoveryRunState:
        """Freeze the current accumulator into the public state value."""
        return RecoveryRunState(
            prepared=self.prepared,
            doptimal_completed=self.doptimal_completed,
            selection_plan=self.selection,
            completed_cells=tuple(self.completed_cells),
            active_cell=self.lasso.key,
            completed_folds=self.lasso.completed_folds,
            lasso_converged=self.lasso.converged,
            lasso_selected_sha256=self.lasso.selected_sha256,
            lasso_fold_sha256=self.lasso.fold_sha256,
            lasso_lambda_ratios=self.lasso.lambda_ratios,
            cell_result_refs=tuple(self.cell_result_refs),
            report_ref=self.report_ref,
            latest_manifest=self.manifests[-1] if self.manifests else None,
            manifest_count=len(self.manifests),
            expected_cell_count=self.protocol.cell_count,
            execution_attempts=tuple(self.execution_attempts),
        )

    def index(self) -> RecoveryManifestIndex:
        """Freeze the lightweight manifest index for numeric journal cursors."""
        return RecoveryManifestIndex(
            doptimal_manifests=tuple(self.doptimal_manifests),
            selection_manifest=self.selection_manifest,
            active_lasso_manifests=tuple(self.active_lasso_manifests),
            cell_manifests=tuple(self.cell_manifests),
            report_manifest=self.report_manifest,
            execution_attempt_manifests=tuple(self.execution_attempt_manifests),
        )


class RecoveryStateCursor:
    """Incremental single-writer recovery state over one verified run-store session."""

    def __init__(self, session: RunStoreSession, accumulator: _RecoveryAccumulator) -> None:
        self._session = session
        self._accumulator = accumulator

    @classmethod
    def open(
        cls,
        session: RunStoreSession,
        *,
        protocol: RecoveryScientificProtocol = REGISTERED_RECOVERY_PROTOCOL,
        execution_policy: RecoveryExecutionPolicy = REGISTERED_EXECUTION_POLICY,
    ) -> RecoveryStateCursor:
        """Replay the session's verified cached chain exactly once."""
        store = session.store
        accumulator = _RecoveryAccumulator(store, protocol, execution_policy)
        for manifest in session.manifests():
            accumulator.apply(manifest)
        return cls(session, accumulator)

    def snapshot(self) -> RecoveryRunState:
        """Return the current immutable state without filesystem access."""
        return self._accumulator.snapshot()

    @property
    def store(self) -> ContentAddressedRunStore:
        """Return the durable store used by this cursor."""
        return self._session.store

    def draft_manifest(
        self, *, entries: Mapping[str, BlobRef], meta: Mapping[str, object]
    ) -> ManifestDraft:
        """Create the next canonical manifest draft from the current verified tip."""
        return self._session.draft_manifest(entries=entries, meta=meta)

    @property
    def index(self) -> RecoveryManifestIndex:
        """Return immutable manifest indices for domain-specific cursors."""
        return self._accumulator.index()

    def lasso_view(self) -> RecoveryLassoView:
        """Return the active cell and its bounded fold manifests without copying history."""
        progress = self._accumulator.lasso
        return RecoveryLassoView(
            active_cell=progress.key,
            completed_folds=progress.completed_folds,
            converged=progress.converged,
            selected_sha256=progress.selected_sha256,
            fold_sha256=progress.fold_sha256,
            lambda_ratios=progress.lambda_ratios,
            manifests=tuple(self._accumulator.active_lasso_manifests),
        )

    def manifest_for_cell(self, key: RecoveryCellKey) -> Manifest | None:
        """Return the durable manifest for one completed cell, if present."""
        try:
            position = self._accumulator.completed_cells.index(key)
        except ValueError:
            return None
        return self._accumulator.cell_manifests[position]

    @property
    def execution_attempts(self) -> tuple[ExecutionAttemptRecord, ...]:
        """Return every invocation in durable start order without replaying history."""
        return tuple(self._accumulator.execution_attempts)

    @property
    def abandoned_execution_attempts(self) -> tuple[ExecutionAttemptRecord, ...]:
        """Return every superseded invocation in durable start order."""
        return self.snapshot().abandoned_execution_attempts

    @property
    def finalized_execution_attempt(self) -> ExecutionAttemptRecord | None:
        """Return the completed invocation, if the run has finalized."""
        return self.snapshot().finalized_execution_attempt

    @property
    def open_execution_attempt(self) -> ExecutionAttemptRecord | None:
        """Return the latest unfinished invocation, if present."""
        return self.snapshot().open_execution_attempt

    def append(self, draft: ManifestDraft) -> Manifest:
        """Validate one candidate transition before publication and adopt only after success."""
        if draft.sequence < len(self._accumulator.manifests):
            published = self._session.publish_manifest(draft)
            expected = self._accumulator.manifests[draft.sequence]
            if published != expected:
                raise RecoveryStateError("exact retry does not match the verified cursor history")
            return published
        candidate_manifest = Manifest(
            sequence=draft.sequence,
            parent_sha256=draft.parent_sha256,
            entries=draft.entries,
            meta=draft.meta,
            sha256=draft.sha256,
        )
        undo = self._accumulator.checkpoint()
        delta = _RecoveryEventDelta()
        try:
            delta = self._accumulator.apply(candidate_manifest)
        except Exception:
            self._accumulator.rollback(undo, delta)
            raise
        try:
            published = self._session.publish_manifest(draft)
        except Exception:
            self._accumulator.rollback(undo, delta)
            raise
        try:
            self._accumulator.replace_latest(published)
        except Exception as error:
            self._accumulator.rollback(undo, delta)
            self._session.poison(f"recovery cursor could not adopt published manifest: {error}")
            raise
        return published


def _replay_manifests(
    store: ContentAddressedRunStore,
    manifests: tuple[Manifest, ...],
    protocol: RecoveryScientificProtocol,
    execution_policy: RecoveryExecutionPolicy,
) -> RecoveryRunState:
    accumulator = _RecoveryAccumulator(store, protocol, execution_policy)
    for manifest in manifests:
        accumulator.apply(manifest)
    return accumulator.snapshot()


def replay_recovery_state(
    store: ContentAddressedRunStore,
    *,
    protocol: RecoveryScientificProtocol = REGISTERED_RECOVERY_PROTOCOL,
    execution_policy: RecoveryExecutionPolicy = REGISTERED_EXECUTION_POLICY,
) -> RecoveryRunState:
    """Replay and validate the complete domain chain without writing to the store."""
    try:
        manifests = store.manifest_chain()
    except RunStoreError as error:
        raise RecoveryStateError("recovery manifest chain cannot be verified") from error
    return _replay_manifests(store, manifests, protocol, execution_policy)


def publish_prepared_run_at(cursor: RecoveryStateCursor, prepared: PreparedRecoveryRun) -> Manifest:
    """Publish the root through an existing incremental cursor."""
    store = cursor._session.store
    state = cursor.snapshot()
    if state.prepared is not None:
        if state.prepared != prepared:
            raise RecoveryStateError("prepared run publication diverged")
        return cursor._accumulator.manifests[0]
    _verify_blob(store, prepared.runtime_record_ref, "runtime record")
    _verify_blob(store, prepared.input_bundle_ref, "input bundle")
    draft = cursor._session.draft_manifest(
        entries={
            "runtime_record": prepared.runtime_record_ref,
            "input_bundle": prepared.input_bundle_ref,
        },
        meta={
            "schema_version": _PREPARED_SCHEMA,
            "state_kind": _PREPARED,
            "prepared": prepared.payload(),
        },
    )
    return cursor.append(draft)


def publish_prepared_run(
    store: ContentAddressedRunStore,
    prepared: PreparedRecoveryRun,
    *,
    protocol: RecoveryScientificProtocol = REGISTERED_RECOVERY_PROTOCOL,
    execution_policy: RecoveryExecutionPolicy = REGISTERED_EXECUTION_POLICY,
) -> Manifest:
    """Publish the unique immutable root record for one recovery run."""
    cursor = RecoveryStateCursor.open(
        RunStoreSession.open(store), protocol=protocol, execution_policy=execution_policy
    )
    return publish_prepared_run_at(cursor, prepared)


def publish_execution_attempt_started_at(
    cursor: RecoveryStateCursor,
    start: ExecutionAttemptStart,
) -> Manifest:
    """Publish one clean invocation start, abandoning a prior unfinished invocation."""
    state = cursor.snapshot()
    prepared = state.prepared
    if prepared is None:
        raise RecoveryStateError("execution attempt requires a prepared run")
    if (
        start.scientific_identity_sha256 != prepared.scientific_identity_sha256
        or start.runtime_record_ref != prepared.runtime_record_ref
        or start.input_bundle_ref != prepared.input_bundle_ref
    ):
        raise RecoveryStateError("execution attempt does not match the prepared run")
    for attempt in state.execution_attempts:
        if attempt.start.attempt_id != start.attempt_id:
            continue
        if attempt.start != start:
            raise RecoveryStateError("execution attempt start publication diverged")
        for manifest in cursor.index.execution_attempt_manifests:
            if (
                manifest.meta.get("state_kind") == _ATTEMPT_STARTED
                and manifest.meta.get("attempt_id") == start.attempt_id
            ):
                return manifest
        raise RecoveryStateError("published execution attempt has no start manifest")
    if state.finalized_execution_attempt is not None:
        raise RecoveryStateError("no execution attempt may start after finalization")
    _verify_blob(cursor.store, start.runtime_record_ref, "attempt runtime record")
    _verify_blob(cursor.store, start.input_bundle_ref, "attempt input bundle")
    reference = cursor.store.put_json(start.payload())
    draft = cursor.draft_manifest(
        entries={"attempt_start": reference},
        meta={
            "schema_version": _ATTEMPT_STARTED_SCHEMA,
            "state_kind": _ATTEMPT_STARTED,
            "attempt_id": start.attempt_id,
            "record_ref": reference.payload(),
        },
    )
    return cursor.append(draft)


def publish_execution_attempt_completed_at(
    cursor: RecoveryStateCursor,
    completion: ExecutionAttemptCompletion,
) -> Manifest:
    """Publish the completion of the latest open invocation."""
    state = cursor.snapshot()
    latest = state.execution_attempts[-1] if state.execution_attempts else None
    if latest is None:
        raise RecoveryStateError("execution completion requires the latest open start")
    if latest.completion is not None:
        if latest.completion != completion:
            raise RecoveryStateError("execution attempt completion publication diverged")
        manifests = cursor.index.execution_attempt_manifests
        if not manifests or manifests[-1].meta.get("state_kind") != _ATTEMPT_COMPLETED:
            raise RecoveryStateError("published execution completion has no manifest")
        return manifests[-1]
    if latest.abandoned or completion.attempt_id != latest.start.attempt_id:
        raise RecoveryStateError("execution completion requires the latest open start")
    if completion.start_ref != latest.start_ref:
        raise RecoveryStateError("execution completion start reference diverged")
    if completion.commit_sha != latest.start.commit_sha:
        raise RecoveryStateError("execution completion commit diverged from its start")
    if _parse_utc_timestamp(completion.completed_utc, "completed_utc") < _parse_utc_timestamp(
        latest.start.started_utc, "started_utc"
    ):
        raise RecoveryStateError("execution-attempt completion precedes its start")
    if len(state.completed_cells) != state.expected_cell_count or state.active_cell is not None:
        raise RecoveryStateError("execution completion requires every cell and no active LASSO fit")
    _verify_blob(cursor.store, completion.start_ref, "execution attempt start")
    reference = cursor.store.put_json(completion.payload())
    draft = cursor.draft_manifest(
        entries={"attempt_completion": reference},
        meta={
            "schema_version": _ATTEMPT_COMPLETED_SCHEMA,
            "state_kind": _ATTEMPT_COMPLETED,
            "attempt_id": completion.attempt_id,
            "record_ref": reference.payload(),
        },
    )
    return cursor.append(draft)


def publish_selection_plan_at(
    cursor: RecoveryStateCursor,
    selection: PublishedSelectionPlan,
) -> Manifest:
    """Publish a selection plan through an existing incremental cursor."""
    store = cursor._session.store
    state = cursor.snapshot()
    protocol = cursor._accumulator.protocol
    if state.prepared is None:
        raise RecoveryStateError("selection plan requires a prepared run")
    if state.selection_plan is not None:
        if state.selection_plan != selection:
            raise RecoveryStateError("selection plan publication diverged")
        manifest = cursor.index.selection_manifest
        if manifest is None:
            raise RecoveryStateError("published selection plan has no manifest")
        return manifest
    if state.doptimal_completed != protocol.budgets[-1]:
        raise RecoveryStateError("selection plan requires complete D-optimal selection")
    if selection.scientific_identity_sha256 != state.prepared.scientific_identity_sha256:
        raise RecoveryStateError("selection-plan scientific identity does not match")
    _verify_blob(store, selection.plan_ref, "selection plan")
    draft = cursor._session.draft_manifest(
        entries={"selection_plan": selection.plan_ref},
        meta={
            "schema_version": _SELECTION_SCHEMA,
            "state_kind": _SELECTION,
            "selection": selection.payload(),
        },
    )
    return cursor.append(draft)


def publish_selection_plan(
    store: ContentAddressedRunStore,
    selection: PublishedSelectionPlan,
    *,
    protocol: RecoveryScientificProtocol = REGISTERED_RECOVERY_PROTOCOL,
    execution_policy: RecoveryExecutionPolicy = REGISTERED_EXECUTION_POLICY,
) -> Manifest:
    """Publish the label-independent plan after complete D-optimal selection."""
    cursor = RecoveryStateCursor.open(
        RunStoreSession.open(store), protocol=protocol, execution_policy=execution_policy
    )
    return publish_selection_plan_at(cursor, selection)


def publish_recovery_cell_at(  # noqa: PLR0912
    cursor: RecoveryStateCursor,
    key: RecoveryCellKey,
    *,
    valid: bool,
    metrics: Mapping[str, object] | None,
    error: str | None,
) -> Manifest:
    """Publish one cell through an existing incremental cursor."""
    store = cursor._session.store
    protocol = cursor._accumulator.protocol
    if type(valid) is not bool:
        raise ValueError("recovery-cell valid flag must be boolean")
    if valid:
        if not isinstance(metrics, Mapping) or not metrics or error is not None:
            raise ValueError("valid recovery cell requires metrics and no error")
    elif metrics is not None or type(error) is not str or not error:
        raise ValueError("invalid recovery cell requires null metrics and a non-empty error")
    metrics_payload = None if metrics is None else dict(metrics)
    _validate_public_json(metrics_payload, "recovery-cell metrics")
    result = {
        "schema_version": _CELL_RESULT_SCHEMA,
        "cell": key.payload(),
        "valid": valid,
        "metrics": metrics_payload,
        "error": error,
    }
    accumulator = cursor._accumulator
    prepared = accumulator.prepared
    selection = accumulator.selection
    if prepared is None or selection is None or not accumulator.manifests:
        raise RecoveryStateError("recovery cell requires a published selection plan")
    expected = accumulator.cell_keys
    completed_count = len(accumulator.completed_cells)
    if completed_count >= len(expected) or key != expected[completed_count]:
        if key not in accumulator.completed_cells:
            raise RecoveryStateError("recovery cell does not follow the registered order")
        manifest = cursor.manifest_for_cell(key)
        if manifest is None:
            raise RecoveryStateError("published recovery cell has no manifest")
        try:
            existing = store.get_json(manifest.entry("result"))
        except (KeyError, RunStoreError, ValueError) as error_value:
            raise RecoveryStateError("published recovery cell cannot be verified") from error_value
        if canonical_json_bytes(existing) == canonical_json_bytes(result):
            return manifest
        raise RecoveryStateError("recovery cell publication diverged")
    lasso = accumulator.lasso
    if lasso.key is not None and lasso.key != key:
        raise RecoveryStateError("recovery cell does not match the active LASSO fit")
    if valid and lasso.completed_folds != protocol.n_folds:
        raise RecoveryStateError("valid recovery cell requires every LASSO fold")
    if valid and not lasso.converged:
        raise RecoveryStateError("valid recovery cell requires a converged LASSO fit")
    if valid:
        _require_valid_metrics(
            metrics_payload,
            key,
            protocol,
            selected_sha256=lasso.selected_sha256,
            fold_sha256=lasso.fold_sha256,
            lambda_ratios=lasso.lambda_ratios,
        )
    result_ref = store.put_json(result)
    draft = cursor._session.draft_manifest(
        entries={"result": result_ref},
        meta={
            "schema_version": _CELL_SCHEMA,
            "state_kind": _CELL,
            "scientific_identity_sha256": prepared.scientific_identity_sha256,
            "selection_plan_sha256": selection.selection_plan_sha256,
            "cell": key.payload(),
            "valid": valid,
            "error": error,
            "result_ref": result_ref.payload(),
        },
    )
    return cursor.append(draft)


def publish_recovery_cell(
    store: ContentAddressedRunStore,
    key: RecoveryCellKey,
    *,
    valid: bool,
    metrics: Mapping[str, object] | None,
    error: str | None,
    protocol: RecoveryScientificProtocol = REGISTERED_RECOVERY_PROTOCOL,
    execution_policy: RecoveryExecutionPolicy = REGISTERED_EXECUTION_POLICY,
) -> Manifest:
    """Publish one valid or explicitly invalid recovery cell in registered order."""
    cursor = RecoveryStateCursor.open(
        RunStoreSession.open(store), protocol=protocol, execution_policy=execution_policy
    )
    return publish_recovery_cell_at(cursor, key, valid=valid, metrics=metrics, error=error)


def publish_recovery_report_at(
    cursor: RecoveryStateCursor,
    report: Mapping[str, object],
) -> Manifest:
    """Publish the final report through an existing incremental cursor."""
    store = cursor._session.store
    protocol = cursor._accumulator.protocol
    report_payload = dict(report)
    _validate_public_json(report_payload, "recovery report")
    state = cursor.snapshot()
    if (
        state.prepared is None
        or state.selection_plan is None
        or state.latest_manifest is None
        or len(state.completed_cells) != protocol.cell_count
    ):
        raise RecoveryStateError("recovery report requires every registered cell")
    if state.active_cell is not None:
        raise RecoveryStateError("recovery report cannot close an active LASSO cell")
    if (
        not state.execution_attempts
        or state.execution_attempts[-1].completion is None
        or state.execution_attempts[-1].abandoned
    ):
        raise RecoveryStateError("recovery report requires a completed latest execution attempt")
    expected_cells = _report_cells_from_results(
        store, state.completed_cells, state.cell_result_refs, protocol
    )
    if canonical_json_bytes(report_payload.get("cells")) != canonical_json_bytes(expected_cells):
        raise RecoveryStateError("recovery report cells do not match the verified cell results")
    cell_results_sha256 = _cell_results_sha256(state.completed_cells, state.cell_result_refs)
    supplied_cell_results = report_payload.get("cell_results_sha256")
    if supplied_cell_results is not None and supplied_cell_results != cell_results_sha256:
        raise RecoveryStateError("recovery report cell-results identity diverged")
    report_payload["cell_results_sha256"] = cell_results_sha256
    if state.report_ref is not None:
        existing = store.get_json(state.report_ref)
        if canonical_json_bytes(existing) == canonical_json_bytes(report_payload):
            manifest = cursor.index.report_manifest
            if manifest is None:
                raise RecoveryStateError("published report has no manifest")
            return manifest
        raise RecoveryStateError("recovery report is already published")
    report_ref = store.put_json(report_payload)
    draft = cursor._session.draft_manifest(
        entries={"report": report_ref},
        meta={
            "schema_version": _REPORT_SCHEMA,
            "state_kind": _REPORT,
            "scientific_identity_sha256": state.prepared.scientific_identity_sha256,
            "selection_plan_sha256": state.selection_plan.selection_plan_sha256,
            "cell_results_sha256": cell_results_sha256,
            "report_ref": report_ref.payload(),
        },
    )
    return cursor.append(draft)


def publish_recovery_report(
    store: ContentAddressedRunStore,
    report: Mapping[str, object],
    *,
    protocol: RecoveryScientificProtocol = REGISTERED_RECOVERY_PROTOCOL,
    execution_policy: RecoveryExecutionPolicy = REGISTERED_EXECUTION_POLICY,
) -> Manifest:
    """Publish the final report only after every registered cell is durably closed."""
    cursor = RecoveryStateCursor.open(
        RunStoreSession.open(store), protocol=protocol, execution_policy=execution_policy
    )
    return publish_recovery_report_at(cursor, report)
