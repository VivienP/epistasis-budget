"""Durable checkpoints for long-running Fourier recovery diagnostics."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import platform
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

import numpy as np
import scipy

from epibudget.fourier_recovery import (
    RecoveryCell,
    SelectionPlan,
    SelectionSequence,
    _fold_sha256,
    _sequence_sha256,
)
from epibudget.recovery_protocol import (
    REGISTERED_EXECUTION_POLICY,
    REGISTERED_RECOVERY_PROTOCOL,
)
from epibudget.run_store import canonical_json_bytes
from epibudget.scored_cache import candidate_sha256
from epibudget.tie_break import canonical_id
from epibudget.types import Variant

_MARKER_SCHEMA = "epibudget-checkpoint-marker-v1"
_MUTATION_WIDTH = 3
_THREAD_ENVIRONMENT = ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
_SELECTION_SCHEMA = "epibudget-fourier-selection-checkpoint-v1"
_BLOCK_SCHEMA = "epibudget-fourier-recovery-block-v1"
_MINIMUM_FOLD_COUNT = 2
_REGISTERED_SEQUENCE_COUNT = REGISTERED_RECOVERY_PROTOCOL.sequence_count
_REGISTERED_CELL_COUNT = REGISTERED_RECOVERY_PROTOCOL.cell_count
REGISTERED_BUDGET_BLOCKS = REGISTERED_EXECUTION_POLICY.budget_blocks(
    REGISTERED_RECOVERY_PROTOCOL.budgets
)
_RECOVERY_CELL_FIELDS = frozenset(field.name for field in fields(RecoveryCell))


@dataclass(frozen=True)
class RecoveryIdentity:
    """Immutable execution and protocol identity shared by recovery checkpoints."""

    execution_commit: str
    candidate_sha256: str
    input_hashes: Mapping[str, str]
    cache_identity: Mapping[str, object]
    numerical_fingerprint: Mapping[str, object]
    protocol: Mapping[str, object]


@dataclass(frozen=True)
class RecoveryBlockWork:
    """One missing four-budget block in canonical execution order."""

    method: str
    seed: int | None
    block_index: int
    budgets: tuple[int, ...]


def _captured_output(function: Callable[[], object]) -> str:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        function()
    return stream.getvalue().strip()


def numerical_fingerprint() -> dict[str, object]:
    """Return the numerical runtime identity required for checkpoint reuse."""
    config = _captured_output(np.__config__.show)
    show_runtime = getattr(np, "show_runtime", None)
    runtime = _captured_output(show_runtime) if callable(show_runtime) else "unavailable"
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "operating_system": platform.platform(),
        "machine": {
            "architecture": platform.machine(),
            "processor": platform.processor(),
        },
        "blas_config_sha256": hashlib.sha256(config.encode("utf-8")).hexdigest(),
        "blas_runtime": runtime,
        "environment_threads": {name: os.environ.get(name) for name in _THREAD_ENVIRONMENT},
    }


def _checkpoint_prefix(kind: str, key: str, digest: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789_-"
    if not kind or any(character not in allowed for character in kind):
        raise ValueError("checkpoint kind must use lowercase ASCII letters, digits, '_' or '-'")
    key_token = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"{kind}--{key_token}--{digest}"


def _copy_bytes_once(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"checkpoint path already exists with different content: {path}")
        return
    with tempfile.TemporaryDirectory(prefix="epibudget-checkpoint-") as temporary:
        local_path = Path(temporary) / path.name
        local_path.write_bytes(content)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, path)


def publish_checkpoint(directory: Path, kind: str, key: str, payload: Mapping[str, object]) -> str:
    """Publish one immutable content-addressed payload and completion marker."""
    payload_bytes = canonical_json_bytes(payload)
    digest = hashlib.sha256(payload_bytes).hexdigest()
    prefix = _checkpoint_prefix(kind, key, digest)
    payload_path = directory / f"{prefix}.payload.json"
    marker_path = directory / f"{prefix}.complete.json"

    if marker_path.exists():
        marker = _load_marker(marker_path, kind)
        if marker is not None:
            completed = discover_checkpoints(directory, kind)
            if completed.get(key) == dict(payload):
                return digest
            raise ValueError(f"completed checkpoint conflicts with key {key!r}")
        marker_path.unlink()
    if payload_path.exists() and payload_path.read_bytes() != payload_bytes:
        payload_path.unlink()
    _copy_bytes_once(payload_path, payload_bytes)
    copied = payload_path.read_bytes()
    if hashlib.sha256(copied).hexdigest() != digest or copied != payload_bytes:
        raise ValueError(f"checkpoint payload digest mismatch after copy: {payload_path}")

    marker = {
        "schema_version": _MARKER_SCHEMA,
        "kind": kind,
        "key": key,
        "payload_filename": payload_path.name,
        "payload_sha256": digest,
    }
    _copy_bytes_once(marker_path, canonical_json_bytes(marker))
    return digest


def _load_marker(path: Path, kind: str) -> dict[str, str] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    expected_fields = {
        "schema_version",
        "kind",
        "key",
        "payload_filename",
        "payload_sha256",
    }
    if set(value) != expected_fields or value.get("schema_version") != _MARKER_SCHEMA:
        return None
    if value.get("kind") != kind or not all(isinstance(value.get(field), str) for field in value):
        return None
    return cast("dict[str, str]", value)


def discover_checkpoints(directory: Path, kind: str) -> dict[str, dict[str, object]]:
    """Load every completed checkpoint of one kind and reject marked corruption."""
    if not directory.exists():
        return {}
    discovered: dict[str, tuple[str, dict[str, object]]] = {}
    for marker_path in sorted(directory.glob(f"{kind}--*.complete.json")):
        marker = _load_marker(marker_path, kind)
        if marker is None:
            continue
        expected_prefix = _checkpoint_prefix(kind, marker["key"], marker["payload_sha256"])
        if marker_path.name != f"{expected_prefix}.complete.json":
            raise ValueError(
                f"checkpoint marker filename does not match its content: {marker_path}"
            )
        payload_name = marker["payload_filename"]
        if Path(payload_name).name != payload_name:
            raise ValueError(f"checkpoint marker contains an unsafe payload path: {marker_path}")
        if payload_name != f"{expected_prefix}.payload.json":
            raise ValueError(
                f"checkpoint payload filename does not match its marker: {marker_path}"
            )
        payload_path = directory / payload_name
        if not payload_path.is_file():
            raise ValueError(f"checkpoint marker references a missing payload: {marker_path}")
        payload_bytes = payload_path.read_bytes()
        digest = hashlib.sha256(payload_bytes).hexdigest()
        if digest != marker["payload_sha256"]:
            raise ValueError(f"checkpoint payload digest mismatch: {payload_path}")
        try:
            decoded = json.loads(payload_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"checkpoint payload is invalid JSON: {payload_path}") from error
        if not isinstance(decoded, dict) or canonical_json_bytes(decoded) != payload_bytes:
            raise ValueError(f"checkpoint payload is not a canonical JSON object: {payload_path}")
        payload = cast("dict[str, object]", decoded)
        key = marker["key"]
        existing = discovered.get(key)
        if existing is not None and existing[0] != digest:
            raise ValueError(f"conflicting completed checkpoints for key {key!r}")
        discovered[key] = (digest, payload)
    return {key: payload for key, (_digest, payload) in discovered.items()}


def _identity_payload(identity: RecoveryIdentity) -> dict[str, object]:
    return {
        "execution_commit": identity.execution_commit,
        "candidate_sha256": identity.candidate_sha256,
        "input_hashes": dict(identity.input_hashes),
        "cache_identity": dict(identity.cache_identity),
        "numerical_fingerprint": dict(identity.numerical_fingerprint),
        "protocol": dict(identity.protocol),
    }


def _selection_plan_body(plan: SelectionPlan) -> dict[str, object]:
    return {
        "budgets": list(plan.budgets),
        "sequences": [
            {
                "method": sequence.method,
                "seed": sequence.seed,
                "selected_ids": [canonical_id(variant) for variant in sequence.selected],
                "selected_sha256": sequence.selected_sha256,
                "tie_break_version": sequence.tie_break_version,
            }
            for sequence in plan.sequences
        ],
    }


def selection_plan_payload(plan: SelectionPlan, identity: RecoveryIdentity) -> dict[str, object]:
    """Serialize a complete label-free selection plan for durable publication."""
    body = _selection_plan_body(plan)
    return {
        "schema_version": _SELECTION_SCHEMA,
        "identity": _identity_payload(identity),
        "selection_plan_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
        "plan": body,
    }


def _variant_from_canonical_id(value: object) -> Variant:
    if not isinstance(value, str):
        raise ValueError("selection plan contains a non-string candidate identity")
    try:
        decoded: Any = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("selection plan contains an invalid candidate identity") from error
    if not isinstance(decoded, list):
        raise ValueError("selection plan candidate identity must contain a mutation list")
    mutations: list[tuple[int, str, str]] = []
    for mutation in decoded:
        if (
            not isinstance(mutation, list)
            or len(mutation) != _MUTATION_WIDTH
            or type(mutation[0]) is not int
            or not isinstance(mutation[1], str)
            or not isinstance(mutation[2], str)
            or len(mutation[1]) != 1
            or len(mutation[2]) != 1
        ):
            raise ValueError("selection plan contains a malformed mutation identity")
        mutations.append((mutation[0], mutation[1], mutation[2]))
    variant = frozenset(mutations)
    if len(variant) != len(mutations) or canonical_id(variant) != value:
        raise ValueError("selection plan candidate identity is not canonical")
    return variant


def _require_mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"selection plan {field} must be a JSON object")
    return cast("dict[str, object]", value)


def _decode_selection_sequence(
    value: object,
    *,
    candidate_by_id: Mapping[str, Variant],
    maximum_budget: int,
) -> SelectionSequence:
    row = _require_mapping(value, "sequence")
    method = row.get("method")
    seed = row.get("seed")
    selected_ids = row.get("selected_ids")
    selected_sha256 = row.get("selected_sha256")
    tie_break_version = row.get("tie_break_version")
    if (
        not isinstance(method, str)
        or not (seed is None or type(seed) is int)
        or not isinstance(selected_ids, list)
        or not isinstance(selected_sha256, str)
        or not isinstance(tie_break_version, str)
    ):
        raise ValueError("selection plan sequence fields are invalid")
    decoded = tuple(_variant_from_canonical_id(identity) for identity in selected_ids)
    if len(set(decoded)) != len(decoded):
        raise ValueError("selection plan sequence contains duplicate candidates")
    if any(canonical_id(variant) not in candidate_by_id for variant in decoded):
        raise ValueError("selection plan sequence contains an unexpected candidate")
    selected = tuple(candidate_by_id[canonical_id(variant)] for variant in decoded)
    if selected_sha256 != _sequence_sha256(selected):
        raise ValueError("selection plan sequence hash does not match")
    if len(selected) < maximum_budget:
        raise ValueError("selection plan sequence does not cover the maximum budget")
    return SelectionSequence(
        method=method,
        seed=seed,
        selected=selected,
        selected_sha256=selected_sha256,
        tie_break_version=tie_break_version,
    )


def _registered_sequence_keys(protocol: Mapping[str, object]) -> list[tuple[str, int | None]]:
    value = protocol.get("sequence_keys")
    if not isinstance(value, list):
        raise ValueError("selection plan protocol has no registered sequence order")
    registered: list[tuple[str, int | None]] = []
    for item in value:
        row = _require_mapping(item, "registered sequence")
        if set(row) != {"method", "seed"}:
            raise ValueError("selection plan registered sequence fields are invalid")
        method = row.get("method")
        seed = row.get("seed")
        if not isinstance(method, str) or not (seed is None or type(seed) is int):
            raise ValueError("selection plan registered sequence identity is invalid")
        registered.append((method, seed))
    return registered


def load_selection_plan(
    payload: Mapping[str, object],
    *,
    expected: RecoveryIdentity,
    expected_candidates: Sequence[Variant],
) -> SelectionPlan:
    """Validate and decode one selection plan without accepting measured labels."""
    if payload.get("schema_version") != _SELECTION_SCHEMA:
        raise ValueError("selection plan checkpoint schema does not match")
    observed_identity = _require_mapping(payload.get("identity"), "identity")
    if canonical_json_bytes(observed_identity) != canonical_json_bytes(_identity_payload(expected)):
        raise ValueError("selection plan checkpoint identity does not match")
    candidates = tuple(expected_candidates)
    if candidate_sha256(candidates) != expected.candidate_sha256:
        raise ValueError("selection plan expected candidate universe does not match")
    candidate_by_id = {canonical_id(variant): variant for variant in candidates}
    if len(candidate_by_id) != len(candidates):
        raise ValueError("selection plan expected candidates contain duplicate identities")

    body = _require_mapping(payload.get("plan"), "body")
    expected_hash = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if payload.get("selection_plan_sha256") != expected_hash:
        raise ValueError("selection plan checkpoint hash does not match")
    budgets_value = body.get("budgets")
    if not isinstance(budgets_value, list) or not all(
        type(budget) is int and budget > 0 for budget in budgets_value
    ):
        raise ValueError("selection plan budgets are invalid")
    budgets = tuple(cast("list[int]", budgets_value))
    if tuple(sorted(set(budgets))) != budgets:
        raise ValueError("selection plan budgets are not strictly increasing")
    if budgets_value != expected.protocol.get("budgets"):
        raise ValueError("selection plan does not match the registered budgets")

    sequences_value = body.get("sequences")
    if not isinstance(sequences_value, list):
        raise ValueError("selection plan sequences must be a list")
    sequences: list[SelectionSequence] = []
    observed_keys: set[tuple[str, int | None]] = set()
    for value in sequences_value:
        sequence = _decode_selection_sequence(
            value,
            candidate_by_id=candidate_by_id,
            maximum_budget=budgets[-1],
        )
        key = (sequence.method, sequence.seed)
        if key in observed_keys:
            raise ValueError(f"selection plan contains duplicate sequence {key!r}")
        observed_keys.add(key)
        sequences.append(sequence)
    registered_keys = _registered_sequence_keys(expected.protocol)
    if [(sequence.method, sequence.seed) for sequence in sequences] != registered_keys:
        raise ValueError("selection plan sequences do not match the registered order")
    return SelectionPlan(budgets=budgets, sequences=tuple(sequences))


def recovery_block_key(method: str, seed: int | None, block_index: int) -> str:
    """Return the stable identity key for one four-budget recovery block."""
    seed_token = "none" if seed is None else str(seed)
    return f"{method}:{seed_token}:{block_index}"


def pending_recovery_blocks(
    plan: SelectionPlan, *, completed_keys: set[str]
) -> tuple[RecoveryBlockWork, ...]:
    """Return missing blocks in selection-plan order without reading measured labels."""
    registered_budgets = tuple(budget for block in REGISTERED_BUDGET_BLOCKS for budget in block)
    if plan.budgets != registered_budgets:
        raise ValueError("selection plan budgets do not match the registered recovery blocks")
    expected_keys = {
        recovery_block_key(sequence.method, sequence.seed, block_index)
        for sequence in plan.sequences
        for block_index in range(len(REGISTERED_BUDGET_BLOCKS))
    }
    unexpected = sorted(completed_keys - expected_keys)
    if unexpected:
        raise ValueError(f"completed recovery checkpoints contain unexpected keys: {unexpected}")
    return tuple(
        RecoveryBlockWork(
            method=sequence.method,
            seed=sequence.seed,
            block_index=block_index,
            budgets=budgets,
        )
        for sequence in plan.sequences
        for block_index, budgets in enumerate(REGISTERED_BUDGET_BLOCKS)
        if recovery_block_key(sequence.method, sequence.seed, block_index) not in completed_keys
    )


def execute_pending_blocks(
    pending: Sequence[RecoveryBlockWork],
    *,
    completed_cell_count: int,
    execute_block: Callable[[RecoveryBlockWork], int],
    report_progress: Callable[[RecoveryBlockWork, int], None],
) -> int:
    """Execute missing blocks while enforcing four-cell durable progress."""
    if type(completed_cell_count) is not int or completed_cell_count < 0:
        raise ValueError("completed recovery cell count is invalid")
    if completed_cell_count % len(REGISTERED_BUDGET_BLOCKS[0]) != 0:
        raise ValueError("completed recovery cell count is not block-aligned")
    current = completed_cell_count
    for work in pending:
        observed = execute_block(work)
        expected = current + len(work.budgets)
        if observed != expected:
            raise ValueError(
                f"recovery checkpoint progress must advance from {current} to {expected}, "
                f"observed {observed}"
            )
        current = observed
        report_progress(work, current)
    return current


def _validate_block_identity(
    *,
    identity: RecoveryIdentity,
    workspace_start: Mapping[str, object],
    workspace_end: Mapping[str, object],
    input_hashes_start: Mapping[str, str],
    input_hashes_end: Mapping[str, str],
) -> None:
    if workspace_start != workspace_end or input_hashes_start != input_hashes_end:
        raise ValueError("recovery block input or workspace drift detected")
    if dict(input_hashes_start) != dict(identity.input_hashes):
        raise ValueError("recovery block input hashes do not match its identity")
    if workspace_start.get("execution_commit") != identity.execution_commit:
        raise ValueError("recovery block execution commit does not match")
    if workspace_start.get("code_state") != "clean":
        raise ValueError("recovery block workspace must be clean")


def _validate_block_cells(
    method: str,
    seed: int | None,
    block_index: int,
    cells: Sequence[RecoveryCell],
) -> tuple[RecoveryCell, ...]:
    if type(block_index) is not int or block_index not in range(len(REGISTERED_BUDGET_BLOCKS)):
        raise ValueError(f"recovery block index {block_index} is not registered")
    expected_budgets = REGISTERED_BUDGET_BLOCKS[block_index]
    observed = tuple(cells)
    if tuple(cell.budget for cell in observed) != expected_budgets:
        raise ValueError("recovery block must contain exactly four canonical budgets")
    if any(cell.method != method or cell.seed != seed for cell in observed):
        raise ValueError("recovery block cell identity does not match its sequence")
    return observed


def recovery_block_payload(
    *,
    identity: RecoveryIdentity,
    selection_plan_sha256: str,
    method: str,
    seed: int | None,
    block_index: int,
    cells: Sequence[RecoveryCell],
    workspace_start: Mapping[str, object],
    workspace_end: Mapping[str, object],
    input_hashes_start: Mapping[str, str],
    input_hashes_end: Mapping[str, str],
    session_id: str,
    started_utc: str,
    completed_utc: str,
) -> dict[str, object]:
    """Build one validated four-budget recovery checkpoint payload."""
    validated_cells = _validate_block_cells(method, seed, block_index, cells)
    _validate_block_identity(
        identity=identity,
        workspace_start=workspace_start,
        workspace_end=workspace_end,
        input_hashes_start=input_hashes_start,
        input_hashes_end=input_hashes_end,
    )
    return {
        "schema_version": _BLOCK_SCHEMA,
        "identity": _identity_payload(identity),
        "selection_plan_sha256": selection_plan_sha256,
        "block": {
            "method": method,
            "seed": seed,
            "block_index": block_index,
            "budgets": list(REGISTERED_BUDGET_BLOCKS[block_index]),
            "cells": [asdict(cell) for cell in validated_cells],
        },
        "provenance": {
            "session_id": session_id,
            "started_utc": started_utc,
            "completed_utc": completed_utc,
            "workspace_start": dict(workspace_start),
            "workspace_end": dict(workspace_end),
            "input_hashes_start": dict(input_hashes_start),
            "input_hashes_end": dict(input_hashes_end),
        },
    }


def _validate_cell_metrics(row: Mapping[str, object]) -> None:
    for name in ("spearman", "relative_sse_gain", "lambda_ratio", "lambda_value"):
        number = row.get(name)
        if number is not None and (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not np.isfinite(number)
        ):
            raise ValueError(f"recovery block cell {name} is invalid")
    spearman = row.get("spearman")
    if spearman is not None and not -1.0 <= cast("float", spearman) <= 1.0:
        raise ValueError("recovery block cell spearman is outside [-1, 1]")


def _recovery_cell_from_payload(value: object, *, expected_coefficient_count: int) -> RecoveryCell:
    row = _require_mapping(value, "recovery cell")
    if set(row) != _RECOVERY_CELL_FIELDS:
        raise ValueError("recovery block cell fields do not match the schema")
    method = row.get("method")
    budget = row.get("budget")
    seed = row.get("seed")
    if not isinstance(method, str) or type(budget) is not int:
        raise ValueError("recovery block cell method or budget is invalid")
    if seed is not None and type(seed) is not int:
        raise ValueError("recovery block cell seed is invalid")
    for name in ("support_size", "coefficient_count"):
        if type(row.get(name)) is not int:
            raise ValueError(f"recovery block cell {name} is invalid")
    if cast("int", row["support_size"]) < 0:
        raise ValueError("recovery block cell support size is invalid")
    if row["coefficient_count"] != expected_coefficient_count:
        raise ValueError("recovery block cell coefficient count does not match the estimand")
    for name in ("selected_sha256", "fold_sha256"):
        if not isinstance(row.get(name), str):
            raise ValueError(f"recovery block cell {name} is invalid")
    if not isinstance(row.get("converged"), bool):
        raise ValueError("recovery block cell convergence flag is invalid")
    error = row.get("error")
    if error is not None and not isinstance(error, str):
        raise ValueError("recovery block cell error is invalid")
    _validate_cell_metrics(row)
    return RecoveryCell(**cast("dict[str, Any]", row))


def _selection_sequence_map(
    plan: SelectionPlan,
) -> dict[tuple[str, int | None], SelectionSequence]:
    registered_budgets = tuple(budget for block in REGISTERED_BUDGET_BLOCKS for budget in block)
    if plan.budgets != registered_budgets:
        raise ValueError("recovery block selection plan has unregistered budgets")
    sequence_by_key = {(sequence.method, sequence.seed): sequence for sequence in plan.sequences}
    if len(sequence_by_key) != len(plan.sequences):
        raise ValueError("recovery block selection plan has duplicate sequences")
    if any(len(sequence.selected) < registered_budgets[-1] for sequence in plan.sequences):
        raise ValueError("recovery block selection plan does not cover the maximum budget")
    return sequence_by_key


def _validate_cells_against_selection(
    cells: Sequence[RecoveryCell],
    *,
    sequence: SelectionSequence,
    n_folds: int,
) -> None:
    for cell in cells:
        selected = sequence.selected[: cell.budget]
        if cell.selected_sha256 != _sequence_sha256(selected):
            raise ValueError("recovery block cell does not match its selection prefix")
        if cell.fold_sha256 != _fold_sha256(selected, n_folds):
            raise ValueError("recovery block cell does not match its registered folds")


def load_recovery_blocks(
    payloads: Mapping[str, Mapping[str, object]],
    *,
    expected_identity: RecoveryIdentity,
    expected_selection_plan_sha256: str,
    expected_plan: SelectionPlan,
) -> dict[str, tuple[RecoveryCell, ...]]:
    """Validate completed recovery payloads against one exact execution identity."""
    sequence_by_key = _selection_sequence_map(expected_plan)
    n_folds = expected_identity.protocol.get("folds")
    coefficient_count = expected_identity.protocol.get("coefficient_count")
    if type(n_folds) is not int or n_folds < _MINIMUM_FOLD_COUNT:
        raise ValueError("recovery block protocol fold count is invalid")
    if type(coefficient_count) is not int or coefficient_count < 1:
        raise ValueError("recovery block protocol coefficient count is invalid")
    loaded: dict[str, tuple[RecoveryCell, ...]] = {}
    for observed_key, payload in payloads.items():
        if payload.get("schema_version") != _BLOCK_SCHEMA:
            raise ValueError(f"recovery block schema does not match for {observed_key!r}")
        identity = _require_mapping(payload.get("identity"), "block identity")
        if canonical_json_bytes(identity) != canonical_json_bytes(
            _identity_payload(expected_identity)
        ):
            raise ValueError(f"recovery block identity does not match for {observed_key!r}")
        if payload.get("selection_plan_sha256") != expected_selection_plan_sha256:
            raise ValueError(f"recovery block selection plan does not match for {observed_key!r}")
        block = _require_mapping(payload.get("block"), "block")
        provenance = _require_mapping(payload.get("provenance"), "block provenance")
        method = block.get("method")
        seed = block.get("seed")
        block_index = block.get("block_index")
        if (
            not isinstance(method, str)
            or not (seed is None or type(seed) is int)
            or type(block_index) is not int
        ):
            raise ValueError(f"recovery block key fields are invalid for {observed_key!r}")
        expected_key = recovery_block_key(method, seed, block_index)
        if observed_key != expected_key:
            raise ValueError(f"recovery block key does not match for {observed_key!r}")
        budgets = block.get("budgets")
        if budgets != list(REGISTERED_BUDGET_BLOCKS[block_index]):
            raise ValueError(f"recovery block budgets do not match for {observed_key!r}")
        cells_value = block.get("cells")
        if not isinstance(cells_value, list):
            raise ValueError(f"recovery block cells are invalid for {observed_key!r}")
        cells = tuple(
            _recovery_cell_from_payload(
                value,
                expected_coefficient_count=coefficient_count,
            )
            for value in cells_value
        )
        cells = _validate_block_cells(method, seed, block_index, cells)
        sequence = sequence_by_key.get((method, seed))
        if sequence is None:
            raise ValueError(f"recovery block has no registered sequence for {observed_key!r}")
        _validate_cells_against_selection(
            cells,
            sequence=sequence,
            n_folds=n_folds,
        )
        workspace_start = _require_mapping(provenance.get("workspace_start"), "workspace start")
        workspace_end = _require_mapping(provenance.get("workspace_end"), "workspace end")
        hashes_start = _require_mapping(provenance.get("input_hashes_start"), "input hashes start")
        hashes_end = _require_mapping(provenance.get("input_hashes_end"), "input hashes end")
        if not all(isinstance(value, str) for value in hashes_start.values()) or not all(
            isinstance(value, str) for value in hashes_end.values()
        ):
            raise ValueError(f"recovery block input hashes are invalid for {observed_key!r}")
        _validate_block_identity(
            identity=expected_identity,
            workspace_start=workspace_start,
            workspace_end=workspace_end,
            input_hashes_start=cast("dict[str, str]", hashes_start),
            input_hashes_end=cast("dict[str, str]", hashes_end),
        )
        loaded[observed_key] = cells
    return loaded


def assemble_recovery_cells(
    blocks: Mapping[str, Sequence[RecoveryCell]],
    *,
    expected_sequences: Sequence[tuple[str, int | None]],
) -> tuple[RecoveryCell, ...]:
    """Assemble exactly 344 cells in frozen sequence and budget order."""
    if len(expected_sequences) != _REGISTERED_SEQUENCE_COUNT or len(set(expected_sequences)) != len(
        expected_sequences
    ):
        raise ValueError("recovery assembly requires exactly 43 unique sequences")
    expected_keys = {
        recovery_block_key(method, seed, block_index)
        for method, seed in expected_sequences
        for block_index in range(len(REGISTERED_BUDGET_BLOCKS))
    }
    observed_keys = set(blocks)
    missing = sorted(expected_keys - observed_keys)
    unexpected = sorted(observed_keys - expected_keys)
    if missing:
        raise ValueError(f"recovery assembly has missing blocks: {missing}")
    if unexpected:
        raise ValueError(f"recovery assembly has unexpected blocks: {unexpected}")
    assembled: list[RecoveryCell] = []
    for method, seed in expected_sequences:
        for block_index in range(len(REGISTERED_BUDGET_BLOCKS)):
            cells = _validate_block_cells(
                method,
                seed,
                block_index,
                blocks[recovery_block_key(method, seed, block_index)],
            )
            assembled.extend(cells)
    if len(assembled) != _REGISTERED_CELL_COUNT:
        raise ValueError(f"recovery assembly expected 344 cells, found {len(assembled)}")
    return tuple(assembled)
