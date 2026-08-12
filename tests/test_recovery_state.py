# ruff: noqa: PLR2004

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from epibudget.recovery_protocol import RecoveryExecutionPolicy, RecoveryScientificProtocol
from epibudget.recovery_state import (
    ExecutionAttemptCompletion,
    ExecutionAttemptStart,
    PreparedRecoveryRun,
    PublishedSelectionPlan,
    RecoveryCellKey,
    RecoveryManifestIndex,
    RecoveryStateCursor,
    RecoveryStateError,
    publish_execution_attempt_completed_at,
    publish_execution_attempt_started_at,
    publish_prepared_run,
    publish_prepared_run_at,
    publish_recovery_cell,
    publish_recovery_cell_at,
    publish_recovery_report,
    publish_recovery_report_at,
    publish_selection_plan,
    publish_selection_plan_at,
    registered_cell_keys,
    replay_recovery_state,
)
from epibudget.run_store import (
    ContentAddressedRunStore,
    Manifest,
    ManifestDraft,
    RunStoreError,
    RunStoreSession,
    canonical_json_bytes,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


@pytest.fixture
def protocol() -> RecoveryScientificProtocol:
    return RecoveryScientificProtocol(
        version="test-protocol-v1",
        dataset="synthetic",
        deterministic_methods=("info",),
        stochastic_methods=("random",),
        budgets=(2, 4),
        seeds=(0,),
        n_folds=2,
        selection_max_order=3,
        estimation_max_order=2,
        coefficient_count=2,
        feature_count=3,
    )


@pytest.fixture
def policy() -> RecoveryExecutionPolicy:
    return RecoveryExecutionPolicy(
        version="test-execution-v1",
        doptimal_block_size=2,
        lasso_checkpoint_unit="fold",
        cell_checkpoint_unit="cell",
        scheduling="breadth-first-budget-major",
        manifest_format="epibudget-run-store-manifest-v1",
        array_format="epibudget-run-store-array-v1",
        legacy_budget_block_size=1,
    )


@pytest.fixture
def store(tmp_path: Path) -> ContentAddressedRunStore:
    root = tmp_path / "run"
    root.mkdir()
    value = ContentAddressedRunStore(root)
    value.initialise()
    return value


def _prepared(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> PreparedRecoveryRun:
    numeric_sha256 = hashlib.sha256(canonical_json_bytes({"probe": "synthetic"})).hexdigest()
    return PreparedRecoveryRun(
        scientific_identity_sha256=SHA_A,
        protocol_semantic_sha256=protocol.semantic_sha256,
        execution_policy_sha256=policy.policy_sha256,
        numerical_compatibility_sha256=numeric_sha256,
        candidate_sha256=SHA_C,
        runtime_record_ref=store.put_json({"runtime": "synthetic"}),
        input_bundle_ref=store.put_json({"inputs": "synthetic"}),
    )


def _attempt_start(
    prepared: PreparedRecoveryRun,
    attempt_id: str,
    started_utc: str,
) -> ExecutionAttemptStart:
    return ExecutionAttemptStart(
        attempt_id=attempt_id,
        scientific_identity_sha256=prepared.scientific_identity_sha256,
        runtime_record_ref=prepared.runtime_record_ref,
        input_bundle_ref=prepared.input_bundle_ref,
        commit_sha="1" * 40,
        workspace_clean=True,
        scientific_diff_sha256=None,
        argv=("python", "scripts/fourier_recovery_curve.py", "run"),
        started_utc=started_utc,
    )


def _complete_attempt_for_report(
    cursor: RecoveryStateCursor,
    prepared: PreparedRecoveryRun,
) -> None:
    start = _attempt_start(prepared, "final-attempt", "2026-08-11T20:00:00Z")
    manifest = publish_execution_attempt_started_at(cursor, start)
    publish_execution_attempt_completed_at(
        cursor,
        ExecutionAttemptCompletion(
            attempt_id=start.attempt_id,
            start_ref=manifest.entry("attempt_start"),
            commit_sha=start.commit_sha,
            workspace_clean=True,
            scientific_diff_sha256=None,
            completed_utc="2026-08-11T22:00:00Z",
        ),
    )


def _append_doptimal(
    store: ContentAddressedRunStore,
    prepared: PreparedRecoveryRun,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
    *,
    start: int,
    stop: int | bool,
    parent: Manifest,
    scientific_sha256: str | None = None,
    selected_values: tuple[int, ...] | None = None,
    prefix_sha256: str | None = None,
    candidate_universe_sha256: str | None = None,
    state_rows: int = 6,
    candidate_count: int = 6,
) -> Manifest:
    stop_value = int(stop)
    selected_array = (
        np.arange(start, stop_value, dtype=np.int64)
        if selected_values is None
        else np.asarray(selected_values, dtype=np.int64)
    )
    selected = store.put_array(selected_array)
    updates = store.put_array(np.zeros((state_rows, stop_value - start), dtype=np.float64))
    posterior = store.put_array(np.ones(state_rows, dtype=np.float64))
    identity = {
        "scientific_identity_sha256": scientific_sha256 or prepared.scientific_identity_sha256,
        "execution_policy": policy.identity_payload(),
        "numerical_compatibility": {"probe": "synthetic"},
        "numerical_compatibility_sha256": prepared.numerical_compatibility_sha256,
        "candidate_universe_sha256": candidate_universe_sha256 or prepared.candidate_sha256,
        "candidate_sequence_sha256": SHA_B,
        "candidate_count": candidate_count,
        "target_budget": protocol.budgets[-1],
        "geometry_sha256": SHA_D,
    }
    arrays = {
        "selected_indices": selected.payload(),
        "updates": updates.payload(),
        "posterior_variance": posterior.payload(),
    }
    return store.publish_manifest(
        entries={
            "selected_indices": selected.blob,
            "updates": updates.blob,
            "posterior_variance": posterior.blob,
        },
        meta={
            "schema_version": "epibudget-reduced-doptimal-delta-v2",
            "state_kind": "reduced_doptimal",
            "identity": identity,
            "start": start,
            "stop": stop,
            "prefix_sha256": prefix_sha256
            or hashlib.sha256(f"prefix-{stop_value}".encode()).hexdigest(),
            "arrays": arrays,
        },
        parent=parent,
    )


def _complete_selection(
    store: ContentAddressedRunStore,
    prepared: PreparedRecoveryRun,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> PublishedSelectionPlan:
    parent = publish_prepared_run(store, prepared, protocol=protocol, execution_policy=policy)
    for start in range(0, protocol.budgets[-1], policy.doptimal_block_size):
        parent = _append_doptimal(
            store,
            prepared,
            protocol,
            policy,
            start=start,
            stop=start + policy.doptimal_block_size,
            parent=parent,
        )
    plan_ref = store.put_json({"selections": ["synthetic"]})
    plan = PublishedSelectionPlan(
        scientific_identity_sha256=prepared.scientific_identity_sha256,
        selection_plan_sha256=plan_ref.sha256,
        plan_ref=plan_ref,
    )
    publish_selection_plan(store, plan, protocol=protocol, execution_policy=policy)
    return plan


def _append_fold(
    store: ContentAddressedRunStore,
    prepared: PreparedRecoveryRun,
    plan: PublishedSelectionPlan,
    key: RecoveryCellKey,
    policy: RecoveryExecutionPolicy,
    *,
    completed: int,
    n_folds: int,
    parent: Manifest | None = None,
    converged: bool = True,
) -> Manifest:
    cv_sse = store.put_array(np.asarray([float(completed), float(completed + 1)]))
    identity = {
        "scientific_identity_sha256": prepared.scientific_identity_sha256,
        "execution_policy": policy.identity_payload(),
        "numerical_compatibility": {"probe": "synthetic"},
        "numerical_compatibility_sha256": prepared.numerical_compatibility_sha256,
        "selection_plan_sha256": plan.selection_plan_sha256,
        "cell": key.payload(),
        "selected_sha256": SHA_C,
        "fold_sha256": SHA_D,
        "problem_sha256": "e" * 64,
        "n_folds": n_folds,
        "lambda_ratios": [1.0, 0.5],
    }
    return store.publish_manifest(
        entries={"cv_sse": cv_sse.blob},
        meta={
            "schema_version": "epibudget-pairwise-lasso-fold-v1",
            "state_kind": "pairwise_lasso_cv",
            "cell": key.payload(),
            "identity": identity,
            "completed_folds": completed,
            "converged": converged,
            "cv_sse": cv_sse.payload(),
        },
        parent=parent or store.latest_manifest(),
    )


def _complete_valid_cell(
    store: ContentAddressedRunStore,
    prepared: PreparedRecoveryRun,
    plan: PublishedSelectionPlan,
    key: RecoveryCellKey,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    for fold in range(1, protocol.n_folds + 1):
        _append_fold(
            store,
            prepared,
            plan,
            key,
            policy,
            completed=fold,
            n_folds=protocol.n_folds,
        )
    publish_recovery_cell(
        store,
        key,
        valid=True,
        metrics=_valid_metrics(key, protocol),
        error=None,
        protocol=protocol,
        execution_policy=policy,
    )


def _valid_metrics(
    key: RecoveryCellKey,
    protocol: RecoveryScientificProtocol,
    *,
    spearman: float = 0.25,
) -> dict[str, object]:
    return {
        "method": key.method,
        "budget": key.budget,
        "seed": key.seed,
        "spearman": spearman,
        "relative_sse_gain": 0.1,
        "support_size": 1,
        "coefficient_count": protocol.coefficient_count,
        "selected_sha256": SHA_C,
        "fold_sha256": SHA_D,
        "lambda_ratio": 0.5,
        "lambda_value": 0.25,
        "converged": True,
        "error": None,
    }


def _invalid_cell_draft(
    session: RunStoreSession,
    store: ContentAddressedRunStore,
    cursor: RecoveryStateCursor,
    key: RecoveryCellKey,
    error: str,
) -> ManifestDraft:
    state = cursor.snapshot()
    assert state.prepared is not None
    assert state.selection_plan is not None
    result = {
        "schema_version": "epibudget-recovery-cell-result-v1",
        "cell": key.payload(),
        "valid": False,
        "metrics": None,
        "error": error,
    }
    result_ref = store.put_json(result)
    return session.draft_manifest(
        entries={"result": result_ref},
        meta={
            "schema_version": "epibudget-recovery-cell-v1",
            "state_kind": "recovery_cell",
            "scientific_identity_sha256": state.prepared.scientific_identity_sha256,
            "selection_plan_sha256": state.selection_plan.selection_plan_sha256,
            "cell": key.payload(),
            "valid": False,
            "error": error,
            "result_ref": result_ref.payload(),
        },
    )


def _doptimal_draft(
    session: RunStoreSession,
    store: ContentAddressedRunStore,
    prepared: PreparedRecoveryRun,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
    *,
    start: int,
    stop: int,
) -> ManifestDraft:
    selected = store.put_array(np.arange(start, stop, dtype=np.int64))
    updates = store.put_array(np.zeros((6, stop - start), dtype=np.float64))
    posterior = store.put_array(np.ones(6, dtype=np.float64))
    identity = {
        "scientific_identity_sha256": prepared.scientific_identity_sha256,
        "execution_policy": policy.identity_payload(),
        "numerical_compatibility": {"probe": "synthetic"},
        "numerical_compatibility_sha256": prepared.numerical_compatibility_sha256,
        "candidate_universe_sha256": prepared.candidate_sha256,
        "candidate_sequence_sha256": SHA_B,
        "candidate_count": 6,
        "target_budget": protocol.budgets[-1],
        "geometry_sha256": SHA_D,
    }
    arrays = {
        "selected_indices": selected.payload(),
        "updates": updates.payload(),
        "posterior_variance": posterior.payload(),
    }
    return session.draft_manifest(
        entries={
            "selected_indices": selected.blob,
            "updates": updates.blob,
            "posterior_variance": posterior.blob,
        },
        meta={
            "schema_version": "epibudget-reduced-doptimal-delta-v2",
            "state_kind": "reduced_doptimal",
            "identity": identity,
            "start": start,
            "stop": stop,
            "prefix_sha256": hashlib.sha256(f"prefix-{stop}".encode()).hexdigest(),
            "arrays": arrays,
        },
    )


def test_registered_cell_keys_are_budget_major(protocol: RecoveryScientificProtocol) -> None:
    assert registered_cell_keys(protocol) == (
        RecoveryCellKey("info", None, 2),
        RecoveryCellKey("random", 0, 2),
        RecoveryCellKey("info", None, 4),
        RecoveryCellKey("random", 0, 4),
    )


def test_replay_empty_store_is_non_mutating(store: ContentAddressedRunStore) -> None:
    before = sorted(
        (path.relative_to(store.root), path.stat().st_size, path.stat().st_mtime_ns)
        for path in store.root.rglob("*")
    )

    state = replay_recovery_state(store)

    after = sorted(
        (path.relative_to(store.root), path.stat().st_size, path.stat().st_mtime_ns)
        for path in store.root.rglob("*")
    )
    assert state.prepared is None
    assert state.latest_manifest is None
    assert before == after


def test_replay_populated_store_is_non_mutating(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    _complete_selection(store, prepared, protocol, policy)
    before = sorted(
        (path.relative_to(store.root), path.stat().st_size, path.stat().st_mtime_ns)
        for path in store.root.rglob("*")
    )

    replay_recovery_state(store, protocol=protocol, execution_policy=policy)

    after = sorted(
        (path.relative_to(store.root), path.stat().st_size, path.stat().st_mtime_ns)
        for path in store.root.rglob("*")
    )
    assert before == after


def test_execution_attempts_survive_crash_restart_and_interleaved_checkpoints(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    cursor = RecoveryStateCursor.open(
        RunStoreSession.open(store), protocol=protocol, execution_policy=policy
    )
    root = publish_prepared_run_at(cursor, prepared)
    start_a = _attempt_start(prepared, "attempt-a", "2026-08-11T20:00:00Z")
    publish_execution_attempt_started_at(cursor, start_a)
    _append_doptimal(
        store,
        prepared,
        protocol,
        policy,
        start=0,
        stop=2,
        parent=store.latest_manifest() or root,
        selected_values=(0, 1),
    )
    _append_doptimal(
        store,
        prepared,
        protocol,
        policy,
        start=2,
        stop=4,
        parent=store.latest_manifest() or root,
        selected_values=(2, 3),
    )
    plan_ref = store.put_json({"selections": ["synthetic"]})
    plan = PublishedSelectionPlan(prepared.scientific_identity_sha256, plan_ref.sha256, plan_ref)
    publish_selection_plan(store, plan, protocol=protocol, execution_policy=policy)
    for key in registered_cell_keys(protocol):
        _complete_valid_cell(store, prepared, plan, key, protocol, policy)
    resumed = RecoveryStateCursor.open(
        RunStoreSession.open(store), protocol=protocol, execution_policy=policy
    )
    start_b = _attempt_start(prepared, "attempt-b", "2026-08-11T21:00:00Z")
    start_b_manifest = publish_execution_attempt_started_at(resumed, start_b)
    completion = ExecutionAttemptCompletion(
        attempt_id=start_b.attempt_id,
        start_ref=start_b_manifest.entry("attempt_start"),
        commit_sha=start_b.commit_sha,
        workspace_clean=True,
        scientific_diff_sha256=None,
        completed_utc="2026-08-11T22:00:00Z",
    )
    completed_manifest = publish_execution_attempt_completed_at(resumed, completion)
    assert publish_execution_attempt_completed_at(resumed, completion) == completed_manifest

    assert tuple(attempt.start.attempt_id for attempt in resumed.execution_attempts) == (
        "attempt-a",
        "attempt-b",
    )
    assert resumed.abandoned_execution_attempts[0].start.attempt_id == "attempt-a"
    assert resumed.finalized_execution_attempt is not None
    assert resumed.open_execution_attempt is None

    state = replay_recovery_state(store, protocol=protocol, execution_policy=policy)

    assert tuple(attempt.start.attempt_id for attempt in state.execution_attempts) == (
        "attempt-a",
        "attempt-b",
    )
    assert tuple(attempt.start.attempt_id for attempt in state.abandoned_execution_attempts) == (
        "attempt-a",
    )
    assert state.finalized_execution_attempt is not None
    assert state.finalized_execution_attempt.start.attempt_id == "attempt-b"
    assert state.open_execution_attempt is None
    assert state.doptimal_completed == 4


def test_execution_attempt_accepts_datetime_isoformat_utc(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    start = _attempt_start(prepared, "attempt-a", "2026-08-11T20:00:00+00:00")

    assert start.started_utc == "2026-08-11T20:00:00+00:00"


def test_execution_attempt_retry_is_exact_and_stale_completion_is_refused(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    cursor = RecoveryStateCursor.open(
        RunStoreSession.open(store), protocol=protocol, execution_policy=policy
    )
    publish_prepared_run_at(cursor, prepared)
    start_a = _attempt_start(prepared, "attempt-a", "2026-08-11T20:00:00Z")
    first = publish_execution_attempt_started_at(cursor, start_a)

    assert publish_execution_attempt_started_at(cursor, start_a) == first
    start_b = _attempt_start(prepared, "attempt-b", "2026-08-11T21:00:00Z")
    second = publish_execution_attempt_started_at(cursor, start_b)
    wrong_commit = ExecutionAttemptCompletion(
        attempt_id=start_b.attempt_id,
        start_ref=second.entry("attempt_start"),
        commit_sha="2" * 40,
        workspace_clean=True,
        scientific_diff_sha256=None,
        completed_utc="2026-08-11T22:00:00Z",
    )
    with pytest.raises(RecoveryStateError, match="commit"):
        publish_execution_attempt_completed_at(cursor, wrong_commit)
    premature = ExecutionAttemptCompletion(
        attempt_id=start_b.attempt_id,
        start_ref=second.entry("attempt_start"),
        commit_sha=start_b.commit_sha,
        workspace_clean=True,
        scientific_diff_sha256=None,
        completed_utc="2026-08-11T22:00:00Z",
    )
    with pytest.raises(RecoveryStateError, match="every cell"):
        publish_execution_attempt_completed_at(cursor, premature)
    stale = ExecutionAttemptCompletion(
        attempt_id=start_a.attempt_id,
        start_ref=first.entry("attempt_start"),
        commit_sha=start_a.commit_sha,
        workspace_clean=True,
        scientific_diff_sha256=None,
        completed_utc="2026-08-11T22:00:00Z",
    )
    before = store.verify()

    with pytest.raises(RecoveryStateError, match="latest open"):
        publish_execution_attempt_completed_at(cursor, stale)

    after = store.verify()
    assert (after.manifest_count, after.blob_count) == (
        before.manifest_count,
        before.blob_count,
    )


def test_execution_attempt_rollback_checkpoint_keeps_only_bounded_tail_state(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared(store, protocol, policy)
    cursor = RecoveryStateCursor.open(
        RunStoreSession.open(store), protocol=protocol, execution_policy=policy
    )
    publish_prepared_run_at(cursor, prepared)
    start_a = _attempt_start(prepared, "attempt-a", "2026-08-11T20:00:00Z")
    publish_execution_attempt_started_at(cursor, start_a)
    attempts_id = id(cursor._accumulator.execution_attempts)
    before = cursor.execution_attempts
    undo = cursor._accumulator.checkpoint()

    assert undo.execution_attempts_length == 1
    assert undo.execution_attempt_previous_last is before[-1]
    assert not hasattr(undo, "execution_attempts")

    start_b = _attempt_start(prepared, "attempt-b", "2026-08-11T21:00:00Z")

    def fail_publication(_draft: ManifestDraft) -> Manifest:
        raise OSError("publication failed")

    monkeypatch.setattr(cursor._session, "publish_manifest", fail_publication)
    with pytest.raises(OSError, match="failed"):
        publish_execution_attempt_started_at(cursor, start_b)

    assert id(cursor._accumulator.execution_attempts) == attempts_id
    assert cursor.execution_attempts == before
    assert cursor.execution_attempts[-1].abandoned is False


def test_execution_attempt_replay_rejects_completion_gap(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    root = publish_prepared_run(store, prepared, protocol=protocol, execution_policy=policy)
    forged_completion = ExecutionAttemptCompletion(
        attempt_id="absent",
        start_ref=prepared.runtime_record_ref,
        commit_sha="1" * 40,
        workspace_clean=True,
        scientific_diff_sha256=None,
        completed_utc="2026-08-11T22:00:00Z",
    )
    completion_ref = store.put_json(forged_completion.payload())
    store.publish_manifest(
        entries={"attempt_completion": completion_ref},
        meta={
            "schema_version": "epibudget-execution-attempt-completed-v1",
            "state_kind": "execution_attempt_completed",
            "attempt_id": forged_completion.attempt_id,
            "record_ref": completion_ref.payload(),
        },
        parent=root,
    )

    with pytest.raises(RecoveryStateError, match="open start"):
        replay_recovery_state(store, protocol=protocol, execution_policy=policy)


def test_execution_attempt_replay_rejects_tampered_record_reference(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    cursor = RecoveryStateCursor.open(
        RunStoreSession.open(store), protocol=protocol, execution_policy=policy
    )
    publish_prepared_run_at(cursor, prepared)
    start = _attempt_start(prepared, "attempt-a", "2026-08-11T20:00:00Z")
    start_manifest = publish_execution_attempt_started_at(cursor, start)
    completion = ExecutionAttemptCompletion(
        attempt_id=start.attempt_id,
        start_ref=start_manifest.entry("attempt_start"),
        commit_sha=start.commit_sha,
        workspace_clean=True,
        scientific_diff_sha256=None,
        completed_utc="2026-08-11T22:00:00Z",
    )
    completion_ref = store.put_json(completion.payload())
    store.publish_manifest(
        entries={"attempt_completion": completion_ref},
        meta={
            "schema_version": "epibudget-execution-attempt-completed-v1",
            "state_kind": "execution_attempt_completed",
            "attempt_id": completion.attempt_id,
            "record_ref": prepared.input_bundle_ref.payload(),
        },
        parent=store.latest_manifest(),
    )

    with pytest.raises(RecoveryStateError, match="reference"):
        replay_recovery_state(store, protocol=protocol, execution_policy=policy)


def test_complete_chain_replays_and_verifies(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    plan = _complete_selection(store, prepared, protocol, policy)
    for key in registered_cell_keys(protocol):
        _complete_valid_cell(store, prepared, plan, key, protocol, policy)
    report_cursor = RecoveryStateCursor.open(
        RunStoreSession.open(store), protocol=protocol, execution_policy=policy
    )
    _complete_attempt_for_report(report_cursor, prepared)
    report = publish_recovery_report(
        store,
        {
            "schema_version": "synthetic-report-v1",
            "cells": [_valid_metrics(key, protocol) for key in registered_cell_keys(protocol)],
        },
        protocol=protocol,
        execution_policy=policy,
    )

    state = replay_recovery_state(store, protocol=protocol, execution_policy=policy)
    verification = state.verification()

    assert state.doptimal_completed == protocol.budgets[-1]
    assert state.selection_plan == plan
    assert state.completed_cells == registered_cell_keys(protocol)
    assert state.report_ref == report.entry("report")
    expected_cell_results_sha256 = hashlib.sha256(
        canonical_json_bytes(
            [
                {"cell": key.payload(), "result_ref": reference.payload()}
                for key, reference in zip(
                    registered_cell_keys(protocol), state.cell_result_refs, strict=True
                )
            ]
        )
    ).hexdigest()
    assert state.cell_results_sha256 == expected_cell_results_sha256
    assert report.meta["cell_results_sha256"] == expected_cell_results_sha256
    report_payload = store.get_json(report.entry("report"))
    assert isinstance(report_payload, dict)
    assert report_payload["cell_results_sha256"] == expected_cell_results_sha256
    assert verification.is_complete
    assert verification.completed_cell_count == protocol.cell_count
    assert verification.manifest_count == len(store.manifest_chain())


@pytest.mark.parametrize(
    ("kind", "meta"),
    [
        ("mystery", {"schema_version": "x", "state_kind": "mystery"}),
        (
            "recovery_prepared",
            {
                "schema_version": "epibudget-recovery-prepared-v1",
                "state_kind": "recovery_prepared",
                "unexpected": True,
            },
        ),
    ],
)
def test_replay_rejects_unknown_kinds_and_fields(
    store: ContentAddressedRunStore, kind: str, meta: dict[str, object]
) -> None:
    del kind
    store.publish_manifest(entries={}, meta=meta, parent=None)

    with pytest.raises(RecoveryStateError):
        replay_recovery_state(store)


def test_replay_rejects_doptimal_before_prepared(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    _append_doptimal(
        store,
        prepared,
        protocol,
        policy,
        start=0,
        stop=2,
        parent=None,  # type: ignore[arg-type]
    )

    with pytest.raises(RecoveryStateError, match="prepared"):
        replay_recovery_state(store, protocol=protocol, execution_policy=policy)


def test_replay_rejects_bool_as_doptimal_integer(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    parent = publish_prepared_run(store, prepared, protocol=protocol, execution_policy=policy)
    _append_doptimal(store, prepared, protocol, policy, start=0, stop=True, parent=parent)

    with pytest.raises(RecoveryStateError, match="integer"):
        replay_recovery_state(store, protocol=protocol, execution_policy=policy)


def test_replay_rejects_doptimal_gap_and_identity_drift(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    parent = publish_prepared_run(store, prepared, protocol=protocol, execution_policy=policy)
    _append_doptimal(
        store,
        prepared,
        protocol,
        policy,
        start=2,
        stop=4,
        parent=parent,
        scientific_sha256=SHA_D,
    )

    with pytest.raises(RecoveryStateError):
        replay_recovery_state(store, protocol=protocol, execution_policy=policy)


def test_replay_rejects_doptimal_candidate_universe_drift(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    parent = publish_prepared_run(store, prepared, protocol=protocol, execution_policy=policy)
    _append_doptimal(
        store,
        prepared,
        protocol,
        policy,
        start=0,
        stop=2,
        parent=parent,
        candidate_universe_sha256=SHA_D,
    )

    with pytest.raises(RecoveryStateError, match="candidate universe"):
        replay_recovery_state(store, protocol=protocol, execution_policy=policy)


def test_replay_rejects_doptimal_indices_repeated_across_blocks(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    parent = publish_prepared_run(store, prepared, protocol=protocol, execution_policy=policy)
    parent = _append_doptimal(
        store,
        prepared,
        protocol,
        policy,
        start=0,
        stop=2,
        parent=parent,
        selected_values=(0, 1),
    )
    _append_doptimal(
        store,
        prepared,
        protocol,
        policy,
        start=2,
        stop=4,
        parent=parent,
        selected_values=(1, 2),
    )

    with pytest.raises(RecoveryStateError, match="unique"):
        replay_recovery_state(store, protocol=protocol, execution_policy=policy)


def test_replay_rejects_repeated_doptimal_prefix_digest(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    parent = publish_prepared_run(store, prepared, protocol=protocol, execution_policy=policy)
    parent = _append_doptimal(
        store,
        prepared,
        protocol,
        policy,
        start=0,
        stop=2,
        parent=parent,
        prefix_sha256=SHA_D,
    )
    _append_doptimal(
        store,
        prepared,
        protocol,
        policy,
        start=2,
        stop=4,
        parent=parent,
        prefix_sha256=SHA_D,
    )

    with pytest.raises(RecoveryStateError, match="prefix"):
        replay_recovery_state(store, protocol=protocol, execution_policy=policy)


def test_replay_rejects_doptimal_state_with_wrong_candidate_axis(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    parent = publish_prepared_run(store, prepared, protocol=protocol, execution_policy=policy)
    _append_doptimal(
        store,
        prepared,
        protocol,
        policy,
        start=0,
        stop=2,
        parent=parent,
        state_rows=5,
    )

    with pytest.raises(RecoveryStateError, match="candidate count"):
        replay_recovery_state(store, protocol=protocol, execution_policy=policy)


def test_selection_requires_complete_doptimal(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    publish_prepared_run(store, prepared, protocol=protocol, execution_policy=policy)
    ref = store.put_json({"selections": []})
    plan = PublishedSelectionPlan(prepared.scientific_identity_sha256, ref.sha256, ref)

    with pytest.raises(RecoveryStateError, match="D-optimal"):
        publish_selection_plan(store, plan, protocol=protocol, execution_policy=policy)


def test_lasso_requires_selection_and_contiguous_folds(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    plan = _complete_selection(store, prepared, protocol, policy)
    key = registered_cell_keys(protocol)[0]
    _append_fold(store, prepared, plan, key, policy, completed=2, n_folds=protocol.n_folds)

    with pytest.raises(RecoveryStateError, match="fold"):
        replay_recovery_state(store, protocol=protocol, execution_policy=policy)


def test_valid_cell_requires_all_folds(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    _complete_selection(store, prepared, protocol, policy)
    key = registered_cell_keys(protocol)[0]

    with pytest.raises(RecoveryStateError, match="fold"):
        publish_recovery_cell(
            store,
            key,
            valid=True,
            metrics={"spearman": 0.1},
            error=None,
            protocol=protocol,
            execution_policy=policy,
        )


def test_valid_cell_requires_converged_lasso(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    plan = _complete_selection(store, prepared, protocol, policy)
    key = registered_cell_keys(protocol)[0]
    _append_fold(
        store,
        prepared,
        plan,
        key,
        policy,
        completed=1,
        n_folds=protocol.n_folds,
    )
    _append_fold(
        store,
        prepared,
        plan,
        key,
        policy,
        completed=2,
        n_folds=protocol.n_folds,
        converged=False,
    )

    with pytest.raises(RecoveryStateError, match="converged"):
        publish_recovery_cell(
            store,
            key,
            valid=True,
            metrics={"spearman": 0.1},
            error=None,
            protocol=protocol,
            execution_policy=policy,
        )


def test_valid_cell_metrics_must_match_cell_and_lasso_identity(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    plan = _complete_selection(store, prepared, protocol, policy)
    key = registered_cell_keys(protocol)[0]
    for fold in range(1, protocol.n_folds + 1):
        _append_fold(
            store,
            prepared,
            plan,
            key,
            policy,
            completed=fold,
            n_folds=protocol.n_folds,
        )
    metrics = _valid_metrics(key, protocol)
    metrics["method"] = "wrong"

    with pytest.raises(RecoveryStateError, match="metrics"):
        publish_recovery_cell(
            store,
            key,
            valid=True,
            metrics=metrics,
            error=None,
            protocol=protocol,
            execution_policy=policy,
        )


def test_valid_cell_requires_selected_lambda_fields(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    plan = _complete_selection(store, prepared, protocol, policy)
    key = registered_cell_keys(protocol)[0]
    for fold in range(1, protocol.n_folds + 1):
        _append_fold(
            store,
            prepared,
            plan,
            key,
            policy,
            completed=fold,
            n_folds=protocol.n_folds,
        )
    metrics = _valid_metrics(key, protocol)
    metrics["lambda_ratio"] = None
    metrics["lambda_value"] = None

    with pytest.raises(RecoveryStateError, match="lambda"):
        publish_recovery_cell(
            store,
            key,
            valid=True,
            metrics=metrics,
            error=None,
            protocol=protocol,
            execution_policy=policy,
        )


def test_invalid_cell_can_close_without_folds(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    _complete_selection(store, prepared, protocol, policy)
    key = registered_cell_keys(protocol)[0]

    publish_recovery_cell(
        store,
        key,
        valid=False,
        metrics=None,
        error="non-finite estimator",
        protocol=protocol,
        execution_policy=policy,
    )

    state = replay_recovery_state(store, protocol=protocol, execution_policy=policy)
    assert state.completed_cells == (key,)
    assert state.active_cell is None
    assert state.completed_folds == 0


def test_republishing_identical_selection_returns_its_original_manifest(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    plan = _complete_selection(store, prepared, protocol, policy)
    key = registered_cell_keys(protocol)[0]
    publish_recovery_cell(
        store,
        key,
        valid=False,
        metrics=None,
        error="synthetic failure",
        protocol=protocol,
        execution_policy=policy,
    )

    manifest = publish_selection_plan(store, plan, protocol=protocol, execution_policy=policy)

    assert manifest.meta["state_kind"] == "selection_plan"


def test_republishing_identical_cell_is_idempotent(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    _complete_selection(store, prepared, protocol, policy)
    key = registered_cell_keys(protocol)[0]
    first = publish_recovery_cell(
        store,
        key,
        valid=False,
        metrics=None,
        error="synthetic failure",
        protocol=protocol,
        execution_policy=policy,
    )

    repeated = publish_recovery_cell(
        store,
        key,
        valid=False,
        metrics=None,
        error="synthetic failure",
        protocol=protocol,
        execution_policy=policy,
    )

    assert repeated == first


def test_republishing_cell_with_different_payload_is_rejected(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    _complete_selection(store, prepared, protocol, policy)
    key = registered_cell_keys(protocol)[0]
    publish_recovery_cell(
        store,
        key,
        valid=False,
        metrics=None,
        error="first failure",
        protocol=protocol,
        execution_policy=policy,
    )

    with pytest.raises(RecoveryStateError, match="diverged"):
        publish_recovery_cell(
            store,
            key,
            valid=False,
            metrics=None,
            error="different failure",
            protocol=protocol,
            execution_policy=policy,
        )


def test_cells_must_follow_exact_budget_major_order(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    _complete_selection(store, prepared, protocol, policy)

    with pytest.raises(RecoveryStateError, match="order"):
        publish_recovery_cell(
            store,
            registered_cell_keys(protocol)[1],
            valid=False,
            metrics=None,
            error="synthetic failure",
            protocol=protocol,
            execution_policy=policy,
        )


def test_report_requires_every_cell_and_is_terminal(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    _complete_selection(store, prepared, protocol, policy)

    with pytest.raises(RecoveryStateError, match="cell"):
        publish_recovery_report(
            store,
            {"cells": []},
            protocol=protocol,
            execution_policy=policy,
        )


def test_report_rejects_cells_not_assembled_from_the_journal(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    plan = _complete_selection(store, prepared, protocol, policy)
    for key in registered_cell_keys(protocol):
        _complete_valid_cell(store, prepared, plan, key, protocol, policy)
    report_cursor = RecoveryStateCursor.open(
        RunStoreSession.open(store), protocol=protocol, execution_policy=policy
    )
    _complete_attempt_for_report(report_cursor, prepared)

    with pytest.raises(RecoveryStateError, match="cells"):
        publish_recovery_report(
            store,
            {"cells": []},
            protocol=protocol,
            execution_policy=policy,
        )


def test_report_requires_the_latest_execution_attempt_to_be_completed(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    plan = _complete_selection(store, prepared, protocol, policy)
    for key in registered_cell_keys(protocol):
        _complete_valid_cell(store, prepared, plan, key, protocol, policy)
    cursor = RecoveryStateCursor.open(
        RunStoreSession.open(store), protocol=protocol, execution_policy=policy
    )
    publish_execution_attempt_started_at(
        cursor, _attempt_start(prepared, "open-attempt", "2026-08-11T20:00:00Z")
    )

    with pytest.raises(RecoveryStateError, match="completed latest execution"):
        publish_recovery_report_at(
            cursor,
            {"cells": [_valid_metrics(key, protocol) for key in registered_cell_keys(protocol)]},
        )


def test_replay_rejects_manifest_after_report(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    plan = _complete_selection(store, prepared, protocol, policy)
    for key in registered_cell_keys(protocol):
        _complete_valid_cell(store, prepared, plan, key, protocol, policy)
    report_cursor = RecoveryStateCursor.open(
        RunStoreSession.open(store), protocol=protocol, execution_policy=policy
    )
    _complete_attempt_for_report(report_cursor, prepared)
    publish_recovery_report(
        store,
        {"cells": [_valid_metrics(key, protocol) for key in registered_cell_keys(protocol)]},
        protocol=protocol,
        execution_policy=policy,
    )
    result_ref = store.put_json(
        {
            "schema_version": "epibudget-recovery-cell-result-v1",
            "cell": registered_cell_keys(protocol)[0].payload(),
            "valid": False,
            "metrics": None,
            "error": "late cell",
        }
    )
    store.publish_manifest(
        entries={"result": result_ref},
        meta={
            "schema_version": "epibudget-recovery-cell-v1",
            "state_kind": "recovery_cell",
            "scientific_identity_sha256": prepared.scientific_identity_sha256,
            "selection_plan_sha256": plan.selection_plan_sha256,
            "cell": registered_cell_keys(protocol)[0].payload(),
            "valid": False,
            "error": "late cell",
            "result_ref": result_ref.payload(),
        },
        parent=store.latest_manifest(),
    )

    with pytest.raises(RecoveryStateError, match="follow"):
        replay_recovery_state(store, protocol=protocol, execution_policy=policy)


def test_publication_rejects_nonfinite_metrics(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    plan = _complete_selection(store, prepared, protocol, policy)
    key = registered_cell_keys(protocol)[0]
    for fold in range(1, protocol.n_folds + 1):
        _append_fold(
            store,
            prepared,
            plan,
            key,
            policy,
            completed=fold,
            n_folds=protocol.n_folds,
        )

    with pytest.raises(ValueError, match="finite"):
        publish_recovery_cell(
            store,
            key,
            valid=True,
            metrics=_valid_metrics(key, protocol, spearman=math.nan),
            error=None,
            protocol=protocol,
            execution_policy=policy,
        )


def test_value_objects_reject_malformed_sha(store: ContentAddressedRunStore) -> None:
    ref = store.put_json({})
    with pytest.raises(ValueError, match="SHA-256"):
        PublishedSelectionPlan("not-a-sha", ref.sha256, ref)


def test_replay_rejects_array_reference_not_matching_entry(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    parent = publish_prepared_run(store, prepared, protocol=protocol, execution_policy=policy)
    first = store.put_array(np.asarray([0, 1], dtype=np.int64))
    other = store.put_array(np.asarray([2, 3], dtype=np.int64))
    updates = store.put_array(np.zeros((6, 2), dtype=np.float64))
    posterior = store.put_array(np.ones(6, dtype=np.float64))
    arrays: dict[str, Any] = {
        "selected_indices": other.payload(),
        "updates": updates.payload(),
        "posterior_variance": posterior.payload(),
    }
    store.publish_manifest(
        entries={
            "selected_indices": first.blob,
            "updates": updates.blob,
            "posterior_variance": posterior.blob,
        },
        meta={
            "schema_version": "epibudget-reduced-doptimal-delta-v2",
            "state_kind": "reduced_doptimal",
            "identity": {
                "scientific_identity_sha256": prepared.scientific_identity_sha256,
                "execution_policy": policy.identity_payload(),
                "numerical_compatibility": {"probe": "synthetic"},
                "numerical_compatibility_sha256": prepared.numerical_compatibility_sha256,
                "candidate_universe_sha256": prepared.candidate_sha256,
                "candidate_sequence_sha256": SHA_B,
                "candidate_count": 6,
                "target_budget": protocol.budgets[-1],
                "geometry_sha256": SHA_D,
            },
            "start": 0,
            "stop": 2,
            "prefix_sha256": SHA_D,
            "arrays": arrays,
        },
        parent=parent,
    )

    with pytest.raises(RecoveryStateError, match="reference"):
        replay_recovery_state(store, protocol=protocol, execution_policy=policy)


def test_cursor_incremental_snapshots_match_full_replay_after_every_append(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    session = RunStoreSession.open(store)
    cursor = RecoveryStateCursor.open(session, protocol=protocol, execution_policy=policy)
    history_list_ids = (
        id(cursor._accumulator.manifests),
        id(cursor._accumulator.doptimal_manifests),
        id(cursor._accumulator.cell_manifests),
        id(cursor._accumulator.completed_cells),
    )

    publish_prepared_run_at(cursor, prepared)
    assert cursor.snapshot() == replay_recovery_state(
        store, protocol=protocol, execution_policy=policy
    )
    for start in range(0, protocol.budgets[-1], policy.doptimal_block_size):
        cursor.append(
            _doptimal_draft(
                session,
                store,
                prepared,
                protocol,
                policy,
                start=start,
                stop=start + policy.doptimal_block_size,
            )
        )
        assert cursor.snapshot() == replay_recovery_state(
            store, protocol=protocol, execution_policy=policy
        )
    plan_ref = store.put_json({"selections": ["synthetic"]})
    plan = PublishedSelectionPlan(prepared.scientific_identity_sha256, plan_ref.sha256, plan_ref)
    publish_selection_plan_at(cursor, plan)
    assert cursor.snapshot() == replay_recovery_state(
        store, protocol=protocol, execution_policy=policy
    )
    assert isinstance(cursor.index, RecoveryManifestIndex)
    assert len(cursor.index.doptimal_manifests) == 2
    for key in registered_cell_keys(protocol):
        publish_recovery_cell_at(
            cursor,
            key,
            valid=False,
            metrics=None,
            error="synthetic failure",
        )
        assert cursor.snapshot() == replay_recovery_state(
            store, protocol=protocol, execution_policy=policy
        )
        assert cursor.manifest_for_cell(key) == cursor.index.cell_manifests[-1]

    report_cells = [
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
            "error": "synthetic failure",
        }
        for key in registered_cell_keys(protocol)
    ]
    _complete_attempt_for_report(cursor, prepared)
    publish_recovery_report_at(cursor, {"cells": report_cells})

    assert cursor.snapshot() == replay_recovery_state(
        store, protocol=protocol, execution_policy=policy
    )
    assert cursor.index.report_manifest == cursor.snapshot().latest_manifest
    assert history_list_ids == (
        id(cursor._accumulator.manifests),
        id(cursor._accumulator.doptimal_manifests),
        id(cursor._accumulator.cell_manifests),
        id(cursor._accumulator.completed_cells),
    )


def test_cursor_append_never_rescans_verified_history(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared(store, protocol, policy)
    _complete_selection(store, prepared, protocol, policy)
    session = RunStoreSession.open(store)
    cursor = RecoveryStateCursor.open(session, protocol=protocol, execution_policy=policy)
    monkeypatch.setattr(
        store,
        "manifest_chain",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected full replay")),
    )
    monkeypatch.setattr(
        session,
        "manifests",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected cached-history read")),
    )

    publish_recovery_cell_at(
        cursor,
        registered_cell_keys(protocol)[0],
        valid=False,
        metrics=None,
        error="synthetic failure",
    )

    assert len(cursor.snapshot().completed_cells) == 1


def test_cursor_validation_and_publication_failure_do_not_mutate_state(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared(store, protocol, policy)
    _complete_selection(store, prepared, protocol, policy)
    session = RunStoreSession.open(store)
    cursor = RecoveryStateCursor.open(session, protocol=protocol, execution_policy=policy)
    before = cursor.snapshot()
    before_internal = (
        tuple(cursor._accumulator.manifests),
        frozenset(cursor._accumulator.doptimal_indices),
        frozenset(cursor._accumulator.doptimal_prefixes),
        id(cursor._accumulator.manifests),
        id(cursor._accumulator.completed_cells),
    )
    unknown = session.draft_manifest(
        entries={}, meta={"schema_version": "x", "state_kind": "unknown"}
    )

    with pytest.raises(RecoveryStateError, match="unknown"):
        cursor.append(unknown)
    assert cursor.snapshot() == before
    assert session.latest_manifest() == before.latest_manifest
    assert before_internal == (
        tuple(cursor._accumulator.manifests),
        frozenset(cursor._accumulator.doptimal_indices),
        frozenset(cursor._accumulator.doptimal_prefixes),
        id(cursor._accumulator.manifests),
        id(cursor._accumulator.completed_cells),
    )

    valid = _invalid_cell_draft(
        session,
        store,
        cursor,
        registered_cell_keys(protocol)[0],
        "synthetic failure",
    )

    def fail_publication(_draft: ManifestDraft) -> Manifest:
        raise OSError("publication failed")

    monkeypatch.setattr(session, "publish_manifest", fail_publication)
    with pytest.raises(OSError, match="failed"):
        cursor.append(valid)
    assert cursor.snapshot() == before
    assert before_internal == (
        tuple(cursor._accumulator.manifests),
        frozenset(cursor._accumulator.doptimal_indices),
        frozenset(cursor._accumulator.doptimal_prefixes),
        id(cursor._accumulator.manifests),
        id(cursor._accumulator.completed_cells),
    )


def test_cursor_exact_retry_and_other_session_divergence_are_fail_closed(
    store: ContentAddressedRunStore,
    protocol: RecoveryScientificProtocol,
    policy: RecoveryExecutionPolicy,
) -> None:
    prepared = _prepared(store, protocol, policy)
    _complete_selection(store, prepared, protocol, policy)
    session_a = RunStoreSession.open(store)
    session_b = RunStoreSession.open(store)
    cursor_a = RecoveryStateCursor.open(session_a, protocol=protocol, execution_policy=policy)
    cursor_b = RecoveryStateCursor.open(session_b, protocol=protocol, execution_policy=policy)
    key = registered_cell_keys(protocol)[0]
    exact_a = _invalid_cell_draft(session_a, store, cursor_a, key, "same failure")
    exact_b = _invalid_cell_draft(session_b, store, cursor_b, key, "same failure")
    rival_b = _invalid_cell_draft(session_b, store, cursor_b, key, "different failure")

    first = cursor_a.append(exact_a)
    assert cursor_a.append(exact_a) == first
    assert cursor_b.append(exact_b) == first
    with pytest.raises(RunStoreError):
        cursor_b.append(rival_b)

    stale_session = RunStoreSession.open(store)
    stale_cursor = RecoveryStateCursor.open(
        stale_session, protocol=protocol, execution_policy=policy
    )
    stale_rival = _invalid_cell_draft(
        stale_session,
        store,
        stale_cursor,
        registered_cell_keys(protocol)[1],
        "stale rival",
    )
    current_before = stale_cursor.snapshot()
    publish_recovery_cell_at(
        cursor_a,
        registered_cell_keys(protocol)[1],
        valid=False,
        metrics=None,
        error="current writer",
    )

    with pytest.raises(RunStoreError):
        stale_cursor.append(stale_rival)
    assert stale_cursor.snapshot() == current_before
