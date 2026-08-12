from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import epibudget.recovery_engine as engine
from epibudget.coeff_recovery import _build_fourier_config
from epibudget.data import DatasetSpec, enumerate_candidates
from epibudget.fourier_recovery import (
    PairwiseTruth,
    RecoveryCell,
    SelectionPlan,
    SelectionSequence,
    _sequence_sha256,
)
from epibudget.recovery_engine import (
    RecoveryEngineError,
    RecoveryProgress,
    RecoveryStatus,
    export_recovery_report,
    prepare_recovery_run,
    recovery_status,
    verify_recovery_run,
)
from epibudget.recovery_protocol import (
    REGISTERED_EXECUTION_POLICY,
    REGISTERED_RECOVERY_PROTOCOL,
    RecoveryExecutionPolicy,
    RecoveryScientificProtocol,
)
from epibudget.recovery_runtime import (
    ArchivedRecoveryInput,
    NumericCompatibility,
    RecoveryInputBundle,
    RecoveryProvenance,
    RecoveryRuntimeRecord,
    RecoveryScientificIdentity,
    ThreadPoolCompatibility,
    materialize_archived_recovery_input,
)
from epibudget.recovery_state import (
    ExecutionAttemptCompletion,
    ExecutionAttemptRecord,
    ExecutionAttemptStart,
    PreparedRecoveryRun,
    PublishedSelectionPlan,
    RecoveryRunState,
    RecoveryStateCursor,
    registered_cell_keys,
    replay_recovery_state,
)
from epibudget.run_store import (
    BlobRef,
    ContentAddressedRunStore,
    RunStoreSession,
    canonical_json_bytes,
)
from epibudget.scored_cache import CacheIdentity, candidate_sha256
from epibudget.types import ScoredVariant, Variant

_PAIRWISE_ORDER = 2

_COMPLETED = 17


def _compatibility() -> NumericCompatibility:
    return NumericCompatibility(
        python_version="3.12",
        numpy_version="2.1",
        scipy_version="1.14",
        blas_sha256="1" * 64,
        thread_environment=(("OMP_NUM_THREADS", "1"),),
        thread_pools=(
            ThreadPoolCompatibility(
                user_api="blas",
                internal_api="openblas",
                prefix="libopenblas",
                version="1",
                threading_layer="pthreads",
                architecture="test",
                num_threads=1,
            ),
        ),
        probe_sha256="2" * 64,
    )


def _registered_inputs() -> engine._RegisteredInputs:
    candidate: Variant = frozenset({(0, "A", "C")})
    candidates = (candidate,)
    identity = CacheIdentity(
        model_id="test-model",
        scorer_seed=0,
        n_perturbations=1,
        candidate_sha256=candidate_sha256(candidates),
        candidate_count=1,
        candidate_alphabet="AC",
        max_order=1,
        wt_sha256="3" * 64,
    )
    return engine._RegisteredInputs(
        specification=DatasetSpec(
            identifier="synthetic",
            loader=lambda _path: {candidate: 1.0},
            sites=(0, 1),
            wt_at_sites=("A", "A"),
            wt_sequence="AA",
            default_data_path="data.csv",
        ),
        candidates=candidates,
        config=_build_fourier_config((0, 1), ("A", "A"), "AC", max_order=2),
        scored=(ScoredVariant(variant=candidate, delta_g=1.0, var_delta_g=1.0),),
        expected_cache_identity=identity,
        observed_cache_identity=identity.model_copy(),
    )


def _integrated_recovery_fixture() -> tuple[
    RecoveryScientificProtocol,
    RecoveryExecutionPolicy,
    engine._RegisteredInputs,
]:
    sites = (0, 1)
    wt_at_sites = ("A", "A")
    alphabet = "ACD"
    candidates = tuple(enumerate_candidates(sites, wt_at_sites, alphabet, max_order=2))
    config = _build_fourier_config(sites, wt_at_sites, alphabet, max_order=2)
    coefficient_count = int(sum(np.count_nonzero(mode) == _PAIRWISE_ORDER for mode in config.modes))
    protocol = replace(
        REGISTERED_RECOVERY_PROTOCOL,
        version="synthetic-integrated-recovery-v1",
        dataset="synthetic_full_factorial",
        budgets=(8,),
        seeds=(0,),
        n_folds=2,
        selection_max_order=2,
        estimation_max_order=2,
        coefficient_count=coefficient_count,
        feature_count=len(config.modes),
    )
    policy = replace(
        REGISTERED_EXECUTION_POLICY,
        version="synthetic-integrated-execution-v1",
        doptimal_block_size=2,
        legacy_budget_block_size=1,
    )
    landscape: dict[Variant, float] = {frozenset(): 0.75}
    for index, variant in enumerate(candidates, start=1):
        order = len(variant)
        residues = sum(ord(mutation[2]) for mutation in variant)
        landscape[variant] = 0.4 + 0.13 * index + 0.17 * order + 0.001 * residues * order
    scored = tuple(
        ScoredVariant(
            variant=variant,
            delta_g=float(index),
            var_delta_g=0.25 + index / 100.0,
        )
        for index, variant in enumerate(candidates, start=1)
    )
    identity = CacheIdentity(
        model_id="synthetic-model",
        scorer_seed=0,
        n_perturbations=1,
        candidate_sha256=candidate_sha256(candidates),
        candidate_count=len(candidates),
        candidate_alphabet=alphabet,
        max_order=2,
        wt_sha256="a" * 64,
    )
    registered = engine._RegisteredInputs(
        specification=DatasetSpec(
            identifier="synthetic_full_factorial",
            loader=lambda _path: landscape,
            sites=sites,
            wt_at_sites=wt_at_sites,
            wt_sequence="AA",
            default_data_path="synthetic.csv",
        ),
        candidates=candidates,
        config=config,
        scored=scored,
        expected_cache_identity=identity,
        observed_cache_identity=identity.model_copy(),
    )
    return protocol, policy, registered


def _input_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    paths = tuple(
        tmp_path / name for name in ("data.csv", "cache.jsonl", "cache.meta.json", "preflight.json")
    )
    for index, path in enumerate(paths):
        path.write_bytes(f"input-{index}".encode())
    return paths  # type: ignore[return-value]


def _patch_prepare(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "_clean_commit", lambda _repo: "a" * 40)
    monkeypatch.setattr(engine, "capture_numeric_compatibility", _compatibility)
    monkeypatch.setattr(
        engine,
        "_validate_registered_inputs",
        lambda *_args, **_kwargs: _registered_inputs(),
    )
    monkeypatch.setattr(engine, "_require_registered_input_digests", lambda _digests: None)


def _runtime_record() -> RecoveryRuntimeRecord:
    reference = BlobRef(sha256="4" * 64, size=1, encoding="binary")
    return RecoveryRuntimeRecord(
        scientific_identity=RecoveryScientificIdentity(
            execution_commit="a" * 40,
            protocol_semantic_sha256=REGISTERED_RECOVERY_PROTOCOL.semantic_sha256,
            candidate_sha256="5" * 64,
            dataset_ref=reference,
            cache_ref=reference,
            sidecar_ref=reference,
            runtime_preflight_ref=reference,
        ),
        numeric_compatibility=_compatibility(),
        provenance=RecoveryProvenance(
            platform="test",
            machine="test",
            argv=("recovery",),
            started_utc="2026-01-01T00:00:00+00:00",
            completed_utc="2026-01-01T00:00:00+00:00",
        ),
    )


def _input_bundle() -> RecoveryInputBundle:
    return RecoveryInputBundle(
        dataset=ArchivedRecoveryInput("dataset.csv", BlobRef("1" * 64, 1, "binary")),
        cache=ArchivedRecoveryInput("cache.jsonl", BlobRef("2" * 64, 1, "binary")),
        sidecar=ArchivedRecoveryInput("cache.meta.json", BlobRef("3" * 64, 1, "binary")),
        runtime_preflight=ArchivedRecoveryInput("preflight.json", BlobRef("4" * 64, 1, "binary")),
    )


def _empty_run_state() -> RecoveryRunState:
    return RecoveryRunState(
        prepared=None,
        doptimal_completed=0,
        selection_plan=None,
        completed_cells=(),
        active_cell=None,
        completed_folds=0,
        lasso_converged=True,
        cell_result_refs=(),
        report_ref=None,
        latest_manifest=None,
        manifest_count=0,
        expected_cell_count=REGISTERED_RECOVERY_PROTOCOL.cell_count,
    )


def test_progress_and_status_are_immutable_value_objects() -> None:
    progress = RecoveryProgress(
        stage="cells",
        completed=_COMPLETED,
        total=344,
        method="random",
        seed=2,
        budget=96,
    )
    status = RecoveryStatus(
        prepared=True,
        selection_complete=True,
        doptimal_completed=3072,
        completed_cells=_COMPLETED,
        total_cells=344,
        active_cell="random:2:96",
        completed_folds=3,
        report_available=False,
        latest_sequence=25,
    )

    assert progress.completed == _COMPLETED
    assert status.completed_cells == _COMPLETED
    with pytest.raises(AttributeError):
        progress.completed = 18  # type: ignore[misc]


def test_uninitialised_directory_is_not_a_recovery_run_and_reads_do_not_mutate(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    before = tuple(run_dir.iterdir())

    with pytest.raises(RecoveryEngineError, match="initialised"):
        recovery_status(run_dir)
    with pytest.raises(RecoveryEngineError, match="initialised"):
        verify_recovery_run(run_dir)
    with pytest.raises(RecoveryEngineError, match="initialised"):
        export_recovery_report(run_dir, tmp_path / "report.json")

    assert tuple(run_dir.iterdir()) == before


def test_prepare_rejects_a_legacy_nonempty_directory_before_mutation(tmp_path: Path) -> None:
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    legacy = run_dir / "selection.registered.json"
    legacy.write_text("{}", encoding="utf-8")
    inputs = []
    for name in ("data.csv", "cache.jsonl", "cache.meta.json", "preflight.json"):
        path = tmp_path / name
        path.write_bytes(b"placeholder")
        inputs.append(path)
    before = {path.name: path.read_bytes() for path in run_dir.iterdir()}

    with pytest.raises(RecoveryEngineError, match="legacy"):
        prepare_recovery_run(
            run_dir,
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            repo=tmp_path,
        )

    assert {path.name: path.read_bytes() for path in run_dir.iterdir()} == before


def test_prepare_rejects_a_noncanonical_registered_input_before_store_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "_clean_commit", lambda _repo: "a" * 40)
    monkeypatch.setattr(engine, "capture_numeric_compatibility", _compatibility)
    monkeypatch.setattr(
        engine,
        "_validate_registered_inputs",
        lambda *_args, **_kwargs: _registered_inputs(),
    )
    protocol = replace(
        REGISTERED_RECOVERY_PROTOCOL,
        dataset_sha256="0" * 64,
        cache_sha256="1" * 64,
        sidecar_sha256="2" * 64,
    )
    monkeypatch.setattr(engine, "REGISTERED_RECOVERY_PROTOCOL", protocol)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(RecoveryEngineError, match="dataset SHA-256"):
        prepare_recovery_run(run_dir, *_input_paths(tmp_path), repo=tmp_path)

    assert tuple(run_dir.iterdir()) == ()


def test_prepare_is_idempotent_and_different_inputs_fail_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_prepare(monkeypatch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    inputs = _input_paths(tmp_path)

    first = prepare_recovery_run(run_dir, *inputs, repo=tmp_path)
    snapshot = {
        path.relative_to(run_dir): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    second = prepare_recovery_run(run_dir, *inputs, repo=tmp_path)

    assert first == second
    assert snapshot == {
        path.relative_to(run_dir): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    inputs[0].write_bytes(b"different")
    with pytest.raises(RecoveryEngineError, match="differ"):
        prepare_recovery_run(run_dir, *inputs, repo=tmp_path)
    assert snapshot == {
        path.relative_to(run_dir): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in run_dir.rglob("*")
        if path.is_file()
    }


def test_prepare_retries_after_initialisation_before_the_root_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_prepare(monkeypatch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    store = ContentAddressedRunStore(run_dir)
    store.initialise()
    store.put_bytes(b"orphan-from-interrupted-prepare")

    status = prepare_recovery_run(run_dir, *_input_paths(tmp_path), repo=tmp_path)

    assert status.prepared is True
    assert recovery_status(run_dir).prepared is True


def test_run_does_not_materialize_dataset_before_selection_is_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_prepare(monkeypatch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    inputs = _input_paths(tmp_path)
    prepare_recovery_run(run_dir, *inputs, repo=tmp_path)
    monkeypatch.setattr(engine, "require_numeric_compatibility", lambda *_args: None)
    materialized: list[str] = []
    original = materialize_archived_recovery_input

    def recording_materialize(
        store: ContentAddressedRunStore,
        archived: ArchivedRecoveryInput,
        destination: Path,
    ) -> Path:
        materialized.append(archived.name)
        return original(store, archived, destination)

    def stop_after_barrier(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("selection sentinel")

    monkeypatch.setattr(engine, "materialize_archived_recovery_input", recording_materialize)
    monkeypatch.setattr(engine, "_complete_selection", stop_after_barrier)

    with pytest.raises(RuntimeError, match="selection sentinel"):
        engine.run_recovery(run_dir, repo=tmp_path)

    assert materialized == [inputs[1].name, inputs[2].name, inputs[3].name]


def test_status_verify_and_export_are_read_only_and_export_is_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    store = ContentAddressedRunStore(run_dir)
    store.initialise()
    report_ref = store.put_json({"schema_version": "synthetic-report", "value": 3})
    fake_state = RecoveryRunState(
        prepared=None,
        doptimal_completed=0,
        selection_plan=None,
        completed_cells=(),
        active_cell=None,
        completed_folds=0,
        lasso_converged=True,
        lasso_selected_sha256=None,
        lasso_fold_sha256=None,
        lasso_lambda_ratios=(),
        cell_result_refs=(),
        report_ref=report_ref,
        latest_manifest=None,
        manifest_count=0,
        expected_cell_count=0,
    )
    monkeypatch.setattr(engine, "replay_recovery_state", lambda _store: fake_state)
    monkeypatch.setattr(
        engine, "_open_journal", lambda _store: SimpleNamespace(snapshot=lambda: fake_state)
    )
    monkeypatch.setattr(
        engine,
        "_runtime_and_bundle",
        lambda *_args: (_runtime_record(), SimpleNamespace()),
    )
    monkeypatch.setattr(engine, "_clean_commit", lambda _repo: "a" * 40)
    monkeypatch.setattr(engine, "require_numeric_compatibility", lambda *_args: None)
    monkeypatch.setattr(engine, "capture_numeric_compatibility", _compatibility)
    monkeypatch.setattr(engine, "_verify_selection_journal", lambda *_args: None)
    before = {
        path.relative_to(run_dir): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in run_dir.rglob("*")
        if path.is_file()
    }

    assert recovery_status(run_dir).report_available is True
    assert verify_recovery_run(run_dir).report_available is True
    output = export_recovery_report(run_dir, tmp_path / "report.json")

    assert output.read_bytes() == store.get_bytes(report_ref)
    assert before == {
        path.relative_to(run_dir): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    with pytest.raises(RecoveryEngineError, match="already exists"):
        export_recovery_report(run_dir, output)
    with pytest.raises(RecoveryEngineError, match="outside"):
        export_recovery_report(run_dir, run_dir / "report.json")


def test_only_declared_numerical_failures_are_scientific() -> None:
    assert engine._scientific_failure(FloatingPointError("non-finite")) is not None
    assert engine._scientific_failure(RuntimeError("FISTA did not converge")) is not None
    assert engine._scientific_failure(ValueError("response is effectively constant")) is not None
    assert engine._scientific_failure(ValueError("internal identity bug")) is None


def test_selection_match_uses_the_canonical_doptimal_candidate_order() -> None:
    first: Variant = frozenset({(0, "A", "C")})
    second: Variant = frozenset({(1, "A", "C")})
    enumerated = (second, first)
    config = _build_fourier_config((0, 1), ("A", "A"), "AC", max_order=2)
    doptimal = engine.initialise_reduced_doptimal(config, enumerated, target_budget=2)
    doptimal.selected_indices.extend((0, 1))
    sequence = SelectionSequence(
        method="doptimal_reduced_pairwise",
        seed=None,
        selected=doptimal.candidates,
        selected_sha256="6" * 64,
        tie_break_version="test",
    )
    plan = SelectionPlan(
        budgets=REGISTERED_RECOVERY_PROTOCOL.budgets,
        sequences=(sequence,),
    )

    assert doptimal.candidates != enumerated
    assert engine._require_selection_matches_doptimal(plan, doptimal) is None


def test_doptimal_checkpoint_identity_uses_the_canonical_state_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first: Variant = frozenset({(0, "A", "C")})
    second: Variant = frozenset({(1, "A", "C")})
    enumerated = (second, first)
    base = _registered_inputs()
    registered = replace(
        base,
        candidates=enumerated,
        config=_build_fourier_config((0, 1), ("A", "A"), "AC", max_order=2),
    )
    monkeypatch.setattr(
        engine,
        "REGISTERED_RECOVERY_PROTOCOL",
        replace(REGISTERED_RECOVERY_PROTOCOL, budgets=(1,)),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    store = ContentAddressedRunStore(run_dir)
    store.initialise()
    journal = RecoveryStateCursor.open(RunStoreSession.open(store))

    observed = engine._restore_doptimal_cursor(
        journal,
        _runtime_record(),
        registered,
    )

    assert observed.state.candidates != enumerated
    assert _sequence_sha256(observed.state.candidates) != candidate_sha256(enumerated)


def test_doptimal_publication_separates_universe_and_sequence_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first: Variant = frozenset({(0, "A", "C")})
    second: Variant = frozenset({(1, "A", "C")})
    enumerated = (second, first)
    base = _registered_inputs()
    registered = replace(
        base,
        candidates=enumerated,
        config=_build_fourier_config((0, 1), ("A", "A"), "AC", max_order=2),
    )
    protocol = replace(REGISTERED_RECOVERY_PROTOCOL, budgets=(1,))
    policy = replace(engine.REGISTERED_EXECUTION_POLICY, doptimal_block_size=1)
    monkeypatch.setattr(engine, "REGISTERED_RECOVERY_PROTOCOL", protocol)
    monkeypatch.setattr(engine, "REGISTERED_EXECUTION_POLICY", policy)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    store = ContentAddressedRunStore(run_dir)
    store.initialise()
    journal = RecoveryStateCursor.open(
        RunStoreSession.open(store),
        protocol=protocol,
        execution_policy=policy,
    )
    runtime = _runtime_record()
    numerical = runtime.numeric_compatibility.payload()
    prepared = PreparedRecoveryRun(
        scientific_identity_sha256=runtime.scientific_identity.sha256,
        protocol_semantic_sha256=protocol.semantic_sha256,
        execution_policy_sha256=policy.policy_sha256,
        numerical_compatibility_sha256=hashlib.sha256(canonical_json_bytes(numerical)).hexdigest(),
        candidate_sha256=candidate_sha256(enumerated),
        runtime_record_ref=store.put_json({"runtime": "synthetic"}),
        input_bundle_ref=store.put_json({"inputs": "synthetic"}),
    )
    engine.publish_prepared_run_at(journal, prepared)

    doptimal = engine._restore_doptimal_cursor(journal, runtime, registered)
    engine.advance_reduced_doptimal(doptimal.state, 1)
    manifest = doptimal.publish(doptimal.state)

    assert manifest.sequence == 1
    assert journal.snapshot().doptimal_completed == 1


def test_integrated_recovery_resumes_after_an_unmarked_doptimal_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, policy, registered = _integrated_recovery_fixture()
    monkeypatch.setattr(engine, "REGISTERED_RECOVERY_PROTOCOL", protocol)
    monkeypatch.setattr(engine, "REGISTERED_EXECUTION_POLICY", policy)
    monkeypatch.setattr(engine, "_require_registered_input_digests", lambda _digests: None)
    monkeypatch.setattr(engine, "_clean_commit", lambda _repo: "a" * 40)
    monkeypatch.setattr(engine, "capture_numeric_compatibility", _compatibility)
    monkeypatch.setattr(
        engine,
        "_validate_registered_inputs",
        lambda *_args, **_kwargs: registered,
    )
    monkeypatch.setattr(
        engine,
        "_open_journal",
        lambda store: RecoveryStateCursor.open(
            RunStoreSession.open(store),
            protocol=protocol,
            execution_policy=policy,
        ),
    )
    monkeypatch.setattr(engine, "registered_cell_keys", lambda: registered_cell_keys(protocol))

    original_publish = ContentAddressedRunStore._publish
    interrupted = False

    def interrupt_first_doptimal_manifest(
        store: ContentAddressedRunStore,
        payload_path: Path,
        content: bytes,
        kind: str,
        expected_sha256: str | None,
    ) -> None:
        nonlocal interrupted
        decoded = json.loads(content) if kind == "manifest" else None
        meta = decoded.get("meta") if isinstance(decoded, dict) else None
        if (
            not interrupted
            and isinstance(meta, dict)
            and meta.get("state_kind") == "reduced_doptimal"
        ):
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload_path.write_bytes(content)
            interrupted = True
            raise OSError("synthetic manifest interruption")
        original_publish(store, payload_path, content, kind, expected_sha256)

    monkeypatch.setattr(ContentAddressedRunStore, "_publish", interrupt_first_doptimal_manifest)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    inputs = _input_paths(tmp_path)
    prepared = prepare_recovery_run(run_dir, *inputs, repo=tmp_path)
    assert prepared.prepared is True

    with pytest.raises(OSError, match="synthetic manifest interruption"):
        engine.run_recovery(run_dir, repo=tmp_path)

    interrupted_audit = ContentAddressedRunStore(run_dir).verify()
    assert interrupted is True
    assert interrupted_audit.has_errors is False
    assert "missing_marker" in interrupted_audit.problems()

    completed = engine.run_recovery(run_dir, repo=tmp_path)
    verified = verify_recovery_run(run_dir)
    exported = export_recovery_report(run_dir, tmp_path / "report.json")
    final_store = ContentAddressedRunStore(run_dir)
    final_state = replay_recovery_state(
        final_store,
        protocol=protocol,
        execution_policy=policy,
    )

    assert completed == verified
    assert completed.report_available is True
    assert completed.doptimal_completed == protocol.budgets[-1]
    assert completed.completed_cells == protocol.cell_count
    assert len(final_state.abandoned_execution_attempts) == 1
    assert final_state.finalized_execution_attempt is not None
    assert exported.stat().st_size > 0
    assert final_store.verify().has_errors is False


def test_execution_history_binds_attempt_times_workspace_and_archived_inputs() -> None:
    runtime = _runtime_record()
    bundle = _input_bundle()
    runtime_ref = BlobRef("8" * 64, 1, "json")
    bundle_ref = BlobRef("9" * 64, 1, "json")
    prepared = PreparedRecoveryRun(
        scientific_identity_sha256=runtime.scientific_identity.sha256,
        protocol_semantic_sha256=REGISTERED_RECOVERY_PROTOCOL.semantic_sha256,
        execution_policy_sha256=engine.REGISTERED_EXECUTION_POLICY.policy_sha256,
        numerical_compatibility_sha256="7" * 64,
        candidate_sha256="6" * 64,
        runtime_record_ref=runtime_ref,
        input_bundle_ref=bundle_ref,
    )
    start = ExecutionAttemptStart(
        attempt_id="attempt-1",
        scientific_identity_sha256=runtime.scientific_identity.sha256,
        runtime_record_ref=runtime_ref,
        input_bundle_ref=bundle_ref,
        commit_sha="a" * 40,
        workspace_clean=True,
        scientific_diff_sha256=None,
        argv=("python", "scripts/fourier_recovery_curve.py", "run"),
        started_utc="2026-08-11T01:00:00+00:00",
    )
    start_ref = BlobRef("5" * 64, 1, "json")
    completion = ExecutionAttemptCompletion(
        attempt_id=start.attempt_id,
        start_ref=start_ref,
        commit_sha="a" * 40,
        workspace_clean=True,
        scientific_diff_sha256=None,
        completed_utc="2026-08-11T02:00:00+00:00",
    )
    state = replace(
        _empty_run_state(),
        prepared=prepared,
        execution_attempts=(
            ExecutionAttemptRecord(
                start=start,
                start_ref=start_ref,
                completion=completion,
                completion_ref=BlobRef("4" * 64, 1, "json"),
            ),
        ),
    )
    payload = engine._execution_history_provenance(state, runtime, bundle)

    assert engine._require_execution_history(payload, state, runtime, bundle) == payload
    drifted = dict(payload)
    drifted["attempts"] = []
    with pytest.raises(RecoveryEngineError, match="history"):
        engine._require_execution_history(drifted, state, runtime, bundle)


def test_report_bytes_depend_only_on_durable_scientific_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registered = _registered_inputs()
    candidate = registered.candidates[0]
    plan = SelectionPlan(
        budgets=REGISTERED_RECOVERY_PROTOCOL.budgets,
        sequences=tuple(
            SelectionSequence(
                method=method,
                seed=seed,
                selected=(candidate,),
                selected_sha256="6" * 64,
                tie_break_version="test",
            )
            for method, seed in REGISTERED_RECOVERY_PROTOCOL.sequence_keys
        ),
    )
    cells = tuple(
        RecoveryCell(
            method=key.method,
            seed=key.seed,
            budget=key.budget,
            spearman=None,
            relative_sse_gain=None,
            support_size=0,
            coefficient_count=REGISTERED_RECOVERY_PROTOCOL.coefficient_count,
            converged=False,
            error="constant response",
        )
        for key in registered_cell_keys()
    )
    monkeypatch.setattr(engine, "_stored_cells", lambda _store, _state: cells)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    store = ContentAddressedRunStore(run_dir)
    truth = PairwiseTruth(
        modes=((1, 1),),
        coefficients=np.array([1.0], dtype=np.float64),
    )

    state = _empty_run_state()
    execution = {"input_sha256": {}, "attempts": []}
    first = engine._report_payload(
        store, state, _runtime_record(), registered, plan, truth, execution
    )
    second = engine._report_payload(
        store, state, _runtime_record(), registered, plan, truth, execution
    )

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    provenance = first["provenance"]
    assert isinstance(provenance, dict)
    assert "started_utc" not in provenance


def test_scientific_verification_rejects_a_forged_report_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    store = ContentAddressedRunStore(run_dir)
    store.initialise()
    forged_ref = store.put_json(
        {
            "cells": [],
            "decision": {"status": "forged"},
            "cell_results_sha256": "0" * 64,
            "provenance": {"execution": {"synthetic": True}},
        }
    )
    plan_ref = store.put_json({"plan": "synthetic"})
    state = replace(
        _empty_run_state(),
        selection_plan=PublishedSelectionPlan("1" * 64, plan_ref.sha256, plan_ref),
        report_ref=forged_ref,
    )
    archived = SimpleNamespace(
        cache=object(), sidecar=object(), runtime_preflight=object(), dataset=object()
    )
    monkeypatch.setattr(engine, "_runtime_and_bundle", lambda *_args: (_runtime_record(), archived))
    monkeypatch.setattr(
        engine,
        "materialize_archived_recovery_input",
        lambda *_args: tmp_path / "materialized",
    )
    monkeypatch.setattr(
        engine,
        "_validate_registered_inputs",
        lambda *_args, **_kwargs: _registered_inputs(),
    )
    monkeypatch.setattr(
        engine,
        "_restore_doptimal_cursor",
        lambda *_args: SimpleNamespace(state=object()),
    )
    monkeypatch.setattr(engine, "_load_selection_plan", lambda *_args: object())
    monkeypatch.setattr(engine, "_require_selection_matches_doptimal", lambda *_args: None)
    monkeypatch.setattr(
        engine,
        "pairwise_truth",
        lambda *_args: PairwiseTruth(
            modes=(),
            coefficients=np.zeros(REGISTERED_RECOVERY_PROTOCOL.coefficient_count, dtype=np.float64),
        ),
    )
    monkeypatch.setattr(
        engine,
        "_report_payload",
        lambda *_args: {"cells": [], "decision": {"status": "recomputed"}},
    )
    monkeypatch.setattr(engine, "_require_execution_history", lambda *_args: {})

    with pytest.raises(RecoveryEngineError, match="recomputed scientific"):
        engine._verify_selection_journal(
            store, cast("RecoveryStateCursor", SimpleNamespace()), state
        )
