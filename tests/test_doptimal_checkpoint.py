"""Offline contracts for the blockwise reduced D-optimal journal."""

# ruff: noqa: PLR2004

from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from epibudget.coeff_recovery import _build_fourier_config
from epibudget.fourier_recovery import (
    ReducedDOptimalState,
    _canonical_identity_sha256,
    _sequence_sha256,
    advance_reduced_doptimal,
    initialise_reduced_doptimal,
    selected_reduced_doptimal,
)
from epibudget.recovery_doptimal import (
    DOptimalCheckpointCursor,
    DOptimalCheckpointError,
    DOptimalCheckpointIdentity,
    doptimal_geometry_sha256,
    publish_doptimal_checkpoint,
    restore_doptimal_checkpoint,
)
from epibudget.recovery_protocol import (
    REGISTERED_EXECUTION_POLICY,
    REGISTERED_RECOVERY_PROTOCOL,
    RecoveryExecutionPolicy,
)
from epibudget.recovery_state import (
    PreparedRecoveryRun,
    RecoveryStateCursor,
    publish_prepared_run_at,
)
from epibudget.run_store import (
    ArrayRef,
    ContentAddressedRunStore,
    Manifest,
    RunStoreSession,
    StoreCorruptionError,
    canonical_json_bytes,
)
from epibudget.types import Variant

_Q4 = "ACDE"
_Q3 = "ACD"
_SITES = (0, 1, 2, 3)
_WT = ("A", "A", "A", "A")
_TARGET = 192


def _candidates() -> list[Variant]:
    return [
        frozenset(
            (site, _WT[index], aa)
            for index, (site, aa) in enumerate(zip(_SITES, residues, strict=True))
            if aa != _WT[index]
        )
        for residues in product(_Q4, repeat=len(_SITES))
        if residues != _WT
    ]


def _initial_state(target_budget: int = _TARGET) -> ReducedDOptimalState:
    return initialise_reduced_doptimal(
        _build_fourier_config(_SITES, _WT, _Q4, max_order=2),
        _candidates(),
        target_budget=target_budget,
    )


def _frozen_state() -> ReducedDOptimalState:
    candidates = [
        frozenset(
            (site, _WT[index], aa)
            for index, (site, aa) in enumerate(zip(_SITES, residues, strict=True))
            if aa != _WT[index]
        )
        for residues in product(_Q3, repeat=len(_SITES))
        if residues != _WT
    ]
    return initialise_reduced_doptimal(
        _build_fourier_config(_SITES, _WT, _Q3, max_order=2),
        candidates,
        target_budget=72,
    )


def _identity(
    state: ReducedDOptimalState,
    policy: RecoveryExecutionPolicy = REGISTERED_EXECUTION_POLICY,
) -> DOptimalCheckpointIdentity:
    numerical = {"numpy": np.__version__, "blas": "synthetic-offline"}
    return DOptimalCheckpointIdentity(
        scientific_identity_sha256="1" * 64,
        execution_policy=policy,
        numerical_compatibility=numerical,
        numerical_compatibility_sha256=hashlib.sha256(canonical_json_bytes(numerical)).hexdigest(),
        candidate_universe_sha256=hashlib.sha256(b"candidate-universe").hexdigest(),
        candidate_sequence_sha256=_sequence_sha256(state.candidates),
        candidate_count=len(state.candidates),
        target_budget=state.target_budget,
        geometry_sha256=doptimal_geometry_sha256(state),
    )


def _store(path: Path) -> ContentAddressedRunStore:
    path.mkdir()
    store = ContentAddressedRunStore(path)
    store.initialise()
    return store


def _array_refs(manifest: Manifest) -> dict[str, ArrayRef]:
    arrays = manifest.meta["arrays"]
    assert isinstance(arrays, dict)
    return {name: ArrayRef.from_payload(payload) for name, payload in arrays.items()}


def _recovery_cursor(
    store: ContentAddressedRunStore,
    identity: DOptimalCheckpointIdentity,
) -> RecoveryStateCursor:
    protocol = replace(REGISTERED_RECOVERY_PROTOCOL, budgets=(64, 128, _TARGET))
    cursor = RecoveryStateCursor.open(
        RunStoreSession.open(store),
        protocol=protocol,
        execution_policy=identity.execution_policy,
    )
    if cursor.snapshot().prepared is None:
        publish_prepared_run_at(
            cursor,
            PreparedRecoveryRun(
                scientific_identity_sha256=identity.scientific_identity_sha256,
                protocol_semantic_sha256=protocol.semantic_sha256,
                execution_policy_sha256=identity.execution_policy.policy_sha256,
                numerical_compatibility_sha256=identity.numerical_compatibility_sha256,
                candidate_sha256=identity.candidate_universe_sha256,
                runtime_record_ref=store.put_json({"runtime": "synthetic"}),
                input_bundle_ref=store.put_json({"inputs": "synthetic"}),
            ),
        )
    return cursor


def test_incremental_cursor_restores_each_old_array_once_and_never_rescans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path / "run")
    initial = _initial_state()
    identity = _identity(initial)
    recovery = _recovery_cursor(store, identity)
    checkpoint = DOptimalCheckpointCursor.restore(recovery, initial, identity)
    advance_reduced_doptimal(checkpoint.state, 64)
    first = checkpoint.publish(checkpoint.state)
    advance_reduced_doptimal(checkpoint.state, 128)
    second = checkpoint.publish(checkpoint.state)
    old_digests = {
        reference.blob.sha256
        for manifest in (first, second)
        for reference in _array_refs(manifest).values()
    }
    reopened = RecoveryStateCursor.open(
        RunStoreSession.open(store),
        protocol=replace(REGISTERED_RECOVERY_PROTOCOL, budgets=(64, 128, _TARGET)),
        execution_policy=identity.execution_policy,
    )
    reads: dict[str, int] = {}
    original_get_array = store.get_array

    def counted_get_array(reference: ArrayRef) -> np.ndarray:
        reads[reference.blob.sha256] = reads.get(reference.blob.sha256, 0) + 1
        return original_get_array(reference)

    monkeypatch.setattr(store, "get_array", counted_get_array)
    resumed = DOptimalCheckpointCursor.restore(reopened, _initial_state(), identity)

    assert {digest: reads[digest] for digest in old_digests} == {
        digest: 1 for digest in old_digests
    }
    monkeypatch.setattr(
        store,
        "manifest_chain",
        lambda: (_ for _ in ()).throw(AssertionError("journal replay")),
    )
    advance_reduced_doptimal(resumed.state, _TARGET)
    resumed.publish(resumed.state)
    assert {digest: reads[digest] for digest in old_digests} == {
        digest: 1 for digest in old_digests
    }


def test_incremental_cursor_retries_exact_and_rejects_divergent_or_stale_state(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "run")
    state = _initial_state()
    identity = _identity(state)
    checkpoint = DOptimalCheckpointCursor.restore(
        _recovery_cursor(store, identity), state, identity
    )
    advance_reduced_doptimal(state, 64)
    first = checkpoint.publish(state)

    assert checkpoint.publish(state) == first
    committed_posterior = state.posterior_variance.copy()
    state.posterior_variance[0] += 1.0
    with pytest.raises(DOptimalCheckpointError, match="committed"):
        checkpoint.publish(state)
    state.posterior_variance[:] = committed_posterior
    advance_reduced_doptimal(state, _TARGET)
    before = store.verify()
    with pytest.raises(DOptimalCheckpointError, match="exactly one"):
        checkpoint.publish(state)
    after = store.verify()
    assert (after.manifest_count, after.blob_count) == (
        before.manifest_count,
        before.blob_count,
    )


def test_incremental_cursor_resume_preserves_the_uninterrupted_oracle(tmp_path: Path) -> None:
    direct = _initial_state()
    advance_reduced_doptimal(direct, _TARGET)
    store = _store(tmp_path / "run")
    identity = _identity(_initial_state())
    checkpoint = DOptimalCheckpointCursor.restore(
        _recovery_cursor(store, identity), _initial_state(), identity
    )
    for stop in (64, 128):
        advance_reduced_doptimal(checkpoint.state, stop)
        checkpoint.publish(checkpoint.state)
    reopened = RecoveryStateCursor.open(
        RunStoreSession.open(store),
        protocol=replace(REGISTERED_RECOVERY_PROTOCOL, budgets=(64, 128, _TARGET)),
        execution_policy=identity.execution_policy,
    )
    resumed = DOptimalCheckpointCursor.restore(reopened, _initial_state(), identity)
    advance_reduced_doptimal(resumed.state, _TARGET)

    assert selected_reduced_doptimal(resumed.state) == selected_reduced_doptimal(direct)
    assert resumed.state.posterior_variance.tobytes() == direct.posterior_variance.tobytes()
    assert resumed.state.updates.tobytes() == direct.updates.tobytes()


def test_checkpoint_identity_owns_an_immutable_numeric_payload() -> None:
    state = _initial_state()
    numerical: dict[str, object] = {"numpy": {"version": "synthetic"}}
    identity = DOptimalCheckpointIdentity(
        scientific_identity_sha256="1" * 64,
        execution_policy=REGISTERED_EXECUTION_POLICY,
        numerical_compatibility=numerical,
        numerical_compatibility_sha256=hashlib.sha256(canonical_json_bytes(numerical)).hexdigest(),
        candidate_universe_sha256=hashlib.sha256(b"candidate-universe").hexdigest(),
        candidate_sequence_sha256=_sequence_sha256(state.candidates),
        candidate_count=len(state.candidates),
        target_budget=state.target_budget,
        geometry_sha256=doptimal_geometry_sha256(state),
    )

    numerical["numpy"] = {"version": "drifted"}

    assert identity.payload()["numerical_compatibility"] == {"numpy": {"version": "synthetic"}}


def test_checkpoint_identity_derives_block_size_from_the_complete_policy() -> None:
    policy = replace(REGISTERED_EXECUTION_POLICY, doptimal_block_size=32)
    identity = _identity(_initial_state(), policy)

    assert identity.block_size == 32
    assert identity.payload()["execution_policy"] == policy.identity_payload()
    assert "policy_sha256" not in identity.payload()
    assert "block_size" not in identity.payload()


def test_checkpoint_at_64_restores_the_frozen_72_pivot_oracle_bitwise(
    tmp_path: Path,
) -> None:
    direct = _frozen_state()
    advance_reduced_doptimal(direct, 72)
    paused = _frozen_state()
    identity = _identity(paused)
    store = _store(tmp_path / "run")

    advance_reduced_doptimal(paused, 64)
    publish_doptimal_checkpoint(store, paused, identity)
    resumed = restore_doptimal_checkpoint(store, _frozen_state(), identity)
    advance_reduced_doptimal(resumed, 72)

    assert _sequence_sha256(selected_reduced_doptimal(resumed)) == (
        "fec1a04174952f0141b18de1df208fb9c10a14501ff925328b592ff0c5d269f1"
    )
    assert resumed.posterior_variance.tobytes() == direct.posterior_variance.tobytes()
    assert resumed.updates.tobytes() == direct.updates.tobytes()


@pytest.mark.parametrize("damage", ["q", "site_indices"])
def test_state_geometry_must_match_the_checkpoint_identity(tmp_path: Path, damage: str) -> None:
    state = _initial_state()
    identity = _identity(state)
    if damage == "q":
        state.q += 1
    else:
        state.site_indices[0, 0] = (state.site_indices[0, 0] + 1) % state.q
    store = _store(tmp_path / "run")

    with pytest.raises(DOptimalCheckpointError, match="geometry"):
        restore_doptimal_checkpoint(store, state, identity)


def test_state_candidate_sequence_must_match_the_checkpoint_identity(tmp_path: Path) -> None:
    state = _initial_state()
    identity = _identity(state)
    state.candidates = tuple(reversed(state.candidates))
    store = _store(tmp_path / "run")

    with pytest.raises(DOptimalCheckpointError, match="candidate hash"):
        restore_doptimal_checkpoint(store, state, identity)


def test_two_deltas_have_exact_entries_array_layout_and_full_capacity_strides(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "run")
    state = _initial_state()
    identity = _identity(state)
    prepare = store.publish_manifest(
        entries={"prepare": store.put_json({"ready": True})},
        meta={"state_kind": "recovery_prepare"},
        parent=None,
    )

    advance_reduced_doptimal(state, 64)
    first = publish_doptimal_checkpoint(store, state, identity)
    advance_reduced_doptimal(state, 128)
    second = publish_doptimal_checkpoint(store, state, identity)

    assert first.parent_sha256 == prepare.sha256
    assert second.parent_sha256 == first.sha256
    assert [manifest.meta["state_kind"] for manifest in store.manifest_chain()] == [
        "recovery_prepare",
        "reduced_doptimal",
        "reduced_doptimal",
    ]
    for manifest, start, stop in ((first, 0, 64), (second, 64, 128)):
        assert set(manifest.entries) == {
            "posterior_variance",
            "selected_indices",
            "updates",
        }
        assert set(manifest.meta) == {
            "arrays",
            "identity",
            "prefix_sha256",
            "schema_version",
            "start",
            "state_kind",
            "stop",
        }
        assert manifest.meta["start"] == start
        assert manifest.meta["stop"] == stop
        references = _array_refs(manifest)
        assert set(references) == set(manifest.entries)
        for name, reference in references.items():
            assert reference.blob == manifest.entries[name]
            assert reference.byteorder == "little"
            assert reference.order == "C"
        assert references["updates"].dtype == "float64"
        assert references["updates"].shape == (len(state.candidates), 64)
        assert references["selected_indices"].dtype == "int64"
        assert references["selected_indices"].shape == (64,)
        assert references["posterior_variance"].dtype == "float64"
        assert references["posterior_variance"].shape == (len(state.candidates),)

    initial = _initial_state()
    original_updates = initial.updates
    restored = restore_doptimal_checkpoint(store, initial, identity)
    assert restored is initial
    assert restored.updates is original_updates
    assert restored.updates.shape == (len(state.candidates), _TARGET)
    assert restored.updates.flags.c_contiguous
    assert restored.updates.strides == (_TARGET * np.dtype(np.float64).itemsize, 8)


def test_checkpoint_publication_never_allocates_a_second_full_update_buffer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path / "run")
    state = _initial_state()
    identity = _identity(state)
    advance_reduced_doptimal(state, 64)
    original_zeros = np.zeros
    full_shape = state.updates.shape

    def guarded_zeros(shape: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        if tuple(shape) == full_shape:
            raise AssertionError("publication allocated a second full update buffer")
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr("epibudget.recovery_doptimal.np.zeros", guarded_zeros)

    publish_doptimal_checkpoint(store, state, identity)


def test_restore_requires_a_fresh_empty_initial_state(tmp_path: Path) -> None:
    store = _store(tmp_path / "run")
    initial = _initial_state()
    identity = _identity(initial)
    advance_reduced_doptimal(initial, 1)

    with pytest.raises(DOptimalCheckpointError, match="empty initial state"):
        restore_doptimal_checkpoint(store, initial, identity)


def test_resume_to_192_is_byte_exact_with_an_uninterrupted_state(tmp_path: Path) -> None:
    direct = _initial_state()
    advance_reduced_doptimal(direct, _TARGET)
    paused = _initial_state()
    identity = _identity(paused)
    store = _store(tmp_path / "run")

    for stop in (64, 128):
        advance_reduced_doptimal(paused, stop)
        publish_doptimal_checkpoint(store, paused, identity)
    resumed = restore_doptimal_checkpoint(store, _initial_state(), identity)
    advance_reduced_doptimal(resumed, _TARGET)

    assert selected_reduced_doptimal(resumed) == selected_reduced_doptimal(direct)
    assert resumed.posterior_variance.tobytes() == direct.posterior_variance.tobytes()
    assert resumed.updates.tobytes() == direct.updates.tobytes()


def test_restore_uses_the_last_complete_manifest_and_keeps_the_incomplete_copy(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "run")
    state = _initial_state()
    identity = _identity(state)
    advance_reduced_doptimal(state, 64)
    first = publish_doptimal_checkpoint(store, state, identity)
    advance_reduced_doptimal(state, 128)
    second = publish_doptimal_checkpoint(store, state, identity)
    payload = next(
        path
        for path in (tmp_path / "run" / "manifests").glob("*.manifest.json")
        if second.sha256 in path.name
    )
    marker = payload.with_name(f"{payload.name}.complete")
    marker.unlink()

    restored = restore_doptimal_checkpoint(store, _initial_state(), identity)

    assert len(restored.selected_indices) == 64
    assert store.latest_manifest() == first
    assert payload.is_file()


def test_identical_checkpoint_publication_is_idempotent_and_conflict_diverges(
    tmp_path: Path,
) -> None:
    writer_a = _store(tmp_path / "run")
    writer_b = ContentAddressedRunStore(tmp_path / "run")
    prepare = writer_a.publish_manifest(
        entries={"prepare": writer_a.put_json({"ready": True})},
        meta={"state_kind": "recovery_prepare"},
        parent=None,
    )
    state = _initial_state()
    identity = _identity(state)
    advance_reduced_doptimal(state, 64)

    first = publish_doptimal_checkpoint(writer_a, state, identity)
    assert first.parent_sha256 == prepare.sha256
    assert publish_doptimal_checkpoint(writer_b, state, identity) == first
    state.posterior_variance[0] += 1.0
    with pytest.raises(DOptimalCheckpointError, match="diverg"):
        publish_doptimal_checkpoint(writer_b, state, identity)


def test_store_corruption_is_not_reclassified_as_checkpoint_divergence(tmp_path: Path) -> None:
    store = _store(tmp_path / "run")
    state = _initial_state()
    identity = _identity(state)
    advance_reduced_doptimal(state, 64)
    manifest = publish_doptimal_checkpoint(store, state, identity)
    references = _array_refs(manifest)
    updates_blob = references["updates"].blob.sha256
    blob_path = tmp_path / "run" / "blobs" / updates_blob[:2] / f"{updates_blob}.blob"
    blob_path.write_bytes(b"corrupt")

    with pytest.raises(StoreCorruptionError):
        restore_doptimal_checkpoint(store, _initial_state(), identity)


def test_restore_rejects_a_structurally_inconsistent_finite_delta(tmp_path: Path) -> None:
    store = _store(tmp_path / "run")
    state = _initial_state()
    identity = _identity(state)
    advance_reduced_doptimal(state, 64)
    initial_variance = _initial_state().posterior_variance
    _publish_raw_delta(
        store,
        state,
        identity,
        start=0,
        stop=64,
        updates=np.zeros((len(state.candidates), 64), dtype=np.float64),
        posterior=initial_variance,
    )

    with pytest.raises(DOptimalCheckpointError, match="structural"):
        restore_doptimal_checkpoint(store, _initial_state(), identity)


def test_restore_rejects_zero_updates_with_hash_order_pivots(tmp_path: Path) -> None:
    store = _store(tmp_path / "run")
    state = _initial_state()
    identity = _identity(state)
    state.selected_indices = sorted(
        range(len(state.candidates)),
        key=lambda index: _canonical_identity_sha256(state.candidates[index]),
    )[:64]
    _publish_raw_delta(
        store,
        state,
        identity,
        start=0,
        stop=64,
        updates=np.zeros((len(state.candidates), 64), dtype=np.float64),
        posterior=state.posterior_variance,
    )

    with pytest.raises(DOptimalCheckpointError, match="first update column"):
        restore_doptimal_checkpoint(store, _initial_state(), identity)


def test_publish_rejects_a_false_new_chunk_without_writing_store_objects(tmp_path: Path) -> None:
    store = _store(tmp_path / "run")
    state = _initial_state()
    identity = _identity(state)
    state.selected_indices = sorted(
        range(len(state.candidates)),
        key=lambda index: _canonical_identity_sha256(state.candidates[index]),
    )[:64]
    before = store.verify()

    with pytest.raises(DOptimalCheckpointError, match="first update column"):
        publish_doptimal_checkpoint(store, state, identity)

    after = store.verify()
    assert (after.manifest_count, after.blob_count) == (
        before.manifest_count,
        before.blob_count,
    )


def _publish_raw_delta(
    store: ContentAddressedRunStore,
    state: ReducedDOptimalState,
    identity: DOptimalCheckpointIdentity,
    *,
    start: int,
    stop: int,
    updates: np.ndarray | None = None,
    selected: np.ndarray | None = None,
    posterior: np.ndarray | None = None,
    identity_payload: dict[str, object] | None = None,
    prefix_sha256: str | None = None,
    parent: Manifest | None = None,
) -> Manifest:
    chosen = np.asarray(
        state.selected_indices[start:stop] if selected is None else selected,
        dtype=np.int64 if selected is None else selected.dtype,
        order="C",
    )
    delta = np.asarray(
        state.updates[:, start:stop] if updates is None else updates,
        dtype=np.float64 if updates is None else updates.dtype,
        order="C" if updates is None else "K",
    )
    variance = np.asarray(
        state.posterior_variance if posterior is None else posterior,
        dtype=np.float64 if posterior is None else posterior.dtype,
        order="C" if posterior is None else "K",
    )
    refs = {
        "selected_indices": store.put_array(chosen),
        "updates": store.put_array(delta),
        "posterior_variance": store.put_array(variance),
    }
    prefix = tuple(state.candidates[index] for index in state.selected_indices[:stop])
    return store.publish_manifest(
        entries={name: reference.blob for name, reference in refs.items()},
        meta={
            "schema_version": "epibudget-reduced-doptimal-delta-v2",
            "state_kind": "reduced_doptimal",
            "identity": identity.payload() if identity_payload is None else identity_payload,
            "start": start,
            "stop": stop,
            "prefix_sha256": _sequence_sha256(prefix) if prefix_sha256 is None else prefix_sha256,
            "arrays": {name: reference.payload() for name, reference in refs.items()},
        },
        parent=parent,
    )


@pytest.mark.parametrize(
    "damage",
    [
        "identity",
        "budget",
        "chunk_width",
        "gap",
        "overlap",
        "dtype",
        "shape",
        "order",
        "index",
        "nan",
        "inf",
        "negative_variance",
        "prefix_sha256",
    ],
)
def test_restore_fails_closed_on_malformed_or_incompatible_delta(
    tmp_path: Path, damage: str
) -> None:
    store = _store(tmp_path / "run")
    state = _initial_state()
    identity = _identity(state)
    advance_reduced_doptimal(state, 129)
    start, stop = 0, 64
    parent: Manifest | None = None
    kwargs: dict[str, Any] = {}
    if damage == "identity":
        payload = identity.payload()
        payload["scientific_identity_sha256"] = "2" * 64
        kwargs["identity_payload"] = payload
    elif damage == "budget":
        payload = identity.payload()
        payload["target_budget"] = _TARGET - 1
        kwargs["identity_payload"] = payload
    elif damage == "chunk_width":
        stop = 63
    elif damage in {"gap", "overlap"}:
        parent = _publish_raw_delta(
            store,
            state,
            identity,
            start=0,
            stop=64,
        )
        start, stop = (65, 129) if damage == "gap" else (63, 127)
    elif damage == "dtype":
        kwargs["updates"] = state.updates[:, :64].astype(np.float32)
    elif damage == "shape":
        kwargs["updates"] = state.updates[:-1, :64]
    elif damage == "order":
        kwargs["updates"] = np.asfortranarray(state.updates[:, :64])
    elif damage == "index":
        selected = np.asarray(state.selected_indices[:64], dtype=np.int64)
        selected[0] = len(state.candidates)
        kwargs["selected"] = selected
    elif damage == "nan":
        updates = state.updates[:, :64].copy()
        updates[0, 0] = np.nan
        kwargs["updates"] = updates
    elif damage == "inf":
        posterior = state.posterior_variance.copy()
        posterior[0] = np.inf
        kwargs["posterior"] = posterior
    elif damage == "negative_variance":
        posterior = state.posterior_variance.copy()
        posterior[0] = -1.0
        kwargs["posterior"] = posterior
    elif damage == "prefix_sha256":
        kwargs["prefix_sha256"] = "f" * 64

    _publish_raw_delta(
        store,
        state,
        identity,
        start=start,
        stop=stop,
        parent=parent,
        **kwargs,
    )

    with pytest.raises(DOptimalCheckpointError):
        restore_doptimal_checkpoint(store, _initial_state(), identity)


def test_doptimal_checkpoint_path_has_no_measured_label_input(tmp_path: Path) -> None:
    state = _initial_state()
    identity = _identity(state)
    store = _store(tmp_path / "run")
    advance_reduced_doptimal(state, 64)

    publish_doptimal_checkpoint(store, state, identity)
    restored = restore_doptimal_checkpoint(store, _initial_state(), identity)

    assert len(restored.selected_indices) == 64
    for function in (publish_doptimal_checkpoint, restore_doptimal_checkpoint):
        parameters = inspect.signature(function).parameters
        assert "landscape" not in parameters
        assert "fitness" not in parameters
        assert "measured" not in parameters
