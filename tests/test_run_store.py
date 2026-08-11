"""Offline adversarial tests for the durable content-addressed run store.

Every test here describes a way a slow, remote, or dishonest filesystem can hurt a long run:
publication interrupted at each step, a payload altered after the fact, a payload altered together
with its own marker so the two agree, a forged or stale parent, and two writers racing on the same
state. The store must reject all of them rather than silently returning partial or wrong data.
"""

# ruff: noqa: PLR2004

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import epibudget.run_store as run_store_module
from epibudget.run_store import (
    ArrayRef,
    BlobRef,
    ContentAddressedRunStore,
    ManifestDraft,
    RunStoreError,
    RunStoreSession,
    StoreCorruptionError,
    StoreDivergenceError,
    StoreDurabilityError,
    StoreReport,
    canonical_json_bytes,
)

_MARKER_SCHEMA = "epibudget-run-store-marker-v1"


def _store(root: Path) -> ContentAddressedRunStore:
    store = ContentAddressedRunStore(root)
    store.initialise()
    return store


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _marker_path(payload_path: Path) -> Path:
    return payload_path.with_name(payload_path.name + ".complete")


def _write_marker(payload_path: Path, content: bytes, kind: str) -> None:
    """Publish a marker that agrees perfectly with ``content``, as a tamperer would."""
    _marker_path(payload_path).write_bytes(
        canonical_json_bytes(
            {
                "schema_version": _MARKER_SCHEMA,
                "kind": kind,
                "name": payload_path.name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
            }
        )
    )


def _retamper(payload_path: Path, content: bytes, kind: str) -> None:
    """Replace a published payload and re-issue a marker consistent with the new bytes."""
    payload_path.write_bytes(content)
    _write_marker(payload_path, content, kind)


def _only_blob(root: Path) -> Path:
    return next((root / "blobs").rglob("*.blob"))


def _only_manifest(root: Path) -> Path:
    return next((root / "manifests").glob("*.manifest.json"))


def _problems(report: StoreReport) -> tuple[str, ...]:
    return report.problems()


# -- construction and durability ---------------------------------------------------


def test_root_must_already_exist(tmp_path: Path) -> None:
    with pytest.raises(RunStoreError, match="existing directory"):
        ContentAddressedRunStore(tmp_path / "absent")


def test_initialise_is_idempotent_and_leaves_no_probe_residue(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = _snapshot(tmp_path)

    store.initialise()

    assert (tmp_path / "store.json").is_file()
    assert list((tmp_path / "probe").iterdir()) == []
    assert _snapshot(tmp_path) == first


def test_durability_capabilities_are_reported_not_assumed(tmp_path: Path) -> None:
    capabilities = ContentAddressedRunStore(tmp_path).durability_capabilities()

    assert capabilities["file_fsync"] is True
    assert isinstance(capabilities["directory_fsync"], bool)
    if not hasattr(os, "O_DIRECTORY"):
        assert capabilities["directory_fsync"] is False


def test_directory_flush_never_propagates_a_filesystem_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Portable guard: the flush reports failure, it never raises out of a publication."""
    monkeypatch.setattr(run_store_module, "_directory_flag", lambda: 0)

    def refuse(*_args: object, **_kwargs: object) -> int:
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr(os, "open", refuse)

    assert run_store_module._sync_directory(tmp_path) is False


def test_a_filesystem_that_refuses_directory_fsync_still_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mounted network or FUSE volume rejects a directory flush; that must not fail a write."""
    original = os.fsync

    def refuse_directories(descriptor: int) -> None:
        if os.fstat(descriptor).st_mode & 0o170000 == 0o040000:
            raise OSError(22, "Invalid argument")
        original(descriptor)

    monkeypatch.setattr(os, "fsync", refuse_directories)
    store = ContentAddressedRunStore(tmp_path)
    store.initialise()

    blob = store.put_json({"step": 1})
    manifest = store.publish_manifest(entries={"a": blob}, meta={"n": 1}, parent=None)

    assert store.durability_capabilities()["directory_fsync"] is False
    assert store.get_json(manifest.entry("a")) == {"step": 1}
    assert store.verify().is_clean is True


def test_probe_rejects_a_filesystem_that_does_not_read_back_what_it_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentAddressedRunStore(tmp_path)
    original = Path.read_bytes

    def lying(self: Path) -> bytes:
        content = original(self)
        return content[:-1] if self.name.endswith(".partial") else content

    monkeypatch.setattr(Path, "read_bytes", lying)

    with pytest.raises(RunStoreError, match="read back"):
        store.probe_durability()


def test_probe_rejects_a_filesystem_that_alters_a_published_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentAddressedRunStore(tmp_path)
    original = Path.read_bytes

    def lying(self: Path) -> bytes:
        content = original(self)
        return content + b"\x00" if self.suffix == ".probe" else content

    monkeypatch.setattr(Path, "read_bytes", lying)

    with pytest.raises(RunStoreError, match="did not publish"):
        store.probe_durability()


# -- content addressing ------------------------------------------------------------


def test_blobs_are_content_addressed_and_archived_once(tmp_path: Path) -> None:
    store = _store(tmp_path)

    first = store.put_bytes(b"payload")
    second = store.put_bytes(b"payload")

    assert first == second
    assert first.sha256 == hashlib.sha256(b"payload").hexdigest()
    assert store.get_bytes(first) == b"payload"
    assert len(list((tmp_path / "blobs").rglob("*.blob"))) == 1


def test_json_blobs_ignore_key_order_and_reject_a_non_canonical_encoding(tmp_path: Path) -> None:
    store = _store(tmp_path)

    forward = store.put_json({"a": 2, "b": 1})
    reversed_keys = store.put_json({"b": 1, "a": 2})

    assert forward == reversed_keys
    assert store.get_json(forward) == {"a": 2, "b": 1}
    with pytest.raises(ValueError, match="not JSON-encoded"):
        store.get_json(store.put_bytes(b"{}"))


def test_a_blob_and_its_marker_altered_together_still_fail_the_address(tmp_path: Path) -> None:
    store = _store(tmp_path)
    reference = store.put_bytes(b"original")
    _retamper(_only_blob(tmp_path), b"tampered", "blob")

    assert len(b"original") == len(b"tampered")
    assert store.has_blob(reference.sha256) is False
    with pytest.raises(StoreCorruptionError, match="does not match its address"):
        store.get_bytes(reference)
    assert "invalid_marker" in _problems(store.verify())


def test_a_manifest_and_its_marker_altered_together_still_fail_the_address(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    blob = store.put_json({"step": 1})
    store.publish_manifest(entries={"a": blob}, meta={"n": 1}, parent=None)
    published = _only_manifest(tmp_path)
    body = json.loads(published.read_bytes())
    body["meta"] = {"n": 9}
    _retamper(published, canonical_json_bytes(body), "manifest")

    with pytest.raises(StoreCorruptionError, match="does not match its address"):
        store.manifest_chain()
    report = store.verify()
    assert report.manifest_count == 0
    assert "invalid_marker" in _problems(report)
    assert report.has_errors is True


def test_a_marker_digest_that_disagrees_with_the_name_is_refused(tmp_path: Path) -> None:
    store = _store(tmp_path)
    reference = store.put_bytes(b"original")
    payload_path = _only_blob(tmp_path)
    marker = json.loads(_marker_path(payload_path).read_bytes())
    marker["sha256"] = hashlib.sha256(b"something else").hexdigest()
    _marker_path(payload_path).write_bytes(canonical_json_bytes(marker))

    assert store.has_blob(reference.sha256) is False
    with pytest.raises(StoreCorruptionError, match="does not match its address"):
        store.get_bytes(reference)
    assert "invalid_marker" in _problems(store.verify())


def test_a_reference_with_the_right_address_but_the_wrong_size_is_refused(tmp_path: Path) -> None:
    store = _store(tmp_path)
    honest = store.put_bytes(b"original")
    lying = BlobRef(sha256=honest.sha256, size=honest.size + 1, encoding=honest.encoding)

    assert store.get_bytes(honest) == b"original"
    with pytest.raises(StoreCorruptionError, match="declares"):
        store.get_bytes(lying)


def test_a_manifest_entry_with_a_wrong_size_is_refused_by_the_whole_chain(tmp_path: Path) -> None:
    store = _store(tmp_path)
    honest = store.put_json({"step": 1})
    lying = BlobRef(sha256=honest.sha256, size=honest.size + 1, encoding="json")

    with pytest.raises(RunStoreError, match="does not resolve to a verified blob"):
        store.publish_manifest(entries={"a": lying}, meta={}, parent=None)


def test_a_payload_altered_after_publication_fails_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    reference = store.put_bytes(b"original")
    _only_blob(tmp_path).write_bytes(b"tampered")

    with pytest.raises(StoreCorruptionError, match="digest does not match"):
        store.get_bytes(reference)
    report = store.verify()
    assert "content_mismatch" in _problems(report)
    assert report.blob_count == 0


# -- interrupted publication -------------------------------------------------------


def test_a_payload_without_its_marker_is_never_valid_and_is_kept_for_diagnosis(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    reference = store.put_bytes(b"interrupted")
    _marker_path(_only_blob(tmp_path)).unlink()

    assert store.has_blob(reference.sha256) is False
    with pytest.raises(StoreCorruptionError, match="no valid completion marker"):
        store.get_bytes(reference)
    report = store.verify()
    assert report.blob_count == 0
    assert report.is_clean is False
    assert "missing_marker" in _problems(report)
    assert _only_blob(tmp_path).read_bytes() == b"interrupted"


def test_a_marker_without_its_payload_is_reported_as_an_orphan(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.put_bytes(b"interrupted")
    _only_blob(tmp_path).unlink()

    report = store.verify()

    assert "orphan_marker" in _problems(report)
    assert report.has_errors is True


def test_an_unreadable_marker_blocks_republication_instead_of_overwriting(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.put_bytes(b"original")
    _marker_path(_only_blob(tmp_path)).write_bytes(b"not json")

    with pytest.raises(StoreCorruptionError, match="unreadable or inconsistent"):
        store.put_bytes(b"original")
    assert "invalid_marker" in _problems(store.verify())


def test_an_interrupted_write_leaves_a_partial_file_that_is_reported(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.put_bytes(b"payload")
    (tmp_path / "blobs" / "orphaned.write.a1b2.partial").write_bytes(b"half")

    report = store.verify()

    assert "interrupted_write" in _problems(report)
    assert report.is_clean is False
    assert (tmp_path / "blobs" / "orphaned.write.a1b2.partial").is_file()


def test_an_unmarked_manifest_leaves_the_previous_state_current(tmp_path: Path) -> None:
    store = _store(tmp_path)
    blob = store.put_json({"step": 1})
    first = store.publish_manifest(entries={"a": blob}, meta={"n": 1}, parent=None)
    store.publish_manifest(entries={"a": blob}, meta={"n": 2}, parent=first)
    sorted((tmp_path / "manifests").glob("*.manifest.json.complete"))[-1].unlink()

    report = store.verify()

    assert store.latest_manifest() == first
    assert report.manifest_count == 1
    assert "missing_marker" in _problems(report)


def test_an_orphan_manifest_marker_is_reported(tmp_path: Path) -> None:
    store = _store(tmp_path)
    blob = store.put_json({"step": 1})
    store.publish_manifest(entries={"a": blob}, meta={"n": 1}, parent=None)
    _only_manifest(tmp_path).unlink()

    report = store.verify()

    assert "orphan_marker" in _problems(report)
    assert report.manifest_count == 0


# -- header --------------------------------------------------------------------------


def test_a_missing_header_is_reported_and_blocks_publication(tmp_path: Path) -> None:
    store = _store(tmp_path)
    blob = store.put_json({"step": 1})
    (tmp_path / "store.json").unlink()
    _marker_path(tmp_path / "store.json").unlink()

    assert store.has_valid_header() is False
    assert "missing_header" in _problems(store.verify())
    with pytest.raises(RunStoreError, match="no valid header"):
        store.publish_manifest(entries={"a": blob}, meta={}, parent=None)


def test_an_unmarked_header_is_reported(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _marker_path(tmp_path / "store.json").unlink()

    assert store.has_valid_header() is False
    assert "missing_marker" in _problems(store.verify())


def test_a_corrupt_header_is_reported(tmp_path: Path) -> None:
    store = _store(tmp_path)
    (tmp_path / "store.json").write_bytes(b'{"schema_version":"x"}')

    assert store.has_valid_header() is False
    assert "invalid_header" in _problems(store.verify())


def test_a_header_with_the_wrong_schema_is_reported(tmp_path: Path) -> None:
    store = _store(tmp_path)
    forged = canonical_json_bytes({"schema_version": "epibudget-run-store-v0"})
    _retamper(tmp_path / "store.json", forged, "header")

    assert store.has_valid_header() is False
    assert "invalid_header" in _problems(store.verify())


# -- manifest chain and parents ------------------------------------------------------


def test_manifest_chain_links_states_without_a_mutable_head(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first_blob = store.put_json({"step": 1})
    second_blob = store.put_json({"step": 2})

    first = store.publish_manifest(entries={"a": first_blob}, meta={"n": 1}, parent=None)
    second = store.publish_manifest(
        entries={"a": first_blob, "b": second_blob}, meta={"n": 2}, parent=first
    )

    assert first.sequence == 0
    assert first.parent_sha256 is None
    assert second.parent_sha256 == first.sha256
    assert store.manifest_chain() == (first, second)
    assert store.latest_manifest() == second
    assert not list(tmp_path.rglob("HEAD*"))
    assert store.get_json(second.entry("b")) == {"step": 2}


def test_a_manifest_cannot_reference_a_blob_that_is_not_published(tmp_path: Path) -> None:
    store = _store(tmp_path)
    absent = BlobRef(sha256="0" * 64, size=3, encoding="json")

    with pytest.raises(RunStoreError, match="does not resolve to a verified blob"):
        store.publish_manifest(entries={"a": absent}, meta={}, parent=None)


def test_a_root_manifest_is_refused_once_the_chain_is_not_empty(tmp_path: Path) -> None:
    store = _store(tmp_path)
    blob = store.put_json({"step": 1})
    store.publish_manifest(entries={"a": blob}, meta={"n": 1}, parent=None)

    with pytest.raises(RunStoreError, match="would diverge from the published chain"):
        store.publish_manifest(entries={"a": blob}, meta={"n": 2}, parent=None)


def test_a_non_root_manifest_is_refused_on_an_empty_chain(tmp_path: Path) -> None:
    store = _store(tmp_path)
    blob = store.put_json({"step": 1})
    forged = store.publish_manifest(entries={"a": blob}, meta={"n": 1}, parent=None)
    for path in sorted((tmp_path / "manifests").iterdir()):
        path.unlink()

    with pytest.raises(RunStoreError, match="only accept a root manifest"):
        store.publish_manifest(entries={"a": blob}, meta={"n": 2}, parent=forged)


def test_a_stale_parent_is_refused(tmp_path: Path) -> None:
    store = _store(tmp_path)
    blob = store.put_json({"step": 1})
    first = store.publish_manifest(entries={"a": blob}, meta={"n": 1}, parent=None)
    store.publish_manifest(entries={"a": blob}, meta={"n": 2}, parent=first)

    with pytest.raises(RunStoreError, match="stale or forged"):
        store.publish_manifest(entries={"a": blob}, meta={"n": 3}, parent=first)


def test_a_forged_parent_is_refused(tmp_path: Path) -> None:
    store = _store(tmp_path)
    blob = store.put_json({"step": 1})
    first = store.publish_manifest(entries={"a": blob}, meta={"n": 1}, parent=None)
    forged = type(first)(
        sequence=first.sequence,
        parent_sha256=None,
        entries=first.entries,
        meta=first.meta,
        sha256="f" * 64,
    )

    with pytest.raises(RunStoreError, match="stale or forged"):
        store.publish_manifest(entries={"a": blob}, meta={"n": 2}, parent=forged)


def test_identical_concurrent_writers_are_idempotent(tmp_path: Path) -> None:
    writer_a = _store(tmp_path)
    writer_b = ContentAddressedRunStore(tmp_path)
    blob = writer_a.put_json({"step": 1})

    from_a = writer_a.publish_manifest(entries={"a": blob}, meta={"n": 1}, parent=None)
    from_b = writer_b.publish_manifest(entries={"a": blob}, meta={"n": 1}, parent=None)
    next_a = writer_a.publish_manifest(entries={"a": blob}, meta={"n": 2}, parent=from_a)
    next_b = writer_b.publish_manifest(entries={"a": blob}, meta={"n": 2}, parent=from_b)

    assert from_a == from_b
    assert next_a == next_b
    assert len(list((tmp_path / "manifests").glob("*.manifest.json"))) == 2
    assert writer_b.verify().is_clean is True


def test_divergent_states_already_on_disk_are_reported_never_resolved(tmp_path: Path) -> None:
    store = _store(tmp_path)
    blob = store.put_json({"step": 1})
    store.publish_manifest(entries={"a": blob}, meta={"n": 1}, parent=None)
    rival = canonical_json_bytes(
        {
            "schema_version": "epibudget-run-store-manifest-v1",
            "sequence": 0,
            "parent_sha256": None,
            "entries": {"a": blob.payload()},
            "meta": {"n": 2},
        }
    )
    digest = hashlib.sha256(rival).hexdigest()
    rival_path = tmp_path / "manifests" / f"{0:012d}.{digest}.manifest.json"
    _retamper(rival_path, rival, "manifest")

    with pytest.raises(StoreDivergenceError, match="two different states"):
        store.manifest_chain()
    report = store.verify()
    assert "divergent_state" in _problems(report)
    assert report.manifest_count == 0


def test_a_gap_in_the_manifest_chain_fails_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    blob = store.put_json({"step": 1})
    first = store.publish_manifest(entries={"a": blob}, meta={"n": 1}, parent=None)
    store.publish_manifest(entries={"a": blob}, meta={"n": 2}, parent=first)
    sorted((tmp_path / "manifests").glob("*.manifest.json.complete"))[0].unlink()

    with pytest.raises(StoreCorruptionError, match="not contiguous"):
        store.manifest_chain()
    assert "broken_chain" in _problems(store.verify())


# -- arrays --------------------------------------------------------------------------


def test_arrays_round_trip_through_an_explicit_binary_description(tmp_path: Path) -> None:
    store = _store(tmp_path)
    array = np.arange(12, dtype=np.float64).reshape(3, 4)

    reference = store.put_array(array)
    restored = store.get_array(reference)

    assert reference.dtype == "float64"
    assert reference.byteorder == "little"
    assert reference.shape == (3, 4)
    assert reference.order == "C"
    assert reference.blob.size == 12 * 8
    assert np.array_equal(restored, array)
    assert restored.dtype == array.dtype
    assert ArrayRef.from_payload(reference.payload()) == reference


def test_fortran_ordered_arrays_keep_their_memory_order(tmp_path: Path) -> None:
    store = _store(tmp_path)
    array = np.asfortranarray(np.arange(6, dtype=np.int64).reshape(2, 3))

    reference = store.put_array(array)

    assert reference.order == "F"
    assert np.array_equal(store.get_array(reference), array)


def test_an_array_description_that_disagrees_with_the_bytes_is_refused(tmp_path: Path) -> None:
    store = _store(tmp_path)
    reference = store.put_array(np.arange(12, dtype=np.float64).reshape(3, 4))
    reshaped = ArrayRef(
        blob=reference.blob,
        dtype=reference.dtype,
        byteorder="little",
        shape=(5, 4),
        order="C",
    )

    with pytest.raises(StoreCorruptionError, match="unexpected byte count"):
        store.get_array(reshaped)


def test_object_arrays_are_refused_so_nothing_is_ever_pickled(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="not a plain numeric type"):
        store.put_array(np.array([object()], dtype=object))


# -- verification ---------------------------------------------------------------------


def test_verify_reads_the_whole_store_without_mutating_it(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first_blob = store.put_json({"step": 1})
    second_blob = store.put_bytes(b"\x00\x01\x02")
    first = store.publish_manifest(entries={"a": first_blob}, meta={"n": 1}, parent=None)
    store.publish_manifest(entries={"a": first_blob, "b": second_blob}, meta={"n": 2}, parent=first)
    before = _snapshot(tmp_path)

    report = store.verify()

    assert report.blob_count == 2
    assert report.manifest_count == 2
    assert report.latest_sequence == 1
    assert report.is_clean is True
    assert report.has_errors is False
    assert _snapshot(tmp_path) == before


def test_verify_stays_non_mutating_on_a_damaged_store(tmp_path: Path) -> None:
    store = _store(tmp_path)
    blob = store.put_json({"step": 1})
    store.publish_manifest(entries={"a": blob}, meta={"n": 1}, parent=None)
    store.put_bytes(b"unreferenced")
    _marker_path(tmp_path / "store.json").unlink()
    before = _snapshot(tmp_path)

    report = store.verify()

    assert report.has_errors is True
    assert _snapshot(tmp_path) == before


def test_a_complete_but_unreferenced_blob_is_reported_as_informational(tmp_path: Path) -> None:
    store = _store(tmp_path)
    referenced = store.put_json({"step": 1})
    store.publish_manifest(entries={"a": referenced}, meta={"n": 1}, parent=None)
    store.put_bytes(b"written before the crash")

    report = store.verify()

    assert report.blob_count == 2
    assert _problems(report) == ("unreferenced_blob",)
    assert report.is_clean is False
    assert report.has_errors is False


def test_an_empty_store_reports_no_state(tmp_path: Path) -> None:
    store = _store(tmp_path)

    report = store.verify()

    assert store.latest_manifest() is None
    assert store.manifest_chain() == ()
    assert report.manifest_count == 0
    assert report.latest_sequence is None
    assert report.is_clean is True


# -- append sessions ---------------------------------------------------------------


def test_session_opens_with_one_chain_scan_then_appends_without_old_scans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    blob = store.put_json({"step": 1})
    first = store.publish_manifest(entries={"a": blob}, meta={"n": 1}, parent=None)
    second = store.publish_manifest(entries={"a": blob}, meta={"n": 2}, parent=first)
    load_count = 0
    original_load = store._load_manifest

    def counted_load(sequence: int, digest: str):  # type: ignore[no-untyped-def]
        nonlocal load_count
        load_count += 1
        return original_load(sequence, digest)

    monkeypatch.setattr(store, "_load_manifest", counted_load)
    session = RunStoreSession.open(store)

    assert load_count == 2
    monkeypatch.setattr(
        store,
        "manifest_chain",
        lambda: (_ for _ in ()).throw(AssertionError("full chain rescan")),
    )
    monkeypatch.setattr(
        store,
        "_manifest_index",
        lambda: (_ for _ in ()).throw(AssertionError("manifest index rescan")),
    )
    monkeypatch.setattr(
        store,
        "_manifest_payloads",
        lambda: (_ for _ in ()).throw(AssertionError("manifest directory enumeration")),
    )
    monkeypatch.setattr(
        store,
        "_marked_manifests",
        lambda: (_ for _ in ()).throw(AssertionError("marked manifest enumeration")),
    )
    original_glob = Path.glob

    def reject_manifest_glob(self: Path, pattern: str):  # type: ignore[no-untyped-def]
        if self == tmp_path / "manifests":
            raise AssertionError("manifest directory glob")
        return original_glob(self, pattern)

    monkeypatch.setattr(Path, "glob", reject_manifest_glob)
    third = session.publish_manifest(session.draft_manifest(entries={"a": blob}, meta={"n": 3}))
    fourth = session.publish_manifest(session.draft_manifest(entries={"a": blob}, meta={"n": 4}))

    assert load_count == 6
    assert session.latest_manifest() == fourth
    assert session.manifests() == (first, second, third, fourth)


def test_session_draft_is_canonical_and_exact_retry_is_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    blob = store.put_json({"step": 1})
    session = RunStoreSession.open(store)
    draft = session.draft_manifest(entries={"a": blob}, meta={"n": 1})

    assert isinstance(draft, ManifestDraft)
    assert session.store is store
    first = session.publish_manifest(draft)
    retried = session.publish_manifest(draft)

    assert retried == first
    assert session.manifests() == (first,)


def test_session_accepts_old_exact_drafts_and_rejects_divergent_and_stale_drafts(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    blob = store.put_json({"step": 1})
    session = RunStoreSession.open(store)
    first = session.draft_manifest(entries={"a": blob}, meta={"n": 1})
    rival = session.draft_manifest(entries={"a": blob}, meta={"n": 2})
    session.publish_manifest(first)

    with pytest.raises(StoreDivergenceError, match="different state"):
        session.publish_manifest(rival)

    second = session.draft_manifest(entries={"a": blob}, meta={"n": 2})
    session.publish_manifest(second)
    assert session.publish_manifest(first) == session.manifests()[0]

    other_root = tmp_path / "other"
    other_root.mkdir()
    other = _store(other_root)
    other_blob = other.put_json({"step": 1})
    other_session = RunStoreSession.open(other)
    for value in range(3):
        other_session.publish_manifest(
            other_session.draft_manifest(entries={"a": other_blob}, meta={"other": value})
        )
    stale = other_session.draft_manifest(entries={"a": other_blob}, meta={"other": 3})
    with pytest.raises(RunStoreError, match="skips"):
        session.publish_manifest(stale)


def test_session_rejects_a_forged_manifest_draft_before_publication(tmp_path: Path) -> None:
    store = _store(tmp_path)
    blob = store.put_json({"step": 1})
    session = RunStoreSession.open(store)
    honest = session.draft_manifest(entries={"a": blob}, meta={"n": 1})
    forged = ManifestDraft(
        sequence=honest.sequence,
        parent_sha256=honest.parent_sha256,
        entries=honest.entries,
        meta={"n": 9},
        sha256=honest.sha256,
        content=honest.content,
    )

    with pytest.raises(RunStoreError, match="canonical"):
        session.publish_manifest(forged)
    assert not list((tmp_path / "manifests").glob("*.manifest.json"))


def test_session_can_be_explicitly_poisoned(tmp_path: Path) -> None:
    store = _store(tmp_path)
    blob = store.put_json({"step": 1})
    session = RunStoreSession.open(store)

    session.poison("writer ownership is uncertain")

    with pytest.raises(RunStoreError, match="ownership is uncertain"):
        session.draft_manifest(entries={"a": blob}, meta={"n": 1})


def test_external_divergent_writers_are_detected_on_full_reopen(tmp_path: Path) -> None:
    store = _store(tmp_path)
    blob = store.put_json({"step": 1})
    writer_a = RunStoreSession.open(store)
    writer_b = RunStoreSession.open(ContentAddressedRunStore(tmp_path))
    exact_a = writer_a.draft_manifest(entries={"a": blob}, meta={"n": 1})
    rival_b = writer_b.draft_manifest(entries={"a": blob}, meta={"n": 2})

    writer_a.publish_manifest(exact_a)
    writer_b.publish_manifest(rival_b)

    with pytest.raises(StoreDivergenceError, match="different state"):
        RunStoreSession.open(store)


def test_stale_external_writer_is_detected_when_it_creates_a_later_divergence(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    blob = store.put_json({"step": 1})
    current = RunStoreSession.open(store)
    stale = RunStoreSession.open(ContentAddressedRunStore(tmp_path))
    first_draft = stale.draft_manifest(entries={"a": blob}, meta={"n": 1})
    current.publish_manifest(current.draft_manifest(entries={"a": blob}, meta={"n": 1}))
    current.publish_manifest(current.draft_manifest(entries={"a": blob}, meta={"n": 2}))

    stale.publish_manifest(first_draft)
    stale.publish_manifest(stale.draft_manifest(entries={"a": blob}, meta={"n": 99}))

    with pytest.raises(StoreDivergenceError, match="different state"):
        RunStoreSession.open(store)


def test_shared_generation_allows_exact_adoption_and_blocks_a_stale_rival(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    blob = store.put_json({"step": 1})
    writer_a = RunStoreSession.open(store)
    writer_b = RunStoreSession.open(store)
    exact_a = writer_a.draft_manifest(entries={"a": blob}, meta={"n": 1})
    exact_b = writer_b.draft_manifest(entries={"a": blob}, meta={"n": 1})
    monkeypatch.setattr(
        store,
        "manifest_chain",
        lambda: (_ for _ in ()).throw(AssertionError("full chain rescan")),
    )
    monkeypatch.setattr(
        store,
        "_manifest_index",
        lambda: (_ for _ in ()).throw(AssertionError("manifest index rescan")),
    )
    monkeypatch.setattr(
        store,
        "_manifest_payloads",
        lambda: (_ for _ in ()).throw(AssertionError("manifest directory enumeration")),
    )
    monkeypatch.setattr(
        store,
        "_marked_manifests",
        lambda: (_ for _ in ()).throw(AssertionError("marked manifest enumeration")),
    )

    first = writer_a.publish_manifest(exact_a)
    writer_a.publish_manifest(writer_a.draft_manifest(entries={"a": blob}, meta={"n": 2}))
    assert writer_b.publish_manifest(exact_b) == first
    rival = writer_b.draft_manifest(entries={"a": blob}, meta={"n": 99})

    with pytest.raises(RunStoreError, match="generation"):
        writer_b.publish_manifest(rival)

    assert not store._manifest_path(rival.sequence, rival.sha256).exists()
    with pytest.raises(RunStoreError, match="poisoned"):
        writer_b.publish_manifest(rival)


def test_session_appends_into_one_internal_list_without_rebuilding_history(tmp_path: Path) -> None:
    store = _store(tmp_path)
    blob = store.put_json({"step": 1})
    session = RunStoreSession.open(store)
    history = session._manifests_cache

    for value in range(20):
        session.publish_manifest(session.draft_manifest(entries={"a": blob}, meta={"n": value}))

    assert session._manifests_cache is history
    assert isinstance(history, list)
    assert len(history) == 20


def test_session_verifies_entries_and_immediate_parent_before_append(tmp_path: Path) -> None:
    store = _store(tmp_path)
    blob = store.put_json({"step": 1})
    session = RunStoreSession.open(store)
    first = session.publish_manifest(session.draft_manifest(entries={"a": blob}, meta={"n": 1}))
    draft = session.draft_manifest(entries={"a": blob}, meta={"n": 2})
    _only_blob(tmp_path).write_bytes(b"tampered")

    with pytest.raises(RunStoreError, match="does not resolve"):
        session.publish_manifest(draft)

    _retamper(_only_blob(tmp_path), canonical_json_bytes({"step": 1}), "blob")
    parent_path = store._manifest_path(first.sequence, first.sha256)
    parent_path.write_bytes(b"tampered")
    with pytest.raises(StoreCorruptionError):
        session.publish_manifest(draft)


def test_session_is_poisoned_after_ambiguous_publication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    blob = store.put_json({"step": 1})
    session = RunStoreSession.open(store)
    draft = session.draft_manifest(entries={"a": blob}, meta={"n": 1})

    def ambiguous(*args: object, **kwargs: object) -> None:
        raise StoreDurabilityError("publication outcome unknown")

    monkeypatch.setattr(store, "_publish", ambiguous)
    with pytest.raises(StoreDurabilityError, match="unknown"):
        session.publish_manifest(draft)
    with pytest.raises(RunStoreError, match="poisoned"):
        session.draft_manifest(entries={"a": blob}, meta={"n": 2})


def test_session_does_not_adopt_a_manifest_it_cannot_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    blob = store.put_json({"step": 1})
    session = RunStoreSession.open(store)
    draft = session.draft_manifest(entries={"a": blob}, meta={"n": 1})
    original_load = store._load_manifest
    loads = 0

    def fail_published_read(sequence: int, digest: str):  # type: ignore[no-untyped-def]
        nonlocal loads
        loads += 1
        if loads == 1:
            raise OSError("remote read failed")
        return original_load(sequence, digest)

    monkeypatch.setattr(store, "_load_manifest", fail_published_read)
    with pytest.raises(OSError, match="remote read"):
        session.publish_manifest(draft)

    assert session.latest_manifest() is None
    with pytest.raises(RunStoreError, match="poisoned"):
        session.publish_manifest(draft)


def test_new_session_recovers_crashes_before_and_after_manifest_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    blob = store.put_json({"step": 1})
    before = RunStoreSession.open(store)
    draft = before.draft_manifest(entries={"a": blob}, meta={"n": 1})
    content = draft.content
    payload_path = store._manifest_path(draft.sequence, draft.sha256)

    def crash_before_marker(*args: object, **kwargs: object) -> None:
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_bytes(content)
        raise OSError("crash before marker")

    monkeypatch.setattr(store, "_publish", crash_before_marker)
    with pytest.raises(OSError, match="before marker"):
        before.publish_manifest(draft)

    monkeypatch.undo()
    recovered = RunStoreSession.open(store)
    first = recovered.publish_manifest(draft)

    after = RunStoreSession.open(store)
    second_draft = after.draft_manifest(entries={"a": blob}, meta={"n": 2})
    original_publish = store._publish

    def crash_after_marker(*args: object, **kwargs: object) -> None:
        original_publish(*args, **kwargs)
        raise OSError("crash after marker")

    monkeypatch.setattr(store, "_publish", crash_after_marker)
    with pytest.raises(OSError, match="after marker"):
        after.publish_manifest(second_draft)

    monkeypatch.undo()
    final = RunStoreSession.open(store)
    assert final.manifests()[0] == first
    assert final.latest_manifest() == final.publish_manifest(second_draft)


def test_session_cached_reads_do_not_mutate_or_rescan_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    blob = store.put_json({"step": 1})
    first = store.publish_manifest(entries={"a": blob}, meta={"n": 1}, parent=None)
    session = RunStoreSession.open(store)
    before = _snapshot(tmp_path)
    monkeypatch.setattr(
        store,
        "manifest_chain",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected rescan")),
    )

    assert session.latest_manifest() == first
    assert session.manifests() == (first,)
    assert _snapshot(tmp_path) == before
