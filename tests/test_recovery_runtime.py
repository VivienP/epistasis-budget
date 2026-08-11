from __future__ import annotations

import copy
import hashlib
import importlib.util
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, cast

import pytest
from threadpoolctl import threadpool_limits

from epibudget import recovery_runtime as runtime
from epibudget.recovery_protocol import REGISTERED_RECOVERY_PROTOCOL
from epibudget.run_store import BlobRef, ContentAddressedRunStore, canonical_json_bytes

_INPUT_COUNT = 4
_SHA256_WIDTH = 64


def _scientific() -> runtime.RecoveryScientificIdentity:
    return runtime.RecoveryScientificIdentity(
        execution_commit="a" * 40,
        protocol_semantic_sha256="b" * 64,
        candidate_sha256="c" * 64,
        dataset_ref=BlobRef(sha256="d" * 64, size=1, encoding="binary"),
        cache_ref=BlobRef(sha256="e" * 64, size=2, encoding="binary"),
        sidecar_ref=BlobRef(sha256="f" * 64, size=3, encoding="json"),
        runtime_preflight_ref=BlobRef(sha256="0" * 64, size=4, encoding="json"),
    )


def _thread_pool(num_threads: int = 1) -> runtime.ThreadPoolCompatibility:
    return runtime.ThreadPoolCompatibility(
        user_api="blas",
        internal_api="openblas",
        prefix="libopenblas",
        version="0.3.27",
        threading_layer="pthreads",
        architecture="Zen",
        num_threads=num_threads,
    )


def _compatibility() -> runtime.NumericCompatibility:
    return runtime.NumericCompatibility(
        python_version="3.12.4",
        numpy_version="2.1.0",
        scipy_version="1.14.0",
        blas_sha256="1" * 64,
        thread_environment=(("OMP_NUM_THREADS", "1"),),
        thread_pools=(_thread_pool(),),
        probe_sha256="2" * 64,
    )


def _runtime_record() -> runtime.RecoveryRuntimeRecord:
    return runtime.RecoveryRuntimeRecord(
        scientific_identity=_scientific(),
        numeric_compatibility=_compatibility(),
        provenance=runtime.RecoveryProvenance(
            platform="Windows-11",
            machine="AMD64",
            argv=("fourier-recovery", "--budget", "48"),
            started_utc="2026-08-10T10:00:00+00:00",
            completed_utc="2026-08-10T10:01:00+00:00",
        ),
    )


def test_recovery_runtime_module_exists() -> None:
    assert importlib.util.find_spec("epibudget.recovery_runtime") is not None


def test_os_metadata_is_separate_from_immutable_identity_and_compatibility() -> None:
    scientific = runtime.RecoveryScientificIdentity(
        execution_commit="a" * 40,
        protocol_semantic_sha256="b" * 64,
        candidate_sha256="c" * 64,
        dataset_ref=BlobRef(sha256="d" * 64, size=1, encoding="binary"),
        cache_ref=BlobRef(sha256="e" * 64, size=2, encoding="binary"),
        sidecar_ref=BlobRef(sha256="f" * 64, size=3, encoding="json"),
        runtime_preflight_ref=BlobRef(sha256="0" * 64, size=4, encoding="json"),
    )
    compatibility = runtime.NumericCompatibility(
        python_version="3.12.4",
        numpy_version="2.1.0",
        scipy_version="1.14.0",
        blas_sha256="1" * 64,
        thread_environment=(("OMP_NUM_THREADS", "1"),),
        thread_pools=(_thread_pool(),),
        probe_sha256="2" * 64,
    )
    windows = runtime.RecoveryProvenance(
        platform="Windows-11",
        machine="AMD64",
        argv=("fourier-recovery", "--budget", "48"),
        started_utc="2026-08-10T10:00:00+00:00",
        completed_utc="2026-08-10T10:01:00+00:00",
    )
    linux = runtime.RecoveryProvenance(
        platform="Linux-6",
        machine="x86_64",
        argv=windows.argv,
        started_utc=windows.started_utc,
        completed_utc=windows.completed_utc,
    )

    assert windows != linux
    assert compatibility == _compatibility()
    with pytest.raises(FrozenInstanceError):
        scientific.execution_commit = "9" * 40  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("execution_commit", "9" * 40),
        ("protocol_semantic_sha256", "9" * 64),
        ("candidate_sha256", "9" * 64),
        ("dataset_ref", BlobRef(sha256="9" * 64, size=1, encoding="binary")),
        ("cache_ref", BlobRef(sha256="9" * 64, size=2, encoding="binary")),
        ("sidecar_ref", BlobRef(sha256="9" * 64, size=3, encoding="json")),
        ("runtime_preflight_ref", BlobRef(sha256="9" * 64, size=4, encoding="json")),
    ],
)
def test_each_scientific_field_changes_the_identity_digest(field: str, value: object) -> None:
    scientific = _scientific()

    changed = replace(scientific, **cast("Any", {field: value}))

    assert changed.sha256 != scientific.sha256


@pytest.mark.parametrize(
    ("field", "value"),
    [("size", 999), ("encoding", "json")],
)
def test_blob_reference_metadata_changes_the_scientific_identity(field: str, value: object) -> None:
    scientific = _scientific()
    changed_ref = replace(scientific.dataset_ref, **cast("Any", {field: value}))

    changed = replace(scientific, dataset_ref=changed_ref)

    assert changed.dataset_ref.sha256 == scientific.dataset_ref.sha256
    assert changed.sha256 != scientific.sha256


def test_float64_linear_algebra_probe_is_repeatable() -> None:
    first = runtime.float64_linear_algebra_probe_sha256()

    assert first == runtime.float64_linear_algebra_probe_sha256()
    assert len(first) == _SHA256_WIDTH
    assert set(first) <= set("0123456789abcdef")


def test_capture_numeric_compatibility_is_exact_and_excludes_os(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread_names = (
        "BLIS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    for name in thread_names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")

    compatibility = runtime.capture_numeric_compatibility()

    assert compatibility.thread_environment == tuple(
        (name, "2" if name == "OMP_NUM_THREADS" else None) for name in thread_names
    )
    assert compatibility.probe_sha256 == runtime.float64_linear_algebra_probe_sha256()
    assert compatibility.python_version
    assert compatibility.numpy_version
    assert compatibility.scipy_version
    assert len(compatibility.blas_sha256) == _SHA256_WIDTH
    assert compatibility.thread_pools
    assert "platform" not in compatibility.payload()
    assert "machine" not in compatibility.payload()


def test_blas_fingerprint_ignores_host_and_path_metadata() -> None:
    first = {
        "Build Dependencies": {
            "blas": {
                "name": "openblas",
                "version": "0.3.27",
                "configuration": "DYNAMIC_ARCH=1 USE64BITINT=0",
                "lib directory": "/opt/host-a/lib",
            },
            "lapack": {
                "name": "openblas",
                "version": "0.3.27",
                "configuration": "LAPACK=1",
                "include directory": "/opt/host-a/include",
            },
        },
        "Machine Information": {"host": "builder-a", "system": "Linux"},
    }
    second = {
        "Build Dependencies": {
            "blas": {
                "name": "openblas",
                "version": "0.3.27",
                "configuration": "DYNAMIC_ARCH=1 USE64BITINT=0",
                "lib directory": "C:\\host-b\\lib",
            },
            "lapack": {
                "name": "openblas",
                "version": "0.3.27",
                "configuration": "LAPACK=1",
                "include directory": "C:\\host-b\\include",
            },
        },
        "Machine Information": {"host": "builder-b", "system": "Windows"},
    }
    simd = {"AVX2": True, "AVX512F": False, "SSE2": True}

    baseline = runtime._blas_sha256(config=first, simd_features=simd)
    assert baseline == runtime._blas_sha256(config=second, simd_features=simd)
    changed = copy.deepcopy(second)
    cast("Any", changed)["Build Dependencies"]["blas"]["version"] = "0.3.28"
    assert baseline != runtime._blas_sha256(config=changed, simd_features=simd)
    assert baseline != runtime._blas_sha256(config=second, simd_features={**simd, "AVX512F": True})


def test_effective_blas_thread_limit_changes_numeric_compatibility() -> None:
    with threadpool_limits(limits=1, user_api="blas"):
        one_thread = runtime.capture_numeric_compatibility()
    with threadpool_limits(limits=2, user_api="blas"):
        two_threads = runtime.capture_numeric_compatibility()

    assert one_thread.thread_pools
    assert {pool.num_threads for pool in one_thread.thread_pools} == {1}
    assert {pool.num_threads for pool in two_threads.thread_pools} == {2}
    assert one_thread.thread_pools != two_threads.thread_pools
    with pytest.raises(runtime.NumericCompatibilityError, match="thread_pools"):
        runtime.require_numeric_compatibility(one_thread, two_threads)


def test_thread_pool_payload_ignores_filepath_and_hostname_and_sorts() -> None:
    first: list[dict[str, object]] = [
        {
            "user_api": "blas",
            "internal_api": "openblas",
            "prefix": "libopenblas",
            "version": "0.3.27",
            "threading_layer": "pthreads",
            "architecture": "Zen",
            "num_threads": 2,
            "filepath": "/host-a/libopenblas.so",
            "hostname": "host-a",
        },
        {
            "user_api": "openmp",
            "internal_api": "openmp",
            "prefix": "libomp",
            "version": None,
            "threading_layer": None,
            "architecture": None,
            "num_threads": 2,
            "filepath": "/host-a/libomp.so",
        },
    ]
    second: list[dict[str, object]] = [
        {**first[1], "filepath": "C:\\host-b\\libomp.dll", "hostname": "host-b"},
        {**first[0], "filepath": "C:\\host-b\\libopenblas.dll", "hostname": "host-b"},
    ]

    assert runtime._canonical_thread_pools(first) == runtime._canonical_thread_pools(second)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("python_version", "3.12.5"),
        ("numpy_version", "2.2.0"),
        ("scipy_version", "1.15.0"),
        ("blas_sha256", "3" * 64),
        ("thread_environment", (("OMP_NUM_THREADS", "2"),)),
        ("probe_sha256", "4" * 64),
    ],
)
def test_numeric_mismatch_rejects_fail_closed_with_typed_error(field: str, value: object) -> None:
    expected = _compatibility()
    actual = replace(expected, **cast("Any", {field: value}))
    assert expected.payload()[field] != actual.payload()[field]

    with pytest.raises(runtime.NumericCompatibilityError, match=field) as caught:
        runtime.require_numeric_compatibility(expected, actual)
    assert caught.value.args == (f"numeric compatibility mismatch: {field}",)


def test_scientific_identity_mismatch_rejects_with_typed_error() -> None:
    expected = _scientific()
    actual = replace(expected, candidate_sha256="9" * 64)
    assert expected.candidate_sha256 != actual.candidate_sha256

    with pytest.raises(runtime.ScientificIdentityError, match="candidate_sha256") as caught:
        runtime.require_scientific_identity(expected, actual)
    assert caught.value.args == ("scientific identity mismatch: candidate_sha256",)


def test_runtime_record_round_trips_as_canonical_json() -> None:
    record = _runtime_record()

    encoded = record.canonical_bytes()

    assert encoded == canonical_json_bytes(record.payload())
    assert runtime.RecoveryRuntimeRecord.from_payload(record.payload()) == record


@pytest.mark.parametrize(
    "section", [None, "scientific_identity", "numeric_compatibility", "provenance"]
)
def test_runtime_payload_rejects_unknown_fields_fail_closed(section: str | None) -> None:
    payload = _runtime_record().payload()
    target = payload if section is None else payload[section]
    assert isinstance(target, dict)
    target["unexpected"] = "value"

    with pytest.raises(runtime.RuntimePayloadError, match="fields"):
        runtime.RecoveryRuntimeRecord.from_payload(payload)


def _input_paths(root: Path) -> dict[str, Path]:
    paths = {
        "dataset": root / "dataset.csv",
        "cache": root / "scores.npz",
        "sidecar": root / "scores.meta.json",
        "runtime_preflight": root / "preflight.json",
    }
    root.mkdir()
    for index, path in enumerate(paths.values()):
        path.write_bytes(f"synthetic-input-{index}".encode())
    return paths


def _store(root: Path) -> ContentAddressedRunStore:
    root.mkdir()
    store = ContentAddressedRunStore(root)
    store.initialise()
    return store


def test_same_input_bundle_archives_each_content_once(tmp_path: Path) -> None:
    store = _store(tmp_path / "store")
    paths = _input_paths(tmp_path / "inputs")

    first = runtime.archive_recovery_inputs(store, **paths)
    second = runtime.archive_recovery_inputs(store, **paths)

    assert first == second
    assert store.verify().blob_count == _INPUT_COUNT
    assert first.dataset.name == "dataset.csv"
    assert first.cache.name == "scores.npz"
    assert first.sidecar.name == "scores.meta.json"
    assert first.runtime_preflight.name == "preflight.json"


def test_scientific_identity_uses_exact_archived_input_and_protocol_digests(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "store")
    paths = _input_paths(tmp_path / "inputs")
    bundle = runtime.archive_recovery_inputs(store, **paths)

    identity = runtime.scientific_identity_from_inputs(
        execution_commit="a" * 40,
        protocol=REGISTERED_RECOVERY_PROTOCOL,
        candidate_sha256="b" * 64,
        inputs=bundle,
    )

    assert identity.protocol_semantic_sha256 == REGISTERED_RECOVERY_PROTOCOL.semantic_sha256
    assert identity.dataset_ref.sha256 == hashlib.sha256(paths["dataset"].read_bytes()).hexdigest()
    assert identity.cache_ref.sha256 == hashlib.sha256(paths["cache"].read_bytes()).hexdigest()
    assert identity.sidecar_ref.sha256 == hashlib.sha256(paths["sidecar"].read_bytes()).hexdigest()
    assert (
        identity.runtime_preflight_ref.sha256
        == hashlib.sha256(paths["runtime_preflight"].read_bytes()).hexdigest()
    )
    assert identity.dataset_ref == bundle.dataset.blob
    assert identity.cache_ref == bundle.cache.blob
    assert identity.sidecar_ref == bundle.sidecar.blob
    assert identity.runtime_preflight_ref == bundle.runtime_preflight.blob


def test_materialize_input_bundle_round_trips_exact_bytes(tmp_path: Path) -> None:
    store = _store(tmp_path / "store")
    source_paths = _input_paths(tmp_path / "inputs")
    bundle = runtime.archive_recovery_inputs(store, **source_paths)
    destination = tmp_path / "restored"
    destination.mkdir()

    restored = runtime.materialize_recovery_inputs(store, bundle, destination)

    assert restored == tuple(destination / item.name for item in bundle.inputs())
    for source, target in zip(source_paths.values(), restored, strict=True):
        assert target.read_bytes() == source.read_bytes()


def test_materialize_reads_and_publishes_one_blob_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path / "store")
    bundle = runtime.archive_recovery_inputs(store, **_input_paths(tmp_path / "inputs"))
    destination = tmp_path / "restored"
    destination.mkdir()
    original_get = store.get_bytes
    reads: list[str] = []

    def observed_get(reference: BlobRef) -> bytes:
        if reads:
            assert (destination / bundle.dataset.name).is_file()
        reads.append(reference.sha256)
        return original_get(reference)

    monkeypatch.setattr(store, "get_bytes", observed_get)

    runtime.materialize_recovery_inputs(store, bundle, destination)

    assert reads == [item.blob.sha256 for item in bundle.inputs()]


def test_materialize_rolls_back_its_files_when_second_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path / "store")
    bundle = runtime.archive_recovery_inputs(store, **_input_paths(tmp_path / "inputs"))
    destination = tmp_path / "restored"
    destination.mkdir()
    existing = destination / "keep.txt"
    existing.write_bytes(b"keep")
    initial = {path.name: path.read_bytes() for path in destination.iterdir()}
    original_open = Path.open

    def failing_open(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if "x" in mode and bundle.cache.name in self.name:
            raise OSError("synthetic second-file failure")
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)

    with pytest.raises(runtime.InputArchiveError, match="materialize"):
        runtime.materialize_recovery_inputs(store, bundle, destination)

    assert {path.name: path.read_bytes() for path in destination.iterdir()} == initial


def test_materialize_refuses_to_overwrite_an_existing_target(tmp_path: Path) -> None:
    store = _store(tmp_path / "store")
    bundle = runtime.archive_recovery_inputs(store, **_input_paths(tmp_path / "inputs"))
    destination = tmp_path / "restored"
    destination.mkdir()
    existing = destination / bundle.dataset.name
    existing.write_bytes(b"keep-me")

    with pytest.raises(runtime.InputArchiveError, match="exists"):
        runtime.materialize_recovery_inputs(store, bundle, destination)

    assert existing.read_bytes() == b"keep-me"
    assert tuple(destination.iterdir()) == (existing,)


def test_materialize_rejects_tampered_archived_content_before_writing(tmp_path: Path) -> None:
    store = _store(tmp_path / "store")
    bundle = runtime.archive_recovery_inputs(store, **_input_paths(tmp_path / "inputs"))
    reference = bundle.cache.blob
    payload_path = store.root / "blobs" / reference.sha256[:2] / f"{reference.sha256}.blob"
    payload_path.write_bytes(b"tampered")
    destination = tmp_path / "restored"
    destination.mkdir()

    with pytest.raises(runtime.InputArchiveError, match="verified"):
        runtime.materialize_recovery_inputs(store, bundle, destination)

    assert not tuple(destination.iterdir())


def test_materialize_rejects_a_malformed_blob_reference(tmp_path: Path) -> None:
    store = _store(tmp_path / "store")
    bundle = runtime.archive_recovery_inputs(store, **_input_paths(tmp_path / "inputs"))
    wrong_blob = replace(bundle.dataset.blob, sha256="not-a-sha256")
    wrong_dataset = replace(bundle.dataset, blob=wrong_blob)
    wrong_bundle = replace(bundle, dataset=wrong_dataset)
    destination = tmp_path / "restored"
    destination.mkdir()

    with pytest.raises(runtime.InputArchiveError, match="verified"):
        runtime.materialize_recovery_inputs(store, wrong_bundle, destination)

    assert not tuple(destination.iterdir())


def test_archived_input_rejects_an_unsafe_materialized_name(tmp_path: Path) -> None:
    store = _store(tmp_path / "store")
    bundle = runtime.archive_recovery_inputs(store, **_input_paths(tmp_path / "inputs"))
    original = bundle.dataset

    with pytest.raises(ValueError, match="portable") as caught:
        replace(bundle.dataset, name="../escape.csv")
    assert caught.value.args == (
        "archived input name is not a portable Windows filename: '../escape.csv'",
    )
    assert bundle.dataset == original


@pytest.mark.parametrize(
    "name",
    [
        "bad*.csv",
        "bad?.csv",
        'bad".csv',
        "bad<.csv",
        "bad>.csv",
        "bad|.csv",
        "bad:stream.csv",
        "NUL",
        "con.txt",
        "PRN.csv",
        "AUX",
        "COM1.dat",
        "com9",
        "LPT1.txt",
        "lpt9",
        "trailing.",
        "trailing ",
        "subdir/file.csv",
        "subdir\\file.csv",
        "nul\x00byte.csv",
    ],
)
def test_archived_input_rejects_nonportable_windows_names(name: str) -> None:
    reference = BlobRef(sha256="a" * 64, size=1, encoding="binary")

    with pytest.raises(ValueError, match="portable"):
        runtime.ArchivedRecoveryInput(name=name, blob=reference)


def test_input_bundle_rejects_duplicate_materialized_names(tmp_path: Path) -> None:
    store = _store(tmp_path / "store")
    paths = _input_paths(tmp_path / "inputs")
    duplicate_directory = tmp_path / "duplicate"
    duplicate_directory.mkdir()
    duplicate = duplicate_directory / paths["dataset"].name
    duplicate.write_bytes(b"different-cache-content")
    paths["cache"] = duplicate

    with pytest.raises(ValueError, match="unique"):
        runtime.archive_recovery_inputs(store, **paths)

    assert store.verify().blob_count == 0


def test_case_insensitive_name_collisions_fail_before_archiving(tmp_path: Path) -> None:
    reference = BlobRef(sha256="a" * 64, size=1, encoding="binary")
    dataset = runtime.ArchivedRecoveryInput(name="dataset.csv", blob=reference)
    cache = runtime.ArchivedRecoveryInput(name="DATASET.CSV", blob=reference)
    unique = runtime.ArchivedRecoveryInput(name="unique.json", blob=reference)

    with pytest.raises(ValueError, match="case-insensitive"):
        runtime.RecoveryInputBundle(
            dataset=dataset,
            cache=cache,
            sidecar=unique,
            runtime_preflight=replace(unique, name="preflight.json"),
        )

    store = _store(tmp_path / "store")
    paths = _input_paths(tmp_path / "inputs")
    alternate = tmp_path / "alternate"
    alternate.mkdir()
    collision = alternate / "DATASET.CSV"
    collision.write_bytes(b"cache")
    paths["cache"] = collision
    with pytest.raises(ValueError, match="case-insensitive"):
        runtime.archive_recovery_inputs(store, **paths)
    assert store.verify().blob_count == 0


def test_input_bundle_payload_round_trip_is_strict(tmp_path: Path) -> None:
    store = _store(tmp_path / "store")
    bundle = runtime.archive_recovery_inputs(store, **_input_paths(tmp_path / "inputs"))

    restored = runtime.RecoveryInputBundle.from_payload(bundle.payload())

    assert restored == bundle
    malformed = dict(bundle.payload())
    malformed["unexpected"] = True
    with pytest.raises(runtime.RuntimePayloadError, match="fields"):
        runtime.RecoveryInputBundle.from_payload(malformed)


def test_materialize_one_archived_input_is_exclusive_and_verified(tmp_path: Path) -> None:
    store = _store(tmp_path / "store")
    bundle = runtime.archive_recovery_inputs(store, **_input_paths(tmp_path / "inputs"))
    destination = tmp_path / "restored"
    destination.mkdir()

    restored = runtime.materialize_archived_recovery_input(store, bundle.cache, destination)

    assert restored == destination / bundle.cache.name
    assert restored.read_bytes() == store.get_bytes(bundle.cache.blob)
    with pytest.raises(runtime.InputArchiveError, match="exists"):
        runtime.materialize_archived_recovery_input(store, bundle.cache, destination)


def test_materialize_one_does_not_remove_a_racing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path / "store")
    bundle = runtime.archive_recovery_inputs(store, **_input_paths(tmp_path / "inputs"))
    destination = tmp_path / "restored"
    destination.mkdir()
    target = destination / bundle.cache.name

    def racing_link(_source: Path, destination_path: Path) -> None:
        destination_path.write_bytes(b"other-writer")
        raise FileExistsError("synthetic race")

    monkeypatch.setattr(runtime.os, "link", racing_link)

    with pytest.raises(runtime.InputArchiveError):
        runtime.materialize_archived_recovery_input(store, bundle.cache, destination)

    assert target.read_bytes() == b"other-writer"
