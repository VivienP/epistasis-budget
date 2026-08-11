"""Runtime identity and input custody for durable Fourier recovery runs."""

from __future__ import annotations

import hashlib
import os
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import scipy
from threadpoolctl import threadpool_info

from epibudget.recovery_protocol import RecoveryScientificProtocol
from epibudget.run_store import (
    BlobRef,
    ContentAddressedRunStore,
    RunStoreError,
    canonical_json_bytes,
)

_THREAD_ENVIRONMENT = (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_GIT_COMMIT_WIDTH = 40
_SHA256_WIDTH = 64
_WINDOWS_FORBIDDEN = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)
_HASH_CHUNK_SIZE = 1024 * 1024
_CONTROL_CHARACTER_LIMIT = 32


class RecoveryRuntimeError(Exception):
    """Base class for invalid or incompatible durable recovery runtime state."""


class ScientificIdentityError(RecoveryRuntimeError):
    """Stored state belongs to a different scientific computation."""


class NumericCompatibilityError(RecoveryRuntimeError):
    """The numerical runtime cannot safely reuse exact computed state."""


@dataclass(frozen=True)
class ThreadPoolCompatibility:
    """Stable numerical properties of one loaded BLAS or OpenMP thread pool."""

    user_api: str
    internal_api: str
    prefix: str
    version: str | None
    threading_layer: str | None
    architecture: str | None
    num_threads: int

    def __post_init__(self) -> None:
        if not self.user_api or not self.internal_api or not self.prefix:
            raise ValueError("thread pool requires user_api, internal_api, and prefix")
        if self.num_threads < 1:
            raise ValueError("thread pool num_threads must be positive")

    def payload(self) -> dict[str, object]:
        """Return stable pool fields, excluding filesystem and host metadata."""
        return {
            "user_api": self.user_api,
            "internal_api": self.internal_api,
            "prefix": self.prefix,
            "version": self.version,
            "threading_layer": self.threading_layer,
            "architecture": self.architecture,
            "num_threads": self.num_threads,
        }

    def sort_key(self) -> tuple[str, str, str, str, str, str, int]:
        """Return the canonical diagnostic order key."""
        return (
            self.user_api,
            self.internal_api,
            self.prefix,
            self.version or "",
            self.threading_layer or "",
            self.architecture or "",
            self.num_threads,
        )


class RuntimePayloadError(RecoveryRuntimeError):
    """A persisted runtime payload is incomplete, malformed, or ambiguous."""


class InputArchiveError(RecoveryRuntimeError):
    """An exact recovery input could not be archived or restored safely."""


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_WIDTH and all(
        character in "0123456789abcdef" for character in value
    )


def _require_sha256(value: str, field: str) -> None:
    if not _is_sha256(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex string")


def _payload_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def float64_linear_algebra_probe_sha256() -> str:
    """Digest deterministic float64 results from core NumPy linear algebra operations."""
    grid = np.arange(1.0, 37.0, dtype=np.float64).reshape(6, 6)
    matrix = grid / np.float64(37.0) + np.eye(6, dtype=np.float64) * np.float64(2.0)
    right_hand_side = np.linspace(-1.0, 1.0, 6, dtype=np.float64)
    solved = np.linalg.solve(matrix, right_hand_side)
    gram = matrix.T @ matrix
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    results = np.concatenate((solved, gram.ravel(order="C"), singular_values))
    little_endian = np.asarray(results, dtype="<f8")
    return hashlib.sha256(little_endian.tobytes(order="C")).hexdigest()


def _mapping_value(mapping: Mapping[object, object], name: str) -> object | None:
    for key, value in mapping.items():
        if isinstance(key, str) and key.casefold() == name.casefold():
            return value
    return None


def _dependency_fingerprint(config: Mapping[object, object], name: str) -> dict[str, object]:
    build_dependencies = _mapping_value(config, "Build Dependencies")
    if not isinstance(build_dependencies, Mapping):
        raise NumericCompatibilityError("NumPy CONFIG has no structured build dependencies")
    dependency = _mapping_value(build_dependencies, name)
    if not isinstance(dependency, Mapping):
        raise NumericCompatibilityError(f"NumPy CONFIG has no structured {name} dependency")
    fingerprint: dict[str, object] = {}
    configuration: dict[str, object] = {}
    for key, value in dependency.items():
        if not isinstance(key, str):
            continue
        normalized = key.casefold().strip()
        if normalized in {"name", "version"} and isinstance(value, (str, int, float, bool)):
            fingerprint[normalized] = value
        elif (
            normalized in {"config", "configuration"} or normalized.endswith(" configuration")
        ) and isinstance(value, (str, int, float, bool)):
            configuration[normalized] = value
    fingerprint["config"] = configuration
    return fingerprint


def _runtime_simd_features() -> dict[str, bool]:
    numpy_core = getattr(np, "_core", None)
    multiarray = getattr(numpy_core, "_multiarray_umath", None)
    features = getattr(multiarray, "__cpu_features__", None)
    if not isinstance(features, Mapping):
        raise NumericCompatibilityError("NumPy exposes no structured SIMD feature mapping")
    return {
        key: enabled
        for key, enabled in features.items()
        if isinstance(key, str) and isinstance(enabled, bool)
    }


def _blas_sha256(
    *,
    config: object | None = None,
    simd_features: Mapping[str, bool] | None = None,
) -> str:
    structured_config = getattr(np.__config__, "CONFIG", None) if config is None else config
    if not isinstance(structured_config, Mapping):
        raise NumericCompatibilityError("NumPy exposes no structured CONFIG mapping")
    features = _runtime_simd_features() if simd_features is None else simd_features
    enabled_simd = sorted(key.upper() for key, enabled in features.items() if enabled is True)
    return _payload_sha256(
        {
            "blas": _dependency_fingerprint(structured_config, "blas"),
            "lapack": _dependency_fingerprint(structured_config, "lapack"),
            "simd": enabled_simd,
        }
    )


def _pool_string(record: Mapping[str, object], field: str, *, optional: bool) -> str | None:
    value = record.get(field)
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise NumericCompatibilityError(f"thread pool {field} must be a non-empty string")
    return value


def _canonical_thread_pools(
    records: Sequence[Mapping[str, object]],
) -> tuple[ThreadPoolCompatibility, ...]:
    pools: list[ThreadPoolCompatibility] = []
    for record in records:
        num_threads = record.get("num_threads")
        if type(num_threads) is not int or num_threads < 1:
            raise NumericCompatibilityError("thread pool num_threads must be a positive integer")
        user_api = _pool_string(record, "user_api", optional=False)
        internal_api = _pool_string(record, "internal_api", optional=False)
        prefix = _pool_string(record, "prefix", optional=False)
        if user_api is None or internal_api is None or prefix is None:
            raise NumericCompatibilityError("thread pool identity fields cannot be null")
        pools.append(
            ThreadPoolCompatibility(
                user_api=user_api,
                internal_api=internal_api,
                prefix=prefix,
                version=_pool_string(record, "version", optional=True),
                threading_layer=_pool_string(record, "threading_layer", optional=True),
                architecture=_pool_string(record, "architecture", optional=True),
                num_threads=num_threads,
            )
        )
    if not pools:
        raise NumericCompatibilityError("no effective numerical thread pools were detected")
    return tuple(sorted(pools, key=ThreadPoolCompatibility.sort_key))


@dataclass(frozen=True)
class RecoveryScientificIdentity:
    """Fields that determine the scientific meaning of one recovery run."""

    execution_commit: str
    protocol_semantic_sha256: str
    candidate_sha256: str
    dataset_ref: BlobRef
    cache_ref: BlobRef
    sidecar_ref: BlobRef
    runtime_preflight_ref: BlobRef

    def __post_init__(self) -> None:
        if len(self.execution_commit) != _GIT_COMMIT_WIDTH or any(
            character not in "0123456789abcdef" for character in self.execution_commit
        ):
            raise ValueError("execution_commit must be a lowercase 40-character Git commit")
        for field, value in (
            ("protocol_semantic_sha256", self.protocol_semantic_sha256),
            ("candidate_sha256", self.candidate_sha256),
        ):
            _require_sha256(value, field)
        for field, reference in (
            ("dataset_ref", self.dataset_ref),
            ("cache_ref", self.cache_ref),
            ("sidecar_ref", self.sidecar_ref),
            ("runtime_preflight_ref", self.runtime_preflight_ref),
        ):
            try:
                BlobRef.from_payload(reference.payload())
            except RunStoreError as error:
                raise ValueError(f"{field} must be a valid immutable blob reference") from error

    def payload(self) -> dict[str, object]:
        """Return the complete canonical scientific identity payload."""
        return {
            "execution_commit": self.execution_commit,
            "protocol_semantic_sha256": self.protocol_semantic_sha256,
            "candidate_sha256": self.candidate_sha256,
            "dataset_ref": self.dataset_ref.payload(),
            "cache_ref": self.cache_ref.payload(),
            "sidecar_ref": self.sidecar_ref.payload(),
            "runtime_preflight_ref": self.runtime_preflight_ref.payload(),
        }

    @property
    def sha256(self) -> str:
        """Return the digest of the canonical scientific identity payload."""
        return _payload_sha256(self.payload())


def require_scientific_identity(
    expected: RecoveryScientificIdentity, actual: RecoveryScientificIdentity
) -> None:
    """Reject any scientific identity difference, naming the first changed field."""
    expected_payload = expected.payload()
    actual_payload = actual.payload()
    for field in expected_payload:
        if actual_payload.get(field) != expected_payload[field]:
            raise ScientificIdentityError(f"scientific identity mismatch: {field}")


@dataclass(frozen=True)
class NumericCompatibility:
    """Exact numerical environment required to resume or combine computations."""

    python_version: str
    numpy_version: str
    scipy_version: str
    blas_sha256: str
    thread_environment: tuple[tuple[str, str | None], ...]
    thread_pools: tuple[ThreadPoolCompatibility, ...]
    probe_sha256: str

    def __post_init__(self) -> None:
        if not self.python_version or not self.numpy_version or not self.scipy_version:
            raise ValueError("numeric compatibility requires exact package versions")
        _require_sha256(self.blas_sha256, "blas_sha256")
        _require_sha256(self.probe_sha256, "probe_sha256")
        names = tuple(name for name, _value in self.thread_environment)
        if not names or names != tuple(sorted(set(names))) or any(not name for name in names):
            raise ValueError("thread_environment names must be non-empty, unique, and sorted")
        if not self.thread_pools or self.thread_pools != tuple(
            sorted(self.thread_pools, key=ThreadPoolCompatibility.sort_key)
        ):
            raise ValueError("thread_pools must be non-empty and canonically sorted")

    def payload(self) -> dict[str, object]:
        """Return the complete canonical numerical compatibility payload."""
        return {
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "scipy_version": self.scipy_version,
            "blas_sha256": self.blas_sha256,
            "thread_environment": [
                {"name": name, "value": value} for name, value in self.thread_environment
            ],
            "thread_pools": [pool.payload() for pool in self.thread_pools],
            "probe_sha256": self.probe_sha256,
        }


def capture_numeric_compatibility() -> NumericCompatibility:
    """Capture every runtime field required for exact numerical reuse."""
    probe_sha256 = float64_linear_algebra_probe_sha256()
    pool_records = cast("list[Mapping[str, object]]", threadpool_info())
    return NumericCompatibility(
        python_version=sys.version,
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        blas_sha256=_blas_sha256(),
        thread_environment=tuple((name, os.environ.get(name)) for name in _THREAD_ENVIRONMENT),
        thread_pools=_canonical_thread_pools(pool_records),
        probe_sha256=probe_sha256,
    )


def require_numeric_compatibility(
    expected: NumericCompatibility, actual: NumericCompatibility
) -> None:
    """Reject any numerical runtime difference, naming the first incompatible field."""
    expected_payload = expected.payload()
    actual_payload = actual.payload()
    for field in expected_payload:
        if actual_payload.get(field) != expected_payload[field]:
            raise NumericCompatibilityError(f"numeric compatibility mismatch: {field}")


@dataclass(frozen=True)
class RecoveryProvenance:
    """Informative execution metadata that cannot change scientific identity."""

    platform: str
    machine: str
    argv: tuple[str, ...]
    started_utc: str
    completed_utc: str

    def __post_init__(self) -> None:
        if not self.platform or not self.machine or not self.argv:
            raise ValueError("recovery provenance requires platform, machine, and argv")
        if any(not isinstance(argument, str) for argument in self.argv):
            raise ValueError("recovery provenance argv must contain strings")
        if not self.started_utc or not self.completed_utc:
            raise ValueError("recovery provenance requires start and completion timestamps")

    def payload(self) -> dict[str, object]:
        """Return the complete informative provenance payload."""
        return {
            "platform": self.platform,
            "machine": self.machine,
            "argv": list(self.argv),
            "started_utc": self.started_utc,
            "completed_utc": self.completed_utc,
        }


@dataclass(frozen=True)
class RecoveryRuntimeRecord:
    """Scientific identity, numerical compatibility, and informative provenance."""

    scientific_identity: RecoveryScientificIdentity
    numeric_compatibility: NumericCompatibility
    provenance: RecoveryProvenance

    def payload(self) -> dict[str, object]:
        """Return the three separate runtime blocks under a versioned schema."""
        return {
            "schema_version": "epibudget-recovery-runtime-v1",
            "scientific_identity": self.scientific_identity.payload(),
            "numeric_compatibility": self.numeric_compatibility.payload(),
            "provenance": self.provenance.payload(),
        }

    def canonical_bytes(self) -> bytes:
        """Return the canonical JSON encoding of the complete runtime record."""
        return canonical_json_bytes(self.payload())

    @classmethod
    def from_payload(cls, payload: object) -> RecoveryRuntimeRecord:
        """Decode a runtime record, rejecting missing, unknown, and malformed fields."""
        try:
            row = _require_fields(
                payload,
                {
                    "schema_version",
                    "scientific_identity",
                    "numeric_compatibility",
                    "provenance",
                },
                "runtime payload",
            )
            if row["schema_version"] != "epibudget-recovery-runtime-v1":
                raise RuntimePayloadError("runtime payload has an unexpected schema")
            scientific = _require_fields(
                row["scientific_identity"],
                {
                    "execution_commit",
                    "protocol_semantic_sha256",
                    "candidate_sha256",
                    "dataset_ref",
                    "cache_ref",
                    "sidecar_ref",
                    "runtime_preflight_ref",
                },
                "scientific identity",
            )
            numeric = _require_fields(
                row["numeric_compatibility"],
                {
                    "python_version",
                    "numpy_version",
                    "scipy_version",
                    "blas_sha256",
                    "thread_environment",
                    "thread_pools",
                    "probe_sha256",
                },
                "numeric compatibility",
            )
            provenance = _require_fields(
                row["provenance"],
                {"platform", "machine", "argv", "started_utc", "completed_utc"},
                "provenance",
            )
            threads_value = numeric["thread_environment"]
            if not isinstance(threads_value, list):
                raise RuntimePayloadError("thread_environment must be a JSON array")
            threads: list[tuple[str, str | None]] = []
            for value in threads_value:
                entry = _require_fields(value, {"name", "value"}, "thread environment entry")
                name = _require_string(entry["name"], "thread environment name")
                setting = entry["value"]
                if setting is not None and not isinstance(setting, str):
                    raise RuntimePayloadError("thread environment value must be a string or null")
                threads.append((name, setting))
            pools = _require_thread_pools(numeric["thread_pools"])
            argv_value = provenance["argv"]
            if not isinstance(argv_value, list) or not all(
                isinstance(argument, str) for argument in argv_value
            ):
                raise RuntimePayloadError("provenance argv must be a JSON string array")
            return cls(
                scientific_identity=RecoveryScientificIdentity(
                    execution_commit=_require_string(
                        scientific["execution_commit"], "execution_commit"
                    ),
                    protocol_semantic_sha256=_require_string(
                        scientific["protocol_semantic_sha256"], "protocol_semantic_sha256"
                    ),
                    candidate_sha256=_require_string(
                        scientific["candidate_sha256"], "candidate_sha256"
                    ),
                    dataset_ref=_require_blob_ref(scientific["dataset_ref"], "dataset_ref"),
                    cache_ref=_require_blob_ref(scientific["cache_ref"], "cache_ref"),
                    sidecar_ref=_require_blob_ref(scientific["sidecar_ref"], "sidecar_ref"),
                    runtime_preflight_ref=_require_blob_ref(
                        scientific["runtime_preflight_ref"], "runtime_preflight_ref"
                    ),
                ),
                numeric_compatibility=NumericCompatibility(
                    python_version=_require_string(numeric["python_version"], "python_version"),
                    numpy_version=_require_string(numeric["numpy_version"], "numpy_version"),
                    scipy_version=_require_string(numeric["scipy_version"], "scipy_version"),
                    blas_sha256=_require_string(numeric["blas_sha256"], "blas_sha256"),
                    thread_environment=tuple(threads),
                    thread_pools=pools,
                    probe_sha256=_require_string(numeric["probe_sha256"], "probe_sha256"),
                ),
                provenance=RecoveryProvenance(
                    platform=_require_string(provenance["platform"], "platform"),
                    machine=_require_string(provenance["machine"], "machine"),
                    argv=tuple(cast("list[str]", argv_value)),
                    started_utc=_require_string(provenance["started_utc"], "started_utc"),
                    completed_utc=_require_string(provenance["completed_utc"], "completed_utc"),
                ),
            )
        except RuntimePayloadError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimePayloadError(f"invalid runtime payload: {error}") from error


def _require_fields(value: object, expected: set[str], description: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RuntimePayloadError(f"{description} must be a JSON object")
    if set(value) != expected:
        raise RuntimePayloadError(f"{description} has unexpected or missing fields")
    return cast("dict[str, object]", value)


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise RuntimePayloadError(f"{field} must be a string")
    return value


def _require_blob_ref(value: object, field: str) -> BlobRef:
    try:
        return BlobRef.from_payload(value)
    except RunStoreError as error:
        raise RuntimePayloadError(f"{field} must be a valid blob reference") from error


def _require_thread_pools(value: object) -> tuple[ThreadPoolCompatibility, ...]:
    if not isinstance(value, list):
        raise RuntimePayloadError("thread_pools must be a JSON array")
    expected_fields = {
        "user_api",
        "internal_api",
        "prefix",
        "version",
        "threading_layer",
        "architecture",
        "num_threads",
    }
    records = [_require_fields(item, expected_fields, "thread pool entry") for item in value]
    try:
        pools = _canonical_thread_pools(records)
    except NumericCompatibilityError as error:
        raise RuntimePayloadError(f"invalid thread_pools: {error}") from error
    if [pool.payload() for pool in pools] != value:
        raise RuntimePayloadError("thread_pools must be canonically sorted")
    return pools


@dataclass(frozen=True)
class ArchivedRecoveryInput:
    """A safe local filename and immutable reference to its exact archived bytes."""

    name: str
    blob: BlobRef

    def __post_init__(self) -> None:
        _require_portable_windows_filename(self.name)

    def payload(self) -> dict[str, object]:
        """Return the canonical archived-input description."""
        return {"name": self.name, "blob": self.blob.payload()}

    @classmethod
    def from_payload(cls, value: object) -> ArchivedRecoveryInput:
        """Decode one archived input with exact field validation."""
        row = _require_fields(value, {"name", "blob"}, "archived recovery input")
        try:
            return cls(
                name=_require_string(row["name"], "archived recovery input name"),
                blob=_require_blob_ref(row["blob"], "archived recovery input blob"),
            )
        except ValueError as error:
            raise RuntimePayloadError(f"invalid archived recovery input: {error}") from error


def _require_portable_windows_filename(name: str) -> None:
    reserved_stem = name.split(".", maxsplit=1)[0].casefold()
    invalid = (
        not name
        or name in {".", ".."}
        or name.endswith((" ", "."))
        or any(
            character in _WINDOWS_FORBIDDEN or ord(character) < _CONTROL_CHARACTER_LIMIT
            for character in name
        )
        or reserved_stem in _WINDOWS_RESERVED
    )
    if invalid:
        raise ValueError(f"archived input name is not a portable Windows filename: {name!r}")


@dataclass(frozen=True)
class RecoveryInputBundle:
    """The four exact files required to reproduce one recovery run."""

    dataset: ArchivedRecoveryInput
    cache: ArchivedRecoveryInput
    sidecar: ArchivedRecoveryInput
    runtime_preflight: ArchivedRecoveryInput

    def __post_init__(self) -> None:
        names = tuple(item.name for item in self.inputs())
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("archived recovery input names must be case-insensitive unique")

    def inputs(self) -> tuple[ArchivedRecoveryInput, ...]:
        """Return inputs in the canonical dataset, cache, sidecar, preflight order."""
        return (self.dataset, self.cache, self.sidecar, self.runtime_preflight)

    def payload(self) -> dict[str, object]:
        """Return the complete canonical input-bundle description."""
        return {
            "dataset": self.dataset.payload(),
            "cache": self.cache.payload(),
            "sidecar": self.sidecar.payload(),
            "runtime_preflight": self.runtime_preflight.payload(),
        }

    @classmethod
    def from_payload(cls, value: object) -> RecoveryInputBundle:
        """Decode a bundle, rejecting missing or additional input roles."""
        row = _require_fields(
            value,
            {"dataset", "cache", "sidecar", "runtime_preflight"},
            "recovery input bundle",
        )
        try:
            return cls(
                dataset=ArchivedRecoveryInput.from_payload(row["dataset"]),
                cache=ArchivedRecoveryInput.from_payload(row["cache"]),
                sidecar=ArchivedRecoveryInput.from_payload(row["sidecar"]),
                runtime_preflight=ArchivedRecoveryInput.from_payload(row["runtime_preflight"]),
            )
        except ValueError as error:
            raise RuntimePayloadError(f"invalid recovery input bundle: {error}") from error


def scientific_identity_from_inputs(
    *,
    execution_commit: str,
    protocol: RecoveryScientificProtocol,
    candidate_sha256: str,
    inputs: RecoveryInputBundle,
) -> RecoveryScientificIdentity:
    """Build scientific identity from the registered protocol and exact archived inputs."""
    return RecoveryScientificIdentity(
        execution_commit=execution_commit,
        protocol_semantic_sha256=protocol.semantic_sha256,
        candidate_sha256=candidate_sha256,
        dataset_ref=inputs.dataset.blob,
        cache_ref=inputs.cache.blob,
        sidecar_ref=inputs.sidecar.blob,
        runtime_preflight_ref=inputs.runtime_preflight.blob,
    )


def _archive_input(store: ContentAddressedRunStore, path: Path) -> ArchivedRecoveryInput:
    if not path.is_file():
        raise InputArchiveError(f"recovery input must be an existing file: {path}")
    content = path.read_bytes()
    reference = store.put_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    if reference.sha256 != digest or store.get_bytes(reference) != content:
        raise InputArchiveError(f"recovery input did not archive exactly: {path}")
    return ArchivedRecoveryInput(name=path.name, blob=reference)


def archive_recovery_inputs(
    store: ContentAddressedRunStore,
    *,
    dataset: Path,
    cache: Path,
    sidecar: Path,
    runtime_preflight: Path,
) -> RecoveryInputBundle:
    """Archive the four explicitly supplied recovery inputs under their exact filenames."""
    paths = (dataset, cache, sidecar, runtime_preflight)
    for path in paths:
        _require_portable_windows_filename(path.name)
    if len({path.name.casefold() for path in paths}) != len(paths):
        raise ValueError("archived recovery input names must be case-insensitive unique")
    return RecoveryInputBundle(
        dataset=_archive_input(store, dataset),
        cache=_archive_input(store, cache),
        sidecar=_archive_input(store, sidecar),
        runtime_preflight=_archive_input(store, runtime_preflight),
    )


def materialize_recovery_inputs(
    store: ContentAddressedRunStore,
    bundle: RecoveryInputBundle,
    destination: Path,
) -> tuple[Path, ...]:
    """Restore one verified blob at a time and roll back files created by this attempt."""
    inputs = bundle.inputs()
    targets = _materialization_targets(inputs, destination)
    created_targets: list[Path] = []
    partials: list[Path] = []
    try:
        for archived, target in zip(inputs, targets, strict=True):
            partial = destination / f".{archived.name}.{uuid.uuid4().hex}.partial"
            partials.append(partial)
            _write_verified_partial(store, archived, partial)
            os.link(partial, target)
            created_targets.append(target)
            partial.unlink()
            partials.remove(partial)
        return targets
    except (InputArchiveError, OSError, RunStoreError, ValueError) as error:
        cleanup_error_count = _remove_created_paths((*partials, *created_targets))
        detail = (
            f"; cleanup failed for {cleanup_error_count} path(s)" if cleanup_error_count else ""
        )
        raise InputArchiveError(
            f"failed to materialize recovery inputs{detail}: {error}"
        ) from error


def materialize_archived_recovery_input(
    store: ContentAddressedRunStore,
    archived: ArchivedRecoveryInput,
    destination: Path,
) -> Path:
    """Restore one archived input exclusively and verify its published bytes."""
    target = _materialization_targets((archived,), destination)[0]
    partial = destination / f".{archived.name}.{uuid.uuid4().hex}.partial"
    target_published = False
    try:
        _write_verified_partial(store, archived, partial)
        os.link(partial, target)
        target_published = True
        partial.unlink()
        if _file_sha256(target) != archived.blob.sha256:
            raise InputArchiveError(
                f"materialized input failed SHA-256 verification: {archived.name}"
            )
        return target
    except (InputArchiveError, OSError, RunStoreError, ValueError) as error:
        created = (partial, target) if target_published else (partial,)
        _remove_created_paths(created)
        if isinstance(error, InputArchiveError):
            raise
        raise InputArchiveError(f"failed to materialize recovery input: {error}") from error


def _materialization_targets(
    inputs: tuple[ArchivedRecoveryInput, ...], destination: Path
) -> tuple[Path, ...]:
    if not destination.is_dir():
        raise InputArchiveError(
            f"materialization destination must be an existing directory: {destination}"
        )
    names = tuple(archived.name for archived in inputs)
    try:
        for name in names:
            _require_portable_windows_filename(name)
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("materialization target names are not case-insensitive unique")
        targets = tuple(destination / name for name in names)
        existing_names = {path.name.casefold() for path in destination.iterdir()}
    except (OSError, ValueError) as error:
        raise InputArchiveError(
            f"failed to prevalidate materialization targets: {error}"
        ) from error
    for target in targets:
        if target.name.casefold() in existing_names:
            raise InputArchiveError(f"materialization target already exists: {target}")
    return targets


def _write_verified_partial(
    store: ContentAddressedRunStore,
    archived: ArchivedRecoveryInput,
    partial: Path,
) -> None:
    try:
        content = store.get_bytes(archived.blob)
    except (OSError, RunStoreError, ValueError) as error:
        raise InputArchiveError(
            f"archived input is not a verified store reference: {archived.name}"
        ) from error
    if hashlib.sha256(content).hexdigest() != archived.blob.sha256:
        raise InputArchiveError(f"archived input failed SHA-256 verification: {archived.name}")
    with partial.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    del content
    if _file_sha256(partial) != archived.blob.sha256:
        raise InputArchiveError(f"materialized input failed SHA-256 verification: {archived.name}")


def _remove_created_paths(paths: tuple[Path, ...]) -> int:
    failures = 0
    for path in reversed(paths):
        try:
            if path.exists():
                path.unlink()
        except OSError:
            failures += 1
    return failures


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()
