"""A durable, content-addressed store for long-running computations.

The store makes no assumption about what it holds and nothing here knows about Fourier recovery. It
assumes only that the filesystem underneath it may be slow, remote, and dishonest: a write may be
acknowledged and then read back differently, a rename may not be atomic, and a process may die
between any two syscalls.

Durability guarantee, stated exactly
------------------------------------
Every published file is written to a sibling temporary path, flushed, ``fsync``-ed, re-read in full,
moved into place, and re-read again; a completion marker is published last, and every read verifies
the recomputed SHA-256 against the marker, against the caller's reference, and against the digest
encoded in the path. From this the store guarantees two things:

* **Process interruption.** A run killed at any point leaves either a complete, verified state or an
  incomplete one that is never mistaken for data.
* **Detectable corruption.** Any later alteration of a payload, a marker, or both together is
  detected on read, because the address is derived from the content rather than recorded beside it.

The store does **not** promise to survive host power loss on every platform. File contents are
fsync-ed always; directory entries are fsync-ed only where the filesystem accepts it, which excludes
Windows and the network and FUSE mounts that reject the call.
:meth:`ContentAddressedRunStore.durability_capabilities` reports what a given root actually
provides, by attempting the flush rather than assuming the platform allows it. Where directory sync
is unavailable, a power loss can lose a whole published file — but it cannot produce a file that
verifies and is wrong, and it never fails a publication that would otherwise have succeeded.

State is a chain of immutable manifests linked by parent digest. There is no mutable ``HEAD``
pointer, so a torn write can never redirect the store to an older or partial state. Two writers that
publish identical content are idempotent; two writers that publish different content at the same
sequence are reported as divergent instead of one being chosen arbitrarily.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt

_STORE_SCHEMA: Final = "epibudget-run-store-v1"
_MANIFEST_SCHEMA: Final = "epibudget-run-store-manifest-v1"
_MARKER_SCHEMA: Final = "epibudget-run-store-marker-v1"
_ARRAY_SCHEMA: Final = "epibudget-run-store-array-v1"
_SHARD_WIDTH: Final = 2
_SEQUENCE_WIDTH: Final = 12
_SHA256_WIDTH: Final = 64
_PROBE_BYTES: Final = 4096
_ENCODINGS: Final = frozenset({"json", "binary"})
_MEMORY_ORDERS: Final = frozenset({"C", "F"})
_MARKER_SUFFIX: Final = ".complete"
_PARTIAL_SUFFIX: Final = ".partial"
_MARKER_FIELDS: Final = frozenset({"schema_version", "kind", "name", "sha256", "size"})
_ERROR: Final = "error"
_INFO: Final = "info"


class RunStoreError(Exception):
    """Base class for every durable-store failure."""


class StoreDurabilityError(RunStoreError):
    """The underlying filesystem did not return the bytes the store just wrote."""


class StoreCorruptionError(RunStoreError):
    """A published payload, marker, or address no longer agrees with the content."""


class StoreDivergenceError(RunStoreError):
    """Two writers published different content for the same manifest sequence."""


@dataclass(frozen=True)
class BlobRef:
    """The identity of one archived payload."""

    sha256: str
    size: int
    encoding: str

    def payload(self) -> dict[str, object]:
        """Return the JSON encoding of this reference."""
        return {"sha256": self.sha256, "size": self.size, "encoding": self.encoding}

    @classmethod
    def from_payload(cls, value: object) -> BlobRef:
        """Decode one reference, rejecting anything that is not a well-formed record."""
        row = _require_object(value, "blob reference")
        digest = row.get("sha256")
        size = row.get("size")
        encoding = row.get("encoding")
        if (
            not _is_sha256(digest)
            or type(size) is not int
            or size < 0
            or encoding not in _ENCODINGS
        ):
            raise StoreCorruptionError("manifest contains a malformed blob reference")
        return cls(sha256=cast("str", digest), size=size, encoding=encoding)


@dataclass(frozen=True)
class ArrayRef:
    """A numeric array archived as explicit little-endian bytes, never as a pickle."""

    blob: BlobRef
    dtype: str
    byteorder: str
    shape: tuple[int, ...]
    order: str

    def payload(self) -> dict[str, object]:
        """Return the JSON encoding of this reference."""
        return {
            "schema_version": _ARRAY_SCHEMA,
            "blob": self.blob.payload(),
            "dtype": self.dtype,
            "byteorder": self.byteorder,
            "shape": list(self.shape),
            "order": self.order,
        }

    @classmethod
    def from_payload(cls, value: object) -> ArrayRef:
        """Decode one array reference, rejecting an incomplete binary description."""
        row = _require_object(value, "array reference")
        if row.get("schema_version") != _ARRAY_SCHEMA:
            raise StoreCorruptionError("array reference has an unexpected schema")
        dtype = row.get("dtype")
        byteorder = row.get("byteorder")
        shape = row.get("shape")
        order = row.get("order")
        if (
            not isinstance(dtype, str)
            or byteorder != "little"
            or order not in _MEMORY_ORDERS
            or not isinstance(shape, list)
            or not all(type(extent) is int and extent >= 0 for extent in shape)
        ):
            raise StoreCorruptionError("array reference has a malformed binary description")
        return cls(
            blob=BlobRef.from_payload(row.get("blob")),
            dtype=dtype,
            byteorder=byteorder,
            shape=tuple(cast("list[int]", shape)),
            order=order,
        )


@dataclass(frozen=True)
class Manifest:
    """One immutable published state, linked to its predecessor by digest."""

    sequence: int
    parent_sha256: str | None
    entries: Mapping[str, BlobRef]
    meta: Mapping[str, object]
    sha256: str

    def entry(self, key: str) -> BlobRef:
        """Return one entry, rejecting a key this manifest does not carry."""
        try:
            return self.entries[key]
        except KeyError as error:
            raise KeyError(f"manifest {self.sequence} has no entry {key!r}") from error


@dataclass(frozen=True)
class ManifestDraft:
    """Canonical bytes and decoded fields for one proposed manifest append."""

    sequence: int
    parent_sha256: str | None
    entries: Mapping[str, BlobRef]
    meta: Mapping[str, object]
    sha256: str
    content: bytes


@dataclass(frozen=True)
class StoreIssue:
    """One problem found by a non-mutating store audit."""

    path: str
    category: str
    problem: str
    severity: str


@dataclass(frozen=True)
class StoreReport:
    """The result of a complete, non-mutating store verification."""

    blob_count: int
    manifest_count: int
    latest_sequence: int | None
    issues: tuple[StoreIssue, ...]

    @property
    def is_clean(self) -> bool:
        """True only when the audit found nothing at all to report."""
        return not self.issues

    @property
    def errors(self) -> tuple[StoreIssue, ...]:
        """Every issue that makes stored state unusable rather than merely untidy."""
        return tuple(issue for issue in self.issues if issue.severity == _ERROR)

    @property
    def has_errors(self) -> bool:
        """True when at least one issue makes stored state unusable."""
        return bool(self.errors)

    def problems(self) -> tuple[str, ...]:
        """Return the distinct problem names found, for concise assertions and logs."""
        return tuple(sorted({issue.problem for issue in self.issues}))


def canonical_json_bytes(payload: object) -> bytes:
    """Serialize one JSON value with the canonical store encoding."""
    try:
        rendered = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"store payload is not canonical JSON: {error}") from error
    return rendered.encode("utf-8")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_WIDTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise StoreCorruptionError(f"store {field} must be a JSON object")
    return cast("dict[str, object]", value)


def _directory_flag() -> int | None:
    flag = getattr(os, "O_DIRECTORY", None)
    return flag if isinstance(flag, int) else None


def _directory_sync_supported() -> bool:
    return _directory_flag() is not None


def _sync_directory(path: Path) -> bool:
    """Flush a directory entry, reporting whether the filesystem actually accepted it.

    A directory flush is a durability enhancement, never a correctness requirement: integrity comes
    from content addressing plus full re-read verification. Network and FUSE filesystems, including
    a mounted Google Drive, refuse the call outright, so a refusal is reported as an unavailable
    capability rather than failing the publication that would otherwise have succeeded.
    """
    flag = _directory_flag()
    if flag is None:
        return False
    try:
        descriptor = os.open(path, os.O_RDONLY | flag)
    except OSError:
        return False
    try:
        os.fsync(descriptor)
    except OSError:
        return False
    else:
        return True
    finally:
        os.close(descriptor)


def _write_verified(path: Path, content: bytes) -> None:
    """Publish exact bytes at ``path``, proving the filesystem returned what it accepted."""
    digest = hashlib.sha256(content).hexdigest()
    if path.exists():
        if path.read_bytes() == content:
            return
        raise StoreCorruptionError(f"store path holds different content than expected: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{uuid.uuid4().hex}{_PARTIAL_SUFFIX}")
    with partial.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    if partial.read_bytes() != content:
        raise StoreDurabilityError(f"store did not read back the bytes it wrote: {partial}")
    os.replace(partial, path)
    _sync_directory(path.parent)
    published = path.read_bytes()
    if published != content or hashlib.sha256(published).hexdigest() != digest:
        raise StoreDurabilityError(f"store did not publish the bytes it wrote: {path}")


class ContentAddressedRunStore:
    """A crash-safe store of content-addressed blobs and an immutable manifest chain."""

    def __init__(self, root: Path) -> None:
        """Bind the store to an existing directory without creating or probing anything."""
        if not root.is_dir():
            raise RunStoreError(f"run-store root must be an existing directory: {root}")
        self._root = root
        self._manifest_generation = 0

    @property
    def root(self) -> Path:
        """The directory this store owns."""
        return self._root

    @property
    def _blobs(self) -> Path:
        return self._root / "blobs"

    @property
    def _manifests(self) -> Path:
        return self._root / "manifests"

    @property
    def _probes(self) -> Path:
        return self._root / "probe"

    @property
    def _header_path(self) -> Path:
        return self._root / "store.json"

    # -- durability ----------------------------------------------------------------

    def durability_capabilities(self) -> dict[str, bool]:
        """Report what this root actually provides, by trying it rather than by assuming it."""
        directory_sync = _directory_sync_supported() and _sync_directory(self._root)
        return {"file_fsync": True, "directory_fsync": directory_sync}

    def initialise(self) -> None:
        """Prove the root is durable, then publish the store header. Safe to repeat."""
        self.probe_durability()
        self._publish(self._header_path, canonical_json_bytes(self._header()), "header", None)

    def probe_durability(self) -> None:
        """Write, re-read, hash, and publish a throwaway payload, then withdraw it."""
        content = os.urandom(_PROBE_BYTES)
        digest = hashlib.sha256(content).hexdigest()
        payload_path = self._probes / f"{digest}.probe"
        self._publish(payload_path, content, "probe", digest)
        if self._read_marked(payload_path, "probe", digest) != content:
            raise StoreDurabilityError(f"run-store probe did not survive publication: {self._root}")
        if payload_path.name not in {entry.name for entry in self._probes.iterdir()}:
            raise StoreDurabilityError(f"run-store probe is not discoverable: {self._probes}")
        # The probe is store scaffolding, never run data; it is withdrawn only once it has passed.
        payload_path.with_name(payload_path.name + _MARKER_SUFFIX).unlink()
        payload_path.unlink()

    def _header(self) -> dict[str, object]:
        return {"schema_version": _STORE_SCHEMA}

    def has_valid_header(self) -> bool:
        """True when the store header is present, marked, intact, and of the expected schema."""
        try:
            content = self._read_marked(self._header_path, "header", None)
        except (StoreCorruptionError, OSError):
            return False
        try:
            decoded: Any = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
        return bool(decoded == self._header())

    def _require_header(self) -> None:
        if not self.has_valid_header():
            raise RunStoreError(
                f"run-store has no valid header; call initialise() first: {self._root}"
            )

    # -- blobs ---------------------------------------------------------------------

    def put_bytes(self, content: bytes, *, encoding: str = "binary") -> BlobRef:
        """Archive raw bytes once and return their reference."""
        if encoding not in _ENCODINGS:
            raise ValueError(f"blob encoding must be one of {sorted(_ENCODINGS)}, got {encoding!r}")
        digest = hashlib.sha256(content).hexdigest()
        self._publish(self._blob_path(digest), content, "blob", digest)
        return BlobRef(sha256=digest, size=len(content), encoding=encoding)

    def put_json(self, value: object) -> BlobRef:
        """Archive one JSON value under its canonical encoding."""
        return self.put_bytes(canonical_json_bytes(value), encoding="json")

    def put_array(self, array: npt.NDArray[Any]) -> ArrayRef:
        """Archive one numeric array as explicit little-endian bytes with a full description."""
        if array.dtype.hasobject or array.dtype.kind in {"O", "V", "U", "S"}:
            raise ValueError(f"array dtype {array.dtype!r} is not a plain numeric type")
        order = "F" if array.flags.f_contiguous and not array.flags.c_contiguous else "C"
        little = array.dtype.newbyteorder("<")
        content = np.asarray(array, dtype=little).tobytes(order=cast("Any", order))
        return ArrayRef(
            blob=self.put_bytes(content),
            dtype=array.dtype.name,
            byteorder="little",
            shape=tuple(int(extent) for extent in array.shape),
            order=order,
        )

    def get_array(self, reference: ArrayRef) -> npt.NDArray[Any]:
        """Return one archived array as a read-only view, rejecting a mismatched description."""
        content = self.get_bytes(reference.blob)
        dtype = np.dtype(reference.dtype).newbyteorder("<")
        expected = int(np.prod(reference.shape, dtype=np.int64))
        if len(content) != expected * dtype.itemsize:
            raise StoreCorruptionError(
                f"array {reference.blob.sha256} has an unexpected byte count"
            )
        flat = np.frombuffer(content, dtype=dtype)
        restored = flat.reshape(reference.shape, order=cast("Any", reference.order))
        return np.asarray(restored, dtype=np.dtype(reference.dtype))

    def get_bytes(self, reference: BlobRef) -> bytes:
        """Return one archived payload, checking the address, the marker, and the reference."""
        content = self._read_marked(self._blob_path(reference.sha256), "blob", reference.sha256)
        if len(content) != reference.size:
            raise StoreCorruptionError(
                f"blob {reference.sha256} is {len(content)} bytes but its reference "
                f"declares {reference.size}"
            )
        return content

    def get_json(self, reference: BlobRef) -> object:
        """Return one archived JSON value, rejecting a non-canonical encoding."""
        if reference.encoding != "json":
            raise ValueError(f"blob {reference.sha256} is not JSON-encoded")
        content = self.get_bytes(reference)
        try:
            decoded: Any = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise StoreCorruptionError(f"blob {reference.sha256} is not valid JSON") from error
        if canonical_json_bytes(decoded) != content:
            raise StoreCorruptionError(f"blob {reference.sha256} is not canonically encoded")
        return decoded

    def has_blob(self, sha256: str) -> bool:
        """True when a marked blob whose marker agrees with its address is present."""
        return self._load_marker(self._blob_path(sha256), "blob", sha256) is not None

    def _blob_path(self, digest: str) -> Path:
        if not _is_sha256(digest):
            raise ValueError(f"blob digest must be a lowercase SHA-256 hex string, got {digest!r}")
        return self._blobs / digest[:_SHARD_WIDTH] / f"{digest}.blob"

    # -- manifests -----------------------------------------------------------------

    def publish_manifest(
        self,
        *,
        entries: Mapping[str, BlobRef],
        meta: Mapping[str, object],
        parent: Manifest | None,
    ) -> Manifest:
        """Publish one immutable state, after proving every referenced blob is durable."""
        self._require_header()
        for key in sorted(entries):
            try:
                self.get_bytes(entries[key])
            except (StoreCorruptionError, ValueError) as error:
                raise RunStoreError(
                    f"manifest entry {key!r} does not resolve to a verified blob: {error}"
                ) from error
        sequence = 0 if parent is None else parent.sequence + 1
        body: dict[str, object] = {
            "schema_version": _MANIFEST_SCHEMA,
            "sequence": sequence,
            "parent_sha256": None if parent is None else parent.sha256,
            "entries": {key: reference.payload() for key, reference in sorted(entries.items())},
            "meta": dict(meta),
        }
        content = canonical_json_bytes(body)
        digest = hashlib.sha256(content).hexdigest()
        published = self._manifest_index().get(sequence)
        if published == digest:
            return self._load_manifest(sequence, digest)
        self._require_current_parent(parent)
        if published is not None:
            raise StoreDivergenceError(
                f"manifest sequence {sequence} already holds a different state: "
                f"{published} and {digest}"
            )
        self._publish(self._manifest_path(sequence, digest), content, "manifest", digest)
        self._manifest_generation += 1
        return self._load_manifest(sequence, digest)

    def _require_current_parent(self, parent: Manifest | None) -> None:
        chain = self.manifest_chain()
        if not chain:
            if parent is not None:
                raise RunStoreError("an empty run store can only accept a root manifest")
            return
        if parent is None:
            raise RunStoreError(
                f"run store already holds {len(chain)} manifests; a root manifest would "
                f"diverge from the published chain"
            )
        current = chain[-1]
        if parent.sequence != current.sequence or parent.sha256 != current.sha256:
            raise RunStoreError(
                f"parent manifest is stale or forged: expected sequence {current.sequence} "
                f"digest {current.sha256}, got sequence {parent.sequence} digest {parent.sha256}"
            )

    def latest_manifest(self) -> Manifest | None:
        """Return the newest state of a verified chain, or None when nothing is published."""
        chain = self.manifest_chain()
        return chain[-1] if chain else None

    def manifest_chain(self) -> tuple[Manifest, ...]:
        """Return every published manifest, oldest first, after verifying the parent links."""
        by_sequence = self._manifest_index()
        if not by_sequence:
            return ()
        expected = list(range(len(by_sequence)))
        if sorted(by_sequence) != expected:
            raise StoreCorruptionError(
                f"manifest chain is not contiguous from zero: {sorted(by_sequence)}"
            )
        chain: list[Manifest] = []
        parent_digest: str | None = None
        for sequence in expected:
            manifest = self._load_manifest(sequence, by_sequence[sequence])
            if manifest.sequence != sequence or manifest.parent_sha256 != parent_digest:
                raise StoreCorruptionError(f"manifest {sequence} does not link to its parent")
            chain.append(manifest)
            parent_digest = manifest.sha256
        return tuple(chain)

    def _manifest_index(self) -> dict[int, str]:
        by_sequence: dict[int, str] = {}
        for sequence, digest in self._marked_manifests():
            existing = by_sequence.get(sequence)
            if existing is not None and existing != digest:
                raise StoreDivergenceError(
                    f"manifest sequence {sequence} holds two different states: {existing} "
                    f"and {digest}"
                )
            by_sequence[sequence] = digest
        return by_sequence

    def _load_manifest(self, sequence: int, digest: str) -> Manifest:
        content = self._read_marked(self._manifest_path(sequence, digest), "manifest", digest)
        return self._decode_manifest(content, digest)

    def _manifest_path(self, sequence: int, digest: str) -> Path:
        return self._manifests / f"{sequence:0{_SEQUENCE_WIDTH}d}.{digest}.manifest.json"

    def _marked_manifests(self) -> Iterator[tuple[int, str]]:
        """Yield completed manifests only, distinguishing an interrupted write from corruption."""
        for payload_path in self._manifest_payloads():
            sequence, digest = self._manifest_address(payload_path)
            marker = self._load_marker(payload_path, "manifest", None)
            if marker is None:
                continue
            if marker["sha256"] != digest:
                raise StoreCorruptionError(
                    f"manifest marker digest does not match its address: {payload_path}"
                )
            yield sequence, digest

    def _manifest_payloads(self) -> list[Path]:
        if not self._manifests.is_dir():
            return []
        return sorted(self._manifests.glob("*.manifest.json"))

    def _manifest_address(self, payload_path: Path) -> tuple[int, str]:
        sequence_token, digest, _rest = payload_path.name.split(".", 2)
        if not sequence_token.isdigit() or not _is_sha256(digest):
            raise StoreCorruptionError(f"manifest filename is malformed: {payload_path}")
        return int(sequence_token), digest

    def _decode_manifest(self, content: bytes, digest: str) -> Manifest:
        if hashlib.sha256(content).hexdigest() != digest:
            raise StoreCorruptionError(f"manifest content does not match its address {digest}")
        try:
            decoded: Any = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise StoreCorruptionError(f"manifest {digest} is not valid JSON") from error
        body = _require_object(decoded, "manifest")
        if canonical_json_bytes(body) != content:
            raise StoreCorruptionError(f"manifest {digest} is not canonically encoded")
        if body.get("schema_version") != _MANIFEST_SCHEMA:
            raise StoreCorruptionError(f"manifest {digest} has an unexpected schema")
        sequence = body.get("sequence")
        parent = body.get("parent_sha256")
        if type(sequence) is not int or sequence < 0:
            raise StoreCorruptionError(f"manifest {digest} has an invalid sequence")
        if parent is not None and not _is_sha256(parent):
            raise StoreCorruptionError(f"manifest {digest} has an invalid parent digest")
        entries_value = _require_object(body.get("entries"), "manifest entries")
        entries = {key: BlobRef.from_payload(value) for key, value in entries_value.items()}
        return Manifest(
            sequence=sequence,
            parent_sha256=cast("str | None", parent),
            entries=entries,
            meta=_require_object(body.get("meta"), "manifest meta"),
            sha256=digest,
        )

    # -- verification --------------------------------------------------------------

    def verify(self) -> StoreReport:
        """Audit every published file, marker, and link. Reads only; never writes or deletes."""
        issues: list[StoreIssue] = []
        issues.extend(self._audit_header())
        blob_digests = self._audit_blobs(issues)
        chain = self._audit_manifests(issues)
        issues.extend(self._audit_partials())
        referenced = self._audit_references(chain, issues)
        issues.extend(
            StoreIssue(
                path=str(self._blob_path(digest).relative_to(self._root)),
                category="blob",
                problem="unreferenced_blob",
                severity=_INFO,
            )
            for digest in sorted(blob_digests - referenced)
        )
        return StoreReport(
            blob_count=len(blob_digests),
            manifest_count=len(chain),
            latest_sequence=chain[-1].sequence if chain else None,
            issues=tuple(issues),
        )

    def _audit_header(self) -> list[StoreIssue]:
        path = self._header_path
        relative = path.name
        if not path.is_file():
            problem = "missing_header" if not _marker_of(path).is_file() else "orphan_marker"
            return [StoreIssue(relative, "header", problem, _ERROR)]
        if not _marker_of(path).is_file():
            return [StoreIssue(relative, "header", "missing_marker", _ERROR)]
        if not self.has_valid_header():
            return [StoreIssue(relative, "header", "invalid_header", _ERROR)]
        return []

    def _audit_blobs(self, issues: list[StoreIssue]) -> set[str]:
        complete: set[str] = set()
        if not self._blobs.is_dir():
            return complete
        for payload_path in sorted(self._blobs.rglob("*.blob")):
            relative = str(payload_path.relative_to(self._root))
            digest = payload_path.name.removesuffix(".blob")
            if not _is_sha256(digest):
                issues.append(StoreIssue(relative, "blob", "malformed_address", _ERROR))
                continue
            if not _marker_of(payload_path).is_file():
                issues.append(StoreIssue(relative, "blob", "missing_marker", _ERROR))
                continue
            if self._load_marker(payload_path, "blob", digest) is None:
                issues.append(StoreIssue(relative, "blob", "invalid_marker", _ERROR))
                continue
            try:
                self._read_marked(payload_path, "blob", digest)
            except StoreCorruptionError:
                issues.append(StoreIssue(relative, "blob", "content_mismatch", _ERROR))
                continue
            complete.add(digest)
        issues.extend(self._orphan_markers(self._blobs, ".blob", "blob"))
        return complete

    def _audit_manifests(self, issues: list[StoreIssue]) -> tuple[Manifest, ...]:
        for payload_path in self._manifest_payloads():
            relative = str(payload_path.relative_to(self._root))
            try:
                _sequence, digest = self._manifest_address(payload_path)
            except StoreCorruptionError:
                issues.append(StoreIssue(relative, "manifest", "malformed_address", _ERROR))
                continue
            if not _marker_of(payload_path).is_file():
                issues.append(StoreIssue(relative, "manifest", "missing_marker", _ERROR))
                continue
            if self._load_marker(payload_path, "manifest", digest) is None:
                issues.append(StoreIssue(relative, "manifest", "invalid_marker", _ERROR))
                continue
            try:
                self._read_marked(payload_path, "manifest", digest)
            except StoreCorruptionError:
                issues.append(StoreIssue(relative, "manifest", "content_mismatch", _ERROR))
        issues.extend(self._orphan_markers(self._manifests, ".manifest.json", "manifest"))
        try:
            return self.manifest_chain()
        except StoreDivergenceError:
            issues.append(StoreIssue("manifests", "manifest", "divergent_state", _ERROR))
        except StoreCorruptionError:
            issues.append(StoreIssue("manifests", "manifest", "broken_chain", _ERROR))
        return ()

    def _audit_references(self, chain: Sequence[Manifest], issues: list[StoreIssue]) -> set[str]:
        referenced: set[str] = set()
        for manifest in chain:
            for key, reference in sorted(manifest.entries.items()):
                referenced.add(reference.sha256)
                try:
                    self.get_bytes(reference)
                except (StoreCorruptionError, ValueError):
                    issues.append(
                        StoreIssue(
                            path=f"manifests/{manifest.sequence}#{key}",
                            category="manifest",
                            problem="unresolvable_entry",
                            severity=_ERROR,
                        )
                    )
        return referenced

    def _audit_partials(self) -> list[StoreIssue]:
        return [
            StoreIssue(
                path=str(path.relative_to(self._root)),
                category="store",
                problem="interrupted_write",
                severity=_INFO,
            )
            for path in sorted(self._root.rglob(f"*{_PARTIAL_SUFFIX}"))
        ]

    def _orphan_markers(self, directory: Path, suffix: str, category: str) -> list[StoreIssue]:
        if not directory.is_dir():
            return []
        return [
            StoreIssue(
                path=str(marker_path.relative_to(self._root)),
                category=category,
                problem="orphan_marker",
                severity=_ERROR,
            )
            for marker_path in sorted(directory.rglob(f"*{suffix}{_MARKER_SUFFIX}"))
            if not marker_path.with_name(marker_path.name.removesuffix(_MARKER_SUFFIX)).is_file()
        ]

    # -- publication primitives ----------------------------------------------------

    def _publish(
        self, payload_path: Path, content: bytes, kind: str, expected_sha256: str | None
    ) -> None:
        marker_path = _marker_of(payload_path)
        if self._load_marker(payload_path, kind, expected_sha256) is not None:
            if payload_path.read_bytes() != content:
                raise StoreCorruptionError(
                    f"completed payload conflicts with the content being published: {payload_path}"
                )
            return
        if marker_path.exists():
            raise StoreCorruptionError(
                f"completion marker is unreadable or inconsistent: {marker_path}"
            )
        _write_verified(payload_path, content)
        marker = {
            "schema_version": _MARKER_SCHEMA,
            "kind": kind,
            "name": payload_path.name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }
        _write_verified(marker_path, canonical_json_bytes(marker))

    def _load_marker(
        self, payload_path: Path, kind: str, expected_sha256: str | None
    ) -> dict[str, object] | None:
        try:
            raw = _marker_of(payload_path).read_bytes()
        except (OSError, ValueError):
            return None
        try:
            decoded: Any = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(decoded, dict) or set(decoded) != _MARKER_FIELDS:
            return None
        consistent = (
            decoded.get("schema_version") == _MARKER_SCHEMA
            and decoded.get("kind") == kind
            and decoded.get("name") == payload_path.name
            and _is_sha256(decoded.get("sha256"))
            and (expected_sha256 is None or decoded.get("sha256") == expected_sha256)
            and type(decoded.get("size")) is int
            and payload_path.is_file()
        )
        return cast("dict[str, object]", decoded) if consistent else None

    def _read_marked(self, payload_path: Path, kind: str, expected_sha256: str | None) -> bytes:
        marker = self._load_marker(payload_path, kind, None)
        if marker is None:
            raise StoreCorruptionError(f"no valid completion marker for {payload_path}")
        if expected_sha256 is not None and marker["sha256"] != expected_sha256:
            raise StoreCorruptionError(f"marker digest does not match its address: {payload_path}")
        content = payload_path.read_bytes()
        if len(content) != marker["size"]:
            raise StoreCorruptionError(f"payload size does not match its marker: {payload_path}")
        observed = hashlib.sha256(content).hexdigest()
        if observed != marker["sha256"]:
            raise StoreCorruptionError(f"payload digest does not match its marker: {payload_path}")
        if expected_sha256 is not None and observed != expected_sha256:
            raise StoreCorruptionError(f"payload digest does not match its address: {payload_path}")
        return content


class RunStoreSession:
    """Append manifests from one verified tip without rescanning the historical chain.

    A session is a single-writer, non-thread-safe cache. Open a new session after any ambiguous I/O
    failure or when transferring ownership to another writer.
    """

    def __init__(self, store: ContentAddressedRunStore, manifests: tuple[Manifest, ...]) -> None:
        self._store = store
        self._manifests_cache = list(manifests)
        self._observed_generation = store._manifest_generation
        self._poisoned_reason: str | None = None

    @classmethod
    def open(cls, store: ContentAddressedRunStore) -> RunStoreSession:
        """Verify the complete chain once and cache its current tip."""
        store._require_header()
        return cls(store, store.manifest_chain())

    @property
    def store(self) -> ContentAddressedRunStore:
        """The durable store backing this session."""
        return self._store

    def latest_manifest(self) -> Manifest | None:
        """Return the cached verified tip without filesystem access."""
        return self._manifests_cache[-1] if self._manifests_cache else None

    def manifests(self) -> tuple[Manifest, ...]:
        """Return the cached verified chain without filesystem access."""
        return tuple(self._manifests_cache)

    def poison(self, reason: str) -> None:
        """Prevent further drafts or writes after an uncertain session outcome."""
        if not reason:
            raise ValueError("poison reason must not be empty")
        if self._poisoned_reason is None:
            self._poisoned_reason = reason

    def draft_manifest(
        self, *, entries: Mapping[str, BlobRef], meta: Mapping[str, object]
    ) -> ManifestDraft:
        """Build the canonical next manifest against the cached tip."""
        self._require_healthy()
        parent = self.latest_manifest()
        sequence = 0 if parent is None else parent.sequence + 1
        parent_sha256 = None if parent is None else parent.sha256
        entry_copy = dict(entries)
        meta_copy = dict(meta)
        content = canonical_json_bytes(
            self._manifest_body(sequence, parent_sha256, entry_copy, meta_copy)
        )
        return ManifestDraft(
            sequence=sequence,
            parent_sha256=parent_sha256,
            entries=MappingProxyType(entry_copy),
            meta=MappingProxyType(meta_copy),
            sha256=hashlib.sha256(content).hexdigest(),
            content=content,
        )

    def publish_manifest(self, draft: ManifestDraft) -> Manifest:  # noqa: PLR0912
        """Publish or recover one exact append while touching only adjacent states."""
        self._require_healthy()
        self._validate_draft(draft)
        self._verify_entries(draft.entries)
        tip = self.latest_manifest()
        if tip is not None and draft.sequence <= tip.sequence:
            cached = self._manifests_cache[draft.sequence]
            if draft.sha256 != cached.sha256:
                raise StoreDivergenceError(
                    f"manifest sequence {draft.sequence} already holds a different state"
                )
            exact = self._load_session_manifest(draft.sequence, draft.sha256)
            if (
                exact != cached
                or exact.parent_sha256 != draft.parent_sha256
                or exact.entries != draft.entries
                or exact.meta != draft.meta
            ):
                raise StoreCorruptionError("published manifest differs from its canonical draft")
            return exact
        expected_sequence = 0 if tip is None else tip.sequence + 1
        if draft.sequence < expected_sequence:
            raise RunStoreError("manifest draft is stale relative to the cached tip")
        if draft.sequence > expected_sequence:
            raise RunStoreError("manifest draft skips the cached manifest tip")
        expected_parent = None if tip is None else tip.sha256
        if draft.parent_sha256 != expected_parent:
            raise RunStoreError("manifest draft has a stale or forged parent")
        if tip is not None:
            verified_parent = self._load_session_manifest(tip.sequence, tip.sha256)
            if verified_parent != tip:
                raise StoreCorruptionError("cached manifest tip no longer matches durable state")
        if self._store._manifest_generation != self._observed_generation:
            return self._adopt_after_generation_change(draft)
        path = self._store._manifest_path(draft.sequence, draft.sha256)
        marker = self._store._load_marker(path, "manifest", draft.sha256)
        if marker is None and _marker_of(path).exists():
            self.poison("candidate manifest has an inconsistent completion marker")
            raise StoreCorruptionError(
                f"completion marker is unreadable or inconsistent: {_marker_of(path)}"
            )
        if marker is None:
            try:
                self._store._publish(path, draft.content, "manifest", draft.sha256)
            except (OSError, StoreDurabilityError) as error:
                self.poison(f"ambiguous manifest publication failure: {error}")
                raise
            self._store._manifest_generation += 1
            self._observed_generation += 1
        try:
            manifest = self._load_session_manifest(draft.sequence, draft.sha256)
        except StoreCorruptionError as error:
            self.poison(f"published manifest could not be verified: {error}")
            raise
        if (
            manifest.sequence != draft.sequence
            or manifest.parent_sha256 != draft.parent_sha256
            or manifest.entries != draft.entries
            or manifest.meta != draft.meta
            or manifest.sha256 != draft.sha256
        ):
            self.poison("published manifest differs from its canonical draft")
            raise StoreCorruptionError("published manifest differs from its canonical draft")
        self._manifests_cache.append(manifest)
        return manifest

    def _require_healthy(self) -> None:
        if self._poisoned_reason is not None:
            raise RunStoreError(f"run-store session is poisoned: {self._poisoned_reason}")

    def _adopt_after_generation_change(self, draft: ManifestDraft) -> Manifest:
        path = self._store._manifest_path(draft.sequence, draft.sha256)
        marker = self._store._load_marker(path, "manifest", draft.sha256)
        if marker is None:
            self.poison("store generation advanced without the expected manifest")
            raise RunStoreError("run-store generation advanced; the expected manifest is absent")
        manifest = self._load_session_manifest(draft.sequence, draft.sha256)
        if (
            manifest.parent_sha256 != draft.parent_sha256
            or manifest.entries != draft.entries
            or manifest.meta != draft.meta
            or manifest.sha256 != draft.sha256
        ):
            self.poison("store generation advanced to a different manifest")
            raise StoreCorruptionError("durable manifest differs from its canonical draft")
        self._manifests_cache.append(manifest)
        self._observed_generation += 1
        return manifest

    @staticmethod
    def _manifest_body(
        sequence: int,
        parent_sha256: str | None,
        entries: Mapping[str, BlobRef],
        meta: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            "schema_version": _MANIFEST_SCHEMA,
            "sequence": sequence,
            "parent_sha256": parent_sha256,
            "entries": {key: reference.payload() for key, reference in sorted(entries.items())},
            "meta": dict(meta),
        }

    def _validate_draft(self, draft: ManifestDraft) -> None:
        if type(draft.sequence) is not int or draft.sequence < 0:
            raise RunStoreError("manifest draft has an invalid sequence")
        if draft.parent_sha256 is not None and not _is_sha256(draft.parent_sha256):
            raise RunStoreError("manifest draft has an invalid parent digest")
        expected = canonical_json_bytes(
            self._manifest_body(draft.sequence, draft.parent_sha256, draft.entries, draft.meta)
        )
        digest = hashlib.sha256(expected).hexdigest()
        if draft.content != expected or draft.sha256 != digest:
            raise RunStoreError("manifest draft is not its canonical encoding")

    def _verify_entries(self, entries: Mapping[str, BlobRef]) -> None:
        for key in sorted(entries):
            try:
                self._store.get_bytes(entries[key])
            except OSError as error:
                self.poison(f"manifest entry could not be verified: {error}")
                raise
            except (StoreCorruptionError, ValueError) as error:
                raise RunStoreError(
                    f"manifest entry {key!r} does not resolve to a verified blob: {error}"
                ) from error

    def _load_session_manifest(self, sequence: int, digest: str) -> Manifest:
        try:
            return self._store._load_manifest(sequence, digest)
        except (OSError, StoreCorruptionError) as error:
            self.poison(f"manifest could not be read reliably: {error}")
            raise


def _marker_of(payload_path: Path) -> Path:
    return payload_path.with_name(payload_path.name + _MARKER_SUFFIX)
