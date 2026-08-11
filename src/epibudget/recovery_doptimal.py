"""Durable block journal for the reduced D-optimal selection state."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import numpy as np
import numpy.typing as npt

from epibudget.coeff_recovery import _order_symmetric_kernel
from epibudget.fourier_recovery import (
    ReducedDOptimalState,
    _canonical_identity_sha256,
    _sequence_sha256,
)
from epibudget.recovery_protocol import RecoveryExecutionPolicy
from epibudget.recovery_state import RecoveryStateCursor
from epibudget.run_store import (
    ArrayRef,
    ContentAddressedRunStore,
    Manifest,
    RunStoreError,
    StoreCorruptionError,
    StoreDivergenceError,
    canonical_json_bytes,
)

_SCHEMA = "epibudget-reduced-doptimal-delta-v1"
_STATE_KIND = "reduced_doptimal"
_SHA256_LENGTH = 64
_ARRAY_NAMES = frozenset({"updates", "selected_indices", "posterior_variance"})
_META_NAMES = frozenset(
    {
        "schema_version",
        "state_kind",
        "identity",
        "start",
        "stop",
        "prefix_sha256",
        "arrays",
    }
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class DOptimalCheckpointError(RunStoreError):
    """A D-optimal checkpoint cannot be published or restored safely."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def doptimal_geometry_sha256(state: ReducedDOptimalState) -> str:
    """Hash every numeric geometry field needed to interpret a D-optimal state."""
    site_indices = np.asarray(state.site_indices)
    if site_indices.dtype.hasobject or site_indices.dtype.kind not in {"i", "u"}:
        raise ValueError("D-optimal site indices must use a plain integer dtype")
    little = np.asarray(site_indices, dtype=site_indices.dtype.newbyteorder("<"), order="C")
    payload = {
        "q": state.q,
        "population_size": state.population_size,
        "site_indices": {
            "dtype": site_indices.dtype.name,
            "byteorder": "little",
            "shape": [int(extent) for extent in site_indices.shape],
            "sha256": hashlib.sha256(little.tobytes(order="C")).hexdigest(),
        },
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class DOptimalCheckpointIdentity:
    """Immutable scientific, execution, numerical, and candidate identity."""

    scientific_identity_sha256: str
    execution_policy: RecoveryExecutionPolicy
    numerical_compatibility: Mapping[str, object]
    numerical_compatibility_sha256: str
    candidate_sha256: str
    candidate_count: int
    target_budget: int
    geometry_sha256: str

    def __post_init__(self) -> None:
        """Reject an identity whose declared digests or dimensions disagree."""
        digests = (
            self.scientific_identity_sha256,
            self.numerical_compatibility_sha256,
            self.candidate_sha256,
            self.geometry_sha256,
        )
        if not all(_is_sha256(value) for value in digests):
            raise ValueError("D-optimal checkpoint identity contains an invalid SHA-256")
        numerical_bytes = canonical_json_bytes(dict(self.numerical_compatibility))
        numerical_digest = hashlib.sha256(numerical_bytes).hexdigest()
        if numerical_digest != self.numerical_compatibility_sha256:
            raise ValueError("D-optimal numerical compatibility digest does not match its payload")
        policy_rendered = json.dumps(
            self.execution_policy.policy_payload(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        policy_digest = hashlib.sha256(policy_rendered.encode("utf-8")).hexdigest()
        if policy_digest != self.execution_policy.policy_sha256:
            raise ValueError("D-optimal execution policy digest does not match its payload")
        decoded: object = json.loads(numerical_bytes)
        frozen = _freeze_json(decoded)
        object.__setattr__(self, "numerical_compatibility", cast("Mapping[str, object]", frozen))
        if (
            type(self.candidate_count) is not int
            or type(self.target_budget) is not int
            or self.candidate_count < 1
            or not 1 <= self.target_budget <= self.candidate_count
        ):
            raise ValueError("D-optimal checkpoint identity has invalid dimensions")

    @property
    def block_size(self) -> int:
        """Return the block size fixed by the complete execution policy."""
        return self.execution_policy.doptimal_block_size

    def payload(self) -> dict[str, object]:
        """Return the complete canonical identity embedded in every delta."""
        numerical = _thaw_json(self.numerical_compatibility)
        return {
            "scientific_identity_sha256": self.scientific_identity_sha256,
            "execution_policy": self.execution_policy.identity_payload(),
            "numerical_compatibility": numerical,
            "numerical_compatibility_sha256": self.numerical_compatibility_sha256,
            "candidate_sha256": self.candidate_sha256,
            "candidate_count": self.candidate_count,
            "target_budget": self.target_budget,
            "geometry_sha256": self.geometry_sha256,
        }


@dataclass(frozen=True)
class _JournalProgress:
    selected_indices: list[int]
    posterior_variance: FloatArray
    manifests: tuple[Manifest, ...]
    latest_doptimal: Manifest | None


def _prefix_sha256(state: ReducedDOptimalState, selected: list[int]) -> str:
    return _sequence_sha256(tuple(state.candidates[index] for index in selected))


def _require_state_identity(  # noqa: PLR0912
    state: ReducedDOptimalState, identity: DOptimalCheckpointIdentity
) -> None:
    if state.target_budget != identity.target_budget:
        raise DOptimalCheckpointError("D-optimal state target budget does not match its identity")
    if len(state.candidates) != identity.candidate_count:
        raise DOptimalCheckpointError("D-optimal state candidate count does not match its identity")
    if _sequence_sha256(state.candidates) != identity.candidate_sha256:
        raise DOptimalCheckpointError("D-optimal state candidate hash does not match its identity")
    try:
        geometry_sha256 = doptimal_geometry_sha256(state)
    except ValueError as error:
        raise DOptimalCheckpointError("D-optimal state geometry is invalid") from error
    if geometry_sha256 != identity.geometry_sha256:
        raise DOptimalCheckpointError("D-optimal state geometry does not match its identity")
    if state.updates.dtype != np.dtype(np.float64):
        raise DOptimalCheckpointError("D-optimal updates must use float64")
    if state.updates.shape != (identity.candidate_count, identity.target_budget):
        raise DOptimalCheckpointError("D-optimal updates have an incompatible shape")
    if not state.updates.flags.c_contiguous:
        raise DOptimalCheckpointError("D-optimal updates must be C-contiguous")
    if state.posterior_variance.dtype != np.dtype(np.float64):
        raise DOptimalCheckpointError("D-optimal posterior variance must use float64")
    if state.posterior_variance.shape != (identity.candidate_count,):
        raise DOptimalCheckpointError("D-optimal posterior variance has an incompatible shape")
    if not np.all(np.isfinite(state.updates)) or not np.all(np.isfinite(state.posterior_variance)):
        raise DOptimalCheckpointError("D-optimal state contains non-finite values")
    if np.any(state.posterior_variance < 0.0):
        raise DOptimalCheckpointError("D-optimal posterior variance must be nonnegative")
    if (
        len(state.selected_indices) > identity.target_budget
        or len(set(state.selected_indices)) != len(state.selected_indices)
        or any(not 0 <= index < identity.candidate_count for index in state.selected_indices)
    ):
        raise DOptimalCheckpointError("D-optimal selected indices are invalid")


def _require_incremental_state(
    state: ReducedDOptimalState,
    identity: DOptimalCheckpointIdentity,
    *,
    start: int,
    stop: int,
) -> None:
    if state.target_budget != identity.target_budget:
        raise DOptimalCheckpointError("D-optimal state target budget does not match its identity")
    if len(state.candidates) != identity.candidate_count:
        raise DOptimalCheckpointError("D-optimal state candidate count does not match its identity")
    if _sequence_sha256(state.candidates) != identity.candidate_sha256:
        raise DOptimalCheckpointError("D-optimal state candidate hash does not match its identity")
    try:
        geometry_sha256 = doptimal_geometry_sha256(state)
    except ValueError as error:
        raise DOptimalCheckpointError("D-optimal state geometry is invalid") from error
    if geometry_sha256 != identity.geometry_sha256:
        raise DOptimalCheckpointError("D-optimal state geometry does not match its identity")
    if state.updates.dtype != np.dtype(np.float64) or state.updates.shape != (
        identity.candidate_count,
        identity.target_budget,
    ):
        raise DOptimalCheckpointError("D-optimal updates have an incompatible layout")
    if not state.updates.flags.c_contiguous:
        raise DOptimalCheckpointError("D-optimal updates must be C-contiguous")
    if state.posterior_variance.dtype != np.dtype(np.float64) or (
        state.posterior_variance.shape != (identity.candidate_count,)
    ):
        raise DOptimalCheckpointError("D-optimal posterior variance has an incompatible layout")
    if not np.all(np.isfinite(state.updates[:, start:stop])) or not np.all(
        np.isfinite(state.posterior_variance)
    ):
        raise DOptimalCheckpointError("D-optimal new block contains non-finite values")
    if np.any(state.posterior_variance < 0.0):
        raise DOptimalCheckpointError("D-optimal posterior variance must be nonnegative")
    if (
        len(state.selected_indices) > identity.target_budget
        or len(set(state.selected_indices)) != len(state.selected_indices)
        or any(not 0 <= index < identity.candidate_count for index in state.selected_indices)
    ):
        raise DOptimalCheckpointError("D-optimal selected indices are invalid")


def _require_mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise DOptimalCheckpointError(f"D-optimal checkpoint {field} must be an object")
    return cast("dict[str, object]", value)


def _array_reference(manifest: Manifest, arrays: Mapping[str, object], name: str) -> ArrayRef:
    try:
        reference = ArrayRef.from_payload(arrays[name])
        entry = manifest.entry(name)
    except (KeyError, RunStoreError, ValueError, TypeError) as error:
        raise DOptimalCheckpointError(
            f"D-optimal checkpoint has an invalid {name} ArrayRef"
        ) from error
    if reference.blob != entry:
        raise DOptimalCheckpointError(
            f"D-optimal checkpoint {name} ArrayRef does not match its manifest entry"
        )
    return reference


def _load_array(
    store: ContentAddressedRunStore,
    reference: ArrayRef,
    *,
    name: str,
    dtype: str,
    shape: tuple[int, ...],
) -> npt.NDArray[np.generic]:
    if (
        reference.dtype != dtype
        or reference.byteorder != "little"
        or reference.order != "C"
        or reference.shape != shape
    ):
        raise DOptimalCheckpointError(
            f"D-optimal checkpoint {name} has an incompatible binary layout"
        )
    try:
        array = store.get_array(reference)
    except StoreCorruptionError:
        raise
    except (RunStoreError, ValueError, TypeError) as error:
        raise DOptimalCheckpointError(
            f"D-optimal checkpoint {name} array cannot be verified"
        ) from error
    if not np.all(np.isfinite(array)):
        raise DOptimalCheckpointError(f"D-optimal checkpoint {name} contains non-finite values")
    return array


def _validate_structural_delta(
    state: ReducedDOptimalState,
    previous_selected: list[int],
    previous_posterior: FloatArray,
    prefix_updates: FloatArray,
    selected_delta: list[int],
    update_delta: FloatArray,
    posterior_snapshot: FloatArray,
) -> None:
    """Validate pivots, one exact covariance column, and the O(NB) variance recurrence.

    Later covariance columns remain trusted producer output protected by the content-addressed store
    and exact run identity. Checking the first column rejects grossly inconsistent chunks without an
    O(NB^2) replay; it is not a cryptographic proof of every covariance column.
    """
    selected = list(previous_selected)
    start = len(selected)
    first_pick = selected_delta[0]
    prior_covariance = (
        state.population_size
        * _order_symmetric_kernel(
            state.site_indices, state.site_indices[[first_pick]], state.q, (1, 2)
        )[:, 0]
    )
    covariance = prior_covariance - prefix_updates[:, :start] @ prefix_updates[first_pick, :start]
    denominator = 1.0 + max(float(covariance[first_pick]), 0.0)
    expected_first = covariance / np.sqrt(denominator)
    if expected_first.tobytes(order="C") != update_delta[:, 0].tobytes(order="C"):
        raise DOptimalCheckpointError(
            "D-optimal checkpoint structural first update column does not match the exact "
            "kernel covariance"
        )
    for column, observed in enumerate(selected_delta):
        available = previous_posterior.copy()
        available[selected] = -np.inf
        maximum = float(np.max(available))
        tied = np.flatnonzero(available == maximum)
        expected = min(
            (int(index) for index in tied),
            key=lambda index: _canonical_identity_sha256(state.candidates[index]),
        )
        if observed != expected:
            raise DOptimalCheckpointError(
                "D-optimal checkpoint structural pivot does not match the variance argmax"
            )
        selected.append(observed)
        previous_posterior[:] = np.maximum(
            previous_posterior - np.square(update_delta[:, column]), 0.0
        )
    if previous_posterior.tobytes(order="C") != posterior_snapshot.tobytes(order="C"):
        raise DOptimalCheckpointError(
            "D-optimal checkpoint structural posterior recurrence does not match its snapshot"
        )


def _initial_posterior(state: ReducedDOptimalState) -> FloatArray:
    diagonal = float(
        state.population_size
        * _order_symmetric_kernel(
            state.site_indices[[0]], state.site_indices[[0]], state.q, (1, 2)
        )[0, 0]
    )
    return np.full(len(state.candidates), diagonal, dtype=np.float64)


def _require_empty_initial_state(
    state: ReducedDOptimalState, identity: DOptimalCheckpointIdentity
) -> None:
    _require_state_identity(state, identity)
    if state.selected_indices or np.any(state.updates != 0.0):
        raise DOptimalCheckpointError("D-optimal restore requires an empty initial state")
    expected_posterior = _initial_posterior(state)
    if state.posterior_variance.tobytes(order="C") != expected_posterior.tobytes(order="C"):
        raise DOptimalCheckpointError("D-optimal restore requires a fresh initial variance")


def _scan_deltas(  # noqa: PLR0912, PLR0915
    store: ContentAddressedRunStore,
    state: ReducedDOptimalState,
    identity: DOptimalCheckpointIdentity,
    *,
    update_buffer: FloatArray,
    posterior: FloatArray,
    copy_updates: bool,
    manifests: tuple[Manifest, ...] | None = None,
) -> _JournalProgress:
    _require_state_identity(state, identity)
    if manifests is None:
        try:
            manifests = store.manifest_chain()
        except StoreCorruptionError:
            raise
        except RunStoreError as error:
            raise DOptimalCheckpointError("D-optimal manifest chain cannot be verified") from error
    selected: list[int] = []
    latest_doptimal: Manifest | None = None
    expected_identity = canonical_json_bytes(identity.payload())
    for manifest in manifests:
        if manifest.meta.get("state_kind") != _STATE_KIND:
            continue
        if set(manifest.meta) != _META_NAMES:
            raise DOptimalCheckpointError("D-optimal checkpoint metadata fields do not match")
        if manifest.meta.get("schema_version") != _SCHEMA:
            raise DOptimalCheckpointError("D-optimal checkpoint schema does not match")
        observed_identity = _require_mapping(manifest.meta.get("identity"), "identity")
        if canonical_json_bytes(observed_identity) != expected_identity:
            raise DOptimalCheckpointError("D-optimal checkpoint identity does not match")
        start = manifest.meta.get("start")
        stop = manifest.meta.get("stop")
        if type(start) is not int or type(stop) is not int:
            raise DOptimalCheckpointError("D-optimal checkpoint bounds must be integers")
        if start != len(selected):
            raise DOptimalCheckpointError("D-optimal checkpoint chain has a gap or overlap")
        if stop - start != identity.block_size or stop > identity.target_budget:
            raise DOptimalCheckpointError("D-optimal checkpoint chunk width does not match")
        if set(manifest.entries) != _ARRAY_NAMES:
            raise DOptimalCheckpointError("D-optimal checkpoint entries do not match")
        arrays = _require_mapping(manifest.meta.get("arrays"), "arrays")
        if set(arrays) != _ARRAY_NAMES:
            raise DOptimalCheckpointError("D-optimal checkpoint ArrayRefs do not match")
        selected_ref = _array_reference(manifest, arrays, "selected_indices")
        updates_ref = _array_reference(manifest, arrays, "updates")
        posterior_ref = _array_reference(manifest, arrays, "posterior_variance")
        selected_delta = _load_array(
            store,
            selected_ref,
            name="selected_indices",
            dtype="int64",
            shape=(identity.block_size,),
        )
        update_delta = _load_array(
            store,
            updates_ref,
            name="updates",
            dtype="float64",
            shape=(identity.candidate_count, identity.block_size),
        )
        posterior_snapshot = _load_array(
            store,
            posterior_ref,
            name="posterior_variance",
            dtype="float64",
            shape=(identity.candidate_count,),
        )
        decoded_selected = [int(index) for index in selected_delta]
        if any(not 0 <= index < identity.candidate_count for index in decoded_selected):
            raise DOptimalCheckpointError("D-optimal checkpoint index is out of range")
        if set(selected).intersection(decoded_selected) or len(set(decoded_selected)) != len(
            decoded_selected
        ):
            raise DOptimalCheckpointError("D-optimal checkpoint indices are not globally unique")
        selected.extend(decoded_selected)
        prefix = manifest.meta.get("prefix_sha256")
        if prefix != _prefix_sha256(state, selected):
            raise DOptimalCheckpointError("D-optimal checkpoint prefix SHA does not match")
        posterior_array = np.asarray(posterior_snapshot, dtype=np.float64)
        if np.any(posterior_array < 0.0):
            raise DOptimalCheckpointError("D-optimal posterior variance must be nonnegative")
        update_array = np.asarray(update_delta, dtype=np.float64)
        if not copy_updates:
            if decoded_selected != state.selected_indices[start:stop]:
                raise DOptimalCheckpointError("D-optimal selected prefix diverges from the journal")
            if update_buffer[:, start:stop].tobytes(order="C") != update_array.tobytes(order="C"):
                raise DOptimalCheckpointError("D-optimal update prefix diverges from the journal")
        _validate_structural_delta(
            state,
            selected[: -len(decoded_selected)],
            posterior,
            update_buffer,
            decoded_selected,
            update_array,
            posterior_array,
        )
        if copy_updates:
            update_buffer[:, start:stop] = update_array
        latest_doptimal = manifest
    return _JournalProgress(
        selected_indices=selected,
        posterior_variance=posterior,
        manifests=manifests,
        latest_doptimal=latest_doptimal,
    )


def _update_block_sha256(state: ReducedDOptimalState, start: int, stop: int) -> str:
    block = np.asarray(state.updates[:, start:stop], dtype=np.dtype("<f8"), order="C")
    return hashlib.sha256(block.tobytes(order="C")).hexdigest()


class DOptimalCheckpointCursor:
    """Incremental D-optimal journal over one already-replayed recovery cursor."""

    def __init__(
        self,
        recovery: RecoveryStateCursor,
        state: ReducedDOptimalState,
        identity: DOptimalCheckpointIdentity,
        committed_selected: list[int],
        committed_posterior: FloatArray,
        committed_update_sha256: list[str],
        latest_manifest: Manifest | None,
    ) -> None:
        self._recovery = recovery
        self._state = state
        self._identity = identity
        self._committed_selected = committed_selected
        self._committed_posterior = committed_posterior
        self._committed_update_sha256 = committed_update_sha256
        self._latest_manifest = latest_manifest

    @classmethod
    def restore(
        cls,
        recovery: RecoveryStateCursor,
        initial: ReducedDOptimalState,
        identity: DOptimalCheckpointIdentity,
    ) -> DOptimalCheckpointCursor:
        """Restore indexed D-optimal blocks once into the supplied full-capacity buffer."""
        _require_empty_initial_state(initial, identity)
        manifests = recovery.index.doptimal_manifests
        restored = _scan_deltas(
            recovery.store,
            initial,
            identity,
            update_buffer=initial.updates,
            posterior=initial.posterior_variance,
            copy_updates=True,
            manifests=manifests,
        )
        initial.selected_indices.extend(restored.selected_indices)
        return cls(
            recovery,
            initial,
            identity,
            list(restored.selected_indices),
            restored.posterior_variance.copy(),
            [manifest.entry("updates").sha256 for manifest in manifests],
            restored.latest_doptimal,
        )

    @property
    def state(self) -> ReducedDOptimalState:
        """The restored state buffer advanced and checkpointed by this cursor."""
        return self._state

    def publish(self, state: ReducedDOptimalState) -> Manifest:
        """Validate and publish one new block without replaying committed blocks."""
        if state is not self._state:
            raise DOptimalCheckpointError("D-optimal cursor cannot publish a different state")
        committed = len(self._committed_selected)
        stop = len(state.selected_indices)
        _require_incremental_state(
            state,
            self._identity,
            start=max(0, min(committed, stop)),
            stop=stop,
        )
        if stop == committed:
            return self._retry_exact(state)
        if stop - committed != self._identity.block_size:
            raise DOptimalCheckpointError(
                "D-optimal checkpoint publication requires exactly one newly completed block"
            )
        if state.selected_indices[:committed] != self._committed_selected:
            raise DOptimalCheckpointError("D-optimal state diverges from its committed prefix")
        posterior = self._committed_posterior.copy()
        selected_delta = state.selected_indices[committed:stop]
        _validate_structural_delta(
            state,
            list(self._committed_selected),
            posterior,
            state.updates,
            selected_delta,
            state.updates[:, committed:stop],
            state.posterior_variance,
        )
        manifest = self._publish_new_block(state, committed, stop)
        self._committed_selected.extend(selected_delta)
        self._committed_posterior = state.posterior_variance.copy()
        self._committed_update_sha256.append(manifest.entry("updates").sha256)
        self._latest_manifest = manifest
        return manifest

    def _retry_exact(self, state: ReducedDOptimalState) -> Manifest:
        if self._latest_manifest is None:
            raise DOptimalCheckpointError("D-optimal cursor has no completed block to retry")
        if state.selected_indices != self._committed_selected:
            raise DOptimalCheckpointError("D-optimal state diverges from its committed indices")
        if state.posterior_variance.tobytes(order="C") != self._committed_posterior.tobytes(
            order="C"
        ):
            raise DOptimalCheckpointError("D-optimal state diverges from committed variance")
        for block, expected in enumerate(self._committed_update_sha256):
            start = block * self._identity.block_size
            stop = start + self._identity.block_size
            if _update_block_sha256(state, start, stop) != expected:
                raise DOptimalCheckpointError("D-optimal state diverges from committed updates")
        return self._latest_manifest

    def _publish_new_block(self, state: ReducedDOptimalState, start: int, stop: int) -> Manifest:
        store = self._recovery.store
        selected = np.asarray(state.selected_indices[start:stop], dtype=np.int64, order="C")
        references = {
            "selected_indices": store.put_array(selected),
            "updates": store.put_array(state.updates[:, start:stop]),
            "posterior_variance": store.put_array(state.posterior_variance),
        }
        draft = self._recovery.draft_manifest(
            entries={name: reference.blob for name, reference in references.items()},
            meta={
                "schema_version": _SCHEMA,
                "state_kind": _STATE_KIND,
                "identity": self._identity.payload(),
                "start": start,
                "stop": stop,
                "prefix_sha256": _prefix_sha256(state, state.selected_indices[:stop]),
                "arrays": {name: reference.payload() for name, reference in references.items()},
            },
        )
        try:
            return self._recovery.append(draft)
        except StoreCorruptionError:
            raise
        except RunStoreError as error:
            message = str(error).lower()
            if "diverg" in message or "stale" in message or "generation" in message:
                raise DOptimalCheckpointError(
                    "D-optimal checkpoint publication diverged"
                ) from error
            raise DOptimalCheckpointError("D-optimal checkpoint publication failed") from error


def _publish_delta(
    store: ContentAddressedRunStore,
    state: ReducedDOptimalState,
    identity: DOptimalCheckpointIdentity,
    *,
    start: int,
    stop: int,
    parent: Manifest | None,
) -> Manifest:
    selected = np.asarray(state.selected_indices[start:stop], dtype=np.int64, order="C")
    references = {
        "selected_indices": store.put_array(selected),
        "updates": store.put_array(state.updates[:, start:stop]),
        "posterior_variance": store.put_array(state.posterior_variance),
    }
    meta = {
        "schema_version": _SCHEMA,
        "state_kind": _STATE_KIND,
        "identity": identity.payload(),
        "start": start,
        "stop": stop,
        "prefix_sha256": _prefix_sha256(state, state.selected_indices[:stop]),
        "arrays": {name: reference.payload() for name, reference in references.items()},
    }
    try:
        return store.publish_manifest(
            entries={name: reference.blob for name, reference in references.items()},
            meta=meta,
            parent=parent,
        )
    except StoreDivergenceError as error:
        raise DOptimalCheckpointError("D-optimal checkpoint publication diverged") from error
    except StoreCorruptionError:
        raise
    except RunStoreError as error:
        message = str(error).lower()
        if "diverg" in message or "stale or forged" in message:
            raise DOptimalCheckpointError("D-optimal checkpoint publication diverged") from error
        raise DOptimalCheckpointError("D-optimal checkpoint publication failed") from error


def publish_doptimal_checkpoint(
    store: ContentAddressedRunStore,
    state: ReducedDOptimalState,
    identity: DOptimalCheckpointIdentity,
) -> Manifest:
    """Publish exactly one completed D-optimal block against the verified global parent."""
    _require_state_identity(state, identity)
    restored = _scan_deltas(
        store,
        state,
        identity,
        update_buffer=state.updates,
        posterior=_initial_posterior(state),
        copy_updates=False,
    )
    completed = len(restored.selected_indices)
    stop = len(state.selected_indices)
    if stop == completed and restored.latest_doptimal is not None:
        if not restored.manifests or restored.manifests[-1] != restored.latest_doptimal:
            raise DOptimalCheckpointError(
                "D-optimal checkpoint cannot be replayed after a newer global manifest"
            )
        parent = restored.manifests[-2] if len(restored.manifests) > 1 else None
        return _publish_delta(
            store,
            state,
            identity,
            start=stop - identity.block_size,
            stop=stop,
            parent=parent,
        )
    if stop - completed != identity.block_size:
        raise DOptimalCheckpointError(
            "D-optimal checkpoint publication requires exactly one newly completed block"
        )
    _validate_structural_delta(
        state,
        restored.selected_indices,
        restored.posterior_variance,
        state.updates,
        state.selected_indices[completed:stop],
        state.updates[:, completed:stop],
        state.posterior_variance,
    )
    parent = restored.manifests[-1] if restored.manifests else None
    return _publish_delta(
        store,
        state,
        identity,
        start=completed,
        stop=stop,
        parent=parent,
    )


def restore_doptimal_checkpoint(
    store: ContentAddressedRunStore,
    initial: ReducedDOptimalState,
    identity: DOptimalCheckpointIdentity,
) -> ReducedDOptimalState:
    """Restore the last complete delta into the supplied empty full-capacity buffer."""
    _require_empty_initial_state(initial, identity)
    restored = _scan_deltas(
        store,
        initial,
        identity,
        update_buffer=initial.updates,
        posterior=initial.posterior_variance,
        copy_updates=True,
    )
    initial.selected_indices.extend(restored.selected_indices)
    return initial
