"""Durable cumulative-fold journal for pairwise LASSO cross-validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType
from typing import cast

import numpy as np

from epibudget.fourier_recovery import PairwiseLassoCVState
from epibudget.recovery_protocol import RecoveryExecutionPolicy
from epibudget.recovery_state import RecoveryCellKey, RecoveryStateCursor
from epibudget.run_store import (
    ArrayRef,
    ContentAddressedRunStore,
    Manifest,
    RunStoreError,
    StoreCorruptionError,
    StoreDivergenceError,
    canonical_json_bytes,
)

_SCHEMA = "epibudget-pairwise-lasso-fold-v1"
_STATE_KIND = "pairwise_lasso_cv"
_SHA256_LENGTH = 64
_ENTRY_NAME = "cv_sse"
_MIN_FOLDS = 2
_META_NAMES = frozenset(
    {
        "schema_version",
        "state_kind",
        "cell",
        "identity",
        "completed_folds",
        "converged",
        "cv_sse",
    }
)


class LassoCheckpointError(RunStoreError):
    """A LASSO checkpoint cannot be published or restored safely."""


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


@dataclass(frozen=True)
class LassoCheckpointIdentity:
    """Scientific, execution, numeric, selection, and cell identity for one fit."""

    scientific_identity_sha256: str
    execution_policy: RecoveryExecutionPolicy
    numerical_compatibility: Mapping[str, object]
    numerical_compatibility_sha256: str
    selection_plan_sha256: str
    method: str
    seed: int | None
    budget: int
    selected_sha256: str
    fold_sha256: str
    problem_sha256: str
    n_folds: int
    lambda_ratios: tuple[float, ...]

    def __post_init__(self) -> None:
        """Reject identities that do not bind every resume-sensitive field exactly."""
        digests = (
            self.scientific_identity_sha256,
            self.numerical_compatibility_sha256,
            self.selection_plan_sha256,
            self.selected_sha256,
            self.fold_sha256,
            self.problem_sha256,
        )
        if not all(_is_sha256(value) for value in digests):
            if not _is_sha256(self.problem_sha256):
                raise ValueError("LASSO checkpoint problem SHA is invalid")
            raise ValueError("LASSO checkpoint identity contains an invalid SHA-256")
        if self.execution_policy.lasso_checkpoint_unit != "fold":
            raise ValueError("LASSO checkpoint execution policy must use the fold unit")
        if type(self.method) is not str or not self.method:
            raise ValueError("LASSO checkpoint method must be a non-empty string")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("LASSO checkpoint seed must be an integer or None")
        if type(self.budget) is not int or self.budget < 1:
            raise ValueError("LASSO checkpoint budget must be positive")
        if type(self.n_folds) is not int or self.n_folds < _MIN_FOLDS:
            raise ValueError("LASSO checkpoint fold count must be at least 2")
        if type(self.lambda_ratios) is not tuple or not self.lambda_ratios:
            raise ValueError("LASSO checkpoint lambda ratios must be a non-empty tuple")
        if any(
            type(value) is not float or not np.isfinite(value) or not 0.0 < value <= 1.0
            for value in self.lambda_ratios
        ) or any(first < second for first, second in pairwise(self.lambda_ratios)):
            raise ValueError("LASSO checkpoint lambda ratios are invalid")
        try:
            numerical_bytes = canonical_json_bytes(_thaw_json(self.numerical_compatibility))
        except (TypeError, ValueError) as error:
            raise ValueError("LASSO numerical compatibility payload is invalid") from error
        if hashlib.sha256(numerical_bytes).hexdigest() != self.numerical_compatibility_sha256:
            raise ValueError("LASSO numerical compatibility digest does not match its payload")
        policy_rendered = json.dumps(
            self.execution_policy.policy_payload(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if hashlib.sha256(policy_rendered.encode("utf-8")).hexdigest() != (
            self.execution_policy.policy_sha256
        ):
            raise ValueError("LASSO execution policy digest does not match its payload")
        frozen = _freeze_json(json.loads(numerical_bytes))
        object.__setattr__(self, "numerical_compatibility", cast("Mapping[str, object]", frozen))

    def cell_payload(self) -> dict[str, object]:
        """Return the method-seed-budget address used to distinguish interleaved cells."""
        return {"method": self.method, "seed": self.seed, "budget": self.budget}

    def payload(self) -> dict[str, object]:
        """Return the complete canonical identity embedded in every fold record."""
        return {
            "scientific_identity_sha256": self.scientific_identity_sha256,
            "execution_policy": self.execution_policy.identity_payload(),
            "numerical_compatibility": _thaw_json(self.numerical_compatibility),
            "numerical_compatibility_sha256": self.numerical_compatibility_sha256,
            "selection_plan_sha256": self.selection_plan_sha256,
            "cell": self.cell_payload(),
            "selected_sha256": self.selected_sha256,
            "fold_sha256": self.fold_sha256,
            "problem_sha256": self.problem_sha256,
            "n_folds": self.n_folds,
            "lambda_ratios": list(self.lambda_ratios),
        }


@dataclass(frozen=True)
class _LassoJournalProgress:
    state: PairwiseLassoCVState
    manifest: Manifest | None
    manifests: tuple[Manifest, ...]


class LassoCheckpointCursor:
    """Incremental fold checkpoint cursor over one verified recovery journal."""

    def __init__(
        self,
        journal: RecoveryStateCursor,
        identity: LassoCheckpointIdentity,
        state: PairwiseLassoCVState,
        manifest: Manifest | None,
    ) -> None:
        self._journal = journal
        self._identity = identity
        self._state = state
        self._manifest = manifest

    @classmethod
    def restore(
        cls,
        journal: RecoveryStateCursor,
        identity: LassoCheckpointIdentity,
    ) -> LassoCheckpointCursor:
        """Restore only the active cell's latest cumulative fold."""
        view = journal.lasso_view()
        key = RecoveryCellKey(identity.method, identity.seed, identity.budget)
        manifests = view.manifests
        if view.active_cell is not None and view.active_cell != key:
            raise LassoCheckpointError("active LASSO cell does not match the requested identity")
        if view.completed_folds == 0:
            if manifests:
                raise LassoCheckpointError("LASSO journal index disagrees with its state")
            return cls(journal, identity, _empty_state(identity), None)
        if view.active_cell != key or not manifests:
            raise LassoCheckpointError("LASSO journal has no checkpoint for the active cell")
        manifest = manifests[-1]
        if manifest.meta.get("state_kind") != _STATE_KIND:
            raise LassoCheckpointError("LASSO journal index contains the wrong state kind")
        observed_identity = _require_mapping(manifest.meta.get("identity"), "identity")
        if canonical_json_bytes(observed_identity) != canonical_json_bytes(identity.payload()):
            raise LassoCheckpointError("LASSO checkpoint identity does not match")
        completed = manifest.meta.get("completed_folds")
        converged = manifest.meta.get("converged")
        if (
            completed != view.completed_folds
            or type(converged) is not bool
            or converged is not view.converged
            or view.selected_sha256 != identity.selected_sha256
            or view.fold_sha256 != identity.fold_sha256
            or view.lambda_ratios != identity.lambda_ratios
        ):
            raise LassoCheckpointError("LASSO journal index disagrees with its state")
        cv_sse = _load_cv_sse(journal.store, manifest, manifest.meta.get("cv_sse"), identity)
        state = PairwiseLassoCVState(
            problem_sha256=identity.problem_sha256,
            completed_folds=view.completed_folds,
            n_folds=identity.n_folds,
            lambda_ratios=identity.lambda_ratios,
            cv_sse=cv_sse,
            converged=converged,
        )
        return cls(journal, identity, state, manifest)

    @property
    def state(self) -> PairwiseLassoCVState:
        """Return the latest durable cumulative fold state."""
        return self._state

    def publish(self, state: PairwiseLassoCVState) -> Manifest:
        """Publish exactly one new cumulative fold, or recover an exact retry."""
        _require_state_identity(state, self._identity)
        if state.completed_folds == self._state.completed_folds:
            if self._manifest is not None and _same_state(state, self._state):
                return self._manifest
            raise LassoCheckpointError("LASSO checkpoint publication diverged from stored state")
        if state.completed_folds != self._state.completed_folds + 1:
            raise LassoCheckpointError(
                "LASSO checkpoint publication requires exactly one newly completed fold"
            )
        if np.any(state.cv_sse < self._state.cv_sse) or (
            not self._state.converged and state.converged
        ):
            raise LassoCheckpointError("LASSO checkpoint cumulative state diverged")
        reference = self._journal.store.put_array(state.cv_sse)
        draft = self._journal.draft_manifest(
            entries={_ENTRY_NAME: reference.blob},
            meta={
                "schema_version": _SCHEMA,
                "state_kind": _STATE_KIND,
                "cell": self._identity.cell_payload(),
                "identity": self._identity.payload(),
                "completed_folds": state.completed_folds,
                "converged": state.converged,
                "cv_sse": reference.payload(),
            },
        )
        try:
            manifest = self._journal.append(draft)
        except StoreDivergenceError as error:
            raise LassoCheckpointError("LASSO checkpoint publication diverged") from error
        except StoreCorruptionError:
            raise
        except RunStoreError as error:
            message = str(error).lower()
            if "diverg" in message or "stale" in message or "generation" in message:
                raise LassoCheckpointError("LASSO checkpoint publication diverged") from error
            raise LassoCheckpointError("LASSO checkpoint publication failed") from error
        self._state = state
        self._manifest = manifest
        return manifest


def _empty_state(identity: LassoCheckpointIdentity) -> PairwiseLassoCVState:
    return PairwiseLassoCVState(
        problem_sha256=identity.problem_sha256,
        completed_folds=0,
        n_folds=identity.n_folds,
        lambda_ratios=identity.lambda_ratios,
        cv_sse=np.zeros(len(identity.lambda_ratios), dtype=np.float64),
        converged=True,
    )


def _same_state(first: PairwiseLassoCVState, second: PairwiseLassoCVState) -> bool:
    return (
        first.problem_sha256 == second.problem_sha256
        and first.completed_folds == second.completed_folds
        and first.n_folds == second.n_folds
        and first.lambda_ratios == second.lambda_ratios
        and first.converged is second.converged
        and first.cv_sse.tobytes(order="C") == second.cv_sse.tobytes(order="C")
    )


def _require_mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise LassoCheckpointError(f"LASSO checkpoint {field} must be an object")
    return cast("dict[str, object]", value)


def _require_state_identity(state: PairwiseLassoCVState, identity: LassoCheckpointIdentity) -> None:
    if state.problem_sha256 != identity.problem_sha256:
        raise LassoCheckpointError("LASSO state problem SHA does not match its identity")
    if state.n_folds != identity.n_folds:
        raise LassoCheckpointError("LASSO state fold count does not match its identity")
    if state.lambda_ratios != identity.lambda_ratios:
        raise LassoCheckpointError("LASSO state lambda ratios do not match its identity")


def _load_cv_sse(
    store: ContentAddressedRunStore,
    manifest: Manifest,
    payload: object,
    identity: LassoCheckpointIdentity,
) -> np.ndarray:
    try:
        reference = ArrayRef.from_payload(payload)
        entry = manifest.entry(_ENTRY_NAME)
    except (KeyError, RunStoreError, TypeError, ValueError) as error:
        raise LassoCheckpointError("LASSO checkpoint has an invalid cv_sse ArrayRef") from error
    if reference.blob != entry:
        raise LassoCheckpointError(
            "LASSO checkpoint cv_sse ArrayRef does not match its manifest entry"
        )
    if (
        reference.dtype != "float64"
        or reference.byteorder != "little"
        or reference.order != "C"
        or reference.shape != (len(identity.lambda_ratios),)
    ):
        raise LassoCheckpointError("LASSO checkpoint cv_sse has an incompatible binary layout")
    try:
        restored = store.get_array(reference)
    except StoreCorruptionError:
        raise
    except (RunStoreError, TypeError, ValueError) as error:
        raise LassoCheckpointError("LASSO checkpoint cv_sse cannot be verified") from error
    array = np.asarray(restored)
    if array.dtype != np.dtype(np.float64):
        raise LassoCheckpointError("LASSO checkpoint cv_sse must use float64")
    if not np.all(np.isfinite(array)):
        raise LassoCheckpointError("LASSO checkpoint cv_sse must be finite")
    if np.any(array < 0.0):
        raise LassoCheckpointError("LASSO checkpoint cv_sse must be nonnegative")
    return array


def _scan_journal(  # noqa: PLR0912
    store: ContentAddressedRunStore, identity: LassoCheckpointIdentity
) -> _LassoJournalProgress:
    try:
        manifests = store.manifest_chain()
    except StoreCorruptionError:
        raise
    except RunStoreError as error:
        raise LassoCheckpointError("LASSO manifest chain cannot be verified") from error
    expected_cell = canonical_json_bytes(identity.cell_payload())
    expected_identity = canonical_json_bytes(identity.payload())
    state = PairwiseLassoCVState(
        problem_sha256=identity.problem_sha256,
        completed_folds=0,
        n_folds=identity.n_folds,
        lambda_ratios=identity.lambda_ratios,
        cv_sse=np.zeros(len(identity.lambda_ratios), dtype=np.float64),
        converged=True,
    )
    latest: Manifest | None = None
    for manifest in manifests:
        if manifest.meta.get("state_kind") != _STATE_KIND:
            continue
        if set(manifest.meta) != _META_NAMES:
            raise LassoCheckpointError("LASSO checkpoint metadata fields do not match")
        if manifest.meta.get("schema_version") != _SCHEMA:
            raise LassoCheckpointError("LASSO checkpoint schema does not match")
        cell = _require_mapping(manifest.meta.get("cell"), "cell")
        if canonical_json_bytes(cell) != expected_cell:
            continue
        observed_identity = _require_mapping(manifest.meta.get("identity"), "identity")
        if canonical_json_bytes(observed_identity) != expected_identity:
            raise LassoCheckpointError("LASSO checkpoint identity does not match")
        completed = manifest.meta.get("completed_folds")
        if type(completed) is not int or completed != state.completed_folds + 1:
            raise LassoCheckpointError("LASSO checkpoint fold chain has a gap or duplicate")
        converged = manifest.meta.get("converged")
        if type(converged) is not bool:
            raise LassoCheckpointError("LASSO checkpoint converged flag is invalid")
        if not state.converged and converged:
            raise LassoCheckpointError("LASSO checkpoint convergence is not cumulative")
        if set(manifest.entries) != {_ENTRY_NAME}:
            raise LassoCheckpointError("LASSO checkpoint entries do not match")
        cv_sse = _load_cv_sse(store, manifest, manifest.meta.get("cv_sse"), identity)
        if np.any(cv_sse < state.cv_sse):
            raise LassoCheckpointError("LASSO checkpoint cv_sse is not cumulative")
        state = PairwiseLassoCVState(
            problem_sha256=identity.problem_sha256,
            completed_folds=completed,
            n_folds=identity.n_folds,
            lambda_ratios=identity.lambda_ratios,
            cv_sse=cv_sse,
            converged=converged,
        )
        latest = manifest
    return _LassoJournalProgress(state=state, manifest=latest, manifests=manifests)


def publish_lasso_checkpoint(
    store: ContentAddressedRunStore,
    state: PairwiseLassoCVState,
    identity: LassoCheckpointIdentity,
) -> Manifest:
    """Publish one newly completed cumulative fold against the latest global parent."""
    _require_state_identity(state, identity)
    if state.completed_folds == 0:
        raise LassoCheckpointError("LASSO checkpoint publication requires a completed fold")
    progress = _scan_journal(store, identity)
    restored = progress.state
    if state.completed_folds == restored.completed_folds:
        if (
            progress.manifest is not None
            and state.converged is restored.converged
            and state.cv_sse.tobytes(order="C") == restored.cv_sse.tobytes(order="C")
        ):
            return progress.manifest
        raise LassoCheckpointError("LASSO checkpoint publication diverged from stored state")
    if state.completed_folds != restored.completed_folds + 1:
        raise LassoCheckpointError(
            "LASSO checkpoint publication requires exactly one newly completed fold"
        )
    if np.any(state.cv_sse < restored.cv_sse) or (not restored.converged and state.converged):
        raise LassoCheckpointError("LASSO checkpoint cumulative state diverged")
    reference = store.put_array(state.cv_sse)
    meta = {
        "schema_version": _SCHEMA,
        "state_kind": _STATE_KIND,
        "cell": identity.cell_payload(),
        "identity": identity.payload(),
        "completed_folds": state.completed_folds,
        "converged": state.converged,
        "cv_sse": reference.payload(),
    }
    parent = progress.manifests[-1] if progress.manifests else None
    try:
        return store.publish_manifest(
            entries={_ENTRY_NAME: reference.blob}, meta=meta, parent=parent
        )
    except StoreDivergenceError as error:
        raise LassoCheckpointError("LASSO checkpoint publication diverged") from error
    except StoreCorruptionError:
        raise
    except RunStoreError as error:
        message = str(error).lower()
        if "diverg" in message or "stale or forged" in message:
            raise LassoCheckpointError("LASSO checkpoint publication diverged") from error
        raise LassoCheckpointError("LASSO checkpoint publication failed") from error


def restore_lasso_checkpoint(
    store: ContentAddressedRunStore, identity: LassoCheckpointIdentity
) -> PairwiseLassoCVState:
    """Restore the last verified contiguous cumulative fold for the requested cell."""
    return _scan_journal(store, identity).state
