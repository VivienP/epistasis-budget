"""Offline contracts for the cumulative foldwise LASSO journal."""

# ruff: noqa: PLR2004

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Mapping
from dataclasses import replace
from itertools import product
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import cast

import numpy as np
import pytest

import epibudget.fourier_recovery as fourier_recovery_module
from epibudget.coeff_recovery import (
    _build_fourier_config,
    _design_matrix,
    _FourierConfig,
    _site_indices,
)
from epibudget.fourier_recovery import (
    PairwiseLassoCVState,
    PairwiseLassoFit,
    RecoveryCell,
    _fold_sha256,
    _sequence_sha256,
    coefficient_metrics,
    fit_pairwise_lasso,
    pairwise_lasso_problem_sha256,
)
from epibudget.recovery_lasso import (
    LassoCheckpointCursor,
    LassoCheckpointError,
    LassoCheckpointIdentity,
    publish_lasso_checkpoint,
    restore_lasso_checkpoint,
)
from epibudget.recovery_protocol import REGISTERED_EXECUTION_POLICY
from epibudget.recovery_state import RecoveryCellKey, RecoveryStateCursor
from epibudget.run_store import (
    ArrayRef,
    BlobRef,
    ContentAddressedRunStore,
    Manifest,
    ManifestDraft,
    RunStoreSession,
    StoreCorruptionError,
    canonical_json_bytes,
)
from epibudget.types import Variant

_RATIOS = (1.0, 0.1, 0.01)
_SITES = (0, 1, 2, 3)
_WT = ("A", "A", "A", "A")
_ALPHABET = "ACD"


def _store(path: Path) -> ContentAddressedRunStore:
    path.mkdir()
    store = ContentAddressedRunStore(path)
    store.initialise()
    return store


def _identity(
    *,
    method: str = "info",
    seed: int | None = None,
    budget: int = 48,
    scientific_sha256: str = "1" * 64,
    problem_sha256: str = "5" * 64,
) -> LassoCheckpointIdentity:
    numerical = {"numpy": np.__version__, "blas": {"name": "synthetic-offline"}}
    return LassoCheckpointIdentity(
        scientific_identity_sha256=scientific_sha256,
        execution_policy=REGISTERED_EXECUTION_POLICY,
        numerical_compatibility=numerical,
        numerical_compatibility_sha256=hashlib.sha256(canonical_json_bytes(numerical)).hexdigest(),
        selection_plan_sha256="2" * 64,
        method=method,
        seed=seed,
        budget=budget,
        selected_sha256="3" * 64,
        fold_sha256="4" * 64,
        problem_sha256=problem_sha256,
        n_folds=5,
        lambda_ratios=_RATIOS,
    )


def _state(
    completed: int, *, converged: bool = True, problem_sha256: str = "5" * 64
) -> PairwiseLassoCVState:
    return PairwiseLassoCVState(
        problem_sha256=problem_sha256,
        completed_folds=completed,
        n_folds=5,
        lambda_ratios=_RATIOS,
        cv_sse=np.array([completed, completed * 2, completed * 3], dtype=np.float64),
        converged=converged,
    )


class _CursorJournal:
    def __init__(
        self,
        store: ContentAddressedRunStore,
        *,
        key: RecoveryCellKey | None = None,
        completed_folds: int = 0,
        manifests: tuple[Manifest, ...] = (),
        identity: LassoCheckpointIdentity | None = None,
    ) -> None:
        self._session = RunStoreSession.open(store)
        self._key = key
        self._completed_folds = completed_folds
        self._active = manifests
        self._identity = identity

    @property
    def store(self) -> ContentAddressedRunStore:
        return self._session.store

    @property
    def index(self) -> SimpleNamespace:
        return SimpleNamespace(active_lasso_manifests=self._active)

    def snapshot(self) -> SimpleNamespace:
        identity = self._identity
        return SimpleNamespace(
            active_cell=self._key,
            completed_folds=self._completed_folds,
            lasso_converged=True,
            lasso_selected_sha256=(None if identity is None else identity.selected_sha256),
            lasso_fold_sha256=(None if identity is None else identity.fold_sha256),
            lasso_lambda_ratios=(() if identity is None else identity.lambda_ratios),
        )

    def lasso_view(self) -> SimpleNamespace:
        identity = self._identity
        return SimpleNamespace(
            active_cell=self._key,
            completed_folds=self._completed_folds,
            converged=True,
            selected_sha256=(None if identity is None else identity.selected_sha256),
            fold_sha256=(None if identity is None else identity.fold_sha256),
            lambda_ratios=(() if identity is None else identity.lambda_ratios),
            manifests=self._active,
        )

    def draft_manifest(
        self, *, entries: Mapping[str, BlobRef], meta: Mapping[str, object]
    ) -> ManifestDraft:
        return self._session.draft_manifest(entries=entries, meta=meta)

    def append(self, draft: ManifestDraft) -> Manifest:
        manifest = self._session.publish_manifest(draft)
        self._active = (*self._active, manifest)
        self._completed_folds = cast("int", manifest.meta["completed_folds"])
        return manifest


def _lasso_fixture() -> tuple[_FourierConfig, list[Variant], np.ndarray, np.ndarray]:
    genotypes = [
        frozenset(
            (site, _WT[index], residue)
            for index, (site, residue) in enumerate(zip(_SITES, residues, strict=True))
            if residue != _WT[index]
        )
        for residues in product(_ALPHABET, repeat=len(_SITES))
    ]
    config = _build_fourier_config(_SITES, _WT, _ALPHABET, max_order=2)
    measured = genotypes[:70]
    design = np.sqrt(len(genotypes)) * _design_matrix(config, _site_indices(config, measured))
    beta = np.zeros(design.shape[1], dtype=np.float64)
    pairwise = np.array(
        [index for index, mode in enumerate(config.modes) if np.count_nonzero(mode) == 2]
    )
    beta[pairwise[[1, 5, 13]]] = (1.5, -0.8, 0.45)
    truth = beta[pairwise]
    response = 0.2 + design @ beta
    return config, measured, response, truth


def _cell_from_fit(
    fit: PairwiseLassoFit, selected: list[Variant], truth: np.ndarray
) -> RecoveryCell:
    metrics = coefficient_metrics(fit.pairwise_coefficients, truth)
    return RecoveryCell(
        method="info",
        budget=len(selected),
        seed=None,
        spearman=metrics.spearman,
        relative_sse_gain=metrics.relative_sse_gain,
        support_size=metrics.support_size,
        coefficient_count=metrics.coefficient_count,
        selected_sha256=_sequence_sha256(selected),
        fold_sha256=_fold_sha256(selected, 5),
        lambda_ratio=fit.lambda_ratio,
        lambda_value=fit.lambda_value,
        converged=fit.converged,
    )


def test_lasso_identity_owns_the_full_immutable_numeric_and_policy_payload() -> None:
    numerical: dict[str, object] = {"numpy": {"version": "synthetic"}}
    digest = hashlib.sha256(canonical_json_bytes(numerical)).hexdigest()
    identity = replace(
        _identity(),
        numerical_compatibility=numerical,
        numerical_compatibility_sha256=digest,
    )

    numerical["numpy"] = {"version": "drifted"}

    assert identity.payload()["numerical_compatibility"] == {"numpy": {"version": "synthetic"}}
    assert identity.payload()["execution_policy"] == (
        REGISTERED_EXECUTION_POLICY.identity_payload()
    )
    with pytest.raises(TypeError):
        identity.numerical_compatibility["numpy"] = "mutated"  # type: ignore[index]


def test_fold_chain_uses_the_latest_global_parent_and_restores_requested_cell(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "run")
    identity = _identity()
    prepare = store.publish_manifest(
        entries={"prepare": store.put_json({"ready": True})},
        meta={"state_kind": "recovery_prepare"},
        parent=None,
    )

    first = publish_lasso_checkpoint(store, _state(1), identity)
    unrelated = store.publish_manifest(
        entries={"selection": store.put_json({"selected": True})},
        meta={"state_kind": "selection_plan"},
        parent=first,
    )
    second = publish_lasso_checkpoint(store, _state(2), identity)

    assert first.parent_sha256 == prepare.sha256
    assert second.parent_sha256 == unrelated.sha256
    first_identity = first.meta["identity"]
    second_identity = second.meta["identity"]
    assert isinstance(first_identity, dict)
    assert isinstance(second_identity, dict)
    assert first_identity["problem_sha256"] == identity.problem_sha256
    assert second_identity["problem_sha256"] == identity.problem_sha256
    assert restore_lasso_checkpoint(store, identity).cv_sse.tobytes() == _state(2).cv_sse.tobytes()


def test_empty_restore_keeps_the_requested_problem_identity(tmp_path: Path) -> None:
    store = _store(tmp_path / "run")
    identity = _identity(problem_sha256="8" * 64)

    restored = restore_lasso_checkpoint(store, identity)

    assert restored.completed_folds == 0
    assert restored.problem_sha256 == identity.problem_sha256


def test_other_lasso_cells_are_allowed_but_same_cell_identity_drift_fails(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "run")
    requested = _identity()
    other = _identity(method="random", seed=3)
    publish_lasso_checkpoint(store, _state(1), other)
    publish_lasso_checkpoint(store, _state(1), requested)

    restored = restore_lasso_checkpoint(store, requested)

    assert restored.completed_folds == 1
    with pytest.raises(LassoCheckpointError, match="identity"):
        restore_lasso_checkpoint(store, _identity(scientific_sha256="9" * 64))


def test_same_state_is_idempotent_and_divergent_state_fails_closed(tmp_path: Path) -> None:
    store = _store(tmp_path / "run")
    identity = _identity()
    first = publish_lasso_checkpoint(store, _state(1), identity)

    assert publish_lasso_checkpoint(store, _state(1), identity) == first
    divergent = PairwiseLassoCVState(
        problem_sha256=identity.problem_sha256,
        completed_folds=1,
        n_folds=5,
        lambda_ratios=_RATIOS,
        cv_sse=np.array([1.0, 2.0, 4.0], dtype=np.float64),
        converged=True,
    )
    with pytest.raises(LassoCheckpointError, match="diverg"):
        publish_lasso_checkpoint(store, divergent, identity)


def test_restore_uses_last_complete_publication_without_mutating_the_store(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "run")
    identity = _identity()
    complete = publish_lasso_checkpoint(store, _state(1), identity)
    second = publish_lasso_checkpoint(store, _state(2), identity)
    payload = next(
        path
        for path in (tmp_path / "run" / "manifests").glob("*.manifest.json")
        if second.sha256 in path.name
    )
    payload.with_name(f"{payload.name}.complete").unlink()
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    restored = restore_lasso_checkpoint(store, identity)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    assert restored.completed_folds == 1
    assert store.latest_manifest() == complete
    assert before == after


def _publish_raw(
    store: ContentAddressedRunStore,
    identity: LassoCheckpointIdentity,
    *,
    completed: int,
    state: PairwiseLassoCVState | None = None,
    parent: Manifest | None = None,
    meta_changes: dict[str, object] | None = None,
    array: np.ndarray | None = None,
) -> Manifest:
    current = _state(completed) if state is None else state
    values = current.cv_sse if array is None else array
    reference = store.put_array(values)
    meta: dict[str, object] = {
        "schema_version": "epibudget-pairwise-lasso-fold-v1",
        "state_kind": "pairwise_lasso_cv",
        "cell": identity.cell_payload(),
        "identity": identity.payload(),
        "completed_folds": completed,
        "converged": current.converged,
        "cv_sse": reference.payload(),
    }
    if meta_changes:
        meta.update(meta_changes)
    return store.publish_manifest(entries={"cv_sse": reference.blob}, meta=meta, parent=parent)


@pytest.mark.parametrize(
    "damage",
    ["schema", "gap", "duplicate", "entries", "dtype", "shape", "nan", "negative"],
)
def test_restore_rejects_malformed_fold_records(tmp_path: Path, damage: str) -> None:
    store = _store(tmp_path / "run")
    identity = _identity()
    parent: Manifest | None = None
    completed = 1
    meta_changes: dict[str, object] = {}
    array: np.ndarray | None = None
    if damage in {"gap", "duplicate"}:
        parent = _publish_raw(store, identity, completed=1)
        completed = 3 if damage == "gap" else 1
    elif damage == "schema":
        meta_changes["schema_version"] = "wrong"
    elif damage == "entries":
        meta_changes["cv_sse"] = {"wrong": True}
    elif damage == "dtype":
        array = np.ones(3, dtype=np.float32)
    elif damage == "shape":
        array = np.ones(2, dtype=np.float64)
    elif damage == "nan":
        array = np.array([1.0, np.nan, 3.0], dtype=np.float64)
    elif damage == "negative":
        array = np.array([1.0, -1.0, 3.0], dtype=np.float64)
    manifest = _publish_raw(
        store,
        identity,
        completed=completed,
        parent=parent,
        meta_changes=meta_changes,
        array=array,
    )
    if damage == "entries":
        assert manifest.entries["cv_sse"]

    with pytest.raises((LassoCheckpointError, StoreCorruptionError)):
        restore_lasso_checkpoint(store, identity)


def test_restore_detects_content_address_tampering(tmp_path: Path) -> None:
    store = _store(tmp_path / "run")
    identity = _identity()
    manifest = publish_lasso_checkpoint(store, _state(1), identity)
    digest = manifest.entries["cv_sse"].sha256
    blob = tmp_path / "run" / "blobs" / digest[:2] / f"{digest}.blob"
    blob.write_bytes(b"tampered")

    with pytest.raises(StoreCorruptionError):
        restore_lasso_checkpoint(store, identity)


def test_identity_rejects_non_fold_policy_and_malformed_fields() -> None:
    with pytest.raises(ValueError, match="fold"):
        replace(
            _identity(),
            execution_policy=replace(REGISTERED_EXECUTION_POLICY, lasso_checkpoint_unit="cell"),
        )
    with pytest.raises(ValueError, match="lambda ratios"):
        replace(_identity(), lambda_ratios=(1.0, float("nan")))
    with pytest.raises(ValueError, match="seed"):
        replace(_identity(), seed=True)
    with pytest.raises(ValueError, match="problem SHA"):
        replace(_identity(), problem_sha256="wrong")


def test_journal_rejects_state_exchange_between_cells_and_response_problems(
    tmp_path: Path,
) -> None:
    config, selected, response, _truth = _lasso_fixture()
    ratios = tuple(float(value) for value in np.geomspace(1.0, 1e-3, 20))
    first_problem = pairwise_lasso_problem_sha256(
        config, selected, response, n_folds=5, lambda_ratios=ratios
    )
    changed_response = response.copy()
    changed_response[0] = np.nextafter(changed_response[0], np.inf)
    second_problem = pairwise_lasso_problem_sha256(
        config, selected, changed_response, n_folds=5, lambda_ratios=ratios
    )
    first_identity = replace(
        _identity(budget=len(selected), problem_sha256=first_problem),
        method="info",
        lambda_ratios=ratios,
    )
    second_identity = replace(
        _identity(budget=len(selected), problem_sha256=second_problem),
        method="random",
        seed=0,
        lambda_ratios=ratios,
    )
    store = _store(tmp_path / "run")
    first_state = PairwiseLassoCVState(
        problem_sha256=first_problem,
        completed_folds=1,
        n_folds=5,
        lambda_ratios=ratios,
        cv_sse=np.arange(len(ratios), dtype=np.float64),
        converged=True,
    )

    publish_lasso_checkpoint(store, first_state, first_identity)

    with pytest.raises(LassoCheckpointError, match="problem"):
        publish_lasso_checkpoint(store, first_state, second_identity)
    with pytest.raises(LassoCheckpointError, match="identity"):
        restore_lasso_checkpoint(
            store,
            replace(first_identity, problem_sha256=second_problem),
        )


@pytest.mark.parametrize("crash_after_fold", [1, 2, 3, 4, 5])
def test_crash_after_each_fold_resumes_to_the_exact_recovery_cell(
    tmp_path: Path, crash_after_fold: int
) -> None:
    config, selected, response, truth = _lasso_fixture()
    oracle_fit = fit_pairwise_lasso(config, selected, response, n_folds=5)
    oracle = _cell_from_fit(oracle_fit, selected, truth)
    identity = replace(
        _identity(
            budget=len(selected),
            problem_sha256=pairwise_lasso_problem_sha256(
                config,
                selected,
                response,
                n_folds=5,
                lambda_ratios=tuple(float(value) for value in np.geomspace(1.0, 1e-3, 20)),
            ),
        ),
        selected_sha256=_sequence_sha256(selected),
        fold_sha256=_fold_sha256(selected, 5),
        lambda_ratios=tuple(float(value) for value in np.geomspace(1.0, 1e-3, 20)),
    )
    store = _store(tmp_path / "run")

    def checkpoint_then_crash(state: PairwiseLassoCVState) -> None:
        publish_lasso_checkpoint(store, state, identity)
        if state.completed_folds == crash_after_fold:
            raise RuntimeError("simulated fold crash")

    with pytest.raises(RuntimeError, match="simulated fold crash"):
        fit_pairwise_lasso(
            config,
            selected,
            response,
            n_folds=5,
            on_fold_completed=checkpoint_then_crash,
        )

    restored = restore_lasso_checkpoint(store, identity)
    resumed_fit = fit_pairwise_lasso(config, selected, response, n_folds=5, resume_cv=restored)

    assert _cell_from_fit(resumed_fit, selected, truth) == oracle


def test_crash_during_refit_resumes_from_the_completed_cv_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, selected, response, truth = _lasso_fixture()
    original_solver = fourier_recovery_module._fista_lasso_path_with_status
    oracle_fit = fit_pairwise_lasso(config, selected, response, n_folds=5)
    oracle = _cell_from_fit(oracle_fit, selected, truth)
    ratios = tuple(float(value) for value in np.geomspace(1.0, 1e-3, 20))
    identity = replace(
        _identity(
            budget=len(selected),
            problem_sha256=pairwise_lasso_problem_sha256(
                config, selected, response, n_folds=5, lambda_ratios=ratios
            ),
        ),
        selected_sha256=_sequence_sha256(selected),
        fold_sha256=_fold_sha256(selected, 5),
        lambda_ratios=ratios,
    )
    store = _store(tmp_path / "run")
    solves = 0

    def crash_on_refit(
        design: np.ndarray, target: np.ndarray, path: list[float]
    ) -> tuple[list[np.ndarray], bool]:
        nonlocal solves
        solves += 1
        if solves == 6:
            raise RuntimeError("simulated refit crash")
        return original_solver(design, target, path)

    monkeypatch.setattr(fourier_recovery_module, "_fista_lasso_path_with_status", crash_on_refit)
    with pytest.raises(RuntimeError, match="simulated refit crash"):
        fit_pairwise_lasso(
            config,
            selected,
            response,
            n_folds=5,
            on_fold_completed=lambda state: publish_lasso_checkpoint(store, state, identity),
        )

    restored = restore_lasso_checkpoint(store, identity)
    assert restored.completed_folds == 5
    resumed_fit = fit_pairwise_lasso(config, selected, response, n_folds=5, resume_cv=restored)

    assert _cell_from_fit(resumed_fit, selected, truth) == oracle


def test_checkpoint_api_has_no_label_or_design_inputs() -> None:
    identity = _identity()

    assert isinstance(identity.numerical_compatibility, MappingProxyType)
    for function in (publish_lasso_checkpoint, restore_lasso_checkpoint):
        parameters = inspect.signature(function).parameters
        assert "landscape" not in parameters
        assert "fitness" not in parameters
        assert "design" not in parameters


def test_lasso_cursor_publishes_without_rescanning_and_restores_only_latest_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path / "cursor")
    identity = _identity()
    key = RecoveryCellKey(identity.method, identity.seed, identity.budget)
    writer = _CursorJournal(store)
    cursor = LassoCheckpointCursor.restore(cast("RecoveryStateCursor", writer), identity)
    first = cursor.publish(_state(1))
    second = cursor.publish(_state(2))
    reader = _CursorJournal(
        store,
        key=key,
        completed_folds=2,
        manifests=(first, second),
        identity=identity,
    )
    reads = 0
    original_get_array = store.get_array

    def counted_get_array(reference: ArrayRef) -> np.ndarray:
        nonlocal reads
        reads += 1
        return original_get_array(reference)

    monkeypatch.setattr(store, "get_array", counted_get_array)
    monkeypatch.setattr(
        store,
        "manifest_chain",
        lambda: (_ for _ in ()).throw(AssertionError("full chain rescan")),
    )
    monkeypatch.setattr(
        reader,
        "snapshot",
        lambda: (_ for _ in ()).throw(AssertionError("full state snapshot")),
    )

    restored = LassoCheckpointCursor.restore(cast("RecoveryStateCursor", reader), identity)

    assert restored.state.completed_folds == 2
    assert restored.state.cv_sse.tobytes() == _state(2).cv_sse.tobytes()
    assert reads == 1


def test_lasso_cursor_exact_retry_is_idempotent_and_divergence_is_rejected(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "cursor")
    identity = _identity()
    journal = _CursorJournal(store)
    cursor = LassoCheckpointCursor.restore(cast("RecoveryStateCursor", journal), identity)
    first = cursor.publish(_state(1))

    assert cursor.publish(_state(1)) == first
    divergent = PairwiseLassoCVState(
        problem_sha256=identity.problem_sha256,
        completed_folds=1,
        n_folds=identity.n_folds,
        lambda_ratios=identity.lambda_ratios,
        cv_sse=np.array([9.0, 9.0, 9.0], dtype=np.float64),
        converged=True,
    )
    with pytest.raises(LassoCheckpointError, match="diverged"):
        cursor.publish(divergent)


def test_lasso_cursor_rejects_a_stale_same_process_writer_without_a_rescan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path / "cursor")
    identity = _identity()
    first_journal = _CursorJournal(store)
    stale_journal = _CursorJournal(store)
    first = LassoCheckpointCursor.restore(cast("RecoveryStateCursor", first_journal), identity)
    stale = LassoCheckpointCursor.restore(cast("RecoveryStateCursor", stale_journal), identity)
    monkeypatch.setattr(
        store,
        "manifest_chain",
        lambda: (_ for _ in ()).throw(AssertionError("full chain rescan")),
    )
    first.publish(_state(1))
    divergent = PairwiseLassoCVState(
        problem_sha256=identity.problem_sha256,
        completed_folds=1,
        n_folds=identity.n_folds,
        lambda_ratios=identity.lambda_ratios,
        cv_sse=np.array([7.0, 8.0, 9.0], dtype=np.float64),
        converged=True,
    )

    with pytest.raises(LassoCheckpointError, match="diverged"):
        stale.publish(divergent)
