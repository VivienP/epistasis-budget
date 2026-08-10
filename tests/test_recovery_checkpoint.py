from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest

from epibudget.fourier_recovery import (
    RecoveryCell,
    SelectionPlan,
    SelectionSequence,
    _fold_sha256,
    _sequence_sha256,
    decide_recovery,
)
from epibudget.recovery_checkpoint import (
    REGISTERED_BUDGET_BLOCKS,
    RecoveryBlockWork,
    RecoveryIdentity,
    assemble_recovery_cells,
    canonical_json_bytes,
    discover_checkpoints,
    execute_pending_blocks,
    load_recovery_blocks,
    load_selection_plan,
    numerical_fingerprint,
    pending_recovery_blocks,
    publish_checkpoint,
    recovery_block_key,
    recovery_block_payload,
    selection_plan_payload,
)
from epibudget.scored_cache import candidate_sha256
from epibudget.types import Variant


def _variant(*mutations: tuple[int, str, str]) -> Variant:
    return frozenset(mutations)


def _selection_fixture() -> tuple[SelectionPlan, tuple[Variant, ...], RecoveryIdentity]:
    candidates = (
        _variant((0, "A", "C")),
        _variant((1, "A", "D")),
        _variant((0, "A", "C"), (1, "A", "D")),
        _variant((2, "A", "E")),
    )
    selected = candidates[:3]
    plan = SelectionPlan(
        budgets=(2, 3),
        sequences=(
            SelectionSequence(
                method="info",
                seed=None,
                selected=selected,
                selected_sha256=_sequence_sha256(selected),
                tie_break_version="test-v1",
            ),
        ),
    )
    identity = RecoveryIdentity(
        execution_commit="a" * 40,
        candidate_sha256=candidate_sha256(candidates),
        input_hashes={"cache_sha256": "c" * 64},
        cache_identity={"model_id": "test-model", "n_perturbations": 16},
        numerical_fingerprint={"python": "3.12", "numpy": "2.0"},
        protocol={
            "budgets": [2, 3],
            "methods": ["info"],
            "seeds": [],
            "sequence_keys": [{"method": "info", "seed": None}],
            "folds": 5,
            "coefficient_count": 2166,
        },
    )
    return plan, candidates, identity


def test_canonical_json_bytes_is_stable_and_rejects_nonfinite_values() -> None:
    first = {"z": [3, None], "a": {"value": 1.25}}
    second = {"a": {"value": 1.25}, "z": [3, None]}

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert canonical_json_bytes(first) == b'{"a":{"value":1.25},"z":[3,null]}'

    with pytest.raises(ValueError, match="JSON"):
        canonical_json_bytes({"bad": float("nan")})


def test_numerical_fingerprint_records_runtime_and_blas_identity() -> None:
    fingerprint = numerical_fingerprint()

    assert fingerprint == numerical_fingerprint()
    assert set(fingerprint) == {
        "blas_config_sha256",
        "blas_runtime",
        "environment_threads",
        "machine",
        "numpy",
        "operating_system",
        "python",
        "scipy",
    }
    environment_threads = fingerprint["environment_threads"]
    assert isinstance(environment_threads, dict)
    assert set(environment_threads) == {
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    }


def test_checkpoint_publication_requires_a_valid_completion_marker(tmp_path: Path) -> None:
    payload = {"schema_version": "test-v1", "records": [1, 2, 3]}

    digest = publish_checkpoint(tmp_path, "block", "info-none-0", payload)

    assert digest == hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    assert discover_checkpoints(tmp_path, "block") == {"info-none-0": payload}


def test_unmarked_or_malformed_checkpoint_upload_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "block--unfinished--deadbeef.payload.json").write_text(
        '{"schema_version":"test-v1"}', encoding="utf-8"
    )
    (tmp_path / "block--unfinished--deadbeef.complete.json").write_text(
        "not-json", encoding="utf-8"
    )

    assert discover_checkpoints(tmp_path, "block") == {}


def test_interrupted_copy_is_replaced_automatically_on_republication(tmp_path: Path) -> None:
    kind = "block"
    key = "info-none-0"
    payload = {"records": [1, 2, 3]}
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    key_token = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    prefix = f"{kind}--{key_token}--{digest}"
    (tmp_path / f"{prefix}.payload.json").write_bytes(b'{"records":[')
    (tmp_path / f"{prefix}.complete.json").write_bytes(b'{"schema_version":')

    assert publish_checkpoint(tmp_path, kind, key, payload) == digest
    assert discover_checkpoints(tmp_path, kind) == {key: payload}


def test_marked_checkpoint_with_altered_payload_fails_closed(tmp_path: Path) -> None:
    publish_checkpoint(tmp_path, "block", "info-none-0", {"records": [1, 2, 3]})
    payload_path = next(tmp_path.glob("*.payload.json"))
    payload_path.write_text(json.dumps({"records": [9]}), encoding="utf-8")

    with pytest.raises(ValueError, match="digest"):
        discover_checkpoints(tmp_path, "block")


def test_checkpoint_marker_filename_must_match_its_declared_content(tmp_path: Path) -> None:
    publish_checkpoint(tmp_path, "block", "info-none-0", {"records": [1, 2, 3]})
    marker_path = next(tmp_path.glob("*.complete.json"))
    marker_path.rename(tmp_path / f"block--{'0' * 16}--{'f' * 64}.complete.json")

    with pytest.raises(ValueError, match="filename"):
        discover_checkpoints(tmp_path, "block")


def test_selection_plan_checkpoint_round_trips_without_a_label_input() -> None:
    plan, candidates, identity = _selection_fixture()

    payload = selection_plan_payload(plan, identity)
    loaded = load_selection_plan(payload, expected=identity, expected_candidates=candidates)

    assert loaded == plan
    assert "landscape" not in inspect.signature(selection_plan_payload).parameters
    assert "landscape" not in inspect.signature(load_selection_plan).parameters


def test_selection_plan_checkpoint_rejects_changed_identity_or_candidates() -> None:
    plan, candidates, identity = _selection_fixture()
    payload = selection_plan_payload(plan, identity)

    with pytest.raises(ValueError, match="identity"):
        load_selection_plan(
            payload,
            expected=replace(identity, execution_commit="d" * 40),
            expected_candidates=candidates,
        )

    with pytest.raises(ValueError, match="candidate"):
        load_selection_plan(payload, expected=identity, expected_candidates=candidates[:-1])


def test_selection_plan_checkpoint_rejects_tampering_and_duplicate_sequences() -> None:
    plan, candidates, identity = _selection_fixture()
    payload = selection_plan_payload(plan, identity)
    plan_body = payload["plan"]
    assert isinstance(plan_body, dict)
    sequences = plan_body["sequences"]
    assert isinstance(sequences, list)
    sequences.append(dict(sequences[0]))

    with pytest.raises(ValueError, match=r"hash|duplicate"):
        load_selection_plan(payload, expected=identity, expected_candidates=candidates)


def test_selection_plan_checkpoint_enforces_registered_budget_and_sequence_order() -> None:
    plan, candidates, identity = _selection_fixture()
    payload = selection_plan_payload(plan, identity)
    body = payload["plan"]
    assert isinstance(body, dict)
    sequences = body["sequences"]
    assert isinstance(sequences, list)
    sequence = sequences[0]
    assert isinstance(sequence, dict)
    sequence["method"] = "unexpected"
    payload["selection_plan_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()

    with pytest.raises(ValueError, match=r"registered|order"):
        load_selection_plan(payload, expected=identity, expected_candidates=candidates)

    payload = selection_plan_payload(plan, identity)
    body = payload["plan"]
    assert isinstance(body, dict)
    body["budgets"] = [2, 4]
    payload["selection_plan_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    with pytest.raises(ValueError, match="registered budgets"):
        load_selection_plan(payload, expected=identity, expected_candidates=candidates)


def _cell(method: str, seed: int | None, budget: int) -> RecoveryCell:
    return RecoveryCell(
        method=method,
        seed=seed,
        budget=budget,
        spearman=0.25,
        relative_sse_gain=0.1,
        support_size=8,
        coefficient_count=2166,
        selected_sha256=f"selected-{method}-{seed}-{budget}",
        fold_sha256=f"fold-{method}-{seed}-{budget}",
        lambda_ratio=0.1,
        lambda_value=0.01,
        converged=True,
    )


def _registered_plan(method: str = "info", seed: int | None = None) -> SelectionPlan:
    selected = tuple(_variant((position, "A", "C")) for position in range(3072))
    return SelectionPlan(
        budgets=tuple(budget for block in REGISTERED_BUDGET_BLOCKS for budget in block),
        sequences=(
            SelectionSequence(
                method=method,
                seed=seed,
                selected=selected,
                selected_sha256=_sequence_sha256(selected),
                tie_break_version="test-v1",
            ),
        ),
    )


def _cell_from_plan(
    plan: SelectionPlan, method: str, seed: int | None, budget: int
) -> RecoveryCell:
    selected = plan.plate(method, seed, budget)
    return replace(
        _cell(method, seed, budget),
        selected_sha256=_sequence_sha256(selected),
        fold_sha256=_fold_sha256(selected, 5),
    )


def test_recovery_budget_blocks_are_two_canonical_groups_of_four() -> None:
    assert REGISTERED_BUDGET_BLOCKS == (
        (48, 96, 192, 384),
        (768, 1536, 2242, 3072),
    )


def test_recovery_block_round_trips_and_rejects_session_drift() -> None:
    _plan, _candidates, identity = _selection_fixture()
    registered_plan = _registered_plan()
    cells = tuple(
        _cell_from_plan(registered_plan, "info", None, budget)
        for budget in REGISTERED_BUDGET_BLOCKS[0]
    )
    workspace = {"execution_commit": identity.execution_commit, "code_state": "clean"}
    payload = recovery_block_payload(
        identity=identity,
        selection_plan_sha256="e" * 64,
        method="info",
        seed=None,
        block_index=0,
        cells=cells,
        workspace_start=workspace,
        workspace_end=workspace,
        input_hashes_start=identity.input_hashes,
        input_hashes_end=identity.input_hashes,
        session_id="session-1",
        started_utc="2026-08-10T12:00:00+00:00",
        completed_utc="2026-08-10T12:30:00+00:00",
    )

    loaded = load_recovery_blocks(
        {recovery_block_key("info", None, 0): payload},
        expected_identity=identity,
        expected_selection_plan_sha256="e" * 64,
        expected_plan=registered_plan,
    )

    assert loaded == {recovery_block_key("info", None, 0): cells}

    with pytest.raises(ValueError, match="drift"):
        recovery_block_payload(
            identity=identity,
            selection_plan_sha256="e" * 64,
            method="info",
            seed=None,
            block_index=0,
            cells=cells,
            workspace_start=workspace,
            workspace_end={**workspace, "code_state": "dirty"},
            input_hashes_start=identity.input_hashes,
            input_hashes_end=identity.input_hashes,
            session_id="session-1",
            started_utc="2026-08-10T12:00:00+00:00",
            completed_utc="2026-08-10T12:30:00+00:00",
        )


def test_recovery_block_rejects_tampered_budget_metadata_and_boolean_integers() -> None:
    _plan, _candidates, identity = _selection_fixture()
    registered_plan = _registered_plan()
    cells = tuple(
        _cell_from_plan(registered_plan, "info", None, budget)
        for budget in REGISTERED_BUDGET_BLOCKS[0]
    )
    workspace = {"execution_commit": identity.execution_commit, "code_state": "clean"}
    payload = recovery_block_payload(
        identity=identity,
        selection_plan_sha256="e" * 64,
        method="info",
        seed=None,
        block_index=0,
        cells=cells,
        workspace_start=workspace,
        workspace_end=workspace,
        input_hashes_start=identity.input_hashes,
        input_hashes_end=identity.input_hashes,
        session_id="session-1",
        started_utc="2026-08-10T12:00:00+00:00",
        completed_utc="2026-08-10T12:30:00+00:00",
    )
    block = payload["block"]
    assert isinstance(block, dict)
    block["budgets"] = [48, 96, 192, 999]

    with pytest.raises(ValueError, match="budgets"):
        load_recovery_blocks(
            {recovery_block_key("info", None, 0): payload},
            expected_identity=identity,
            expected_selection_plan_sha256="e" * 64,
            expected_plan=registered_plan,
        )

    block["budgets"] = list(REGISTERED_BUDGET_BLOCKS[0])
    block["block_index"] = True
    with pytest.raises(ValueError, match="key fields"):
        load_recovery_blocks(
            {recovery_block_key("info", None, 0): payload},
            expected_identity=identity,
            expected_selection_plan_sha256="e" * 64,
            expected_plan=registered_plan,
        )


def test_recovery_block_rejects_cells_from_a_different_selection_or_estimand() -> None:
    _plan, _candidates, identity = _selection_fixture()
    registered_plan = _registered_plan()
    cells = tuple(
        _cell_from_plan(registered_plan, "info", None, budget)
        for budget in REGISTERED_BUDGET_BLOCKS[0]
    )
    workspace = {"execution_commit": identity.execution_commit, "code_state": "clean"}
    payload = recovery_block_payload(
        identity=identity,
        selection_plan_sha256="e" * 64,
        method="info",
        seed=None,
        block_index=0,
        cells=cells,
        workspace_start=workspace,
        workspace_end=workspace,
        input_hashes_start=identity.input_hashes,
        input_hashes_end=identity.input_hashes,
        session_id="session-1",
        started_utc="2026-08-10T12:00:00+00:00",
        completed_utc="2026-08-10T12:30:00+00:00",
    )
    block = payload["block"]
    assert isinstance(block, dict)
    cells_payload = block["cells"]
    assert isinstance(cells_payload, list)
    first = cells_payload[0]
    assert isinstance(first, dict)
    first["selected_sha256"] = "wrong-selection"

    with pytest.raises(ValueError, match="selection prefix"):
        load_recovery_blocks(
            {recovery_block_key("info", None, 0): payload},
            expected_identity=identity,
            expected_selection_plan_sha256="e" * 64,
            expected_plan=registered_plan,
        )

    first["selected_sha256"] = cells[0].selected_sha256
    first["coefficient_count"] = 1
    with pytest.raises(ValueError, match="coefficient count"):
        load_recovery_blocks(
            {recovery_block_key("info", None, 0): payload},
            expected_identity=identity,
            expected_selection_plan_sha256="e" * 64,
            expected_plan=registered_plan,
        )


def test_recovery_block_requires_exactly_four_canonical_cells() -> None:
    _plan, _candidates, identity = _selection_fixture()
    workspace = {"execution_commit": identity.execution_commit, "code_state": "clean"}
    cells = tuple(_cell("info", None, budget) for budget in REGISTERED_BUDGET_BLOCKS[0][:-1])

    with pytest.raises(ValueError, match=r"four|budgets"):
        recovery_block_payload(
            identity=identity,
            selection_plan_sha256="e" * 64,
            method="info",
            seed=None,
            block_index=0,
            cells=cells,
            workspace_start=workspace,
            workspace_end=workspace,
            input_hashes_start=identity.input_hashes,
            input_hashes_end=identity.input_hashes,
            session_id="session-1",
            started_utc="2026-08-10T12:00:00+00:00",
            completed_utc="2026-08-10T12:30:00+00:00",
        )


def test_complete_recovery_assembly_requires_exactly_344_cells() -> None:
    sequences: list[tuple[str, int | None]] = [
        (method, None) for method in ("info", "fitness", "doptimal_reduced_pairwise")
    ]
    sequences.extend((method, seed) for method in ("random", "structural") for seed in range(20))
    blocks = {
        recovery_block_key(method, seed, block_index): tuple(
            _cell(method, seed, budget) for budget in budgets
        )
        for method, seed in sequences
        for block_index, budgets in enumerate(REGISTERED_BUDGET_BLOCKS)
    }

    assembled = assemble_recovery_cells(blocks, expected_sequences=sequences)

    assert len(assembled) == len(sequences) * sum(map(len, REGISTERED_BUDGET_BLOCKS))
    missing = dict(blocks)
    missing.pop(recovery_block_key("structural", 19, 1))
    with pytest.raises(ValueError, match="missing"):
        assemble_recovery_cells(missing, expected_sequences=sequences)


def test_pending_recovery_blocks_skips_only_completed_four_budget_groups() -> None:
    plan = SelectionPlan(
        budgets=tuple(budget for block in REGISTERED_BUDGET_BLOCKS for budget in block),
        sequences=(
            SelectionSequence(
                method="info",
                seed=None,
                selected=(),
                selected_sha256="selection",
                tie_break_version="test-v1",
            ),
        ),
    )

    pending = pending_recovery_blocks(
        plan,
        completed_keys={recovery_block_key("info", None, 0)},
    )

    assert [(item.method, item.seed, item.block_index, item.budgets) for item in pending] == [
        ("info", None, 1, REGISTERED_BUDGET_BLOCKS[1])
    ]


def test_runner_resumes_completed_blocks_and_recomputes_an_incomplete_copy(
    tmp_path: Path,
) -> None:
    plan = _registered_plan()
    first_key = recovery_block_key("info", None, 0)
    publish_checkpoint(tmp_path, "block", first_key, {"block_index": 0})
    (tmp_path / "block--unfinished--deadbeef.payload.json").write_text(
        '{"block_index":1', encoding="utf-8"
    )
    (tmp_path / "block--unfinished--deadbeef.complete.json").write_text(
        "not-json", encoding="utf-8"
    )
    completed = discover_checkpoints(tmp_path, "block")
    pending = pending_recovery_blocks(plan, completed_keys=set(completed))
    processed: list[int] = []
    progress: list[int] = []

    def execute(work: RecoveryBlockWork) -> int:
        block_index = work.block_index
        processed.append(block_index)
        key = recovery_block_key("info", None, block_index)
        publish_checkpoint(tmp_path, "block", key, {"block_index": block_index})
        return len(discover_checkpoints(tmp_path, "block")) * 4

    completed_cells = execute_pending_blocks(
        pending,
        completed_cell_count=len(completed) * 4,
        execute_block=execute,
        report_progress=lambda _work, count: progress.append(count),
    )

    assert processed == [1]
    assert progress == [8]
    assert completed_cells == sum(len(block) for block in REGISTERED_BUDGET_BLOCKS)


def test_runner_persists_or_loads_the_selection_plan_before_landscape_labels() -> None:
    source = Path("scripts/fourier_recovery_curve.py").read_text(encoding="utf-8")
    landscape_load = source.index("landscape = specification.loader(args.data)")

    assert "execute_pending_blocks(" in source
    assert '"coefficient_count": pairwise_coefficient_count' in source
    assert '"coefficient_count": len(config.modes)' not in source
    assert source.index("load_selection_plan(") < landscape_load
    assert source.index('publish_checkpoint(args.checkpoint_dir, "selection"') < landscape_load


def test_resumed_and_uninterrupted_cells_produce_the_same_decision() -> None:
    sequences: list[tuple[str, int | None]] = [
        (method, None) for method in ("info", "fitness", "doptimal_reduced_pairwise")
    ]
    sequences.extend((method, seed) for method in ("random", "structural") for seed in range(20))
    blocks = {
        recovery_block_key(method, seed, block_index): tuple(
            _cell(method, seed, budget) for budget in budgets
        )
        for method, seed in sequences
        for block_index, budgets in enumerate(REGISTERED_BUDGET_BLOCKS)
    }
    direct = tuple(cell for cells in blocks.values() for cell in cells)
    resumed = assemble_recovery_cells(blocks, expected_sequences=sequences)
    stochastic_seeds = tuple(range(20))
    expected_methods = (
        "info",
        "fitness",
        "doptimal_reduced_pairwise",
        "random",
        "structural",
    )
    expected_budgets = tuple(budget for block in REGISTERED_BUDGET_BLOCKS for budget in block)

    assert decide_recovery(
        resumed,
        stochastic_seeds=stochastic_seeds,
        expected_methods=expected_methods,
        expected_budgets=expected_budgets,
    ) == decide_recovery(
        direct,
        stochastic_seeds=stochastic_seeds,
        expected_methods=expected_methods,
        expected_budgets=expected_budgets,
    )
