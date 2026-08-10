"""Run the registered TrpB pairwise Fourier recovery diagnostic from a validated score cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:  # noqa: PLR0915
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    from epibudget.coeff_recovery import AA20, _build_fourier_config  # noqa: PLC0415
    from epibudget.data import enumerate_candidates, resolve_dataset  # noqa: PLC0415
    from epibudget.fourier_recovery import (  # noqa: PLC0415
        RecoveryCell,
        _fold_sha256,
        _sequence_sha256,
        build_selection_plan,
        decide_recovery,
        evaluate_plate,
        pairwise_truth,
        registered_fit_count,
        validate_recovery_dataset,
        validate_runtime_preflight,
    )
    from epibudget.labels import training_target  # noqa: PLC0415
    from epibudget.provenance import write_json_atomic  # noqa: PLC0415
    from epibudget.recovery_checkpoint import (  # noqa: PLC0415
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
    from epibudget.recovery_protocol import (  # noqa: PLC0415
        REGISTERED_EXECUTION_POLICY,
        REGISTERED_RECOVERY_PROTOCOL,
    )
    from epibudget.scored_cache import (  # noqa: PLC0415
        CacheIdentity,
        cache_metadata_path,
        candidate_sha256,
        validate_cache_against_universe,
    )
    from epibudget.spectrum_diagnostic import (  # noqa: PLC0415
        _capture_workspace_snapshot,
        _sha256_file,
    )
    from epibudget.tie_break import canonical_id  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/proteingym/trpb_johnston2024.csv"))
    parser.add_argument("--cache", type=Path, default=Path("report/scored_trpb_650m_n16.jsonl"))
    parser.add_argument(
        "--runtime-preflight",
        type=Path,
        default=Path("report/diagnostics/fourier_recovery_runtime_v4.json"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("report/diagnostics/fourier_recovery_trpb.json")
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Durable directory containing resumable selection and recovery checkpoints.",
    )
    args = parser.parse_args()

    protocol = REGISTERED_RECOVERY_PROTOCOL
    budgets = protocol.budgets
    seeds = protocol.seeds
    methods = protocol.methods
    pairwise_order = protocol.estimation_max_order
    sequence_keys: list[dict[str, str | int | None]] = [
        {"method": method, "seed": seed} for method, seed in protocol.sequence_keys
    ]
    dataset = protocol.dataset
    validate_recovery_dataset(dataset)
    model_id = "facebook/esm2_t33_650M_UR50D"
    repo = Path(__file__).resolve().parent.parent
    sidecar = cache_metadata_path(args.cache)
    started_utc = datetime.now(UTC)
    session_id = uuid.uuid4().hex
    argv = [sys.executable, *sys.argv]
    workspace_start = _capture_workspace_snapshot(repo)
    input_hashes_start = {
        "dataset_sha256": _sha256_file(args.data),
        "cache_sha256": _sha256_file(args.cache),
        "cache_sidecar_sha256": _sha256_file(sidecar),
        "runtime_preflight_sha256": _sha256_file(args.runtime_preflight),
    }

    specification = resolve_dataset(dataset)
    candidates = enumerate_candidates(
        specification.sites, specification.wt_at_sites, AA20, max_order=protocol.selection_max_order
    )
    config = _build_fourier_config(
        specification.sites, specification.wt_at_sites, AA20, max_order=pairwise_order
    )
    pairwise_coefficient_count = sum(
        sum(value != 0 for value in mode) == pairwise_order for mode in config.modes
    )
    if pairwise_coefficient_count != protocol.coefficient_count:
        raise ValueError("candidate universe does not yield the registered coefficient count")
    if len(config.modes) != protocol.feature_count:
        raise ValueError("design basis does not yield the registered feature count")
    runtime = json.loads(args.runtime_preflight.read_text(encoding="utf-8"))
    validate_runtime_preflight(
        runtime,
        expected_commit=workspace_start.execution_commit,
        expected_candidate_count=len(candidates),
        expected_candidate_sha256=candidate_sha256(candidates),
        expected_budgets=budgets,
        expected_fit_count=registered_fit_count(budgets, seeds),
        expected_feature_count=len(config.modes),
    )
    cache, metadata, expected_identity = validate_cache_against_universe(
        args.cache,
        candidates,
        candidate_alphabet=AA20,
        max_order=3,
        model_id=model_id,
        scorer_seed=0,
        n_perturbations=16,
        wt_sequence=specification.wt_sequence,
    )
    scored = [cache[variant] for variant in candidates]

    checkpoint_identity = RecoveryIdentity(
        execution_commit=workspace_start.execution_commit,
        candidate_sha256=candidate_sha256(candidates),
        input_hashes=input_hashes_start,
        cache_identity=expected_identity.model_dump(mode="json"),
        numerical_fingerprint=numerical_fingerprint(),
        protocol={
            "schema_version": "epibudget-fourier-recovery-checkpoint-v1",
            "methods": list(methods),
            "budgets": list(budgets),
            "seeds": list(seeds),
            "sequence_keys": sequence_keys,
            "budget_block_size": REGISTERED_EXECUTION_POLICY.legacy_budget_block_size,
            "folds": protocol.n_folds,
            "coefficient_count": pairwise_coefficient_count,
            "maximum_order": protocol.selection_max_order,
        },
    )
    selection_checkpoints = discover_checkpoints(args.checkpoint_dir, "selection")
    if selection_checkpoints and set(selection_checkpoints) != {"registered"}:
        raise ValueError("checkpoint directory contains unexpected selection-plan identities")
    if selection_checkpoints:
        selection_payload = selection_checkpoints["registered"]
        plan = load_selection_plan(
            selection_payload,
            expected=checkpoint_identity,
            expected_candidates=candidates,
        )
    else:
        selection_workspace_start = _capture_workspace_snapshot(repo)
        selection_inputs_start = {
            "dataset_sha256": _sha256_file(args.data),
            "cache_sha256": _sha256_file(args.cache),
            "cache_sidecar_sha256": _sha256_file(sidecar),
            "runtime_preflight_sha256": _sha256_file(args.runtime_preflight),
        }
        # This is the label barrier: the complete plan is durable before the landscape is loaded.
        generated_plan = build_selection_plan(
            scored, budgets=budgets, seeds=seeds, max_order=protocol.selection_max_order
        )
        selection_workspace_end = _capture_workspace_snapshot(repo)
        selection_inputs_end = {
            "dataset_sha256": _sha256_file(args.data),
            "cache_sha256": _sha256_file(args.cache),
            "cache_sidecar_sha256": _sha256_file(sidecar),
            "runtime_preflight_sha256": _sha256_file(args.runtime_preflight),
        }
        if (
            selection_workspace_start != selection_workspace_end
            or selection_workspace_start.code_state != "clean"
            or selection_inputs_start != selection_inputs_end
            or selection_inputs_start != input_hashes_start
        ):
            raise ValueError("selection-plan input or workspace drift detected")
        selection_payload = selection_plan_payload(generated_plan, checkpoint_identity)
        publish_checkpoint(args.checkpoint_dir, "selection", "registered", selection_payload)
        published_selection = discover_checkpoints(args.checkpoint_dir, "selection")
        if set(published_selection) != {"registered"}:
            raise ValueError("selection-plan checkpoint publication did not complete")
        selection_payload = published_selection["registered"]
        plan = load_selection_plan(
            selection_payload,
            expected=checkpoint_identity,
            expected_candidates=candidates,
        )
    selection_plan_sha256 = selection_payload.get("selection_plan_sha256")
    if not isinstance(selection_plan_sha256, str):
        raise ValueError("selection-plan checkpoint has no canonical hash")

    landscape = specification.loader(args.data)
    transformed = {variant: training_target(value) for variant, value in landscape.items()}
    truth = pairwise_truth(transformed, specification.sites)
    if len(truth.coefficients) != pairwise_coefficient_count:
        raise ValueError("pairwise coefficient count does not match the registered protocol")

    block_payloads = discover_checkpoints(args.checkpoint_dir, "block")
    completed_blocks = load_recovery_blocks(
        block_payloads,
        expected_identity=checkpoint_identity,
        expected_selection_plan_sha256=selection_plan_sha256,
        expected_plan=plan,
    )
    completed_cell_count = sum(len(block) for block in completed_blocks.values())
    print(f"recovery progress: {completed_cell_count} / {protocol.cell_count} cells")
    sequence_by_key = {(sequence.method, sequence.seed): sequence for sequence in plan.sequences}

    def execute_block(work: RecoveryBlockWork) -> int:
        sequence = sequence_by_key[(work.method, work.seed)]
        block_started = datetime.now(UTC)
        block_workspace_start = _capture_workspace_snapshot(repo)
        block_input_hashes_start = {
            "dataset_sha256": _sha256_file(args.data),
            "cache_sha256": _sha256_file(args.cache),
            "cache_sidecar_sha256": _sha256_file(sidecar),
            "runtime_preflight_sha256": _sha256_file(args.runtime_preflight),
        }
        block_cells = []
        for budget in work.budgets:
            selected = sequence.selected[:budget]
            try:
                cell = evaluate_plate(
                    config,
                    selected,
                    landscape,
                    truth.coefficients,
                    method=sequence.method,
                    seed=sequence.seed,
                    budget=budget,
                    n_folds=protocol.n_folds,
                )
            except (FloatingPointError, RuntimeError, ValueError) as error:
                cell = RecoveryCell(
                    method=sequence.method,
                    budget=budget,
                    seed=sequence.seed,
                    spearman=None,
                    relative_sse_gain=None,
                    support_size=0,
                    coefficient_count=pairwise_coefficient_count,
                    selected_sha256=_sequence_sha256(selected),
                    fold_sha256=_fold_sha256(selected, protocol.n_folds),
                    converged=False,
                    error=f"{type(error).__name__}: {error}",
                )
            block_cells.append(cell)
        block_input_hashes_end = {
            "dataset_sha256": _sha256_file(args.data),
            "cache_sha256": _sha256_file(args.cache),
            "cache_sidecar_sha256": _sha256_file(sidecar),
            "runtime_preflight_sha256": _sha256_file(args.runtime_preflight),
        }
        block_workspace_end = _capture_workspace_snapshot(repo)
        block_key = recovery_block_key(work.method, work.seed, work.block_index)
        block_payload = recovery_block_payload(
            identity=checkpoint_identity,
            selection_plan_sha256=selection_plan_sha256,
            method=work.method,
            seed=work.seed,
            block_index=work.block_index,
            cells=block_cells,
            workspace_start=block_workspace_start.model_dump(mode="json"),
            workspace_end=block_workspace_end.model_dump(mode="json"),
            input_hashes_start=block_input_hashes_start,
            input_hashes_end=block_input_hashes_end,
            session_id=session_id,
            started_utc=block_started.isoformat(),
            completed_utc=datetime.now(UTC).isoformat(),
        )
        publish_checkpoint(args.checkpoint_dir, "block", block_key, block_payload)
        published_blocks = discover_checkpoints(args.checkpoint_dir, "block")
        validated_blocks = load_recovery_blocks(
            published_blocks,
            expected_identity=checkpoint_identity,
            expected_selection_plan_sha256=selection_plan_sha256,
            expected_plan=plan,
        )
        return sum(len(block) for block in validated_blocks.values())

    def report_progress(work: RecoveryBlockWork, cell_count: int) -> None:
        print(
            f"recovery checkpoint: {work.method} seed={work.seed} budgets={work.budgets}; "
            f"{cell_count} / {protocol.cell_count} cells; {args.checkpoint_dir}"
        )

    pending = pending_recovery_blocks(plan, completed_keys=set(completed_blocks))
    execute_pending_blocks(
        pending,
        completed_cell_count=completed_cell_count,
        execute_block=execute_block,
        report_progress=report_progress,
    )
    block_payloads = discover_checkpoints(args.checkpoint_dir, "block")
    completed_blocks = load_recovery_blocks(
        block_payloads,
        expected_identity=checkpoint_identity,
        expected_selection_plan_sha256=selection_plan_sha256,
        expected_plan=plan,
    )
    cells = assemble_recovery_cells(
        completed_blocks,
        expected_sequences=[(sequence.method, sequence.seed) for sequence in plan.sequences],
    )
    decision = decide_recovery(
        cells,
        stochastic_seeds=seeds,
        expected_methods=methods,
        expected_budgets=budgets,
    )

    input_hashes_end = {
        "dataset_sha256": _sha256_file(args.data),
        "cache_sha256": _sha256_file(args.cache),
        "cache_sidecar_sha256": _sha256_file(sidecar),
        "runtime_preflight_sha256": _sha256_file(args.runtime_preflight),
    }
    workspace_end = _capture_workspace_snapshot(repo)
    observed_identity = CacheIdentity.from_metadata(metadata)
    provenance_stable = input_hashes_start == input_hashes_end and workspace_start == workspace_end
    architecture_decision_eligible = (
        provenance_stable
        and workspace_start.code_state == "clean"
        and expected_identity == observed_identity
        and decision.status != "invalid_coverage"
    )
    truth_sha256 = hashlib.sha256(truth.coefficients.tobytes()).hexdigest()
    modes_payload = json.dumps(truth.modes, separators=(",", ":")).encode("ascii")
    payload = {
        "schema_version": "epibudget-fourier-recovery-v2",
        "public_claim_eligible": False,
        "architecture_decision_eligible": architecture_decision_eligible,
        "dataset": dataset,
        "label_transform": protocol.label_transform,
        "candidate_count": len(candidates),
        "candidate_composition": {"1": 76, "2": 2166, "3": 27436},
        "budgets": list(budgets),
        "stochastic_seeds": list(seeds),
        "pairwise_coefficient_count": pairwise_coefficient_count,
        "pairwise_truth_sha256": truth_sha256,
        "pairwise_modes_sha256": hashlib.sha256(modes_payload).hexdigest(),
        "imputation_note": (
            "The redistributed target contains 159,129 measured values and 871 source-imputed "
            "values whose identities are not exposed by the mirror."
        ),
        "cache_identity_expected": expected_identity.model_dump(mode="json"),
        "cache_identity_observed": observed_identity.model_dump(mode="json"),
        "selection_sequences": [
            {
                "method": sequence.method,
                "seed": sequence.seed,
                "selected_sha256": sequence.selected_sha256,
                "tie_break_version": sequence.tie_break_version,
                "selected_ids": [canonical_id(variant) for variant in sequence.selected],
            }
            for sequence in plan.sequences
        ],
        "cells": [asdict(cell) for cell in cells],
        "aggregates": [asdict(item) for item in decision.aggregates],
        "decision": {
            "status": decision.status,
            "passing_cells": [list(item) for item in decision.passing_cells],
            "reasons": list(decision.reasons),
        },
        "provenance": {
            "started_utc": started_utc.isoformat(),
            "completed_utc": datetime.now(UTC).isoformat(),
            "argv": argv,
            "exact_command": subprocess.list2cmdline(argv),
            "workspace_start": workspace_start.model_dump(mode="json"),
            "workspace_end": workspace_end.model_dump(mode="json"),
            "input_hashes_start": input_hashes_start,
            "input_hashes_end": input_hashes_end,
            "provenance_stable": provenance_stable,
            "session_id": session_id,
            "selection_checkpoint_sha256": hashlib.sha256(
                canonical_json_bytes(selection_payload)
            ).hexdigest(),
            "block_checkpoint_sha256": {
                key: hashlib.sha256(canonical_json_bytes(block_payload)).hexdigest()
                for key, block_payload in sorted(block_payloads.items())
            },
        },
    }
    write_json_atomic(args.out, payload)
    print(args.out)


if __name__ == "__main__":
    main()
