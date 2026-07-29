"""Run the registered TrpB pairwise Fourier recovery diagnostic from a validated score cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
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
    args = parser.parse_args()

    budgets = (48, 96, 192, 384, 768, 1536, 2242, 3072)
    seeds = tuple(range(20))
    methods = ("info", "fitness", "doptimal_reduced_pairwise", "random", "structural")
    dataset = "trpb_johnston2024"
    validate_recovery_dataset(dataset)
    model_id = "facebook/esm2_t33_650M_UR50D"
    repo = Path(__file__).resolve().parent.parent
    sidecar = cache_metadata_path(args.cache)
    started_utc = datetime.now(UTC)
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
        specification.sites, specification.wt_at_sites, AA20, max_order=3
    )
    config = _build_fourier_config(
        specification.sites, specification.wt_at_sites, AA20, max_order=2
    )
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

    # This is the label barrier: every sequence and hash is fixed before the landscape is loaded.
    plan = build_selection_plan(scored, budgets=budgets, seeds=seeds, max_order=3)
    landscape = specification.loader(args.data)
    transformed = {variant: training_target(value) for variant, value in landscape.items()}
    truth = pairwise_truth(transformed, specification.sites)

    cells = []
    for sequence in plan.sequences:
        for budget in budgets:
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
                    n_folds=5,
                )
            except (FloatingPointError, RuntimeError, ValueError) as error:
                cell = RecoveryCell(
                    method=sequence.method,
                    budget=budget,
                    seed=sequence.seed,
                    spearman=None,
                    relative_sse_gain=None,
                    support_size=0,
                    coefficient_count=len(truth.coefficients),
                    selected_sha256=_sequence_sha256(selected),
                    fold_sha256=_fold_sha256(selected, 5),
                    converged=False,
                    error=f"{type(error).__name__}: {error}",
                )
            cells.append(cell)
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
        "schema_version": "epibudget-fourier-recovery-v1",
        "public_claim_eligible": False,
        "architecture_decision_eligible": architecture_decision_eligible,
        "dataset": dataset,
        "label_transform": "log1p(fitness)",
        "candidate_count": len(candidates),
        "candidate_composition": {"1": 76, "2": 2166, "3": 27436},
        "budgets": list(budgets),
        "stochastic_seeds": list(seeds),
        "pairwise_coefficient_count": len(truth.coefficients),
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
        },
    }
    write_json_atomic(args.out, payload)
    print(args.out)


if __name__ == "__main__":
    main()
