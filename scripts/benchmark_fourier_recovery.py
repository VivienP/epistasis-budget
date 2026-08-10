"""Benchmark the registered TrpB Fourier recovery fit without reading measured labels."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    from epibudget.coeff_recovery import AA20, _build_fourier_config  # noqa: PLC0415
    from epibudget.data import (  # noqa: PLC0415
        TRPB_SITES,
        TRPB_WT_AT_SITES,
        enumerate_candidates,
    )
    from epibudget.fourier_recovery import (  # noqa: PLC0415
        benchmark_doptimal_prefix,
        benchmark_synthetic_fit,
        registered_fit_count,
    )
    from epibudget.provenance import write_json_atomic  # noqa: PLC0415
    from epibudget.recovery_protocol import REGISTERED_RECOVERY_PROTOCOL  # noqa: PLC0415
    from epibudget.scored_cache import candidate_sha256  # noqa: PLC0415
    from epibudget.spectrum_diagnostic import _capture_workspace_snapshot  # noqa: PLC0415

    protocol = REGISTERED_RECOVERY_PROTOCOL
    registered_budgets = protocol.budgets
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", default=",".join(str(budget) for budget in registered_budgets))
    parser.add_argument(
        "--out", type=Path, default=Path("report/diagnostics/fourier_recovery_runtime_v4.json")
    )
    args = parser.parse_args()

    budgets = tuple(int(value) for value in args.budgets.split(","))
    if budgets != registered_budgets:
        raise ValueError("the runtime preflight must measure every frozen registered budget")
    repo = Path(__file__).resolve().parent.parent
    candidates = enumerate_candidates(
        TRPB_SITES, TRPB_WT_AT_SITES, AA20, max_order=protocol.selection_max_order
    )
    config = _build_fourier_config(
        TRPB_SITES, TRPB_WT_AT_SITES, AA20, max_order=protocol.estimation_max_order
    )
    started = datetime.now(UTC)
    workspace_start = _capture_workspace_snapshot(repo)
    measurements = [
        benchmark_synthetic_fit(config, candidates, budget=budget, seed=0, n_folds=protocol.n_folds)
        for budget in budgets
    ]
    doptimal = benchmark_doptimal_prefix(
        config, candidates, pilot_budget=budgets[0], maximum_budget=budgets[-1]
    )
    fits_per_budget = registered_fit_count((1,), protocol.seeds)
    projected_lasso_seconds = sum(item.fit_seconds * fits_per_budget for item in measurements)
    projected_seconds = projected_lasso_seconds + doptimal.projected_maximum_seconds
    maximum_doptimal_bytes = max(item.doptimal_update_bytes for item in measurements)
    workspace_end = _capture_workspace_snapshot(repo)
    payload = {
        "schema_version": "epibudget-fourier-runtime-v4",
        "public_claim_eligible": False,
        "uses_measured_labels": False,
        "candidate_count": len(candidates),
        "candidate_sha256": candidate_sha256(candidates),
        "registered_fit_count": registered_fit_count(registered_budgets, protocol.seeds),
        "measured_budgets": list(budgets),
        "started_utc": started.isoformat(),
        "completed_utc": datetime.now(UTC).isoformat(),
        "measurements": [asdict(item) for item in measurements],
        "doptimal_pilot": asdict(doptimal),
        "projected_lasso_seconds": projected_lasso_seconds,
        "projected_seconds": projected_seconds,
        "maximum_doptimal_bytes": maximum_doptimal_bytes,
        "argv": [sys.executable, *sys.argv],
        "provenance": {
            "workspace_start": workspace_start.model_dump(mode="json"),
            "workspace_end": workspace_end.model_dump(mode="json"),
            "workspace_state_matches": workspace_start == workspace_end,
        },
    }
    write_json_atomic(args.out, payload)
    print(args.out)


if __name__ == "__main__":
    main()
