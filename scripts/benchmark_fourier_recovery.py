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
    from epibudget.scored_cache import candidate_sha256  # noqa: PLC0415
    from epibudget.spectrum_diagnostic import _capture_workspace_snapshot  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", default="48,96,192,384,768,1536,2242,3072")
    parser.add_argument(
        "--out", type=Path, default=Path("report/diagnostics/fourier_recovery_runtime_v3.json")
    )
    args = parser.parse_args()

    budgets = tuple(int(value) for value in args.budgets.split(","))
    registered_budgets = (48, 96, 192, 384, 768, 1536, 2242, 3072)
    if budgets != registered_budgets:
        raise ValueError("the runtime preflight must measure every frozen registered budget")
    repo = Path(__file__).resolve().parent.parent
    candidates = enumerate_candidates(TRPB_SITES, TRPB_WT_AT_SITES, AA20, max_order=3)
    config = _build_fourier_config(TRPB_SITES, TRPB_WT_AT_SITES, AA20, max_order=2)
    started = datetime.now(UTC)
    workspace_start = _capture_workspace_snapshot(repo)
    measurements = [
        benchmark_synthetic_fit(config, candidates, budget=budget, seed=0, n_folds=5)
        for budget in budgets
    ]
    doptimal = benchmark_doptimal_prefix(
        config, candidates, pilot_budget=48, maximum_budget=budgets[-1]
    )
    fits_per_budget = registered_fit_count((1,), tuple(range(20)))
    projected_lasso_seconds = sum(item.fit_seconds * fits_per_budget for item in measurements)
    projected_seconds = projected_lasso_seconds + doptimal.projected_maximum_seconds
    maximum_doptimal_bytes = max(item.doptimal_update_bytes for item in measurements)
    max_projected_hours = 8.0
    max_doptimal_gib = 2.0
    max_projected_seconds = max_projected_hours * 3600.0
    max_doptimal_bytes = int(max_doptimal_gib * 1024**3)
    workspace_end = _capture_workspace_snapshot(repo)
    payload = {
        "schema_version": "epibudget-fourier-runtime-v3",
        "public_claim_eligible": False,
        "uses_measured_labels": False,
        "candidate_count": len(candidates),
        "candidate_sha256": candidate_sha256(candidates),
        "registered_fit_count": registered_fit_count(registered_budgets, tuple(range(20))),
        "measured_budgets": list(budgets),
        "started_utc": started.isoformat(),
        "completed_utc": datetime.now(UTC).isoformat(),
        "measurements": [asdict(item) for item in measurements],
        "doptimal_pilot": asdict(doptimal),
        "projected_lasso_seconds": projected_lasso_seconds,
        "projected_seconds": projected_seconds,
        "maximum_doptimal_bytes": maximum_doptimal_bytes,
        "limits": {
            "max_projected_hours": max_projected_hours,
            "max_doptimal_gib": max_doptimal_gib,
        },
        "schedule_real_curve": (
            projected_seconds <= max_projected_seconds
            and maximum_doptimal_bytes <= max_doptimal_bytes
            and all(item.converged for item in measurements)
        ),
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
