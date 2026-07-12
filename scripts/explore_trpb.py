"""Exploratory TrpB profile — a Phase 1.5, non-decision-eligible transfer check.

Orchestrates the importable profiler (``epibudget.trpb_explore.profile_trpb``) over a TrpB CSV and
prints/writes a run-headed exploratory profile. It computes **no** benchmark number: every output is
stamped ``run_type = exploratory_non_decision_eligible`` and is not part of the frozen GB1 claim
(docs/VALIDATION.md). The reusable analysis lives in the module; this script only orchestrates and
serializes it, so a notebook or Colab cell can call ``profile_trpb`` directly instead.

No ESM, no torch, no network: it reads the CSV and the candidate enumeration only.

Get the data first (git-ignored, never committed):
    python scripts/fetch_trpb.py     # writes data/proteingym/trpb_johnston2024.csv + provenance

Then, locally:
    python scripts/explore_trpb.py          # defaults to data/proteingym/trpb_johnston2024.csv
    python scripts/explore_trpb.py --data /path/to/trpb.csv --out report/trpb_explore/profile.json

In Colab (no hardcoded local paths): upload or mount the CSV, then pass its path, e.g.
    !python scripts/explore_trpb.py --data /content/trpb_johnston2024.csv

A methods/metrics run is a SEPARATE, expensive step (it needs ESM scoring) and is intentionally NOT
run here. When the GB1 headline is interpreted and the deferred TrpB confirmatory run is authorized,
the command is the frozen `validate` on the native TrpB dataset (docs/VALIDATION.md §"Second
landscape"); the `trpb_johnston2024` identifier selects the TrpB loader, sites and reference, and
`--data` defaults to `data/proteingym/trpb_johnston2024.csv`:
    epibudget validate --dataset trpb_johnston2024 \\
        --model esm2_t33_650M --alphabet ACDEFGHIKLMNPQRSTVWY --budgets 48,96,192 \\
        --seeds 20 --n-perturbations 16 --device cuda --out report/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from epibudget.trpb_explore import RUN_TYPE, TRPB_SOURCE, TrpbProfile, profile_trpb

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str:
    """Best-effort git query; returns 'unknown' when git or the repo is unavailable (e.g. Colab)."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError):
        return "unknown"
    value = out.stdout.strip()
    return value if (out.returncode == 0 and value) else "unknown"


def _source_and_version(data_path: Path) -> tuple[str, str]:
    """Read source/version from the fetch script's provenance sidecar if present; never invented."""
    prov = data_path.parent / "provenance_trpb.json"
    if prov.exists():
        try:
            record = json.loads(prov.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return TRPB_SOURCE, "provenance_trpb.json unreadable"
        source = str(record.get("source_dataset", TRPB_SOURCE))
        version = str(record.get("downloaded_utc", "unknown"))
        return source, version
    return TRPB_SOURCE, "provenance_trpb.json absent (run scripts/fetch_trpb.py)"


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory, non-decision-eligible TrpB profile.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/proteingym/trpb_johnston2024.csv"),
        help="Path to the TrpB CSV (protein,label). Not committed; see scripts/fetch_trpb.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("report/trpb_explore/profile.json"),
        help="Where to write the JSON profile (report/ is git-ignored).",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"TrpB CSV not found at {args.data}. Run `python scripts/fetch_trpb.py` first, or pass "
            f"--data PATH (in Colab, upload/mount the file and pass its path)."
        )

    profile = profile_trpb(args.data)
    source, version = _source_and_version(args.data)
    prov_checksum = None
    prov_path = args.data.parent / "provenance_trpb.json"
    if prov_path.exists():
        try:
            prov_checksum = json.loads(prov_path.read_text(encoding="utf-8")).get("sha256")
        except (OSError, json.JSONDecodeError):
            prov_checksum = None

    run_header = {
        "repository_sha": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dataset_source": source,
        "dataset_version": version,
        "dataset_checksum_sha256": profile.dataset_checksum_sha256,
        "dataset_checksum_matches_provenance": (
            None if prov_checksum is None else prov_checksum == profile.dataset_checksum_sha256
        ),
        "run_type": RUN_TYPE,
        "decision_eligible": False,
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }

    payload = {"run_header": run_header, "profile": profile.model_dump()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _print_summary(profile, run_header, source, version, args.out)


def _print_summary(
    profile: TrpbProfile,
    run_header: dict[str, object],
    source: str,
    version: str,
    out_path: Path,
) -> None:
    """ASCII-only stdout summary (Windows consoles default to cp1252 and choke on non-ASCII)."""
    cov = profile.coverage
    fit = profile.fitness
    dup = profile.duplicates
    full = profile.all_sites_fully_covered
    lines = [
        "=== Exploratory TrpB profile (NON-DECISION-ELIGIBLE) ===",
        f"  run_type            : {run_header['run_type']}",
        f"  repository_sha      : {run_header['repository_sha']}",
        f"  branch              : {run_header['branch']}",
        f"  dataset_source      : {source}",
        f"  dataset_version     : {version}",
        f"  dataset_checksum    : {profile.dataset_checksum_sha256}",
        f"  checksum==provenance: {run_header['dataset_checksum_matches_provenance']}",
        f"  generated_at_utc    : {run_header['generated_at_utc']}",
        "  --- dimensions ---",
        f"  rows                : {profile.n_rows:,}",
        f"  unique variants     : {dup.n_unique_variants:,}",
        f"  measured / missing  : {profile.n_measured:,} / {profile.n_missing_label:,}",
        f"  invalid records     : {profile.n_invalid_records:,}  {profile.status_counts}",
        f"  wild type present   : {profile.wt_present}  (label={_fmt(profile.wt_label)})",
        f"  mutated positions   : {len(profile.sites_0indexed)}  sites={profile.sites_0indexed}",
        f"  order distribution  : {profile.order_distribution}",
        f"  singles/doubles/triples/quads: "
        f"{profile.n_singles:,}/{profile.n_doubles:,}/{profile.n_triples:,}/{profile.n_quadruples:,}",
        f"  AA coverage / site  : {profile.aa_coverage_counts}  (full={full})",
        "  --- duplicates ---",
        f"  duplicate variants  : {dup.n_duplicate_variants:,}  "
        f"(identical={dup.n_identical_duplicate_variants:,}, "
        f"conflicting={dup.n_conflicting_duplicate_variants:,})",
        "  --- fitness (measured labels) ---",
        f"  n / positive / <=0  : {fit.n:,} / {fit.n_positive:,} / {fit.n_nonpositive:,}",
        f"  min/median/max      : {_fmt(fit.min)} / {_fmt(fit.median)} / {_fmt(fit.max)}  "
        f"(mean={_fmt(fit.mean)}, std={_fmt(fit.std)})",
        "  --- order-1..3 universe coverage ---",
        f"  max selection order : {cov.max_selection_order}",
        f"  universe size       : {cov.universe_size:,}",
        f"  measured / positive : {cov.n_universe_measured:,} "
        f"({cov.coverage_fraction:.4f}) / {cov.n_universe_measured_positive:,} "
        f"({cov.positive_coverage_fraction:.4f})",
        f"  order>3 (out-scope) : {cov.n_beyond_selection_order_measured:,}",
        "  --- budgets (proposed) ---",
        f"  exploratory         : B={profile.budget_recommendation.exploratory_budgets}, "
        f"partitions={profile.budget_recommendation.exploratory_partitions}",
        f"  confirmatory (frozen): B={profile.budget_recommendation.confirmatory_budgets}",
        "  --- GB1-specific assumptions to respect ---",
        *(f"  [{i}] {note}" for i, note in enumerate(profile.gb1_incompatibilities, 1)),
        f"\n  written to {out_path}",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
