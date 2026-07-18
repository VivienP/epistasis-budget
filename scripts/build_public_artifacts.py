"""Build provisional public JSON artifacts from the audited local result files."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from epibudget.artifacts import public_artifact_payload
from epibudget.data import load_gb1
from epibudget.provenance import workspace_code_diff_sha256, write_json_exclusive

_BASE_COMMIT_SHA = "1a1f30aabd11bb50af6208bef983f2d017352b97"
_DATA_SHA256 = "2f115d4eaf03b6083dcc22f7451b3ddfad41c9d8e519286c4e69b6d06db78f1c"


class ArtifactSpec(TypedDict):
    filename: str
    source: str
    source_run_id: str
    generation_command: str
    model_id: str
    data_sha256: str | None
    configuration: dict[str, object]
    status: str
    evidence_classification: str


_SPECS: tuple[ArtifactSpec, ...] = (
    {
        "filename": "signal_650m.json",
        "source": "report/spike_gb1_650M.json",
        "source_run_id": "spike_gb1_650M",
        "generation_command": (
            "python scripts/gb1_epistasis_signal.py "
            "--model facebook/esm2_t33_650M_UR50D --seed 0 "
            "--out report/spike_gb1_650M.json"
        ),
        "model_id": "facebook/esm2_t33_650M_UR50D",
        "data_sha256": _DATA_SHA256,
        "configuration": {"seed": 0, "n_perturbations": 0},
        "status": "supplementary",
        "evidence_classification": "traceable_not_rerun",
    },
    {
        "filename": "smoke_recovery_35m.json",
        "source": "report/20260709T105647Z/metrics.json",
        "source_run_id": "20260709T105647Z",
        "generation_command": (
            "epibudget validate --model esm2_t12_35M --alphabet ADEF "
            "--budgets 48,96 --seeds 20 --n-perturbations 16 --device cpu"
        ),
        "model_id": "facebook/esm2_t12_35M_UR50D",
        "data_sha256": _DATA_SHA256,
        "configuration": {
            "alphabet": "ADEF",
            "budgets": [48, 96],
            "seeds": 20,
            "n_perturbations": 16,
        },
        "status": "smoke_test",
        "evidence_classification": "traceable_not_rerun",
    },
    {
        "filename": "calibration_35m.json",
        "source": "report/calibration_20260710T121638Z/metrics.json",
        "source_run_id": "calibration_20260710T121638Z",
        "generation_command": (
            "python scripts/calibrate_uncertainty.py --model esm2_t12_35M --n 300 "
            "--n-perturbations 16 --alphabet ACDEFGHIKLMNPQRSTVWY --seed 0 --device cpu"
        ),
        "model_id": "facebook/esm2_t12_35M_UR50D",
        "data_sha256": _DATA_SHA256,
        "configuration": {
            "n": 300,
            "seed": 0,
            "n_perturbations": 16,
            "alphabet": "ACDEFGHIKLMNPQRSTVWY",
        },
        "status": "smoke_test",
        "evidence_classification": "traceable_not_rerun",
    },
    {
        "filename": "calibration_650m.json",
        "source": "report/calibration_20260710T120730Z/metrics.json",
        "source_run_id": "calibration_20260710T120730Z",
        "generation_command": (
            "python scripts/calibrate_uncertainty.py --model esm2_t33_650M --n 300 "
            "--n-perturbations 16 --alphabet ACDEFGHIKLMNPQRSTVWY --seed 0 --device cpu"
        ),
        "model_id": "facebook/esm2_t33_650M_UR50D",
        "data_sha256": _DATA_SHA256,
        "configuration": {
            "n": 300,
            "seed": 0,
            "n_perturbations": 16,
            "alphabet": "ACDEFGHIKLMNPQRSTVWY",
        },
        "status": "supplementary",
        "evidence_classification": "traceable_not_rerun",
    },
    {
        "filename": "supplementary_recovery_650m.json",
        "source": "report/20260710T101945Z/metrics.json",
        "source_run_id": "20260710T101945Z",
        "generation_command": (
            "python scripts/headline_650m_supplementary.py --model esm2_t33_650M "
            "--alphabet ACDEFGHIKLMNPQRSTVWY --budgets 48,96,192 --seeds 20 --device cpu"
        ),
        "model_id": "facebook/esm2_t33_650M_UR50D",
        "data_sha256": _DATA_SHA256,
        "configuration": {
            "alphabet": "ACDEFGHIKLMNPQRSTVWY",
            "budgets": [48, 96, 192],
            "seeds": 20,
            "n_perturbations": 0,
        },
        "status": "supplementary",
        "evidence_classification": "traceable_not_rerun",
    },
    {
        "filename": "headline_650m.json",
        "source": "report/20260711T091947Z/metrics.json",
        "source_run_id": "20260711T091947Z",
        "generation_command": (
            "epibudget validate --model esm2_t33_650M --alphabet ACDEFGHIKLMNPQRSTVWY "
            "--budgets 48,96,192 --seeds 20 --n-perturbations 16 --device cuda "
            "--batch-size 128 --scored-cache scored_650m.jsonl --out report/"
        ),
        "model_id": "facebook/esm2_t33_650M_UR50D",
        "data_sha256": _DATA_SHA256,
        "configuration": {
            "alphabet": "ACDEFGHIKLMNPQRSTVWY",
            "budgets": [48, 96, 192],
            "seeds": 20,
            "n_perturbations": 16,
            "device": "cuda",
            # Colab executed the GitHub branch tip; an ancestor of the local manifest base.
            "colab_commit": "3ba75ebbe700247654c824627fa98c4b8ba4010c",
            "cache_wt_sha256": ("7e859d82171047700fd3e9632f7a47eab4a39baedc8c3316d2fc62d3ce2260bb"),
            "cache_candidate_sha256": (
                "0822f65ec14183af7a534ec30cbf5f54e9864f7c96677d0aa1e0e046e4258c41"
            ),
        },
        "status": "primary",
        "evidence_classification": "traceable_not_rerun",
    },
    {
        "filename": "robustness_650m.json",
        "source": "report/20260711T102440Z/robustness.json",
        "source_run_id": "20260711T102440Z",
        "generation_command": (
            "epibudget robustness --scored-cache report/scored_650m.jsonl "
            "--data data/proteingym/gb1_wu2016.csv --alphabet ACDEFGHIKLMNPQRSTVWY "
            "--budgets 48,96,192 --seeds 20 --max-order 3 --n-folds 5 --out report/"
        ),
        "model_id": "facebook/esm2_t33_650M_UR50D",
        "data_sha256": _DATA_SHA256,
        "configuration": {
            "alphabet": "ACDEFGHIKLMNPQRSTVWY",
            "budgets": [48, 96, 192],
            "seeds": 20,
            "max_order": 3,
            "n_folds": 5,
            "analysis": (
                "post-hoc A1 common-predicted-term precision, A2 cross-fit scale "
                "sensitivity, A3 paired difference CIs; descriptive, not hypothesis tests"
            ),
        },
        "status": "primary",
        # Derived from the Colab-produced (uncommitted, GPU) scored cache, so it is not
        # regenerable from a clean checkout — traceable, not reproduced.
        "evidence_classification": "traceable_not_rerun",
    },
    {
        "filename": "bench_35m.json",
        "source": "report/bench_35M.json",
        "source_run_id": "bench_35M",
        "generation_command": (
            "python scripts/bench_scoring.py --model esm2_t12_35M --alphabet ACD "
            "--n-perturbations 4 --batch-size 32 --threads 12 --device cpu --seed 0"
        ),
        "model_id": "facebook/esm2_t12_35M_UR50D",
        "data_sha256": None,
        "configuration": {
            "alphabet": "ACD",
            "n_perturbations": 4,
            "batch_size": 32,
            "threads": 12,
            "seed": 0,
        },
        "status": "benchmark",
        "evidence_classification": "traceable_not_rerun",
    },
    {
        "filename": "bench_650m.json",
        "source": "report/bench_650m.json",
        "source_run_id": "bench_650m",
        "generation_command": (
            "python scripts/bench_scoring.py --model esm2_t33_650M --alphabet AC "
            "--n-perturbations 4 --batch-size 32 --threads 12 --device cpu --seed 0"
        ),
        "model_id": "facebook/esm2_t33_650M_UR50D",
        "data_sha256": None,
        "configuration": {
            "alphabet": "AC",
            "n_perturbations": 4,
            "batch_size": 32,
            "threads": 12,
            "seed": 0,
        },
        "status": "benchmark",
        "evidence_classification": "traceable_not_rerun",
    },
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_payload(repo: Path) -> dict[str, object]:
    csv_path = repo / "data" / "proteingym" / "gb1_wu2016.csv"
    landscape = load_gb1(csv_path)  # raises on any off-site mutation or a missing WT row
    counts: Counter[tuple[int, str]] = Counter()
    for variant, fitness in landscape.items():
        status = "live" if fitness > 0.0 else "dead"
        counts[(len(variant), status)] += 1
    measured = sum(counts.values())
    theoretical_by_order = {0: 1, 1: 76, 2: 2166, 3: 27436, 4: 130321}
    return {
        "schema_version": 1,
        "dataset": "gb1_wu2016",
        "source_path": "data/proteingym/gb1_wu2016.csv",
        "data_sha256": _sha256(csv_path),
        "theoretical_genotypes": 160000,
        "measured_rows": measured,
        "live_rows": sum(value for (order, status), value in counts.items() if status == "live"),
        "dead_rows": sum(value for (order, status), value in counts.items() if status == "dead"),
        "missing_rows": 160000 - measured,
        "by_order": {
            str(order): {
                "theoretical": theoretical_by_order[order],
                "live": counts[(order, "live")],
                "dead": counts[(order, "dead")],
                "missing": theoretical_by_order[order]
                - counts[(order, "live")]
                - counts[(order, "dead")],
            }
            for order in range(5)
        },
        "conditioning": (
            "Calibration and real-valued epistasis truth use positive-fitness, "
            "log-transformable rows with complete loops."
        ),
    }


def _git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    artifacts_dir = repo / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    head = _git_head(repo)
    if head != _BASE_COMMIT_SHA:
        raise RuntimeError(f"expected base {_BASE_COMMIT_SHA}, found {head}")
    code_diff = workspace_code_diff_sha256(repo, _BASE_COMMIT_SHA)
    manifest_entries: list[dict[str, object]] = []

    dataset_path = artifacts_dir / "dataset_gb1.json"
    write_json_exclusive(dataset_path, _dataset_payload(repo))
    manifest_entries.append(
        {
            "path": "artifacts/dataset_gb1.json",
            "sha256": _sha256(dataset_path),
            "source_run_id": "data-provenance-20260710",
            "generation_command": "python scripts/build_public_artifacts.py",
            "base_commit_sha": _BASE_COMMIT_SHA,
            "code_state": "dirty",
            "code_diff_sha256": code_diff,
            "model_id": None,
            "data_sha256": _DATA_SHA256,
            "configuration": {"theoretical_alphabet_size": 20, "sites": 4},
            "status": "primary",
            "evidence_classification": "reproduced",
        }
    )

    for spec in _SPECS:
        source = repo / spec["source"]
        destination = artifacts_dir / spec["filename"]
        payload = public_artifact_payload(
            source,
            source_path=spec["source"],
            source_run_id=spec["source_run_id"],
            evidence_classification=spec["evidence_classification"],
        )
        write_json_exclusive(destination, payload)
        manifest_entries.append(
            {
                "path": f"artifacts/{spec['filename']}",
                "sha256": _sha256(destination),
                "source_run_id": spec["source_run_id"],
                "generation_command": spec["generation_command"],
                "base_commit_sha": _BASE_COMMIT_SHA,
                "code_state": "dirty",
                "code_diff_sha256": code_diff,
                "model_id": spec["model_id"],
                "data_sha256": spec["data_sha256"],
                "configuration": spec["configuration"],
                "status": spec["status"],
                "evidence_classification": spec["evidence_classification"],
            }
        )

    claim_map_path = artifacts_dir / "claim_map.json"
    manifest_entries.append(
        {
            "path": "artifacts/claim_map.json",
            "sha256": _sha256(claim_map_path),
            "source_run_id": "claim-map-v1",
            "generation_command": (
                "maintained with README.md; validated by python scripts/validate_artifacts.py"
            ),
            "base_commit_sha": _BASE_COMMIT_SHA,
            "code_state": "dirty",
            "code_diff_sha256": code_diff,
            "model_id": None,
            "data_sha256": None,
            "configuration": {"schema_version": 1},
            "status": "primary",
            "evidence_classification": "reproduced",
        }
    )
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "provisional": True,
        "requires_remanifest_after_commit": True,
        "artifacts": manifest_entries,
    }
    write_json_exclusive(artifacts_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
