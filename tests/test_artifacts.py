"""Offline tests for public artifact provenance and documented-claim validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from epibudget.artifacts import (
    ArtifactValidationError,
    public_artifact_payload,
    validate_public_artifacts,
)

_COMMIT_SHA = "a" * 40
_MINUS = "\N{MINUS SIGN}"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _fixture_repo(tmp_path: Path) -> Path:
    artifact = tmp_path / "artifacts" / "result.json"
    _write_json(artifact, {"result": {"spearman": -0.11281947577195303}})
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "generated_at_utc": "2026-07-10T00:00:00Z",
        "provisional": True,
        "requires_remanifest_after_commit": True,
        "artifacts": [
            {
                "path": "artifacts/result.json",
                "sha256": artifact_sha,
                "source_run_id": "run-1",
                "generation_command": "python generate.py",
                "base_commit_sha": _COMMIT_SHA,
                "code_state": "dirty",
                "code_diff_sha256": "b" * 64,
                "model_id": "esm2_t33_650M",
                "data_sha256": "c" * 64,
                "configuration": {"n": 300},
                "status": "supplementary",
                "evidence_classification": "traceable_not_rerun",
            }
        ],
    }
    _write_json(tmp_path / "artifacts" / "manifest.json", manifest)
    claim_map = {
        "schema_version": 1,
        "forbidden_literals": ["+0.139", "+0.036", "n=150"],
        "claims": [
            {
                "id": "calibration.spearman",
                "document": "README.md",
                "artifact": "artifacts/result.json",
                "json_pointer": "/result/spearman",
                "transform": {"name": "round", "digits": 3, "unicode_minus": True},
                "rendered": f"{_MINUS}0.113",
                "anchor": f"Spearman = {_MINUS}0.113",
            }
        ],
    }
    _write_json(tmp_path / "artifacts" / "claim_map.json", claim_map)
    (tmp_path / "README.md").write_text(f"Spearman = {_MINUS}0.113\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "VALIDATION.md").write_text("Frozen protocol.\n", encoding="utf-8")
    return tmp_path


def test_validate_public_artifacts_accepts_matching_claims(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    validate_public_artifacts(repo)


def test_validate_public_artifacts_rejects_changed_checksum(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    (repo / "artifacts" / "result.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="checksum mismatch"):
        validate_public_artifacts(repo)


def test_validate_public_artifacts_rejects_documented_number_mismatch(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    (repo / "README.md").write_text(f"Spearman = {_MINUS}0.114\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="documented claim missing"):
        validate_public_artifacts(repo)


def test_validate_public_artifacts_rejects_forbidden_historical_number(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    (repo / "docs" / "VALIDATION.md").write_text("Historical result +0.139.\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="forbidden historical literal"):
        validate_public_artifacts(repo)


def test_validate_public_artifacts_rejects_non_allowlisted_transform(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    claim_map_path = repo / "artifacts" / "claim_map.json"
    claim_map = json.loads(claim_map_path.read_text(encoding="utf-8"))
    claim_map["claims"][0]["transform"] = {"name": "python_eval"}
    _write_json(claim_map_path, claim_map)

    with pytest.raises(ValueError, match="transform"):
        validate_public_artifacts(repo)


def test_validate_public_artifacts_rejects_manifest_path_escaping_repo(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    manifest_path = tmp_path / "artifacts" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["path"] = "../outside.json"
    _write_json(manifest_path, manifest)
    outside = tmp_path.parent / "outside.json"
    _write_json(outside, {"result": {}})

    try:
        with pytest.raises(ArtifactValidationError, match="escapes repository"):
            validate_public_artifacts(repo)
    finally:
        outside.unlink(missing_ok=True)


def test_validate_public_artifacts_accepts_identity_and_ratio_transforms(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    artifact_path = tmp_path / "artifacts" / "result.json"
    _write_json(
        artifact_path,
        {
            "result": {
                "spearman": -0.11281947577195303,
                "dataset": "gb1_wu2016",
                "hits": 3,
                "total": 12,
            }
        },
    )
    manifest_path = tmp_path / "artifacts" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    _write_json(manifest_path, manifest)

    claim_map_path = tmp_path / "artifacts" / "claim_map.json"
    claim_map = json.loads(claim_map_path.read_text(encoding="utf-8"))
    claim_map["claims"].append(
        {
            "id": "dataset.name",
            "document": "README.md",
            "artifact": "artifacts/result.json",
            "json_pointer": "/result/dataset",
            "transform": {"name": "identity"},
            "rendered": "gb1_wu2016",
            "anchor": "dataset gb1_wu2016 used",
        }
    )
    claim_map["claims"].append(
        {
            "id": "dataset.hit_ratio",
            "document": "README.md",
            "artifact": "artifacts/result.json",
            "json_pointer": "/result/hits",
            "transform": {
                "name": "ratio",
                "digits": 3,
                "denominator_json_pointer": "/result/total",
            },
            "rendered": "0.250",
            "anchor": "hit ratio 0.250",
        }
    )
    claim_map["claims"].append(
        {
            "id": "dataset.total_grouped",
            "document": "README.md",
            "artifact": "artifacts/result.json",
            "json_pointer": "/result/total",
            "transform": {"name": "grouped"},
            "rendered": "12",
            "anchor": "12 total rows",
        }
    )
    _write_json(claim_map_path, claim_map)
    (tmp_path / "README.md").write_text(
        f"Spearman = {_MINUS}0.113\ndataset gb1_wu2016 used\nhit ratio 0.250\n12 total rows\n",
        encoding="utf-8",
    )

    validate_public_artifacts(repo)


def test_public_artifact_payload_preserves_source_and_checksum(tmp_path: Path) -> None:
    source = tmp_path / "metrics.json"
    _write_json(source, {"spearman": 0.25})

    payload = public_artifact_payload(
        source,
        source_path="report/metrics.json",
        source_run_id="20260710T000000Z",
        evidence_classification="traceable_not_rerun",
    )

    assert payload["result"] == {"spearman": 0.25}
    provenance = payload["provenance"]
    assert provenance["source_path"] == "report/metrics.json"
    assert provenance["source_run_id"] == "20260710T000000Z"
    assert provenance["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_committed_artifacts_use_lf_line_endings() -> None:
    """Manifest checksums are byte-level, so committed artifacts must stay LF across platforms."""
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((repo / "artifacts" / "manifest.json").read_text(encoding="utf-8"))
    offenders = [
        entry["path"]
        for entry in manifest["artifacts"]
        if b"\r" in (repo / entry["path"]).read_bytes()
    ]
    assert not offenders, (
        f"CRLF found in checksummed artifacts {offenders}; "
        "ensure .gitattributes pins artifacts/**/*.json to eol=lf"
    )
