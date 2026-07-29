"""Provenance-complete reporting for the TrpB Fourier-spectrum diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from epibudget.data import resolve_dataset
from epibudget.labels import training_target
from epibudget.provenance import (
    changed_scientific_files,
    workspace_code_diff_sha256,
    write_json_atomic,
)
from epibudget.sparsity import SpectrumSummary, summarize_spectrum
from epibudget.types import Variant

SCHEMA_VERSION: Literal["epibudget-spectrum-diagnostic-v1"] = "epibudget-spectrum-diagnostic-v1"
_DATASET = "trpb_johnston2024"
_EXPECTED_ROWS = 160_000
_UNFLAGGED_IMPUTED_VALUES = 871
_IMPUTATION_NOTE = (
    "The redistributed TrpB target contains 159,129 measured values and 871 source-imputed values; "
    "the mirror does not identify the imputed rows."
)


class WorkspaceSnapshot(BaseModel):
    """Repository state captured at one process boundary."""

    model_config = ConfigDict(frozen=True)

    execution_commit: str
    code_state: Literal["clean", "dirty", "unavailable"]
    code_diff_sha256: str
    changed_scientific_files: list[str]


class SpectrumDiagnosticReport(BaseModel):
    """Validated, non-promotional output of the complete-grid spectrum diagnostic."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    schema_version: Literal["epibudget-spectrum-diagnostic-v1"] = SCHEMA_VERSION
    decision_eligible: Literal[False] = False
    dataset: str
    label_transform: Literal["log1p(fitness)"] = "log1p(fitness)"
    row_count: int = Field(ge=1)
    expected_row_count: int = Field(ge=1)
    unflagged_imputed_values: int = Field(ge=0)
    imputation_note: str
    dataset_sha256_start: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_sha256_end: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_state_matches: bool
    process_argv: list[str]
    started_utc: datetime
    completed_utc: datetime
    workspace_start: WorkspaceSnapshot
    workspace_end: WorkspaceSnapshot
    workspace_state_matches: bool
    summary: SpectrumSummary


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_lines(repo: Path, *args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return [line for line in result.stdout.splitlines() if line]


def _capture_workspace_snapshot(repo: Path) -> WorkspaceSnapshot:
    """Capture commit and scientific working-tree state without mutating the repository."""
    try:
        head = _git_lines(repo, "rev-parse", "HEAD")
        if len(head) != 1:
            raise ValueError("git rev-parse returned no unique commit")
        commit = head[0]
        dirty = bool(_git_lines(repo, "status", "--porcelain"))
        changed = changed_scientific_files(repo, commit) if dirty else []
        diff_hash = workspace_code_diff_sha256(repo, commit) if dirty else ""
    except (OSError, subprocess.CalledProcessError, ValueError):
        return WorkspaceSnapshot(
            execution_commit="",
            code_state="unavailable",
            code_diff_sha256="",
            changed_scientific_files=[],
        )
    return WorkspaceSnapshot(
        execution_commit=commit,
        code_state="dirty" if dirty else "clean",
        code_diff_sha256=diff_hash,
        changed_scientific_files=changed,
    )


def diagnose_landscape(
    landscape: Mapping[Variant, float],
    *,
    sites: Sequence[int],
    dataset: str,
    dataset_sha256_start: str,
    dataset_sha256_end: str,
    expected_rows: int,
    unflagged_imputed_values: int,
    imputation_note: str,
    process_argv: list[str],
    started_utc: datetime,
    completed_utc: datetime,
    workspace_start: WorkspaceSnapshot,
    workspace_end: WorkspaceSnapshot,
) -> SpectrumDiagnosticReport:
    """Transform finite labels and build one provenance-complete spectrum report."""
    if len(landscape) != expected_rows:
        raise ValueError(f"expected {expected_rows} unique genotypes, got {len(landscape)}")
    transformed = {variant: training_target(value) for variant, value in landscape.items()}
    summary = summarize_spectrum(transformed, sites)
    return _build_report(
        summary=summary,
        row_count=len(landscape),
        dataset=dataset,
        dataset_sha256_start=dataset_sha256_start,
        dataset_sha256_end=dataset_sha256_end,
        expected_rows=expected_rows,
        unflagged_imputed_values=unflagged_imputed_values,
        imputation_note=imputation_note,
        process_argv=process_argv,
        started_utc=started_utc,
        completed_utc=completed_utc,
        workspace_start=workspace_start,
        workspace_end=workspace_end,
    )


def _build_report(
    *,
    summary: SpectrumSummary,
    row_count: int,
    dataset: str,
    dataset_sha256_start: str,
    dataset_sha256_end: str,
    expected_rows: int,
    unflagged_imputed_values: int,
    imputation_note: str,
    process_argv: list[str],
    started_utc: datetime,
    completed_utc: datetime,
    workspace_start: WorkspaceSnapshot,
    workspace_end: WorkspaceSnapshot,
) -> SpectrumDiagnosticReport:
    return SpectrumDiagnosticReport(
        dataset=dataset,
        row_count=row_count,
        expected_row_count=expected_rows,
        unflagged_imputed_values=unflagged_imputed_values,
        imputation_note=imputation_note,
        dataset_sha256_start=dataset_sha256_start,
        dataset_sha256_end=dataset_sha256_end,
        input_state_matches=dataset_sha256_start == dataset_sha256_end,
        process_argv=process_argv,
        started_utc=started_utc,
        completed_utc=completed_utc,
        workspace_start=workspace_start,
        workspace_end=workspace_end,
        workspace_state_matches=workspace_start == workspace_end,
        summary=summary,
    )


def publish_report(report: SpectrumDiagnosticReport, path: Path) -> None:
    """Publish a complete report atomically and refuse to replace an existing file."""
    write_json_atomic(path, report.model_dump(mode="json"))


def _run_diagnostic(
    *,
    data_path: Path,
    out_path: Path,
    repo: Path,
    process_argv: list[str],
    dataset: str,
    loader: Callable[[Path], Mapping[Variant, float]],
    sites: Sequence[int],
    expected_rows: int,
    unflagged_imputed_values: int,
    imputation_note: str,
) -> SpectrumDiagnosticReport:
    started_utc = datetime.now(UTC)
    workspace_start = _capture_workspace_snapshot(repo)
    dataset_sha256_start = _sha256_file(data_path)
    landscape = loader(data_path)
    if len(landscape) != expected_rows:
        raise ValueError(f"expected {expected_rows} unique genotypes, got {len(landscape)}")
    transformed = {variant: training_target(value) for variant, value in landscape.items()}
    summary = summarize_spectrum(transformed, sites)
    dataset_sha256_end = _sha256_file(data_path)
    workspace_end = _capture_workspace_snapshot(repo)
    completed_utc = datetime.now(UTC)
    report = _build_report(
        summary=summary,
        row_count=len(landscape),
        dataset=dataset,
        dataset_sha256_start=dataset_sha256_start,
        dataset_sha256_end=dataset_sha256_end,
        expected_rows=expected_rows,
        unflagged_imputed_values=unflagged_imputed_values,
        imputation_note=imputation_note,
        process_argv=process_argv,
        started_utc=started_utc,
        completed_utc=completed_utc,
        workspace_start=workspace_start,
        workspace_end=workspace_end,
    )
    publish_report(report, out_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/proteingym/trpb_johnston2024.csv"))
    parser.add_argument("--out", type=Path, default=Path("report/diagnostics/spectrum_trpb.json"))
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent.parent
    specification = resolve_dataset(_DATASET)
    _run_diagnostic(
        data_path=args.data,
        out_path=args.out,
        repo=repo,
        process_argv=[sys.executable, *sys.argv],
        dataset=_DATASET,
        loader=specification.loader,
        sites=specification.sites,
        expected_rows=_EXPECTED_ROWS,
        unflagged_imputed_values=_UNFLAGGED_IMPUTED_VALUES,
        imputation_note=_IMPUTATION_NOTE,
    )
    print(args.out)
