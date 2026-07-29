from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import pytest

from epibudget.epistasis import _orthonormal_contrast_basis
from epibudget.sparsity import summarize_spectrum
from epibudget.spectrum_diagnostic import (
    SCHEMA_VERSION,
    WorkspaceSnapshot,
    _run_diagnostic,
    diagnose_landscape,
    publish_report,
)
from epibudget.types import Variant

_PAIRWISE_ORDER = 2
_N_BINARY_CELLS = 4
_FLOAT_TOL = 1e-15
_SHA_A = "a" * 64
_SHA_B = "b" * 64


def _binary_pair_landscape() -> dict[Variant, float]:
    basis = _orthonormal_contrast_basis(2)
    landscape: dict[Variant, float] = {}
    for first in range(2):
        for second in range(2):
            mutations = set()
            if first:
                mutations.add((0, "A", "C"))
            if second:
                mutations.add((1, "A", "C"))
            landscape[frozenset(mutations)] = float(basis[1, first] * basis[1, second])
    return landscape


def test_summarize_spectrum_recovers_one_pairwise_coefficient() -> None:
    summary = summarize_spectrum(_binary_pair_landscape(), sites=(0, 1))

    pairwise = next(item for item in summary.by_order if item.order == _PAIRWISE_ORDER)
    assert summary.n_cells == _N_BINARY_CELLS
    assert summary.target_variance == pytest.approx(0.25)
    assert summary.nonconstant_variance == pytest.approx(0.25)
    assert summary.parseval_abs_error <= _FLOAT_TOL
    assert pairwise.coefficient_count == 1
    assert pairwise.variance_contribution == pytest.approx(0.25)
    assert pairwise.magnitude_quantiles.maximum == pytest.approx(1.0)
    assert pairwise.effective_counts.k90 == 1
    assert pairwise.effective_counts.k95 == 1
    assert pairwise.effective_counts.k99 == 1
    assert summary.overall_effective_counts.k99 == 1


def test_summarize_spectrum_rejects_non_finite_values() -> None:
    landscape = _binary_pair_landscape()
    landscape[frozenset()] = math.inf

    with pytest.raises(ValueError, match="finite"):
        summarize_spectrum(landscape, sites=(0, 1))


def test_effective_counts_are_invariant_to_uniform_rescaling() -> None:
    landscape = _binary_pair_landscape()
    original = summarize_spectrum(landscape, sites=(0, 1))
    scaled = summarize_spectrum(
        {variant: -3.0 * value for variant, value in landscape.items()}, sites=(0, 1)
    )

    assert scaled.overall_effective_counts == original.overall_effective_counts
    assert scaled.overall_magnitude_quantiles.maximum == pytest.approx(
        3.0 * original.overall_magnitude_quantiles.maximum
    )


def test_constant_landscape_has_zero_effective_counts() -> None:
    landscape = {variant: 2.0 for variant in _binary_pair_landscape()}

    summary = summarize_spectrum(landscape, sites=(0, 1))

    assert summary.nonconstant_variance == pytest.approx(0.0, abs=_FLOAT_TOL)
    assert summary.overall_effective_counts.k90 == 0
    assert summary.overall_effective_counts.k95 == 0
    assert summary.overall_effective_counts.k99 == 0
    assert summary.overall_magnitude_quantiles.maximum == pytest.approx(0.0, abs=_FLOAT_TOL)


def _workspace_snapshot() -> WorkspaceSnapshot:
    return WorkspaceSnapshot(
        execution_commit="1" * 40,
        code_state="clean",
        code_diff_sha256="",
        changed_scientific_files=[],
    )


def test_diagnostic_report_records_non_promotional_provenance() -> None:
    started = datetime(2026, 7, 29, 10, 0, tzinfo=UTC)
    completed = datetime(2026, 7, 29, 10, 1, tzinfo=UTC)
    workspace = _workspace_snapshot()

    report = diagnose_landscape(
        _binary_pair_landscape(),
        sites=(0, 1),
        dataset="synthetic",
        dataset_sha256_start=_SHA_A,
        dataset_sha256_end=_SHA_A,
        expected_rows=_N_BINARY_CELLS,
        unflagged_imputed_values=0,
        imputation_note="Synthetic fixture; no imputed values.",
        process_argv=["spectrum_diagnostic.py", "--data", "fixture.csv"],
        started_utc=started,
        completed_utc=completed,
        workspace_start=workspace,
        workspace_end=workspace,
    )

    assert report.schema_version == SCHEMA_VERSION
    assert report.decision_eligible is False
    assert report.label_transform == "log1p(fitness)"
    assert report.input_state_matches is True
    assert report.workspace_state_matches is True
    assert report.summary.n_cells == _N_BINARY_CELLS


def test_diagnostic_report_records_input_drift() -> None:
    workspace = _workspace_snapshot()
    report = diagnose_landscape(
        _binary_pair_landscape(),
        sites=(0, 1),
        dataset="synthetic",
        dataset_sha256_start=_SHA_A,
        dataset_sha256_end=_SHA_B,
        expected_rows=_N_BINARY_CELLS,
        unflagged_imputed_values=0,
        imputation_note="Synthetic fixture; no imputed values.",
        process_argv=["spectrum_diagnostic.py"],
        started_utc=datetime(2026, 7, 29, 10, 0, tzinfo=UTC),
        completed_utc=datetime(2026, 7, 29, 10, 1, tzinfo=UTC),
        workspace_start=workspace,
        workspace_end=workspace,
    )

    assert report.input_state_matches is False


def test_publish_report_is_create_only(tmp_path: Path) -> None:
    workspace = _workspace_snapshot()
    report = diagnose_landscape(
        _binary_pair_landscape(),
        sites=(0, 1),
        dataset="synthetic",
        dataset_sha256_start=_SHA_A,
        dataset_sha256_end=_SHA_A,
        expected_rows=_N_BINARY_CELLS,
        unflagged_imputed_values=0,
        imputation_note="Synthetic fixture; no imputed values.",
        process_argv=["spectrum_diagnostic.py"],
        started_utc=datetime(2026, 7, 29, 10, 0, tzinfo=UTC),
        completed_utc=datetime(2026, 7, 29, 10, 1, tzinfo=UTC),
        workspace_start=workspace,
        workspace_end=workspace,
    )
    target = tmp_path / "spectrum.json"

    publish_report(report, target)
    first_payload = target.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        publish_report(report, target)

    assert target.read_text(encoding="utf-8") == first_payload


def test_run_diagnostic_captures_input_state_after_spectrum_computation(tmp_path: Path) -> None:
    data_path = tmp_path / "input.csv"
    data_path.write_text("initial", encoding="utf-8")
    output_path = tmp_path / "spectrum.json"

    class MutatingLandscape(dict[Variant, float]):
        def items(self):  # type: ignore[no-untyped-def]
            data_path.write_text("changed during analysis", encoding="utf-8")
            return super().items()

    landscape = MutatingLandscape(_binary_pair_landscape())
    report = _run_diagnostic(
        data_path=data_path,
        out_path=output_path,
        repo=tmp_path,
        process_argv=["spectrum_diagnostic.py"],
        dataset="synthetic",
        loader=lambda _path: landscape,
        sites=(0, 1),
        expected_rows=_N_BINARY_CELLS,
        unflagged_imputed_values=0,
        imputation_note="Synthetic fixture; no imputed values.",
    )

    assert report.input_state_matches is False
    assert output_path.is_file()
