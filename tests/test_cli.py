"""Offline tests for the CLI's pure helpers (FASTA reading, variant parsing).

The ``score`` command itself needs an ESM-2 forward pass, so it is not exercised here; only the
input-parsing helpers are, which is where the surprising failure modes (indexing, WT mismatch) live.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from epibudget import scoring
from epibudget.cli import app, parse_variant, read_fasta_sequence, read_variant_specs
from epibudget.data import GB1_WT_SEQUENCE
from epibudget.types import ScoredVariant, Variant


def test_read_fasta_sequence_concatenates_body_and_drops_header(tmp_path: Path) -> None:
    f = tmp_path / "wt.fasta"
    f.write_text(">gb1 B1 domain\nMTYKLILNGK\nTLKGETTTEA\n", encoding="utf-8")
    assert read_fasta_sequence(f) == "MTYKLILNGKTLKGETTTEA"


def test_read_fasta_sequence_rejects_empty(tmp_path: Path) -> None:
    f = tmp_path / "empty.fasta"
    f.write_text(">only a header\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no sequence"):
        read_fasta_sequence(f)


def test_parse_variant_single_and_multi() -> None:
    # 1-indexed DMS notation; V39 is 0-indexed position 38 in the GB1 WT.
    v = parse_variant("V39A", GB1_WT_SEQUENCE)
    assert v == frozenset({(38, "V", "A")})
    v2 = parse_variant("V39A D40C", GB1_WT_SEQUENCE)
    assert v2 == frozenset({(38, "V", "A"), (39, "D", "C")})


def test_parse_variant_accepts_common_separators() -> None:
    expected = frozenset({(38, "V", "A"), (39, "D", "C")})
    for spec in ("V39A;D40C", "V39A,D40C", "V39A+D40C", "V39A  D40C"):
        assert parse_variant(spec, GB1_WT_SEQUENCE) == expected


def test_parse_variant_empty_or_wt_is_wild_type() -> None:
    assert parse_variant("", GB1_WT_SEQUENCE) == frozenset()
    assert parse_variant("WT", GB1_WT_SEQUENCE) == frozenset()


def test_parse_variant_rejects_wt_letter_mismatch() -> None:
    # WT at position 39 (1-indexed) is V, not Q.
    with pytest.raises(ValueError, match="WT mismatch"):
        parse_variant("Q39A", GB1_WT_SEQUENCE)


def test_parse_variant_rejects_synonymous() -> None:
    with pytest.raises(ValueError, match="synonymous"):
        parse_variant("V39V", GB1_WT_SEQUENCE)


def test_parse_variant_rejects_out_of_range_position() -> None:
    with pytest.raises(ValueError, match="out of range"):
        parse_variant("V999A", GB1_WT_SEQUENCE)


def test_parse_variant_rejects_unknown_residue() -> None:
    with pytest.raises(ValueError, match="not a valid amino acid"):
        parse_variant("V39Z", GB1_WT_SEQUENCE)


def test_parse_variant_rejects_malformed_token() -> None:
    with pytest.raises(ValueError, match="malformed"):
        parse_variant("hello", GB1_WT_SEQUENCE)


def test_read_variant_specs_with_header(tmp_path: Path) -> None:
    f = tmp_path / "variants.csv"
    f.write_text("variant\nV39A\nV39A D40C\n\n", encoding="utf-8")
    assert read_variant_specs(f) == ["V39A", "V39A D40C"]


def test_read_variant_specs_headerless(tmp_path: Path) -> None:
    f = tmp_path / "variants.csv"
    f.write_text("V39A\nD40C G41W\n", encoding="utf-8")
    assert read_variant_specs(f) == ["V39A", "D40C G41W"]


class _FakeScorer:
    """Stand-in for ConjointScorer: no ESM-2 forward pass, deterministic stub scores."""

    def __init__(self, model_id: str, n_perturbations: int = 16, seed: int = 0) -> None:
        self.model_id = model_id

    def score_batch(self, wt: str, variants: Sequence[Variant]) -> list[ScoredVariant]:
        return [
            ScoredVariant(variant=v, delta_g=0.1 * i, var_delta_g=0.01)
            for i, v in enumerate(variants)
        ]


def test_score_command_runs_offline_with_a_stubbed_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fasta = tmp_path / "wt.fasta"
    fasta.write_text(f">gb1\n{GB1_WT_SEQUENCE}\n", encoding="utf-8")
    variants_file = tmp_path / "variants.csv"
    variants_file.write_text("V39A\nV39A D40C\n", encoding="utf-8")

    monkeypatch.setattr(scoring, "ConjointScorer", _FakeScorer)

    result = CliRunner().invoke(
        app, ["score", "--fasta", str(fasta), "--variants", str(variants_file)]
    )
    assert result.exit_code == 0, result.output
    assert "single" in result.output  # one row per spec, labelled by order
    assert "double" in result.output
