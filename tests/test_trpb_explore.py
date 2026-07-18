"""Tests for the exploratory TrpB profiler (src/epibudget/trpb_explore.py).

Offline and synthetic: a tiny TrpB-shaped CSV is built from the frozen TRPB_WT_SEQUENCE with
apply_mutations — no network, no ESM, no copy of the ~160k-row external data. The tests assert the
exploratory contract — duplicate/conflict/missing/invalid classification, order-independence,
checksum stability, and that nothing here assumes exactly four mutated positions or a GB1 shape.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from epibudget.data import (
    TRPB_SITES,
    TRPB_WT_AT_SITES,
    TRPB_WT_SEQUENCE,
    apply_mutations,
)
from epibudget.trpb_explore import (
    CANONICAL_ALPHABET,
    MAX_SELECTION_ORDER,
    RUN_TYPE,
    build_profile,
    canonical_variant_id,
    profile_trpb,
    read_raw_records,
    sha256_file,
)
from epibudget.types import Variant

_P0, _P1, _P2, _P3 = TRPB_SITES
_WT = TRPB_WT_SEQUENCE


def _seq(variant: Variant) -> str:
    return apply_mutations(_WT, variant)


def _single(pos: int, mut: str) -> Variant:
    return frozenset({(pos, _WT[pos], mut)})


def _write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    """Write a protein/label CSV; an empty label string encodes a missing measurement."""
    lines = ["protein,label"] + [f"{seq},{label}" for seq, label in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _valid_rows() -> list[tuple[str, str]]:
    """WT + one single/double/triple/quadruple — the full order 0..4 shape of TrpB."""
    single = _single(_P0, "A")
    double = frozenset({(_P0, _WT[_P0], "A"), (_P1, _WT[_P1], "G")})
    triple = double | {(_P2, _WT[_P2], "L")}
    quad = triple | {(_P3, _WT[_P3], "T")}
    return [
        (_WT, "0.41"),
        (_seq(single), "0.9"),
        (_seq(double), "0.2"),
        (_seq(triple), "-0.3"),  # inactive: TrpB marks it with a non-positive label
        (_seq(quad), "1.1"),  # order 4: outside the order-1..3 selection universe
    ]


# --- canonical encoding / order ------------------------------------------------------------------


def test_canonical_variant_id_is_order_independent() -> None:
    a = frozenset({(_P0, _WT[_P0], "A"), (_P1, _WT[_P1], "G")})
    b = frozenset({(_P1, _WT[_P1], "G"), (_P0, _WT[_P0], "A")})
    assert canonical_variant_id(a) == canonical_variant_id(b)
    assert canonical_variant_id(frozenset()) == "WT"


def test_read_records_preserve_file_order(tmp_path: Path) -> None:
    rows = _valid_rows()
    csv = tmp_path / "trpb.csv"
    _write_csv(csv, rows)
    records = read_raw_records(csv)
    assert [r.row_index for r in records] == list(range(len(rows)))


# --- valid loading / WT / order ------------------------------------------------------------------


def test_profile_valid_dataset(tmp_path: Path) -> None:
    csv = tmp_path / "trpb.csv"
    _write_csv(csv, _valid_rows())
    profile = profile_trpb(csv)

    assert profile.run_type == RUN_TYPE
    assert profile.decision_eligible is False
    assert profile.n_rows == len(_valid_rows())
    assert profile.n_measured == len(_valid_rows())  # every valid row carries a label
    assert profile.n_invalid_records == 0


def test_wild_type_identified_regardless_of_position(tmp_path: Path) -> None:
    rows = _valid_rows()
    rows_wt_last = rows[1:] + rows[:1]  # move the WT row to the end
    csv = tmp_path / "trpb.csv"
    _write_csv(csv, rows_wt_last)
    profile = profile_trpb(csv)
    assert profile.wt_present is True
    assert profile.wt_label == pytest.approx(0.41)
    assert profile.wt_canonical_id == "WT"


def test_mutation_order_counts(tmp_path: Path) -> None:
    csv = tmp_path / "trpb.csv"
    _write_csv(csv, _valid_rows())
    profile = profile_trpb(csv)
    assert profile.order_distribution == {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
    assert (profile.n_singles, profile.n_doubles, profile.n_triples, profile.n_quadruples) == (
        1,
        1,
        1,
        1,
    )


def test_order4_counted_beyond_selection_universe(tmp_path: Path) -> None:
    csv = tmp_path / "trpb.csv"
    _write_csv(csv, _valid_rows())
    profile = profile_trpb(csv)
    # The order-4 quadruple is measured but sits outside the order-1..3 selection universe.
    assert profile.coverage.max_selection_order == MAX_SELECTION_ORDER
    assert profile.coverage.n_beyond_selection_order_measured == 1
    # single/double/triple are inside the universe and measured; the inactive (<=0) triple counts
    # as measured but not as positive coverage, so 3 measured of which 2 are positive.
    assert (
        profile.coverage.n_universe_measured,
        profile.coverage.n_universe_measured_positive,
    ) == (3, 2)


# --- sequence-length / invalid AA / off-site -----------------------------------------------------


def test_wrong_length_classified_not_raised(tmp_path: Path) -> None:
    csv = tmp_path / "trpb.csv"
    _write_csv(csv, [(_WT, "0.41"), ("MKGYFG", "0.5")])
    records = read_raw_records(csv)
    assert records[1].status == "wrong_length"
    assert records[1].variant is None
    profile = build_profile(records, csv, TRPB_SITES, TRPB_WT_AT_SITES, CANONICAL_ALPHABET)
    assert profile.status_counts["wrong_length"] == 1
    assert profile.n_invalid_records == 1


def test_invalid_amino_acid_classified(tmp_path: Path) -> None:
    bad = apply_mutations(_WT, frozenset({(_P0, _WT[_P0], "X")}))  # X is not a standard residue
    csv = tmp_path / "trpb.csv"
    _write_csv(csv, [(_WT, "0.41"), (bad, "0.5")])
    records = read_raw_records(csv)
    assert records[1].status == "invalid_amino_acid"
    assert records[1].variant is None
    assert "X" in records[1].detail


def test_off_site_mutation_classified(tmp_path: Path) -> None:
    off = apply_mutations(_WT, frozenset({(5, _WT[5], "A")}))  # position 5 is not a TrpB site
    csv = tmp_path / "trpb.csv"
    _write_csv(csv, [(_WT, "0.41"), (off, "0.5")])
    records = read_raw_records(csv)
    assert records[1].status == "off_site_mutation"
    assert "5" in records[1].detail


# --- missing labels ------------------------------------------------------------------------------


def test_missing_label_preserved_not_dropped(tmp_path: Path) -> None:
    single = _seq(_single(_P0, "A"))
    csv = tmp_path / "trpb.csv"
    _write_csv(csv, [(_WT, "0.41"), (single, "")])  # empty label = missing measurement
    records = read_raw_records(csv)
    assert records[1].status == "missing_label"
    assert records[1].variant is not None  # the genotype is still recovered
    assert records[1].label is None
    profile = build_profile(records, csv, TRPB_SITES, TRPB_WT_AT_SITES, CANONICAL_ALPHABET)
    assert profile.n_missing_label == 1
    assert profile.n_measured == 1  # only the WT has a label
    assert profile.fitness.n == 1


# --- duplicates: identical vs conflicting --------------------------------------------------------


def test_identical_duplicate_rows(tmp_path: Path) -> None:
    single = _seq(_single(_P0, "A"))
    csv = tmp_path / "trpb.csv"
    _write_csv(csv, [(_WT, "0.41"), (single, "0.9"), (single, "0.9")])
    profile = profile_trpb(csv)
    assert profile.duplicates.n_duplicate_variants == 1
    assert profile.duplicates.n_identical_duplicate_variants == 1
    assert profile.duplicates.n_conflicting_duplicate_variants == 0


def test_conflicting_duplicate_measurements(tmp_path: Path) -> None:
    single = _seq(_single(_P0, "A"))
    csv = tmp_path / "trpb.csv"
    _write_csv(csv, [(_WT, "0.41"), (single, "0.9"), (single, "0.2")])  # disagree on the label
    profile = profile_trpb(csv)
    assert profile.duplicates.n_conflicting_duplicate_variants == 1
    assert profile.duplicates.n_identical_duplicate_variants == 0
    assert canonical_variant_id(_single(_P0, "A")) in profile.duplicates.conflicting_samples


def test_present_vs_missing_duplicate_is_conflicting(tmp_path: Path) -> None:
    single = _seq(_single(_P0, "A"))
    csv = tmp_path / "trpb.csv"
    _write_csv(csv, [(_WT, "0.41"), (single, "0.9"), (single, "")])  # one measured, one missing
    profile = profile_trpb(csv)
    assert profile.duplicates.n_conflicting_duplicate_variants == 1


# --- determinism / order independence / checksum -------------------------------------------------


def test_build_profile_is_invariant_to_record_order(tmp_path: Path) -> None:
    csv = tmp_path / "trpb.csv"
    _write_csv(csv, [*_valid_rows(), (_seq(_single(_P1, "W")), "0.7")])
    records = read_raw_records(csv)
    forward = build_profile(records, csv, TRPB_SITES, TRPB_WT_AT_SITES, CANONICAL_ALPHABET)
    reverse = build_profile(
        list(reversed(records)), csv, TRPB_SITES, TRPB_WT_AT_SITES, CANONICAL_ALPHABET
    )
    assert forward.model_dump() == reverse.model_dump()  # byte-identical, not just approximately


def test_checksum_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    c = tmp_path / "c.csv"
    _write_csv(a, _valid_rows())
    _write_csv(b, _valid_rows())  # identical content
    _write_csv(c, _valid_rows()[:-1])  # one row fewer
    assert sha256_file(a) == sha256_file(b)
    assert sha256_file(a) != sha256_file(c)
    assert profile_trpb(a).dataset_checksum_sha256 == sha256_file(a)


# --- structural guards ---------------------------------------------------------------------------


def test_missing_required_column_raises(tmp_path: Path) -> None:
    csv = tmp_path / "bad.csv"
    csv.write_text("protein,fitness\n" + _WT + ",0.41\n", encoding="utf-8")  # 'label' absent
    with pytest.raises(ValueError, match="expected columns"):
        read_raw_records(csv)


def test_wrong_reference_construct_raises(tmp_path: Path) -> None:
    csv = tmp_path / "trpb.csv"
    _write_csv(csv, [(_WT, "0.41")])
    wrong = ("A", *TRPB_WT_AT_SITES[1:])  # claim 'A' where the Tm9D8* parent has 'V'
    with pytest.raises(ValueError, match="reference residue at site"):
        read_raw_records(csv, _WT, TRPB_SITES, wrong)


# --- no assumption of exactly four positions -----------------------------------------------------


def test_profile_handles_non_four_site_landscape(tmp_path: Path) -> None:
    # A two-site landscape over the same WT: the profiler must not assume four positions, and the
    # coverage universe must cap the order at the number of sites (min(3, 2) == 2).
    two_sites = (_P0, _P1)
    two_wt = (_WT[_P0], _WT[_P1])
    single = frozenset({(_P0, _WT[_P0], "A")})
    double = frozenset({(_P0, _WT[_P0], "A"), (_P1, _WT[_P1], "G")})
    csv = tmp_path / "two.csv"
    _write_csv(csv, [(_WT, "1.0"), (_seq(single), "0.5"), (_seq(double), "0.2")])
    profile = profile_trpb(csv, sites=two_sites, wt_at_sites=two_wt)
    assert profile.coverage.max_selection_order == len(two_sites)  # min(3, 2) == 2
    assert profile.coverage.universe_size == 2 * 19 + 19 * 19  # order-1 + order-2 over 2 sites
    assert profile.n_singles == 1
    assert profile.n_doubles == 1
    assert profile.coverage.n_beyond_selection_order_measured == 0


# --- repo hygiene: raw data stays git-ignored ----------------------------------------------------


def test_gitignore_excludes_downloaded_trpb_data() -> None:
    gitignore = Path(__file__).resolve().parents[1] / ".gitignore"
    text = gitignore.read_text(encoding="utf-8")
    # The fetch script writes data/proteingym/trpb_johnston2024.csv; it must never be committable.
    assert "data/proteingym/" in text


def test_raw_record_is_frozen(tmp_path: Path) -> None:
    csv = tmp_path / "trpb.csv"
    _write_csv(csv, _valid_rows())
    records = read_raw_records(csv)
    with pytest.raises(dataclasses.FrozenInstanceError):
        records[0].status = "ok"  # type: ignore[misc]  # frozen dataclass rejects mutation
