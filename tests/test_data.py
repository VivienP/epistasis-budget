"""Tests for the GB1 loaders and candidate enumeration (docs/SPEC.md#10, Step 1 de-risk gate).

The enumeration tests are pure and offline. The real-landscape loader test is marked ``data``
because it needs ``scripts/fetch_gb1.py`` to have run; the parsing/assertion logic is covered
offline with a tiny synthetic fixture so the WT-residue guard is exercised without any network.
"""

from __future__ import annotations

from math import comb
from pathlib import Path

import pytest

from epibudget.data import (
    GB1_SITES,
    GB1_WT_AT_SITES,
    GB1_WT_SEQUENCE,
    apply_mutations,
    enumerate_candidates,
    load_gb1,
    variant_from_sequence,
    variant_order_composition,
)
from epibudget.types import Variant

_AA20 = "ACDEFGHIKLMNPQRSTVWY"

# Expected candidate counts over the four GB1 sites, 20-letter alphabet (19 non-WT choices/site):
# order-k = C(4, k) * 19**k.
_N_ORDER_1 = 76
_N_ORDER_2 = 2166
_N_ORDER_3 = 27436
_GB1_LANDSCAPE_MIN_ROWS = 100_000  # ~149k measured of the 160k possible four-site variants


def _order(v: Variant) -> int:
    return len(v)


def test_enumerate_counts_by_order_match_combinatorics() -> None:
    # Over n positions with a 20-letter alphabet, order-k variants = C(n,k) * 19**k.
    cands = enumerate_candidates(GB1_SITES, GB1_WT_AT_SITES, allowed_aa=_AA20, max_order=3)
    by_order = {k: sum(1 for v in cands if _order(v) == k) for k in (1, 2, 3)}
    n = len(GB1_SITES)
    assert by_order[1] == comb(n, 1) * 19**1 == _N_ORDER_1
    assert by_order[2] == comb(n, 2) * 19**2 == _N_ORDER_2
    assert by_order[3] == comb(n, 3) * 19**3 == _N_ORDER_3
    assert len(cands) == _N_ORDER_1 + _N_ORDER_2 + _N_ORDER_3


def test_enumerate_respects_max_order() -> None:
    cands = enumerate_candidates(GB1_SITES, GB1_WT_AT_SITES, allowed_aa=_AA20, max_order=2)
    assert {_order(v) for v in cands} == {1, 2}


def test_enumerate_never_emits_the_wt_residue_and_stays_on_sites() -> None:
    cands = enumerate_candidates(GB1_SITES, GB1_WT_AT_SITES, allowed_aa=_AA20, max_order=3)
    wt_of = dict(zip(GB1_SITES, GB1_WT_AT_SITES, strict=True))
    for v in cands:
        for pos, wt_aa, mut_aa in v:
            assert pos in wt_of
            assert wt_aa == wt_of[pos]  # mutation carries the correct WT residue
            assert mut_aa != wt_aa  # never a synonymous "mutation"
            assert mut_aa in _AA20


def test_enumerate_produces_unique_variants() -> None:
    cands = enumerate_candidates(GB1_SITES, GB1_WT_AT_SITES, allowed_aa=_AA20, max_order=3)
    assert len(set(cands)) == len(cands)


def test_enumerate_rejects_mismatched_positions_and_residues() -> None:
    with pytest.raises(ValueError, match="length"):
        enumerate_candidates((38, 39), ("V",), max_order=2)


def test_enumerate_rejects_max_order_above_site_count() -> None:
    with pytest.raises(ValueError, match="max_order"):
        enumerate_candidates((38, 39), ("V", "D"), max_order=3)


def test_variant_from_sequence_diffs_against_wt() -> None:
    v = frozenset({(38, "V", "A"), (40, "G", "W")})
    seq = apply_mutations(GB1_WT_SEQUENCE, v)
    assert variant_from_sequence(seq) == v
    assert variant_from_sequence(GB1_WT_SEQUENCE) == frozenset()  # WT diffs to the empty variant


def test_variant_from_sequence_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="length"):
        variant_from_sequence("MTY")  # not 56 residues


def test_variant_order_composition_counts_by_order() -> None:
    landscape: dict[Variant, float] = {
        frozenset(): 1.0,
        frozenset({(38, "V", "A")}): 0.5,
        frozenset({(39, "D", "C")}): 0.4,
        frozenset({(38, "V", "A"), (39, "D", "C")}): 0.2,
    }
    assert variant_order_composition(landscape) == {0: 1, 1: 2, 2: 1}


def _write_csv(path: Path, rows: list[tuple[str, float]]) -> None:
    lines = ["protein,label"] + [f"{seq},{label}" for seq, label in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_gb1_parses_sequences_into_variants(tmp_path: Path) -> None:
    wt = GB1_WT_SEQUENCE
    f_wt, f_single, f_double = 1.0, 0.5, 0.25
    single = apply_mutations(wt, frozenset({(38, "V", "A")}))
    double = apply_mutations(wt, frozenset({(38, "V", "A"), (39, "D", "C")}))
    csv = tmp_path / "toy.csv"
    _write_csv(csv, [(wt, f_wt), (single, f_single), (double, f_double)])

    landscape = load_gb1(csv)
    assert landscape[frozenset()] == f_wt  # wild type
    assert landscape[frozenset({(38, "V", "A")})] == f_single
    assert landscape[frozenset({(38, "V", "A"), (39, "D", "C")})] == f_double


def test_load_gb1_rejects_off_site_mutation(tmp_path: Path) -> None:
    # A mutation at position 5 (outside the four sites) must be caught, not silently accepted.
    off_site = apply_mutations(GB1_WT_SEQUENCE, frozenset({(5, GB1_WT_SEQUENCE[5], "A")}))
    csv = tmp_path / "bad.csv"
    _write_csv(csv, [(GB1_WT_SEQUENCE, 1.0), (off_site, 0.5)])
    with pytest.raises(ValueError, match="off-target"):
        load_gb1(csv)


def test_load_gb1_requires_wild_type_row(tmp_path: Path) -> None:
    single = apply_mutations(GB1_WT_SEQUENCE, frozenset({(38, "V", "A")}))
    csv = tmp_path / "nowt.csv"
    _write_csv(csv, [(single, 0.5)])
    with pytest.raises(ValueError, match="wild type"):
        load_gb1(csv)


@pytest.mark.data
def test_gb1_loads_complete_landscape() -> None:
    """Real-data gate: the fetched GB1 four-site landscape loads and the WT residues assert.

    Requires ``python scripts/fetch_gb1.py`` to have populated data/proteingym/.
    """
    landscape = load_gb1(Path("data/proteingym/gb1_wu2016.csv"))
    assert len(landscape) > _GB1_LANDSCAPE_MIN_ROWS
    wt = frozenset()  # the wild type is the empty variant
    assert wt in landscape
