"""Offline tests for the public-artifact dataset payload (no ESM-2, no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_public_artifacts import _dataset_payload

from epibudget.data import GB1_WT_SEQUENCE

_THEORETICAL_GENOTYPES = 160000
_MEASURED_ROWS = 4
_LIVE_ROWS = 3  # WT + single_live + double_live


def _mutate(seq: str, position: int, aa: str) -> str:
    chars = list(seq)
    chars[position] = aa
    return "".join(chars)


def _write_csv(path: Path, rows: list[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["protein,label"] + [f"{seq},{label}" for seq, label in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_dataset_payload_counts_live_and_dead_by_order(tmp_path: Path) -> None:
    repo = tmp_path
    csv_path = repo / "data" / "proteingym" / "gb1_wu2016.csv"
    wt = GB1_WT_SEQUENCE
    single_live = _mutate(wt, 38, "A")
    single_dead = _mutate(wt, 39, "A")
    double_live = _mutate(_mutate(wt, 38, "A"), 40, "C")
    _write_csv(
        csv_path,
        [
            (wt, 1.0),
            (single_live, 0.5),
            (single_dead, 0.0),
            (double_live, 0.3),
        ],
    )

    payload = _dataset_payload(repo)

    assert payload["measured_rows"] == _MEASURED_ROWS
    assert payload["live_rows"] == _LIVE_ROWS
    assert payload["dead_rows"] == 1
    assert payload["missing_rows"] == _THEORETICAL_GENOTYPES - _MEASURED_ROWS
    by_order = payload["by_order"]
    assert by_order["0"] == {"theoretical": 1, "live": 1, "dead": 0, "missing": 0}
    assert by_order["1"]["live"] == 1
    assert by_order["1"]["dead"] == 1
    assert by_order["2"]["live"] == 1


def test_dataset_payload_rejects_off_site_mutation(tmp_path: Path) -> None:
    repo = tmp_path
    csv_path = repo / "data" / "proteingym" / "gb1_wu2016.csv"
    wt = GB1_WT_SEQUENCE
    off_site = _mutate(wt, 0, "A")  # position 0 is not one of GB1_SITES
    _write_csv(csv_path, [(wt, 1.0), (off_site, 0.5)])

    with pytest.raises(ValueError, match="off-target"):
        _dataset_payload(repo)
