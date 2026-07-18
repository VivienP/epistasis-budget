"""Fetch the TrpB four-site combinatorially complete landscape (Johnston et al. 2024, PNAS).

A SECOND validation landscape alongside GB1 — enzyme catalysis (β-subunit of tryptophan synthase)
rather than GB1's IgG-Fc binding, so a shared recovery across both is an independent generalization
claim. Data is written to data/ (git-ignored) and NEVER committed; this script records provenance
(source URL, download date, sha256, row count, reference sequence, order composition). See
docs/VALIDATION.md for the (deferred) TrpB protocol.

Usage:
    python scripts/fetch_trpb.py [--out data/proteingym]

Source. The combinatorially complete 20^4 = 160,000-variant landscape at active-site positions
183/184/227/228 of Tm9D8* (a thermostable TmTrpB variant; parent genotype "VFVS") is mirrored on the
Hugging Face Hub as ``SeprotHub/Dataset-TrpB_fitness_landsacpe`` (the source's literal spelling).
Variants are full 397-residue sequences, so the genotype is recovered by diffing against the parent.
Label = an aggregated catalytic-fitness score (Kowalsky et al.); <= 0 is inactive (like a dead row).

Honesty note. Per the paper, 159,129 of the 160,000 variants (99.45%) had sufficient sequencing
coverage to be measured; the remaining 871 (0.54%) were IMPUTED for downstream analyses. The Hugging
Face redistribution ships all 160,000 rows and does NOT flag which are imputed, so this fetch cannot
separate them — any TrpB result must state that ~0.5% of fitness values are imputed, not measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from _console import configure_utf8_stdout

from epibudget.data import (
    TRPB_SITES,
    TRPB_WT_AT_SITES,
    TRPB_WT_SEQUENCE,
    load_trpb,
    variant_order_composition,
)

SOURCE_URL = "https://huggingface.co/datasets/SeprotHub/Dataset-TrpB_fitness_landsacpe/resolve/main/dataset.csv"
EXPECTED_ROWS = (
    160_000  # full 20^4 combinatorial space (159,129 measured + 871 imputed, per the paper)
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/proteingym"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    csv_path = args.out / "trpb_johnston2024.csv"
    print(f"Downloading TrpB (Johnston-2024) landscape from {SOURCE_URL}")
    urllib.request.urlretrieve(SOURCE_URL, csv_path)

    checksum = _sha256(csv_path)
    size_bytes = csv_path.stat().st_size

    # Load through the real loader so the fetch validates exactly what validation will read: the
    # reference-residue asserts and the on-sites guard run here, failing loudly on a wrong parent.
    landscape = load_trpb(csv_path)
    composition = variant_order_composition(landscape)
    n_rows = len(landscape)

    if n_rows != EXPECTED_ROWS:
        print(f"  WARNING: expected {EXPECTED_ROWS} genotypes, loaded {n_rows} (source changed?)")

    provenance = {
        "source_url": SOURCE_URL,
        "source_dataset": (
            "SeprotHub/Dataset-TrpB_fitness_landsacpe "
            "(Johnston et al. 2024, PNAS 121(32) e2400439121)"
        ),
        "downloaded_utc": datetime.now(UTC).isoformat(),
        "file": csv_path.name,
        "sha256": checksum,
        "size_bytes": size_bytes,
        "n_genotypes": n_rows,
        "reference_sequence": TRPB_WT_SEQUENCE,
        "reference_note": "Tm9D8* parent = VFVS at sites 183/184/227/228 (1-indexed)",
        "sites_0indexed": list(TRPB_SITES),
        "wt_at_sites": list(TRPB_WT_AT_SITES),
        "order_composition": composition,
        "label_semantics": (
            "aggregated catalytic fitness (Kowalsky et al.); <= 0 is inactive. ~871 of 160,000 "
            "values are imputed (not measured) per the paper and are not flagged in this mirror."
        ),
    }
    prov_path = args.out / "provenance_trpb.json"
    prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(f"  saved   {csv_path}  ({size_bytes:,} bytes)")
    print(f"  sha256  {checksum}")
    print(f"  rows    {n_rows:,}")
    print(f"  orders  {composition}")
    print(f"  prov    {prov_path}")


if __name__ == "__main__":
    main()
