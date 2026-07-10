"""Fetch the measured GB1 four-site dataset (Wu et al. 2016, eLife) for validation.

Data is written to data/ (git-ignored) and NEVER committed. This script records provenance: source
URL, download date, sha256 checksum, row count, WT sequence, and the mutation-order composition of
the landscape. See docs/VALIDATION.md.

Usage:
    python scripts/fetch_gb1.py [--out data/proteingym]

Source. A 149,361-genotype measured subset of the theoretical 20^4 four-site space at
V39/D40/G41/V54 is mirrored on the
Hugging Face Hub as ``SaProtHub/Dataset-GB1-fitness`` (a faithful redistribution of Wu-2016; label =
fitness relative to the wild type, WT = 1.0). Variants are stored as full 56-residue sequences, so
the genotype is recovered by diffing each sequence against the wild type — robust to any
mutant-string formatting quirk.

The official ProteinGym Hugging Face mirror (``OATML-Markslab/ProteinGym``) does NOT contain this
assay (it ships GB1's Olson-2014 pairwise set only); do not point this fetch there.
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

from epibudget.data import (
    GB1_SITES,
    GB1_WT_AT_SITES,
    GB1_WT_SEQUENCE,
    load_gb1,
    variant_order_composition,
)

SOURCE_URL = (
    "https://huggingface.co/datasets/SaProtHub/Dataset-GB1-fitness/resolve/main/dataset.csv"
)
EXPECTED_ROWS = 149_361  # Wu-2016 measured genotypes (of the 160,000 = 20^4 combinatorial space)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/proteingym"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    csv_path = args.out / "gb1_wu2016.csv"
    print(f"Downloading GB1 (Wu-2016) landscape from {SOURCE_URL}")
    urllib.request.urlretrieve(SOURCE_URL, csv_path)

    checksum = _sha256(csv_path)
    size_bytes = csv_path.stat().st_size

    # Load through the real loader so the fetch validates exactly what validation will read: the
    # WT-residue asserts and the on-sites-only guard run here, failing loudly on a wrong construct.
    landscape = load_gb1(csv_path)  # WT-present + on-sites-only guards run inside
    composition = variant_order_composition(landscape)
    n_rows = len(landscape)

    if n_rows != EXPECTED_ROWS:
        print(f"  WARNING: expected {EXPECTED_ROWS} genotypes, loaded {n_rows} (source changed?)")

    provenance = {
        "source_url": SOURCE_URL,
        "source_dataset": "SaProtHub/Dataset-GB1-fitness (Wu et al. 2016, eLife 5:e16965)",
        "downloaded_utc": datetime.now(UTC).isoformat(),
        "file": csv_path.name,
        "sha256": checksum,
        "size_bytes": size_bytes,
        "n_genotypes": n_rows,
        "wt_sequence": GB1_WT_SEQUENCE,
        "sites_0indexed": list(GB1_SITES),
        "wt_at_sites": list(GB1_WT_AT_SITES),
        "order_composition": composition,
        "label_semantics": "fitness relative to wild type (WT == 1.0); dead variants == 0.0",
    }
    prov_path = args.out / "provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(f"  saved   {csv_path}  ({size_bytes:,} bytes)")
    print(f"  sha256  {checksum}")
    print(f"  rows    {n_rows:,}")
    print(f"  orders  {composition}")
    print(f"  prov    {prov_path}")


if __name__ == "__main__":
    main()
