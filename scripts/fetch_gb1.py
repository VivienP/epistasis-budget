"""Fetch the complete GB1 four-site landscape (Wu et al. 2016, eLife) for validation.

Data is written to data/ (git-ignored) and NEVER committed. This script records provenance: source
URL, download date, checksum, row count, and the WT sequence. See docs/VALIDATION.md.

Usage:
    python scripts/fetch_gb1.py [--out data/proteingym]

Week-0 task (docs/ROADMAP.md). Implementation notes for whoever fills this in:
  * Source: ProteinGym substitution assays include GB1; alternatively the original Wu-2016 supplement.
  * Expect the *complete* four-site set: 20**4 = 160,000 variants at positions V39, D40, G41, V54.
  * After download: assert row count, compute a sha256 checksum, write a provenance.json next to the
    data, and print both. Do NOT impute missing rows — if coverage is partial, report it honestly.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/proteingym"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        "Week 0 — implement the ProteinGym/Wu-2016 GB1 fetch with checksum + provenance. "
        "See docstring and .claude/agents/proteingym-data-engineer.md."
    )


if __name__ == "__main__":
    main()
