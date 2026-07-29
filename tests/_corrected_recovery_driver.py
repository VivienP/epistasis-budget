"""Drive the real ``scripts/corrected_recovery.py`` over a tiny synthetic landscape.

Not a test module (the leading underscore keeps pytest from collecting it): it is the subprocess
body of ``test_emitter_writes_a_file_that_validates_against_its_own_schema``. It lives in a file
rather than a string literal so it is linted and type-checked like the rest of the suite.

Building a synthetic landscape here, instead of pointing at GB1, is what makes the emitter test
cheap enough to keep in the offline suite: 64 candidates against 29 678, and no 10 MB CSV.

Usage: python tests/_corrected_recovery_driver.py <repo-root> <work-dir>
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

_REPO = Path(sys.argv[1])
_WORK = Path(sys.argv[2])
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from epibudget import data as data_module  # noqa: E402
from epibudget.data import DatasetSpec, enumerate_candidates  # noqa: E402
from epibudget.scored_cache import cache_metadata_path, candidate_sha256  # noqa: E402
from epibudget.types import Variant  # noqa: E402

SITES = (0, 1, 2, 3)
WT_AT = ("A", "A", "A", "A")
WT_SEQUENCE = "AAAA"
ALPHABET = "CD"  # two non-WT residues per site -> 64 candidates over orders 1..3


def _fitness(variant: Variant) -> float:
    """Additive effects passed through a saturating link: real epistasis, all values positive.

    The saturation is what makes it epistatic -- a concave g() turns additive DG into a non-zero
    inclusion-exclusion contrast -- and the strictly positive range keeps every variant inside the
    log-ratio domain, so the label accounting has no non-finite bucket to fall into.
    """
    total = 0.0
    for site, _wt, mutant in sorted(variant):
        total += (0.35 if mutant == "C" else -0.2) * (1.0 + 0.1 * site)
    return 2.0 / (1.0 + math.exp(-total))


def _write_landscape(candidates: list[Variant]) -> Path:
    rows = [[sorted([list(m) for m in v]), _fitness(v)] for v in [frozenset(), *candidates]]
    path = _WORK / "landscape.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _load_landscape(path: Path) -> dict[Variant, float]:
    out: dict[Variant, float] = {}
    for mutations, value in json.loads(Path(path).read_text(encoding="utf-8")):
        out[frozenset(tuple(m) for m in mutations)] = float(value)
    return out


def _write_cache(candidates: list[Variant]) -> Path:
    path = _WORK / "scored.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for i, variant in enumerate(candidates):
            row = {
                "variant": sorted([list(m) for m in variant]),
                # A deliberately imperfect surrogate for the measured fitness.
                "delta_g": _fitness(variant) - 1.0 + 0.05 * ((i % 7) - 3),
                # Dispersion must vary across candidates, or the `info` weight is constant and the
                # acquisition refuses to rank on it (audit M-2).
                "var_delta_g": 0.01 + 0.001 * (i % 11),
            }
            handle.write(json.dumps(row) + "\n")

    cache_metadata_path(path).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "synthetic/test-scorer",
                "wt_sha256": hashlib.sha256(WT_SEQUENCE.encode("ascii")).hexdigest(),
                "candidate_sha256": candidate_sha256(candidates),
                "candidate_count": len(candidates),
                "candidate_alphabet": ALPHABET,
                "max_order": 3,
                "scorer_seed": 0,
                "n_perturbations": 16,
                "device": "cpu",
                "mask_fraction": 0.15,
                "batch_size": 8,
                "num_threads": None,
            }
        ),
        encoding="utf-8",
    )
    return path


def main() -> None:
    candidates = enumerate_candidates(SITES, WT_AT, allowed_aa=ALPHABET, max_order=3)
    landscape_path = _write_landscape(candidates)
    cache_path = _write_cache(candidates)

    data_module.DATASETS["synthetic_tiny"] = DatasetSpec(
        identifier="synthetic_tiny",
        loader=_load_landscape,
        sites=SITES,
        wt_at_sites=WT_AT,
        wt_sequence=WT_SEQUENCE,
        default_data_path=str(landscape_path),
    )

    import corrected_recovery  # noqa: PLC0415

    sys.argv = [
        "corrected_recovery.py",
        "--dataset",
        "synthetic_tiny",
        "--scored-cache",
        str(cache_path),
        "--alphabet",
        ALPHABET,
        "--model-id",
        # Overridable so a test can assert the cache-identity gate actually rejects a mismatch.
        os.environ.get("EPIBUDGET_DRIVER_MODEL_ID", "synthetic/test-scorer"),
        "--budgets",
        "8,20",
        "--tie-seeds",
        "4",
        "--random-seeds",
        "3",
        "--bootstrap",
        "40",
        "--out",
        str(_WORK / "out"),
    ]
    corrected_recovery.main()


if __name__ == "__main__":
    main()
