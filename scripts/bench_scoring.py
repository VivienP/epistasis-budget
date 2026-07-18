"""Throughput benchmark: the batched, de-duplicated ConjointScorer vs the per-variant reference.

Measures, on a real GB1 candidate slice, the wall-clock and variants/s of ``score`` (the reference,
one forward per variant) against ``score_batch`` (de-duplicated cross-variant batches), plus the
masked-row de-duplication ratio. Numbers are real and machine-specific; run it to produce the
before/after figures cited in docs/LIMITATIONS.md. Also asserts the two paths agree (sanity), so a
speedup that silently changed the numbers would be caught here too.

Requires an ESM-2 forward pass (CPU minutes on 35M; use a small pool for 650M). Usage:

    python scripts/bench_scoring.py --model esm2_t12_35M --alphabet ACDEF --max-order 3 \
        --n-perturbations 4 --threads 12 --batch-size 32 [--device cpu] [--out report/bench.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from _console import configure_utf8_stdout

from epibudget.data import GB1_SITES, GB1_WT_AT_SITES, GB1_WT_SEQUENCE, enumerate_candidates
from epibudget.scoring import ConjointScorer
from epibudget.scoring_plan import dedup, plan_variant
from epibudget.types import Variant

_MODEL_ALIASES = {
    "esm2_t12_35M": "facebook/esm2_t12_35M_UR50D",
    "esm2_t30_150M": "facebook/esm2_t30_150M_UR50D",
    "esm2_t33_650M": "facebook/esm2_t33_650M_UR50D",
}

# A batched-vs-reference gap above this points to a parity bug, not float noise; flag it loudly.
_DIVERGENCE_WARN = 1e-3


def _dedup_stats(candidates: list[Variant], seed: int, n_perturbations: int) -> tuple[int, int]:
    """Total planned masked rows vs unique rows after de-duplication over the candidate pool."""
    rows = []
    offset = 0
    for v in candidates:
        passes, r = plan_variant(
            GB1_WT_SEQUENCE, v, seed=seed, n_perturbations=n_perturbations, mask_fraction=0.15
        )
        for masked_seq, read_pos, mut_aa, wt_aa, pass_id, site_index in r:
            rows.append((masked_seq, read_pos, mut_aa, wt_aa, pass_id + offset, site_index))
        offset += len(passes)
    unique_seqs, _, _ = dedup(rows)
    return len(rows), len(unique_seqs)


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="esm2_t12_35M")
    parser.add_argument("--alphabet", default="ACDEF", help="Per-site candidate alphabet.")
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--n-perturbations", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=0, help="torch threads; 0 = default.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    model_id = _MODEL_ALIASES.get(args.model, args.model)
    num_threads = args.threads if args.threads > 0 else None
    candidates = enumerate_candidates(
        GB1_SITES, GB1_WT_AT_SITES, allowed_aa=args.alphabet, max_order=args.max_order
    )
    total_rows, unique_rows = _dedup_stats(candidates, args.seed, args.n_perturbations)

    scorer = ConjointScorer(
        model_id,
        device=args.device,
        n_perturbations=args.n_perturbations,
        seed=args.seed,
        batch_size=args.batch_size,
        num_threads=num_threads,
    )
    scorer._ensure_loaded()  # warm up: exclude model load from timing

    t0 = time.perf_counter()
    reference = [scorer.score(GB1_WT_SEQUENCE, v) for v in candidates]
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    optimized = scorer.score_batch(GB1_WT_SEQUENCE, candidates)
    t_opt = time.perf_counter() - t0

    ref_by = {sv.variant: sv for sv in reference}
    max_dg = max(abs(o.delta_g - ref_by[o.variant].delta_g) for o in optimized)
    max_var = max(abs(o.var_delta_g - ref_by[o.variant].var_delta_g) for o in optimized)

    n = len(candidates)
    result = {
        "model_id": model_id,
        "device": scorer.device,
        "alphabet": args.alphabet,
        "max_order": args.max_order,
        "n_perturbations": args.n_perturbations,
        "batch_size": args.batch_size,
        "num_threads": num_threads,
        "n_variants": n,
        "planned_rows": total_rows,
        "unique_rows": unique_rows,
        "dedup_ratio": round(total_rows / unique_rows, 3) if unique_rows else None,
        "reference_seconds": round(t_ref, 3),
        "optimized_seconds": round(t_opt, 3),
        "reference_variants_per_s": round(n / t_ref, 3) if t_ref else None,
        "optimized_variants_per_s": round(n / t_opt, 3) if t_opt else None,
        "speedup": round(t_ref / t_opt, 3) if t_opt else None,
        "max_abs_delta_g_gap": max_dg,
        "max_abs_var_delta_g_gap": max_var,
    }

    print("=== scoring throughput benchmark ===")
    for key, value in result.items():
        print(f"  {key:26s}: {value}")
    if max_dg > _DIVERGENCE_WARN or max_var > _DIVERGENCE_WARN:
        print("  WARNING: batched path diverges from reference beyond 1e-3 - investigate parity")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
