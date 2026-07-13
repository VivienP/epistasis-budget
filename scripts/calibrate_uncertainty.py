"""Uncertainty-prior calibration: does var_delta_g (σ²) track the model's real per-variant error?

The info-optimal method's only ingredient beyond structure is ``var_delta_g`` — the
masking-perturbation dispersion used as a per-variant uncertainty prior (τ²). It is a useful
acquisition signal only if it is larger where the model is more wrong. This script measures that
directly at a given model size: for a seeded sample of covered variants it computes σ² =
``var_delta_g`` and the calibrated absolute prediction error ``|b·ΔĜ_ESM − ΔG_measured|`` (``b`` the
through-origin ΔG-scale slope; ``ΔG_measured = ln fitness``), then reports Spearman/Pearson(σ²,
|error|) with a bootstrap 95% CI.

Measured fitness enters here for the *error* term only after the sample is fixed — never selection.
No label or live/dead status reaches the acquisition path. Because only positive-fitness rows are
log-transformable, the analysis is conditional on measurable positive fitness. Correlations and
their intervals do not convert a near-zero estimate into proof of calibration or anti-calibration.

Usage:
    python scripts/calibrate_uncertainty.py --model esm2_t33_650M --n 300 --n-perturbations 16 \
        --threads 12 --batch-size 32 [--device cpu] [--seed 0] [--out report/]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from _console import configure_utf8_stdout

from epibudget.calibrate import calibrate
from epibudget.data import (
    GB1_SITES,
    GB1_WT_AT_SITES,
    GB1_WT_SEQUENCE,
    enumerate_candidates,
    load_gb1,
)
from epibudget.scoring import ConjointScorer
from epibudget.types import Variant

_AA20 = "ACDEFGHIKLMNPQRSTVWY"


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="esm2_t33_650M")
    parser.add_argument("--n", type=int, default=300, help="Number of covered variants to sample.")
    parser.add_argument("--n-perturbations", type=int, default=16)
    parser.add_argument("--alphabet", default=_AA20)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=Path, default=Path("data/proteingym/gb1_wu2016.csv"))
    parser.add_argument("--out", type=Path, default=Path("report/"))
    args = parser.parse_args()

    model_id = {
        "esm2_t12_35M": "facebook/esm2_t12_35M_UR50D",
        "esm2_t30_150M": "facebook/esm2_t30_150M_UR50D",
        "esm2_t33_650M": "facebook/esm2_t33_650M_UR50D",
    }.get(args.model, args.model)

    landscape = load_gb1(args.data)
    data_sha256 = hashlib.sha256(args.data.read_bytes()).hexdigest()
    candidates = enumerate_candidates(
        GB1_SITES, GB1_WT_AT_SITES, allowed_aa=args.alphabet, max_order=args.max_order
    )
    covered = [v for v in candidates if landscape.get(v, 0.0) > 0.0]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(covered), size=min(args.n, len(covered)), replace=False)
    sample: list[Variant] = [covered[i] for i in sorted(idx)]
    order_counts = dict(sorted(Counter(len(v) for v in sample).items()))
    print(
        f"[calibrate] scoring {len(sample)} covered variants (orders {order_counts}) with "
        f"{args.model} on {args.device}, n_perturbations={args.n_perturbations} ..."
    )

    scorer = ConjointScorer(
        model_id,
        device=args.device,
        n_perturbations=args.n_perturbations,
        seed=args.seed,
        batch_size=args.batch_size,
        num_threads=args.threads if args.threads > 0 else None,
    )
    scored = scorer.score_batch(GB1_WT_SEQUENCE, sample)

    esm_dg = [sv.delta_g for sv in scored]
    sigma2 = [sv.var_delta_g for sv in scored]
    measured_dg = [float(np.log(landscape[sv.variant])) for sv in scored]
    result = calibrate(esm_dg, sigma2, measured_dg, seed=args.seed)

    payload = {
        "calibration": "uncertainty_prior_vs_prediction_error",
        "question": "Does var_delta_g (sigma^2) correlate with |calibrated ESM error| per variant?",
        "model_id": args.model,
        "device": scorer.device,
        "n": result.n,
        "order_composition": order_counts,
        "seed": args.seed,
        "n_perturbations": args.n_perturbations,
        "candidate_alphabet": args.alphabet,
        "max_order": args.max_order,
        "data_sha256": data_sha256,
        "calibration_slope_b": result.calibration_slope_b,
        "spearman_sigma2_abserror": result.spearman,
        "pearson_sigma2_abserror": result.pearson,
        "spearman_ci95": result.spearman_ci95,
        "pearson_ci95": result.pearson_ci95,
        "pairs_sigma2_abserror": [
            [s, e] for s, e in zip(result.sigma2, result.abs_error, strict=True)
        ],
    }
    run_dir = args.out / f"calibration_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "metrics.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"[calibrate] Spearman(sigma2,|error|)={result.spearman:+.3f} CI95={result.spearman_ci95}"
    )
    print(
        f"[calibrate] Pearson={result.pearson:+.3f} CI95={result.pearson_ci95}  "
        f"b={result.calibration_slope_b:.3f}"
    )
    print(f"[calibrate] wrote {out_path}")


if __name__ == "__main__":
    main()
