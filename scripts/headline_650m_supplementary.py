"""Supplementary 650M full-alphabet recovery — the var-independent methods, in-session.

NOT the frozen headline. With ``n_perturbations=0`` there is no ``var_delta_g`` pass, so the full
20-letter four-site pool (~29,678 variants) collapses to the ~4,564 de-duplicated deterministic
forwards and scores in ~40 min on this CPU — versus ~8–9 days for the frozen 16-perturbation run
(docs/LIMITATIONS.md §1). Because ``var_delta_g`` is absent, **info-optimal is omitted** (it
degenerates at τ²≡0); this run reports only the methods that do not need it: fitness-greedy, random,
practice, and the structural-only ablation (τ²≡1, rank by loop count).

Why it is worth running: it is the first full-alphabet, ``pool ≫ B`` evidence at 650M — it
un-tautologises breadth vs precision (docs/LIMITATIONS.md §4), which the reduced-alphabet 35M smoke
cannot, and it establishes the structural-only recovery line at scale (the baseline the confirmed
null rests on). The frozen headline (docs/headline_650m_colab.md) later supplies info-optimal to
compare against this same line. Additive evidence, never a substitute for the frozen run (inv. #2).

Usage:
    python scripts/headline_650m_supplementary.py --model esm2_t33_650M --budgets 48,96,192 \
        --seeds 20 --threads 12 --batch-size 32 [--device cpu] [--alphabet ACDEFGHIKLMNPQRSTVWY] \
        [--scored-cache report/scored_650m_det.jsonl] [--out report/]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from epibudget.acquisition import allocate, fitness_greedy
from epibudget.data import (
    GB1_SITES,
    GB1_WT_AT_SITES,
    GB1_WT_SEQUENCE,
    enumerate_candidates,
    load_gb1,
)
from epibudget.epistasis import ground_truth_epistasis
from epibudget.scored_cache import score_with_cache
from epibudget.scoring import ConjointScorer
from epibudget.validate import (
    MethodResult,
    Term,
    _candidate_fitness,
    _candidate_terms,
    _measured_dg,
    _random_result,
    hit_rate,
    infer_epistasis,
    map_recovery,
    practice_heuristic,
    random_selection,
    structural_graph,
)

_AA20 = "ACDEFGHIKLMNPQRSTVWY"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="esm2_t33_650M")
    parser.add_argument("--budgets", default="48,96,192")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--alphabet", default=_AA20)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data", type=Path, default=Path("data/proteingym/gb1_wu2016.csv"))
    parser.add_argument("--scored-cache", default="")
    parser.add_argument("--out", type=Path, default=Path("report/"))
    args = parser.parse_args()

    model_id = {
        "esm2_t12_35M": "facebook/esm2_t12_35M_UR50D",
        "esm2_t30_150M": "facebook/esm2_t30_150M_UR50D",
        "esm2_t33_650M": "facebook/esm2_t33_650M_UR50D",
    }.get(args.model, args.model)

    budgets = [int(b) for b in args.budgets.split(",")]
    landscape = load_gb1(args.data)
    data_sha256 = hashlib.sha256(args.data.read_bytes()).hexdigest()
    candidates = enumerate_candidates(
        GB1_SITES, GB1_WT_AT_SITES, allowed_aa=args.alphabet, max_order=args.max_order
    )
    print(
        f"[supplementary] scoring {len(candidates)} candidates (alphabet={args.alphabet!r}) "
        f"with {args.model} on {args.device}, n_perturbations=0 (deterministic only) ..."
    )

    scorer = ConjointScorer(
        model_id,
        device=args.device,
        n_perturbations=0,  # var-free: the whole point of the ~4,564-forward de-dup regime
        seed=0,
        batch_size=args.batch_size,
        num_threads=args.threads if args.threads > 0 else None,
    )
    if args.scored_cache:
        scored = score_with_cache(scorer, GB1_WT_SEQUENCE, candidates, Path(args.scored_cache))
    else:
        scored = scorer.score_batch(GB1_WT_SEQUENCE, candidates)

    max_order = args.max_order
    term_set = set(_candidate_terms(scored, max_order))
    landscape_dg = {v: float(np.log(f)) for v, f in landscape.items() if f > 0.0}
    truth_by_term: dict[Term, float] = {
        interaction.mutations: interaction.epsilon_hat
        for interaction in ground_truth_epistasis(landscape_dg, max_order)
        if interaction.mutations in term_set
    }
    var_epsilon = float(np.var(np.array(list(truth_by_term.values())))) if truth_by_term else 0.0
    structural = structural_graph(scored, max_order)  # τ² ≡ 1 ablation (var-independent)
    candidate_fitness = _candidate_fitness(scored, landscape)

    results: list[MethodResult] = []
    for budget in budgets:
        deterministic = {
            "fitness": fitness_greedy(scored, budget),
            "structural": allocate(structural, scored, budget, lambda_=0.0).selected,
            "practice": practice_heuristic(scored, budget),
        }
        random_sels = [random_selection(scored, budget, s) for s in range(args.seeds)]
        for method, selected in deterministic.items():
            measured_dg = _measured_dg(landscape, selected)
            inferred = infer_epistasis(measured_dg, scored, max_order)
            metrics = map_recovery(inferred, truth_by_term, frozenset(measured_dg), seed=budget)
            results.append(
                MethodResult(
                    method=method,
                    budget=budget,
                    ci_method="bootstrap-over-terms",
                    hit_rate=hit_rate(selected, candidate_fitness, budget),
                    metrics=metrics,
                )
            )
        results.append(
            _random_result(
                random_sels, scored, landscape, truth_by_term, candidate_fitness, budget, max_order
            )
        )

    payload = {
        "supplementary": True,
        "info_optimal": "deferred (needs the var_delta_g pass; run the frozen headline on a GPU, "
        "see docs/headline_650m_colab.md)",
        "note": "650M full-alphabet DETERMINISTIC-ONLY (n_perturbations=0): var-independent "
        "methods only. NOT the frozen headline; the VALIDATION.md decision rule (info vs fitness "
        "vs random) is not evaluated here.",
        "dataset": "gb1_wu2016",
        "model_id": args.model,
        "device": scorer.device,
        "budgets": budgets,
        "seeds": args.seeds,
        "candidate_alphabet": args.alphabet,
        "scorer_seed": 0,
        "n_perturbations": 0,
        "max_order": max_order,
        "data_sha256": data_sha256,
        "n_candidates": len(scored),
        "n_truth_terms": len(truth_by_term),
        "var_epsilon": var_epsilon,
        "results": [r.model_dump() for r in results],
    }
    run_dir = args.out / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "metrics.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[supplementary] var_epsilon={var_epsilon:.4f}  truth_terms={len(truth_by_term)}")
    for r in results:
        pooled = next(m for m in r.metrics if m.order == "pairwise")
        print(f"  B={r.budget:>3} {r.method:<10} pairwise spearman={pooled.spearman}")
    print(f"[supplementary] wrote {out_path}")


if __name__ == "__main__":
    main()
