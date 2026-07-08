"""Step 1 de-risk spike: does ESM-2 conjoint scoring carry GB1 epistasis signal?

Answers the two gate questions of docs/ROADMAP.md Step 1, on real data, before anything is built on
top:

  1. Is Var[ε_pred] > 0 (conjoint scoring is genuinely non-additive — invariant #1)?
  2. Does ESM-predicted ε correlate with measured ε (Spearman ≳ 0.2)?

Method. Measured ΔG(v) = ln(fitness(v)) (fitness relative to WT, WT == 1 ⇒ ΔG(∅) = 0); dead variants
(fitness == 0) have no log-fitness, so any interaction whose constituents include one is dropped —
never imputed. Predicted ΔG(v) is the conjoint ESM-2 conditional log-likelihood ratio. Pairwise and
third-order ε are the WT-referenced inclusion–exclusion terms (epistasis.epsilon_pairwise / _third),
computed identically from the measured and predicted ΔG maps over a sampled slice of amino-acid
instances. The correlation is Spearman over those ε instances.

Requires: python scripts/fetch_gb1.py (GB1 on disk) and an ESM-2 forward pass (CPU minutes on 35M).

Usage:
    python scripts/spike_gb1_epistasis.py [--data data/proteingym/gb1_wu2016.csv]
        [--model facebook/esm2_t12_35M_UR50D] [--k-pair 50] [--k-tri 40] [--seed 0]
        [--out report/spike_gb1.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from itertools import combinations
from math import log
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from epibudget.data import GB1_SITES, GB1_WT_SEQUENCE, load_gb1
from epibudget.epistasis import epsilon_pairwise, epsilon_third
from epibudget.scoring import ConjointScorer
from epibudget.types import Mutation, Variant

_AA20 = "ACDEFGHIKLMNPQRSTVWY"
_MIN_N_FOR_CORR = 3  # too few ε instances for a meaningful Spearman


def _mutant(site: int, mut_aa: str) -> Mutation:
    return (site, GB1_WT_SEQUENCE[site], mut_aa)


def _non_wt(site: int) -> list[str]:
    return [a for a in _AA20 if a != GB1_WT_SEQUENCE[site]]


def _sample(rng: np.random.Generator, grid: list[tuple[str, ...]], k: int) -> list[tuple[str, ...]]:
    idx = rng.choice(len(grid), size=min(k, len(grid)), replace=False)
    return [grid[i] for i in idx]


def sample_pairwise(rng: np.random.Generator, k: int) -> list[tuple[Mutation, Mutation]]:
    out: list[tuple[Mutation, Mutation]] = []
    for pi, pj in combinations(GB1_SITES, 2):
        grid = [(a, b) for a in _non_wt(pi) for b in _non_wt(pj)]
        for a, b in _sample(rng, grid, k):
            out.append((_mutant(pi, a), _mutant(pj, b)))
    return out


def sample_third(rng: np.random.Generator, k: int) -> list[tuple[Mutation, Mutation, Mutation]]:
    out: list[tuple[Mutation, Mutation, Mutation]] = []
    for pi, pj, pk in combinations(GB1_SITES, 3):
        grid = [(a, b, c) for a in _non_wt(pi) for b in _non_wt(pj) for c in _non_wt(pk)]
        for a, b, c in _sample(rng, grid, k):
            out.append((_mutant(pi, a), _mutant(pj, b), _mutant(pk, c)))
    return out


def _subsets(muts: tuple[Mutation, ...]) -> list[Variant]:
    """All non-empty sub-variants needed by the inclusion–exclusion term for ``muts``."""
    subs: list[Variant] = []
    for r in range(1, len(muts) + 1):
        subs.extend(frozenset(c) for c in combinations(muts, r))
    return subs


def _covered(muts: tuple[Mutation, ...], landscape: dict[Variant, float]) -> bool:
    """True iff every sub-variant is measured with strictly positive (log-defined) fitness."""
    return all(landscape.get(v, 0.0) > 0.0 for v in _subsets(muts))


def _spearman(pred: list[float], true: list[float]) -> float | None:
    if len(pred) < _MIN_N_FOR_CORR:
        return None
    rho = float(spearmanr(pred, true).statistic)
    return rho if np.isfinite(rho) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/proteingym/gb1_wu2016.csv"))
    parser.add_argument("--model", default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument(
        "--k-pair", type=int, default=50, help="aa-combos sampled per position-pair"
    )
    parser.add_argument(
        "--k-tri", type=int, default=40, help="aa-combos sampled per position-triple"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("report/spike_gb1.json"))
    args = parser.parse_args()

    landscape = load_gb1(args.data)
    rng = np.random.default_rng(args.seed)

    pairs = sample_pairwise(rng, args.k_pair)
    triples = sample_third(rng, args.k_tri)
    pairs_cov = [m for m in pairs if _covered(m, landscape)]
    triples_cov = [m for m in triples if _covered(m, landscape)]

    # Distinct variants to score conjointly = the union of every instance's sub-variants.
    needed: set[Variant] = set()
    for m in pairs_cov:
        needed.update(_subsets(m))
    for m in triples_cov:
        needed.update(_subsets(m))
    needed_list = sorted(needed, key=lambda v: (len(v), sorted(v)))

    print(
        f"Scoring {len(needed_list)} distinct variants with {args.model} | "
        f"covered: pairs {len(pairs_cov)}/{len(pairs)}, triples {len(triples_cov)}/{len(triples)}"
    )
    scorer = ConjointScorer(args.model, seed=args.seed, n_perturbations=0)
    scored = scorer.score_batch(GB1_WT_SEQUENCE, needed_list)
    dg_pred: dict[Variant, float] = {sv.variant: sv.delta_g for sv in scored}
    dg_true: dict[Variant, float] = {v: log(landscape[v]) for v in needed_list}

    def eps(
        order2: bool, instances: Iterable[tuple[Mutation, ...]]
    ) -> tuple[list[float], list[float]]:
        pred, true = [], []
        for m in instances:
            if order2:
                pred.append(epsilon_pairwise(dg_pred, m[0], m[1]))
                true.append(epsilon_pairwise(dg_true, m[0], m[1]))
            else:
                pred.append(epsilon_third(dg_pred, m[0], m[1], m[2]))
                true.append(epsilon_third(dg_true, m[0], m[1], m[2]))
        return pred, true

    pair_pred, pair_true = eps(True, pairs_cov)
    tri_pred, tri_true = eps(False, triples_cov)
    pooled_pred = pair_pred + tri_pred
    pooled_true = pair_true + tri_true

    rho_pair = _spearman(pair_pred, pair_true)
    rho_third = _spearman(tri_pred, tri_true)
    result = {
        "model": args.model,
        "seed": args.seed,
        "data_sha_note": "see data/proteingym/provenance.json",
        "n_pairwise": len(pair_pred),
        "n_third": len(tri_pred),
        "var_eps_pred_pooled": float(np.var(pooled_pred)) if pooled_pred else None,
        # The gate is judged PER ORDER: pooling pairwise (3-term) and third-order (7-term) ε into
        # one Spearman can distort it (different scales) and overstates significance (shared
        # sub-terms make instances non-independent). Pooled is context only, never the headline.
        "spearman_pairwise": rho_pair,
        "spearman_third": rho_third,
        "spearman_pooled_context_only": _spearman(pooled_pred, pooled_true),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # ASCII-only stdout (Windows consoles default to cp1252 and choke on Greek letters).
    print("\n=== Step 1 de-risk gate ===")
    print(f"  gate #1  Var[eps_pred] > 0        : {_fmt4(result['var_eps_pred_pooled'])}")
    print(
        f"  gate #2  Spearman(eps_pred,eps_true): "
        f"pairwise={_fmt(rho_pair)} (n={result['n_pairwise']}) "
        f"third={_fmt(rho_third)} (n={result['n_third']})  [target: each order >~ 0.2]"
    )
    print(f"  context  pooled Spearman         : {_fmt(result['spearman_pooled_context_only'])}")
    print(f"  written to {args.out}")


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _fmt4(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4f}"


if __name__ == "__main__":
    main()
