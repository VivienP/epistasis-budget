"""GB1 validation harness. Frozen protocol in docs/VALIDATION.md.

Compares four methods at each budget — info-optimal, fitness-greedy, random, practice-heuristic — on
epistasis-map recovery against the full GB1 ground truth. The first three are the frozen decision
rule; the practice heuristic is a companion.

Isolation. The SAME ``infer_epistasis`` runs per method; only the *selected set* differs, so the
comparison isolates selection, not inference. Selection is zero-shot (ESM predictions + the factor
graph). Measured fitness enters exactly once, via ``data.reveal_measured_fitness``, strictly after a
method has returned its selection (docs/skills/no-label-leakage).

Inference (docs/VALIDATION.md §Simulation). ``infer_epistasis`` is the closed-form posterior mean of
the linear-Gaussian model in graph.py: measuring a variant pins its ΔG, and every unmeasured loop
member keeps its unit-calibrated ESM prior mean. This is a Tikhonov estimator (prior mean = the
calibrated ESM ΔĜ, precision = 1/var_delta_g) — a precisification of "regularised least squares over
the interaction basis" that keeps selection and grading on one coherent model.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from math import log
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from scipy.stats import pearsonr, spearmanr

from epibudget.acquisition import allocate, fitness_greedy
from epibudget.data import reveal_measured_fitness
from epibudget.epistasis import (
    epsilon_pairwise,
    epsilon_third,
    ground_truth_epistasis,
    interaction_loop,
    predicted_epistasis,
)
from epibudget.graph import EpistasisFactorGraph
from epibudget.types import Interaction, Mutation, ScoredVariant, Variant

_PAIRWISE_ORDER = 2
_MIN_POINTS_FOR_CORR = 3
_N_BOOTSTRAP = 1000

Term = tuple[Mutation, ...]


class OrderMetric(BaseModel):
    """Recovery for one interaction order (or pooled): correlations, CIs, coverage.

    ``coverage_fraction`` / ``n_informed`` count terms this method's selection informs (breadth).
    ``pearson_informed`` / ``spearman_informed`` are over the *shared* union subset (fixed across
    methods, so those correlations stay comparable).
    """

    order: str  # "pairwise" | "third" | "pooled"
    n_terms: int
    pearson: float | None
    spearman: float | None
    pearson_ci95: tuple[float, float] | None
    spearman_ci95: tuple[float, float] | None
    n_informed: int  # per-method: terms informed by this method's selection
    coverage_fraction: float  # per-method: n_informed / n_terms
    n_informed_union: int  # size of the shared informed-union subset (context)
    pearson_informed: float | None  # over the shared informed-union subset
    spearman_informed: float | None


class MethodResult(BaseModel):
    method: str  # "info" | "fitness" | "random" | "practice"
    budget: int
    ci_method: str  # "bootstrap-over-terms" | "bootstrap-over-seeds"
    hit_rate: float
    metrics: list[OrderMetric]


class Report(BaseModel):
    dataset: str
    model_id: str
    seeds: int
    budgets: list[int]
    candidate_alphabet: str  # provenance: the per-site alphabet the candidate pool was drawn from
    n_candidates: int
    n_truth_terms: int
    var_epsilon: float  # invariant #1 sanity: must be > 0
    results: list[MethodResult]


# --------------------------------------------------------------------- inference


def _calibrate_slope(esm: Sequence[float], measured: Sequence[float]) -> float:
    """Through-origin least-squares slope b of measured ΔG ≈ b·ΔĜ_ESM over the revealed set.

    ESM pseudo-log-likelihood and ln-fitness live on different, uncalibrated scales, so the ESM
    prior must be put on the measured scale before it is mixed with measured values in an ε sum. The
    fit is intercept-free because both scales are anchored at the WT reference (ΔĜ(WT)=ΔG(WT)=0, the
    same convention epsilon_pairwise/epsilon_third rely on): a free intercept breaks that anchor
    and — since the ±1 inclusion–exclusion coefficients over a non-empty loop sum to −(−1)ⁿ, not 0 —
    inject a term-dependent offset into every partially-measured (informed) ε. Degenerate input
    (no points, or all ΔĜ=0) falls back to b=1.
    """
    x = np.asarray(esm, dtype=np.float64)
    y = np.asarray(measured, dtype=np.float64)
    denominator = float(np.dot(x, x))
    if denominator == 0.0:
        return 1.0
    return float(np.dot(x, y) / denominator)


def _candidate_terms(scored: Sequence[ScoredVariant], max_order: int) -> list[Term]:
    """Order-2..max_order variants present in ``scored``, as canonical mutation tuples."""
    return [
        tuple(sorted(sv.variant))
        for sv in scored
        if _PAIRWISE_ORDER <= len(sv.variant) <= max_order
    ]


def infer_epistasis(
    revealed: Mapping[Variant, float],
    scored: Sequence[ScoredVariant],
    max_order: int = 3,
) -> list[Interaction]:
    """Posterior-mean ε̂ over the order-2..max_order terms from the measured ΔG of revealed variants.

    ``revealed`` maps each measured variant to its ΔG (ln fitness); ``scored`` supplies the ESM ΔĜ
    and var_delta_g for every candidate (all orders, as loop members). Each unmeasured loop member
    keeps its unit-calibrated ESM prior mean; measured members are pinned to their true ΔG.
    Round-trips: empty ``revealed`` reproduces the ESM ε̂; a fully-measured loop reproduces true ε.
    """
    esm = {sv.variant: sv.delta_g for sv in scored}
    tau2 = {sv.variant: sv.var_delta_g for sv in scored}
    b = _calibrate_slope([esm[v] for v in revealed], [revealed[v] for v in revealed])
    mu: dict[Variant, float] = {v: (revealed[v] if v in revealed else b * esm[v]) for v in esm}

    interactions: list[Interaction] = []
    for term in _candidate_terms(scored, max_order):
        if len(term) == _PAIRWISE_ORDER:
            eps = epsilon_pairwise(mu, term[0], term[1])
        else:
            eps = epsilon_third(mu, term[0], term[1], term[2])
        sigma2 = sum(tau2[member] for member in interaction_loop(term) if member not in revealed)
        interactions.append(Interaction.of(term, epsilon_hat=eps, sigma2=sigma2))
    return interactions


# --------------------------------------------------------------------- recovery


def _pearson_spearman(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    """Raw Pearson and Spearman; scipy's near-constant caution muted (degeneracy handled above)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ConstantInputWarning / NearConstantInputWarning
        return float(pearsonr(pred, true).statistic), float(spearmanr(pred, true).statistic)


def _corr(pred: np.ndarray, true: np.ndarray) -> tuple[float | None, float | None]:
    """Pearson and Spearman, or (None, None) if undefined (too few points or a constant vector)."""
    if len(pred) < _MIN_POINTS_FOR_CORR or float(np.std(pred)) == 0.0 or float(np.std(true)) == 0.0:
        return None, None
    return _pearson_spearman(pred, true)


def _bootstrap_ci(
    pred: np.ndarray, true: np.ndarray, statistic: str, seed: int
) -> tuple[float, float] | None:
    """Percentile 95% CI of a correlation by resampling the evaluated terms with replacement."""
    n = len(pred)
    if n < _MIN_POINTS_FOR_CORR:
        return None
    rng = np.random.default_rng(seed)
    stats: list[float] = []
    for _ in range(_N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        p, t = pred[idx], true[idx]
        if float(np.std(p)) == 0.0 or float(np.std(t)) == 0.0:
            continue
        pearson, spearman = _pearson_spearman(p, t)
        stats.append(pearson if statistic == "pearson" else spearman)
    if len(stats) < _MIN_POINTS_FOR_CORR:
        return None
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def _informed(term: Term, measured: frozenset[Variant]) -> bool:
    """True iff at least one loop member of ``term`` was measured (posterior mean can move it)."""
    return any(member in measured for member in interaction_loop(term))


def _order_label(order: int) -> str:
    return "pairwise" if order == _PAIRWISE_ORDER else "third"


def _order_metric(
    order_name: str,
    rows: list[tuple[Term, float, float]],
    measured: frozenset[Variant],
    informed_union: frozenset[Term],
    seed: int,
    with_ci: bool,
) -> OrderMetric:
    """Build one OrderMetric from (term, inferred, true) rows for a given order (or pooled)."""
    pred = np.array([r[1] for r in rows], dtype=np.float64)
    true = np.array([r[2] for r in rows], dtype=np.float64)
    pearson, spearman = _corr(pred, true)
    pearson_ci = _bootstrap_ci(pred, true, "pearson", seed) if with_ci else None
    spearman_ci = _bootstrap_ci(pred, true, "spearman", seed + 1) if with_ci else None

    n_covered = sum(1 for r in rows if _informed(r[0], measured))  # this method's own coverage
    union_rows = [r for r in rows if r[0] in informed_union]  # the shared comparable subset
    up = np.array([r[1] for r in union_rows], dtype=np.float64)
    ut = np.array([r[2] for r in union_rows], dtype=np.float64)
    pearson_inf, spearman_inf = _corr(up, ut)

    return OrderMetric(
        order=order_name,
        n_terms=len(rows),
        pearson=pearson,
        spearman=spearman,
        pearson_ci95=pearson_ci,
        spearman_ci95=spearman_ci,
        n_informed=n_covered,
        coverage_fraction=n_covered / len(rows) if rows else 0.0,
        n_informed_union=len(union_rows),
        pearson_informed=pearson_inf,
        spearman_informed=spearman_inf,
    )


def map_recovery(
    inferred: Sequence[Interaction],
    truth_by_term: Mapping[Term, float],
    measured: frozenset[Variant],
    informed_union: frozenset[Term],
    seed: int = 0,
    with_ci: bool = True,
) -> list[OrderMetric]:
    """Per-order and pooled recovery of inferred ε̂ vs ground-truth ε over the evaluated terms.

    ``measured`` is this method's selection (drives per-method coverage); ``informed_union`` is the
    shared subset (fixed across methods) for the comparable informed-subset correlation.
    """
    rows_by_order: dict[str, list[tuple[Term, float, float]]] = {"pairwise": [], "third": []}
    for interaction in inferred:
        if interaction.mutations in truth_by_term:
            row = (
                interaction.mutations,
                interaction.epsilon_hat,
                truth_by_term[interaction.mutations],
            )
            rows_by_order[_order_label(interaction.order)].append(row)
    pooled = rows_by_order["pairwise"] + rows_by_order["third"]

    metrics: list[OrderMetric] = []
    for name, rows in (
        ("pairwise", rows_by_order["pairwise"]),
        ("third", rows_by_order["third"]),
        ("pooled", pooled),
    ):
        metrics.append(_order_metric(name, rows, measured, informed_union, seed, with_ci))
    return metrics


# --------------------------------------------------------------------- baselines


def random_selection(candidates: Sequence[ScoredVariant], budget: int, seed: int) -> list[Variant]:
    """Uniform sample of ``budget`` distinct candidates (RNG touches identity only, not a label)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(candidates), size=budget, replace=False)
    return [candidates[i].variant for i in idx]


def practice_heuristic(candidates: Sequence[ScoredVariant], budget: int) -> list[Variant]:
    """Top beneficial singles (by predicted ΔG) → their cross-site pairwise combos (MULTI-evolve).

    Zero-shot: ranks on predicted ``delta_g`` only. Enough top singles are taken to fill ``budget``
    valid cross-site pairs; the pairs are then ranked by their own predicted ΔG and the top
    ``budget`` kept. Only order-2 variants spend budget. If the pool has fewer valid cross-site
    pairs than ``budget`` it under-fills rather than raising; the frozen budgets fit.
    """
    pair_dg = {
        tuple(sorted(sv.variant)): sv.delta_g
        for sv in candidates
        if len(sv.variant) == _PAIRWISE_ORDER
    }
    singles = sorted(
        (sv for sv in candidates if len(sv.variant) == 1), key=lambda s: s.delta_g, reverse=True
    )
    single_muts = [next(iter(s.variant)) for s in singles]

    pairs: list[tuple[Mutation, Mutation]] = []
    seen: set[Term] = set()
    for k in range(2, len(single_muts) + 1):
        pairs = []
        seen = set()
        for i in range(k):
            for j in range(i + 1, k):
                mi, mj = single_muts[i], single_muts[j]
                if mi[0] == mj[0]:
                    continue  # same site: not a valid variant
                key = tuple(sorted((mi, mj)))
                if key in pair_dg and key not in seen:
                    seen.add(key)
                    pairs.append((mi, mj))
        if len(pairs) >= budget:
            break

    ranked = sorted(pairs, key=lambda p: pair_dg[tuple(sorted(p))], reverse=True)
    return [frozenset(p) for p in ranked[:budget]]


def hit_rate(selected: Sequence[Variant], fitness: Mapping[Variant, float], budget: int) -> float:
    """Fraction of the true top-``budget`` fitness variants (over the candidate universe) captured.

    Uses measured fitness, but computed strictly after selection is fixed — never an input to it.
    """
    ranked = sorted(fitness, key=lambda v: fitness[v], reverse=True)
    top = set(ranked[:budget])
    return len(set(selected) & top) / budget if budget else 0.0


# --------------------------------------------------------------------- harness


def _measured_dg(
    landscape: Mapping[Variant, float], selected: Sequence[Variant]
) -> dict[Variant, float]:
    """Reveal fitness for the selection; live values become ΔG=ln f (drop dead, never impute)."""
    revealed = reveal_measured_fitness(dict(landscape), selected)
    return {v: log(f) for v, f in revealed.items() if f > 0.0}


def _candidate_fitness(
    scored: Sequence[ScoredVariant], landscape: Mapping[Variant, float]
) -> dict[Variant, float]:
    """Measured fitness restricted to the candidate universe (for hit_rate's honest denominator)."""
    return {sv.variant: landscape[sv.variant] for sv in scored if sv.variant in landscape}


def run_validation(
    scored: Sequence[ScoredVariant],
    landscape: Mapping[Variant, float],
    budgets: Sequence[int],
    seeds: int,
    model_id: str,
    out_dir: Path,
    dataset: str = "gb1_wu2016",
    max_order: int = 3,
    candidate_alphabet: str = "",
) -> Report:
    """Execute the frozen protocol and write ``<out_dir>/metrics.json``. See docs/VALIDATION.md.

    ``scored`` are the ESM-scored candidates (orders 1..max_order); ``landscape`` is the full GB1
    {variant → fitness}. The selection graph is built from ``predicted_epistasis`` (ESM) only, never
    from ``ground_truth_epistasis`` — so no label can leak into selection.
    """
    term_set = set(_candidate_terms(scored, max_order))
    landscape_dg = {v: log(f) for v, f in landscape.items() if f > 0.0}
    truth_by_term: dict[Term, float] = {
        interaction.mutations: interaction.epsilon_hat
        for interaction in ground_truth_epistasis(landscape_dg, max_order)
        if interaction.mutations in term_set
    }
    var_epsilon = float(np.var(np.array(list(truth_by_term.values())))) if truth_by_term else 0.0

    graph = EpistasisFactorGraph(
        predicted_epistasis(scored, max_order), {sv.variant: sv.var_delta_g for sv in scored}
    )
    candidate_fitness = _candidate_fitness(scored, landscape)

    results: list[MethodResult] = []
    for budget in budgets:
        deterministic = {
            "info": allocate(graph, scored, budget, lambda_=0.0).selected,
            "fitness": fitness_greedy(scored, budget),
            "practice": practice_heuristic(scored, budget),
        }
        random_sels = [random_selection(scored, budget, s) for s in range(seeds)]

        # "measured" = the LIVE (non-dead) variants a method reveals: what the posterior uses.
        det_measured = {
            m: frozenset(_measured_dg(landscape, sel)) for m, sel in deterministic.items()
        }
        random_measured = [frozenset(_measured_dg(landscape, sel)) for sel in random_sels]
        informed_union = frozenset(
            term
            for term in term_set
            if any(_informed(term, m) for m in (*det_measured.values(), *random_measured))
        )

        for method, selected in deterministic.items():
            measured_dg = _measured_dg(landscape, selected)
            inferred = infer_epistasis(measured_dg, scored, max_order)
            metrics = map_recovery(
                inferred, truth_by_term, det_measured[method], informed_union, seed=budget
            )
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
                random_sels,
                random_measured,
                scored,
                landscape,
                truth_by_term,
                informed_union,
                candidate_fitness,
                budget,
                max_order,
            )
        )

    report = Report(
        dataset=dataset,
        model_id=model_id,
        seeds=seeds,
        budgets=list(budgets),
        candidate_alphabet=candidate_alphabet,
        n_candidates=len(scored),
        n_truth_terms=len(truth_by_term),
        var_epsilon=var_epsilon,
        results=results,
    )
    _write_report(report, out_dir)
    return report


def _random_result(
    random_sels: Sequence[Sequence[Variant]],
    random_measured: Sequence[frozenset[Variant]],
    scored: Sequence[ScoredVariant],
    landscape: Mapping[Variant, float],
    truth_by_term: Mapping[Term, float],
    informed_union: frozenset[Term],
    candidate_fitness: Mapping[Variant, float],
    budget: int,
    max_order: int,
) -> MethodResult:
    """Random baseline: recovery averaged over seeds; CI bootstrapped over the per-seed scalars."""
    per_seed = [
        map_recovery(
            infer_epistasis(_measured_dg(landscape, sel), scored, max_order),
            truth_by_term,
            measured,
            informed_union,
            with_ci=False,
        )
        for sel, measured in zip(random_sels, random_measured, strict=True)
    ]
    hit = float(np.mean([hit_rate(sel, candidate_fitness, budget) for sel in random_sels]))

    metrics: list[OrderMetric] = []
    for i, order_name in enumerate(("pairwise", "third", "pooled")):
        template = per_seed[0][i]
        metrics.append(
            OrderMetric(
                order=order_name,
                n_terms=template.n_terms,
                pearson=_mean_metric(per_seed, i, "pearson"),
                spearman=_mean_metric(per_seed, i, "spearman"),
                pearson_ci95=_seed_ci(per_seed, i, "pearson"),
                spearman_ci95=_seed_ci(per_seed, i, "spearman"),
                n_informed=round(float(np.mean([s[i].n_informed for s in per_seed]))),
                coverage_fraction=float(np.mean([s[i].coverage_fraction for s in per_seed])),
                n_informed_union=template.n_informed_union,
                pearson_informed=_mean_metric(per_seed, i, "pearson_informed"),
                spearman_informed=_mean_metric(per_seed, i, "spearman_informed"),
            )
        )
    return MethodResult(
        method="random",
        budget=budget,
        ci_method="bootstrap-over-seeds",
        hit_rate=hit,
        metrics=metrics,
    )


def _values(per_seed: Sequence[Sequence[OrderMetric]], order_idx: int, field: str) -> list[float]:
    return [v for s in per_seed if (v := getattr(s[order_idx], field)) is not None]


def _mean_metric(
    per_seed: Sequence[Sequence[OrderMetric]], order_idx: int, field: str
) -> float | None:
    vals = _values(per_seed, order_idx, field)
    return float(np.mean(vals)) if vals else None


def _seed_ci(
    per_seed: Sequence[Sequence[OrderMetric]], order_idx: int, field: str
) -> tuple[float, float] | None:
    vals = _values(per_seed, order_idx, field)
    if len(vals) < _MIN_POINTS_FOR_CORR:
        return None
    rng = np.random.default_rng(0)
    means = [
        float(np.mean(rng.choice(vals, size=len(vals), replace=True))) for _ in range(_N_BOOTSTRAP)
    ]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _write_report(report: Report, out_dir: Path) -> None:
    """Write metrics.json under ``out_dir`` (which is treated as the run directory)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
