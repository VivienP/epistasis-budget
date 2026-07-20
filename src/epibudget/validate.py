"""Landscape validation harness (dataset-generic; registered landscapes in data.DATASETS).

Frozen protocol in docs/VALIDATION.md. Compares five methods at each budget: info-optimal,
fitness-greedy, structural-only, random, and practice. Recovery is evaluated against the eligible
ground-truth terms of the selected landscape. Info, fitness, and random form the frozen decision
rule; structural and practice are companions.

Isolation. The SAME ``infer_epistasis`` runs per method; only the *selected set* differs, so the
comparison isolates selection, not inference. Selection is zero-shot (ESM predictions + the factor
graph). Measured fitness enters exactly once, via ``data.reveal_measured_fitness``, strictly after a
method has returned its selection (docs/VALIDATION.md threats table).

Inference (docs/VALIDATION.md §Simulation). ``infer_epistasis`` is the closed-form posterior mean of
the linear-Gaussian model in graph.py: measuring a variant pins its ΔG, and every unmeasured loop
member keeps its unit-calibrated ESM prior mean. This is a Tikhonov estimator (prior mean = the
calibrated ESM ΔĜ, precision = 1/var_delta_g) — a precisification of "regularised least squares over
the interaction basis" that keeps selection and grading on one coherent model.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping, Sequence
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
    wt_centered_log_fitness,
)
from epibudget.graph import EpistasisFactorGraph
from epibudget.provenance import write_json_exclusive
from epibudget.types import Interaction, Mutation, ScoredVariant, Variant

_PAIRWISE_ORDER = 2
_MIN_POINTS_FOR_CORR = 3
_N_BOOTSTRAP = 1000

Term = tuple[Mutation, ...]


class OrderMetric(BaseModel):
    """Recovery for one order (or pooled): the primary correlation plus a breadth/precision split.

    ``coverage_fraction`` / ``n_informed`` count terms this method's selection touches. ``n_pinned``
    counts terms whose *entire* loop is measured — recovered exactly (the breadth signal).
    ``pearson_predicted`` / ``spearman_predicted`` are over terms the method informs but does not
    fully pin, where it must genuinely predict ε (the precision signal). A non-tautological
    info-optimal advantage must appear in precision, not only breadth.
    """

    order: str  # "pairwise" | "third" | "pooled"
    n_terms: int
    pearson: float | None
    spearman: float | None
    pearson_ci95: tuple[float, float] | None
    spearman_ci95: tuple[float, float] | None
    n_informed: int  # terms this method's selection touches (loop ∩ measured ≠ ∅)
    coverage_fraction: float  # n_informed / n_terms
    n_pinned: int  # terms whose full loop is measured (recovered exactly) — breadth
    pearson_predicted: float | None  # over informed-but-not-pinned terms — precision
    spearman_predicted: float | None


class MethodResult(BaseModel):
    method: str  # "info" | "fitness" | "structural" | "random" | "practice"
    budget: int
    ci_method: str  # "bootstrap-over-terms" | "bootstrap-over-seeds"
    hit_rate: float
    metrics: list[OrderMetric]


class Report(BaseModel):
    dataset: str
    model_id: str
    seeds: int  # random-baseline seed count
    budgets: list[int]
    # Provenance — every run embeds enough to reproduce it (docs/VALIDATION.md §Reproducibility).
    candidate_alphabet: str  # the per-site alphabet the candidate pool was drawn from
    scorer_seed: int  # the ConjointScorer seed (deterministic var_delta_g)
    n_perturbations: int  # masking passes behind var_delta_g
    device: str  # the resolved compute device the scoring ran on (cpu / cuda)
    max_order: int
    data_sha256: str  # checksum of the dataset the landscape/truth came from
    sites: list[int]  # 0-indexed target sites of the landscape (dataset-specific)
    wt_sha256: str  # checksum of the reference (WT) sequence the ε anchor is defined against
    n_candidates: int
    n_truth_terms: int
    var_epsilon: float  # ground-truth variance, reported separately from invariant #1
    var_predicted_epsilon: float  # descriptive spread; not the invariant #1 gate
    predicted_epistasis_signal: bool
    predicted_epistasis_tolerance: float
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


def _predicted_epistasis_tolerance(scored: Sequence[ScoredVariant], max_order: int) -> float:
    """Roundoff threshold ``eps64 * max|ΔĜ| * (2**max_order - 1)``.

    The final factor bounds the number of signed additions in the largest inclusion-exclusion loop.
    """
    max_abs_delta_g = max((abs(sv.delta_g) for sv in scored), default=0.0)
    loop_operation_bound = (1 << max_order) - 1
    return float(np.finfo(np.float64).eps * max_abs_delta_g * loop_operation_bound)


def esm_prior_mu(
    scored: Sequence[ScoredVariant], revealed: Mapping[Variant, float]
) -> dict[Variant, float]:
    """The posterior-mean ΔG map ``infer_epistasis`` conditions on: measured pinned, else prior.

    ``mu[v] = revealed[v]`` if ``v`` was measured, else ``b * esm[v]`` with ``b`` the through-origin
    slope calibrated on the revealed set (:func:`_calibrate_slope`). For an unrevealed (e.g.
    held-out) variant this is exactly the ESM prior on the measured scale — the circular quantity
    that makes a prediction for such a variant unable to demonstrate learned downstream
    information. Used by both :func:`infer_epistasis` and the downstream benchmark's
    ``esm_circular_diagnostic``.
    """
    esm = {sv.variant: sv.delta_g for sv in scored}
    b = _calibrate_slope([esm[v] for v in revealed], [revealed[v] for v in revealed])
    return {v: (revealed[v] if v in revealed else b * esm[v]) for v in esm}


def infer_epistasis(
    revealed: Mapping[Variant, float],
    scored: Sequence[ScoredVariant],
    max_order: int = 3,
) -> list[Interaction]:
    """Posterior-mean ε̂ over the order-2..max_order terms from the measured ΔG of revealed variants.

    ``revealed`` maps each measured variant to its WT-centred log-fitness ΔG; ``scored`` supplies
    the ESM ΔĜ and var_delta_g for every candidate (all orders, as loop members). Each unmeasured
    loop member keeps its unit-calibrated ESM prior mean; measured members are pinned to their
    true ΔG.
    Round-trips: empty ``revealed`` reproduces the ESM ε̂; a fully-measured loop reproduces true ε.
    """
    tau2 = {sv.variant: sv.var_delta_g for sv in scored}
    mu = esm_prior_mu(scored, revealed)

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


def _pinned(term: Term, measured: frozenset[Variant]) -> bool:
    """True iff every loop member of ``term`` was measured — its ε is then recovered exactly."""
    return all(member in measured for member in interaction_loop(term))


def _order_label(order: int) -> str:
    return "pairwise" if order == _PAIRWISE_ORDER else "third"


def _order_metric(
    order_name: str,
    rows: list[tuple[Term, float, float]],
    measured: frozenset[Variant],
    seed: int,
    with_ci: bool,
) -> OrderMetric:
    """Build one OrderMetric from (term, inferred, true) rows, with the breadth/precision split."""
    pred = np.array([r[1] for r in rows], dtype=np.float64)
    true = np.array([r[2] for r in rows], dtype=np.float64)
    pearson, spearman = _corr(pred, true)
    pearson_ci = _bootstrap_ci(pred, true, "pearson", seed) if with_ci else None
    spearman_ci = _bootstrap_ci(pred, true, "spearman", seed + 1) if with_ci else None

    n_covered = sum(1 for r in rows if _informed(r[0], measured))
    n_pinned = sum(1 for r in rows if _pinned(r[0], measured))
    # Precision: terms informed but NOT fully pinned — the method had to predict, not just read off.
    predicted = [r for r in rows if _informed(r[0], measured) and not _pinned(r[0], measured)]
    pp = np.array([r[1] for r in predicted], dtype=np.float64)
    pt = np.array([r[2] for r in predicted], dtype=np.float64)
    pearson_pred, spearman_pred = _corr(pp, pt)

    return OrderMetric(
        order=order_name,
        n_terms=len(rows),
        pearson=pearson,
        spearman=spearman,
        pearson_ci95=pearson_ci,
        spearman_ci95=spearman_ci,
        n_informed=n_covered,
        coverage_fraction=n_covered / len(rows) if rows else 0.0,
        n_pinned=n_pinned,
        pearson_predicted=pearson_pred,
        spearman_predicted=spearman_pred,
    )


def map_recovery(
    inferred: Sequence[Interaction],
    truth_by_term: Mapping[Term, float],
    measured: frozenset[Variant],
    seed: int = 0,
    with_ci: bool = True,
) -> list[OrderMetric]:
    """Per-order and pooled recovery of inferred ε̂ vs ground-truth ε over the evaluated terms.

    ``measured`` is this method's (live) selection, driving the per-method breadth/precision split.
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
        metrics.append(_order_metric(name, rows, measured, seed, with_ci))
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


def structural_graph(scored: Sequence[ScoredVariant], max_order: int = 3) -> EpistasisFactorGraph:
    """A factor graph with τ² ≡ 1, so ``info_gain(∅, v) = n(v)`` — the structural-only ablation.

    Ranking by this weight ignores the ESM masking-perturbation dispersion: it selects by how many
    interaction loops a variant braces. If info-optimal (which uses the real τ²) does not beat
    selection by this graph, the ESM uncertainty prior contributes nothing to the allocation.
    """
    return EpistasisFactorGraph(
        predicted_epistasis(scored, max_order), {sv.variant: 1.0 for sv in scored}
    )


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
    """Reveal WT plus the selection.

    Return live selected values on the WT-centered log scale.
    """
    wt: Variant = frozenset()
    requested = [wt, *(variant for variant in selected if variant != wt)]
    centered = wt_centered_log_fitness(reveal_measured_fitness(dict(landscape), requested))
    return {variant: centered[variant] for variant in selected if variant and variant in centered}


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
    scorer_seed: int = 0,
    n_perturbations: int = 0,
    device: str = "cpu",
    data_sha256: str = "",
    wt_sequence: str = "",
    sites: Sequence[int] = (),
) -> Report:
    """Execute the frozen protocol and write ``<out_dir>/metrics.json``. See docs/VALIDATION.md.

    ``scored`` are the ESM-scored candidates (orders 1..max_order); ``landscape`` is the full
    {variant → fitness} landscape. The selection graph is built from ``predicted_epistasis`` (ESM)
    only, never from ``ground_truth_epistasis`` — so no label can leak into selection.

    ``dataset``, ``wt_sequence`` and ``sites`` are recorded verbatim as provenance so a report is
    self-describing about which landscape and reference construct produced it (GB1 vs TrpB).
    """
    term_set = set(_candidate_terms(scored, max_order))
    landscape_dg = wt_centered_log_fitness(landscape)
    truth_by_term: dict[Term, float] = {
        interaction.mutations: interaction.epsilon_hat
        for interaction in ground_truth_epistasis(landscape_dg, max_order)
        if interaction.mutations in term_set
    }
    var_epsilon = float(np.var(np.array(list(truth_by_term.values())))) if truth_by_term else 0.0

    predicted = predicted_epistasis(scored, max_order)
    predicted_values = np.array([interaction.epsilon_hat for interaction in predicted])
    var_predicted_epsilon = float(np.var(predicted_values)) if len(predicted_values) else 0.0
    predicted_epistasis_tolerance = _predicted_epistasis_tolerance(scored, max_order)
    predicted_epistasis_signal = any(
        abs(interaction.epsilon_hat) > predicted_epistasis_tolerance for interaction in predicted
    )
    graph = EpistasisFactorGraph(predicted, {sv.variant: sv.var_delta_g for sv in scored})
    # τ² ≡ const ablation
    structural = EpistasisFactorGraph(predicted, {sv.variant: 1.0 for sv in scored})
    candidate_fitness = _candidate_fitness(scored, landscape)

    results: list[MethodResult] = []
    for budget in budgets:
        deterministic = {
            "info": allocate(graph, scored, budget, lambda_=0.0).selected,
            "fitness": fitness_greedy(scored, budget),
            "structural": allocate(structural, scored, budget, lambda_=0.0).selected,
            "practice": practice_heuristic(scored, budget),
        }
        random_sels = [random_selection(scored, budget, s) for s in range(seeds)]

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

    report = Report(
        dataset=dataset,
        model_id=model_id,
        seeds=seeds,
        budgets=list(budgets),
        candidate_alphabet=candidate_alphabet,
        scorer_seed=scorer_seed,
        n_perturbations=n_perturbations,
        device=device,
        max_order=max_order,
        data_sha256=data_sha256,
        sites=list(sites),
        wt_sha256=hashlib.sha256(wt_sequence.encode("ascii")).hexdigest() if wt_sequence else "",
        n_candidates=len(scored),
        n_truth_terms=len(truth_by_term),
        var_epsilon=var_epsilon,
        var_predicted_epsilon=var_predicted_epsilon,
        predicted_epistasis_signal=predicted_epistasis_signal,
        predicted_epistasis_tolerance=predicted_epistasis_tolerance,
        results=results,
    )
    _write_report(report, out_dir)
    return report


def _random_result(
    random_sels: Sequence[Sequence[Variant]],
    scored: Sequence[ScoredVariant],
    landscape: Mapping[Variant, float],
    truth_by_term: Mapping[Term, float],
    candidate_fitness: Mapping[Variant, float],
    budget: int,
    max_order: int,
) -> MethodResult:
    """Random baseline: recovery averaged over seeds; CI bootstrapped over the per-seed scalars."""
    per_seed = [
        map_recovery(
            infer_epistasis(_measured_dg(landscape, sel), scored, max_order),
            truth_by_term,
            frozenset(_measured_dg(landscape, sel)),
            with_ci=False,
        )
        for sel in random_sels
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
                n_pinned=round(float(np.mean([s[i].n_pinned for s in per_seed]))),
                pearson_predicted=_mean_metric(per_seed, i, "pearson_predicted"),
                spearman_predicted=_mean_metric(per_seed, i, "spearman_predicted"),
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
    write_json_exclusive(out_dir / "metrics.json", report.model_dump(mode="json"))
