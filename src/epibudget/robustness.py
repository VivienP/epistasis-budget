"""Phase B robustness analyses (docs/specs/robustness.md, docs/VALIDATION.md post-registration).

Three POST-HOC companion analyses on a completed run's inputs (ESM-scored candidates + the full
measured landscape). They never feed selection and never alter the frozen decision rule:

- A1 common-predicted-term precision: compare two methods' precision on the SAME (intersected)
  informed-but-not-pinned terms, so a breadth advantage cannot masquerade as a precision advantage.
- A2 method-independent, five-fold cross-fitted scale sensitivity: re-evaluate recovery with one
  cross-fitted slope per fold (fit out-of-fold on the full measurable landscape) to test whether the
  method ranking survives removing per-method slope-fitting noise. Cross-fitting guards only against
  a member's own fold leaking into its slope; it still consumes full-landscape labels, so the
  numbers are a robustness probe, never an operational recovery figure and never a headline ranking.
- A3 paired / hierarchical difference CIs: bootstrap the difference of two methods' correlations on
  identical terms. Descriptive intervals, never a hypothesis test.

No torch/model/network import: this runs on already-computed scores. Reuses validate.py internals
(imported directly, same package) so the analysed selections and inference match the frozen harness.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isnan, log
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from epibudget.acquisition import allocate, fitness_greedy
from epibudget.epistasis import (
    epsilon_pairwise,
    epsilon_third,
    ground_truth_epistasis,
    interaction_loop,
    predicted_epistasis,
)
from epibudget.graph import EpistasisFactorGraph
from epibudget.provenance import write_json_exclusive
from epibudget.scoring_plan import variant_key
from epibudget.types import Interaction, ScoredVariant, Variant
from epibudget.validate import (
    _MIN_POINTS_FOR_CORR,
    _N_BOOTSTRAP,
    Term,
    _calibrate_slope,
    _candidate_terms,
    _corr,
    _informed,
    _measured_dg,
    _pinned,
    infer_epistasis,
    random_selection,
    structural_graph,
)

_N_FOLDS = 5
_MIN_FOLDS = 2
_ORDERS: tuple[tuple[str, int], ...] = (("pairwise", 2), ("third", 3))
_DET_PAIRS: tuple[tuple[str, str], ...] = (("info", "fitness"), ("info", "structural"))
_STATS: tuple[str, ...] = ("spearman", "pearson")

_DIFF_INTERPRETATION = "descriptive difference on matched terms; NOT a hypothesis test"
_CROSSFIT_CAVEAT = (
    "cross-fitted on full-landscape labels (more label information than an operational run); a "
    "robustness probe of the frozen ranking, NOT an operational recovery number; never quote as a "
    "headline figure and never adopt crossfit_ranking as the reported method order"
)
_REPORT_NOTE = (
    "post-hoc, descriptive; does not alter the frozen decision rule; difference CIs are not "
    "hypothesis tests; A2 crossfit numbers are non-operational robustness probes"
)


class PairDifference(BaseModel):
    """A descriptive difference of two methods' correlations on matched terms (never a test)."""

    method_a: str
    method_b: str
    order: str
    statistic: str
    delta: float | None
    delta_ci95: tuple[float, float] | None
    excludes_zero: bool
    interpretation: str = _DIFF_INTERPRETATION


class CommonPrecision(BaseModel):
    """Precision of two methods on the terms both inform but do not pin (identity intersection)."""

    method_a: str
    method_b: str
    order: str
    n_common: int
    spearman_a: float | None
    spearman_b: float | None
    pearson_a: float | None
    pearson_b: float | None
    # Mean over common terms of (loop members measured / loop size): unequal depth means the
    # precision comparison still partly reflects coverage depth, not only skill — reported here.
    mean_informed_fraction_a: float | None
    mean_informed_fraction_b: float | None
    difference: PairDifference


class ScaleSensitivity(BaseModel):
    """Whether the method ranking survives replacing per-method slopes with a cross-fitted one."""

    order: str
    n_folds: int
    global_ranking: list[str]
    crossfit_ranking: list[str]
    ranking_agrees: bool
    per_method_spearman_global: dict[str, float | None]
    per_method_spearman_crossfit: dict[str, float | None]
    caveat: str = _CROSSFIT_CAVEAT


class RobustnessReport(BaseModel):
    """The three Phase B analyses over the frozen budget grid, per order (never pooled)."""

    dataset: str
    model_id: str
    budgets: list[int]
    seeds: int
    max_order: int
    n_candidates: int
    n_folds: int
    note: str
    common_precision: list[CommonPrecision]
    scale_sensitivity: list[ScaleSensitivity]
    pair_differences: list[PairDifference]


# --------------------------------------------------------------------- folds + cross-fit


def variant_fold(variant: Variant, n_folds: int) -> int:
    """Deterministic, salt-free fold of a variant by its stable identity key (label-free)."""
    return variant_key(sorted(variant)) % n_folds


def crossfit_slopes(
    scored: Sequence[ScoredVariant], landscape: Mapping[Variant, float], n_folds: int = _N_FOLDS
) -> dict[int, float]:
    """One through-origin slope per fold, fit on the measurable candidates OUTSIDE that fold.

    Method-independent (uses the full measurable candidate set, not any method's reveal).
    Out-of-fold fitting keeps a member's own fold from leaking into the slope that prices it.
    """
    if n_folds < _MIN_FOLDS:
        raise ValueError(f"n_folds must be >= {_MIN_FOLDS} for out-of-fold slopes, got {n_folds}")
    rows = [
        (sv.variant, sv.delta_g, log(landscape[sv.variant]))
        for sv in scored
        if landscape.get(sv.variant, 0.0) > 0.0
    ]
    slopes: dict[int, float] = {}
    for fold in range(n_folds):
        esm = [dg for (v, dg, _y) in rows if variant_fold(v, n_folds) != fold]
        measured = [y for (v, _dg, y) in rows if variant_fold(v, n_folds) != fold]
        slopes[fold] = _calibrate_slope(esm, measured)
    return slopes


def infer_epistasis_crossfit(
    revealed: Mapping[Variant, float],
    scored: Sequence[ScoredVariant],
    slopes: Mapping[int, float],
    max_order: int = 3,
    n_folds: int = _N_FOLDS,
) -> list[Interaction]:
    """``infer_epistasis`` with each unmeasured member priced by its own fold's slope."""
    esm = {sv.variant: sv.delta_g for sv in scored}
    tau2 = {sv.variant: sv.var_delta_g for sv in scored}
    mu: dict[Variant, float] = {
        v: (revealed[v] if v in revealed else slopes[variant_fold(v, n_folds)] * esm[v])
        for v in esm
    }
    interactions: list[Interaction] = []
    for term in _candidate_terms(scored, max_order):
        if len(term) == 2:  # noqa: PLR2004 — pairwise vs third, mirrors validate.infer_epistasis
            eps = epsilon_pairwise(mu, term[0], term[1])
        else:
            eps = epsilon_third(mu, term[0], term[1], term[2])
        sigma2 = sum(tau2[member] for member in interaction_loop(term) if member not in revealed)
        interactions.append(Interaction.of(term, epsilon_hat=eps, sigma2=sigma2))
    return interactions


# --------------------------------------------------------------------- correlation helpers


def _finite(value: float | None) -> float | None:
    """Map a NaN correlation (scipy on a near-constant vector) to None, so no NaN reaches JSON."""
    return None if value is None or isnan(value) else value


def _safe_corr(pred: np.ndarray, true: np.ndarray) -> tuple[float | None, float | None]:
    """``_corr`` but NaN → None: it guards exactly-constant vectors, not near-constant ones."""
    pearson, spearman = _corr(pred, true)
    return _finite(pearson), _finite(spearman)


def _stat(pred: np.ndarray, true: np.ndarray, statistic: str) -> float | None:
    pearson, spearman = _safe_corr(pred, true)
    return pearson if statistic == "pearson" else spearman


def _delta(
    pred_a: np.ndarray, pred_b: np.ndarray, true: np.ndarray, statistic: str
) -> float | None:
    ca = _stat(pred_a, true, statistic)
    cb = _stat(pred_b, true, statistic)
    return None if (ca is None or cb is None) else ca - cb


def _excludes_zero(ci: tuple[float, float] | None) -> bool:
    return ci is not None and (ci[0] > 0.0 or ci[1] < 0.0)


def _seed_for(*parts: object) -> int:
    """Stable, salt-free seed from the analysis key, so CIs are reproducible across processes."""
    key = 0
    for part in parts:
        for ch in str(part):
            key = (key * 131 + ord(ch)) % (2**31)
    return key


def paired_difference(
    pred_a: np.ndarray, pred_b: np.ndarray, true: np.ndarray, statistic: str, seed: int
) -> tuple[float | None, tuple[float, float] | None]:
    """Percentile CI of ``corr(A) − corr(B)`` on index-aligned terms (one shared resample)."""
    n = len(true)
    if n < _MIN_POINTS_FOR_CORR:
        return None, None
    delta = _delta(pred_a, pred_b, true, statistic)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(_N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        drawn = _delta(pred_a[idx], pred_b[idx], true[idx], statistic)
        if drawn is not None:
            deltas.append(drawn)
    if len(deltas) < _MIN_POINTS_FOR_CORR:
        return delta, None
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return delta, (float(lo), float(hi))


# --------------------------------------------------------------------- per-method state


class _MethodState:
    """A method's live-revealed set and its ε̂ over the truth terms (frozen global inference)."""

    __slots__ = ("hat", "measured")

    def __init__(self, measured: frozenset[Variant], hat: dict[Term, float]) -> None:
        self.measured = measured
        self.hat = hat


def _method_state(
    selected: Sequence[Variant],
    scored: Sequence[ScoredVariant],
    landscape: Mapping[Variant, float],
    truth_by_term: Mapping[Term, float],
    max_order: int,
) -> _MethodState:
    measured_dg = _measured_dg(landscape, selected)
    inferred = infer_epistasis(measured_dg, scored, max_order)
    hat = {i.mutations: i.epsilon_hat for i in inferred if i.mutations in truth_by_term}
    return _MethodState(frozenset(measured_dg), hat)


def _informed_fraction(term: Term, measured: frozenset[Variant]) -> float:
    loop = interaction_loop(term)
    return sum(1 for member in loop if member in measured) / len(loop)


def _predicted_terms(state: _MethodState, truth_terms: Sequence[Term]) -> set[Term]:
    """Terms this method informs but does not pin — where it had to predict ε, not read it off."""
    return {
        t for t in truth_terms if _informed(t, state.measured) and not _pinned(t, state.measured)
    }


# --------------------------------------------------------------------- A1: common precision


def common_precision(
    method_a: str,
    method_b: str,
    order: str,
    truth_by_term: Mapping[Term, float],
    truth_terms: Sequence[Term],
    states: Mapping[str, _MethodState],
    seed: int,
) -> CommonPrecision:
    """Precision of A and B on ``sorted(predicted(A) ∩ predicted(B))`` — the matched set."""
    sa, sb = states[method_a], states[method_b]
    common = sorted(_predicted_terms(sa, truth_terms) & _predicted_terms(sb, truth_terms))
    if not common:
        empty = PairDifference(
            method_a=method_a,
            method_b=method_b,
            order=order,
            statistic="spearman",
            delta=None,
            delta_ci95=None,
            excludes_zero=False,
        )
        return CommonPrecision(
            method_a=method_a,
            method_b=method_b,
            order=order,
            n_common=0,
            spearman_a=None,
            spearman_b=None,
            pearson_a=None,
            pearson_b=None,
            mean_informed_fraction_a=None,
            mean_informed_fraction_b=None,
            difference=empty,
        )
    true = np.array([truth_by_term[t] for t in common], dtype=np.float64)
    pred_a = np.array([sa.hat[t] for t in common], dtype=np.float64)
    pred_b = np.array([sb.hat[t] for t in common], dtype=np.float64)
    pearson_a, spearman_a = _safe_corr(pred_a, true)
    pearson_b, spearman_b = _safe_corr(pred_b, true)
    delta, ci = paired_difference(pred_a, pred_b, true, "spearman", seed)
    difference = PairDifference(
        method_a=method_a,
        method_b=method_b,
        order=order,
        statistic="spearman",
        delta=delta,
        delta_ci95=ci,
        excludes_zero=_excludes_zero(ci),
    )
    return CommonPrecision(
        method_a=method_a,
        method_b=method_b,
        order=order,
        n_common=len(common),
        spearman_a=spearman_a,
        spearman_b=spearman_b,
        pearson_a=pearson_a,
        pearson_b=pearson_b,
        mean_informed_fraction_a=float(
            np.mean([_informed_fraction(t, sa.measured) for t in common])
        ),
        mean_informed_fraction_b=float(
            np.mean([_informed_fraction(t, sb.measured) for t in common])
        ),
        difference=difference,
    )


# ----------------------------------------------------------------- A3: full-set + hierarchical diff


def _full_paired_diff(
    method_a: str,
    method_b: str,
    order: str,
    statistic: str,
    truth_by_term: Mapping[Term, float],
    truth_terms: Sequence[Term],
    states: Mapping[str, _MethodState],
    seed: int,
) -> PairDifference:
    sa, sb = states[method_a], states[method_b]
    true = np.array([truth_by_term[t] for t in truth_terms], dtype=np.float64)
    pred_a = np.array([sa.hat[t] for t in truth_terms], dtype=np.float64)
    pred_b = np.array([sb.hat[t] for t in truth_terms], dtype=np.float64)
    delta, ci = paired_difference(pred_a, pred_b, true, statistic, seed)
    return PairDifference(
        method_a=method_a,
        method_b=method_b,
        order=order,
        statistic=statistic,
        delta=delta,
        delta_ci95=ci,
        excludes_zero=_excludes_zero(ci),
    )


def hierarchical_random_difference(
    order: str,
    statistic: str,
    truth_by_term: Mapping[Term, float],
    truth_terms: Sequence[Term],
    info_state: _MethodState,
    random_states: Sequence[_MethodState],
    seeds: int,
    seed: int,
) -> PairDifference:
    """info vs random: seeds resampled (outer), a fresh term-resample per drawn seed (inner).

    The random arm's reported recovery is the mean over seeds (matching ``_random_result``), so the
    hierarchical CI brackets its own point estimate.
    """
    true = np.array([truth_by_term[t] for t in truth_terms], dtype=np.float64)
    info_pred = np.array([info_state.hat[t] for t in truth_terms], dtype=np.float64)
    random_preds = [
        np.array([rs.hat[t] for t in truth_terms], dtype=np.float64) for rs in random_states
    ]
    none = PairDifference(
        method_a="info",
        method_b="random",
        order=order,
        statistic=statistic,
        delta=None,
        delta_ci95=None,
        excludes_zero=False,
    )
    n = len(true)
    n_seeds = len(random_preds)
    if n < _MIN_POINTS_FOR_CORR or n_seeds == 0:
        return none

    info_corr = _stat(info_pred, true, statistic)
    rand_corrs = [c for rp in random_preds if (c := _stat(rp, true, statistic)) is not None]
    delta = (
        info_corr - float(np.mean(rand_corrs)) if (info_corr is not None and rand_corrs) else None
    )

    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(_N_BOOTSTRAP):
        info_vals: list[float] = []
        rand_vals: list[float] = []
        for label in rng.integers(0, n_seeds, size=seeds):
            idx = rng.integers(0, n, size=n)
            ic = _stat(info_pred[idx], true[idx], statistic)
            rc = _stat(random_preds[label][idx], true[idx], statistic)
            if ic is not None and rc is not None:
                info_vals.append(ic)
                rand_vals.append(rc)
        if info_vals:
            deltas.append(float(np.mean(info_vals)) - float(np.mean(rand_vals)))
    if len(deltas) < _MIN_POINTS_FOR_CORR:
        return PairDifference(
            method_a="info",
            method_b="random",
            order=order,
            statistic=statistic,
            delta=delta,
            delta_ci95=None,
            excludes_zero=False,
        )
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    ci = (float(lo), float(hi))
    return PairDifference(
        method_a="info",
        method_b="random",
        order=order,
        statistic=statistic,
        delta=delta,
        delta_ci95=ci,
        excludes_zero=_excludes_zero(ci),
    )


# --------------------------------------------------------------------- A2: scale sensitivity


def _rank(scores: Mapping[str, float | None]) -> list[str]:
    """Methods best-first by score (None last; ties by name for cross-process determinism)."""
    return sorted(scores, key=lambda m: (scores[m] is None, -(scores[m] or 0.0), m))


def _scale_sensitivity(
    order: str,
    truth_by_term: Mapping[Term, float],
    truth_terms: Sequence[Term],
    scored: Sequence[ScoredVariant],
    landscape: Mapping[Variant, float],
    slopes: Mapping[int, float],
    selections: Mapping[str, Sequence[Variant]],
    states: Mapping[str, _MethodState],
    max_order: int,
    n_folds: int,
) -> ScaleSensitivity:
    true = np.array([truth_by_term[t] for t in truth_terms], dtype=np.float64)
    global_sp: dict[str, float | None] = {}
    crossfit_sp: dict[str, float | None] = {}
    for method in ("info", "fitness", "structural"):
        global_pred = np.array([states[method].hat[t] for t in truth_terms], dtype=np.float64)
        _, global_sp[method] = _safe_corr(global_pred, true)

        measured_dg = _measured_dg(landscape, selections[method])
        crossfit = infer_epistasis_crossfit(measured_dg, scored, slopes, max_order, n_folds)
        cf_hat = {i.mutations: i.epsilon_hat for i in crossfit if i.mutations in truth_by_term}
        crossfit_pred = np.array([cf_hat[t] for t in truth_terms], dtype=np.float64)
        _, crossfit_sp[method] = _safe_corr(crossfit_pred, true)

    global_ranking = _rank(global_sp)
    crossfit_ranking = _rank(crossfit_sp)
    return ScaleSensitivity(
        order=order,
        n_folds=n_folds,
        global_ranking=global_ranking,
        crossfit_ranking=crossfit_ranking,
        ranking_agrees=global_ranking == crossfit_ranking,
        per_method_spearman_global=global_sp,
        per_method_spearman_crossfit=crossfit_sp,
    )


# --------------------------------------------------------------------- orchestration


def _truth_by_term(
    scored: Sequence[ScoredVariant], landscape: Mapping[Variant, float], max_order: int
) -> dict[Term, float]:
    # Mirrors run_validation's truth construction (validate.py) so both grade the same terms.
    term_set = set(_candidate_terms(scored, max_order))
    landscape_dg = {v: log(f) for v, f in landscape.items() if f > 0.0}
    return {
        interaction.mutations: interaction.epsilon_hat
        for interaction in ground_truth_epistasis(landscape_dg, max_order)
        if interaction.mutations in term_set
    }


def _deterministic_selections(
    scored: Sequence[ScoredVariant], budget: int, max_order: int
) -> dict[str, list[Variant]]:
    graph = EpistasisFactorGraph(
        predicted_epistasis(scored, max_order), {sv.variant: sv.var_delta_g for sv in scored}
    )
    structural = structural_graph(scored, max_order)
    return {
        "info": allocate(graph, scored, budget, lambda_=0.0).selected,
        "fitness": fitness_greedy(scored, budget),
        "structural": allocate(structural, scored, budget, lambda_=0.0).selected,
    }


def robustness_report(
    scored: Sequence[ScoredVariant],
    landscape: Mapping[Variant, float],
    budgets: Sequence[int],
    seeds: int,
    *,
    max_order: int = 3,
    n_folds: int = _N_FOLDS,
    dataset: str = "gb1_wu2016",
    model_id: str = "",
    out_dir: Path | None = None,
) -> RobustnessReport:
    """Run A1/A2/A3 over the budget grid, per order; optionally write ``robustness.json``."""
    truth = _truth_by_term(scored, landscape, max_order)
    slopes = crossfit_slopes(scored, landscape, n_folds)

    common_precision_out: list[CommonPrecision] = []
    scale_out: list[ScaleSensitivity] = []
    pair_diff_out: list[PairDifference] = []

    for budget in budgets:
        deterministic = _deterministic_selections(scored, budget, max_order)
        states = {
            method: _method_state(sel, scored, landscape, truth, max_order)
            for method, sel in deterministic.items()
        }
        random_states = [
            _method_state(random_selection(scored, budget, s), scored, landscape, truth, max_order)
            for s in range(seeds)
        ]

        for order, order_n in _ORDERS:
            truth_terms = sorted(t for t in truth if len(t) == order_n)
            if len(truth_terms) < _MIN_POINTS_FOR_CORR:
                continue
            for method_a, method_b in _DET_PAIRS:
                seed = _seed_for(budget, order, method_a, method_b)
                common_precision_out.append(
                    common_precision(method_a, method_b, order, truth, truth_terms, states, seed)
                )
                for statistic in _STATS:
                    pair_diff_out.append(
                        _full_paired_diff(
                            method_a,
                            method_b,
                            order,
                            statistic,
                            truth,
                            truth_terms,
                            states,
                            _seed_for(budget, order, method_a, method_b, statistic),
                        )
                    )
            for statistic in _STATS:
                pair_diff_out.append(
                    hierarchical_random_difference(
                        order,
                        statistic,
                        truth,
                        truth_terms,
                        states["info"],
                        random_states,
                        seeds,
                        _seed_for(budget, order, "info", "random", statistic),
                    )
                )
            scale_out.append(
                _scale_sensitivity(
                    order,
                    truth,
                    truth_terms,
                    scored,
                    landscape,
                    slopes,
                    deterministic,
                    states,
                    max_order,
                    n_folds,
                )
            )

    report = RobustnessReport(
        dataset=dataset,
        model_id=model_id,
        budgets=list(budgets),
        seeds=seeds,
        max_order=max_order,
        n_candidates=len(scored),
        n_folds=n_folds,
        note=_REPORT_NOTE,
        common_precision=common_precision_out,
        scale_sensitivity=scale_out,
        pair_differences=pair_diff_out,
    )
    if out_dir is not None:
        write_json_exclusive(out_dir / "robustness.json", report.model_dump(mode="json"))
    return report
