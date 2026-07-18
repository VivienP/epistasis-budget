"""Gate 3 — correlated-error inference probe (cache-only, zero-GPU).

See docs/specs/gate3-correlated-inference.md. Gate 2 showed that revealing info-optimal measurements
improves the rank of recovered epistasis but nearly doubles the squared error, because ε is an
inclusion–exclusion difference that cancels positively-correlated nested ESM error and hard-pinning
breaks that cancellation. This module swaps only the inferrer: it conditions a correlated-error
prior over ΔG (additive random effects across shared sub-mutations) on the measured set, and asks
whether that closes the SSE gap without destroying the rank gain. It decides repair vs replace for
the inference model; it never selects, and its hyper-parameter is fit on measured errors only.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import combinations

import numpy as np
import numpy.typing as npt
from scipy.stats import pearsonr, rankdata, spearmanr

from epibudget.acquisition import allocate
from epibudget.epistasis import interaction_loop, predicted_epistasis
from epibudget.gate2 import (
    Term,
    _canonical_scored,
    _center_positive,
    _epsilon,
    _reveal_selection,
    _shared_slopes,
    _truth_terms,
)
from epibudget.graph import EpistasisFactorGraph
from epibudget.robustness import variant_fold
from epibudget.types import Mutation, ScoredVariant, Variant

FloatArray = npt.NDArray[np.float64]

_ORDERS: tuple[tuple[int, str], ...] = ((2, "pairwise"), (3, "third"))
_MIN_CORR_POINTS = 3
# λ frontier: ∞ (pin) → 0⁺ (full additive correction). np.inf is the pin baseline.
_LAMBDA_GRID: tuple[float, ...] = (
    float("inf"),
    1e3,
    1e2,
    3e1,
    1e1,
    3.0,
    1.0,
    3e-1,
    1e-1,
    3e-2,
    1e-2,
    1e-3,
)


@dataclass(frozen=True)
class OrderCell:
    """Recovery of one interaction order at one budget, for prior / pin / correlated inferrers."""

    order: str
    n_terms: int
    n_measured: int
    sse_prior: float
    pin_pearson: float | None
    pin_spearman: float | None
    pin_sse_gain: float | None
    corr_lambda: float
    corr_pearson: float | None
    corr_spearman: float | None
    corr_sse_gain: float | None
    corr_delta_spearman: float | None
    corr_delta_spearman_ci95: tuple[float, float] | None
    corr_delta_pearson: float | None
    # Main-effect-controlled (residualized) recovery: rank after removing the shared measured-member
    # skeleton k(S). Isolates genuine interaction recovery from the main-effect-sharing confound.
    skeleton_confound_spearman: float | None
    residual_prior_spearman: float | None
    residual_pin_spearman: float | None
    residual_corr_spearman: float | None
    residual_corr_pearson: float | None
    residual_corr_delta_spearman: float | None
    residual_corr_delta_spearman_ci95: tuple[float, float] | None
    # (λ, sse_gain, delta_spearman) along the whole frontier, pin at λ=inf.
    frontier: list[tuple[float, float | None, float | None]]


@dataclass(frozen=True)
class BudgetResult:
    budget: int
    basis: str
    lambda_star: float
    lambda_identified: bool
    cells: list[OrderCell]


@dataclass(frozen=True)
class Gate3Result:
    dataset: str
    model_id: str
    budgets: list[int]
    bases: list[str]
    public_claim_eligible: bool
    results: list[BudgetResult]
    decision: str
    decision_reason: str
    per_basis_decision: dict[str, str] = field(default_factory=dict)


def _safe_corr(pred: FloatArray, truth: FloatArray) -> tuple[float | None, float | None]:
    if len(truth) < _MIN_CORR_POINTS or float(np.std(pred)) == 0.0 or float(np.std(truth)) == 0.0:
        return None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pearson = float(pearsonr(pred, truth).statistic)
        spearman = float(spearmanr(pred, truth).statistic)
    p = pearson if np.isfinite(pearson) else None
    s = spearman if np.isfinite(spearman) else None
    return p, s


def _spearman(pred: FloatArray, truth: FloatArray) -> float | None:
    return _safe_corr(pred, truth)[1]


def _sub_effects(variant: Variant, max_effect_order: int) -> list[frozenset[Mutation]]:
    """Non-empty sub-mutation groups of ``variant`` up to ``max_effect_order`` (the RE basis)."""
    members = sorted(variant)
    out: list[frozenset[Mutation]] = []
    for size in range(1, min(max_effect_order, len(members)) + 1):
        out.extend(frozenset(combo) for combo in combinations(members, size))
    return out


def _effect_index(
    measured: Sequence[Variant], max_effect_order: int
) -> dict[frozenset[Mutation], int]:
    """Columns of the additive design: every sub-effect present in the measured set (estimable)."""
    effects: set[frozenset[Mutation]] = set()
    for variant in measured:
        effects.update(_sub_effects(variant, max_effect_order))
    return {effect: index for index, effect in enumerate(sorted(effects, key=sorted))}


def _incidence(
    variants: Sequence[Variant],
    effect_index: Mapping[frozenset[Mutation], int],
    max_effect_order: int,
) -> FloatArray:
    """Row-per-variant 0/1 design ``G[v, effect] = 1[effect ⊆ v]`` over the effect columns."""
    matrix = np.zeros((len(variants), len(effect_index)), dtype=np.float64)
    for row, variant in enumerate(variants):
        for effect in _sub_effects(variant, max_effect_order):
            column = effect_index.get(effect)
            if column is not None:
                matrix[row, column] = 1.0
    return matrix


def _ridge_blup(design_m: FloatArray, error_m: FloatArray, lam: float) -> FloatArray:
    """â = (GᵀG + λI)⁻¹ Gᵀe — the additive random-effect BLUP / Gaussian-conditioning weights."""
    p = design_m.shape[1]
    if p == 0:
        return np.zeros(0, dtype=np.float64)
    gram = design_m.T @ design_m + lam * np.eye(p, dtype=np.float64)
    return np.asarray(np.linalg.solve(gram, design_m.T @ error_m), dtype=np.float64)


def _gcv_lambda(design_m: FloatArray, error_m: FloatArray, grid: Sequence[float]) -> float:
    """Pick λ by generalized cross-validation on the measured errors only (leakage-safe)."""
    n, p = design_m.shape
    if n == 0 or p == 0:
        return float("inf")
    gram = design_m.T @ design_m
    best_lambda = float("inf")
    best_score = float("inf")
    for lam in grid:
        if not np.isfinite(lam):
            residual = error_m  # λ=∞ ⇒ â=0 ⇒ fitted 0
            trace_h = 0.0
        else:
            inverse = np.linalg.solve(gram + lam * np.eye(p, dtype=np.float64), np.eye(p))
            hat_diag = np.einsum("ij,jk,ik->i", design_m, inverse, design_m)
            trace_h = float(np.sum(hat_diag))
            residual = error_m - design_m @ (inverse @ (design_m.T @ error_m))
        denom = n - trace_h
        if denom <= 0.0:
            continue
        score = n * float(np.dot(residual, residual)) / (denom * denom)
        if score < best_score:
            best_score, best_lambda = score, lam
    return best_lambda


def _predicted_error(
    needed: Sequence[Variant],
    effect_index: Mapping[frozenset[Mutation], int],
    coefficients: FloatArray,
    max_effect_order: int,
) -> dict[Variant, float]:
    if len(effect_index) == 0:
        return dict.fromkeys(needed, 0.0)
    design = _incidence(needed, effect_index, max_effect_order)
    predicted = design @ coefficients
    return {variant: float(predicted[row]) for row, variant in enumerate(needed)}


def _bootstrap_delta_spearman_ci(
    pred_prior: FloatArray, pred_post: FloatArray, truth: FloatArray, iterations: int, seed: int
) -> tuple[float, float] | None:
    n = len(truth)
    if n < _MIN_CORR_POINTS:
        return None
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        prior_s = _spearman(pred_prior[idx], truth[idx])
        post_s = _spearman(pred_post[idx], truth[idx])
        if prior_s is not None and post_s is not None:
            samples.append(post_s - prior_s)
    if len(samples) < iterations // 2:
        return None
    array = np.array(samples, dtype=np.float64)
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def _skeleton(term: Term, measured: frozenset[Variant], true_dg: Mapping[Variant, float]) -> float:
    """k(S) = Σ_{T ∈ loop(S), T measured} c_T·true(T): the component shared by ε̂ and truth.

    Both the pinned ε̂ and the truth use the true ΔG for measured loop members, so this is the shared
    measured skeleton; controlling for it isolates interaction recovery. Measured labels only.
    """
    order = len(term)
    total = 0.0
    for member in interaction_loop(term):
        if member in measured:
            sign = 1.0 if (order - len(member)) % 2 == 0 else -1.0
            total += sign * true_dg[member]
    return total


def _residualize(values: FloatArray, control: FloatArray) -> FloatArray:
    """OLS residual of ``values`` on ``[1, control]`` (centering if ``control`` is constant)."""
    if float(np.std(control)) == 0.0:
        return values - float(np.mean(values))
    design = np.column_stack([np.ones_like(control), control])
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    return np.asarray(values - design @ beta, dtype=np.float64)


def _partial_pearson(pred: FloatArray, truth: FloatArray, control: FloatArray) -> float | None:
    """Pearson of ``pred`` vs ``truth`` after residualizing both on ``control``."""
    rp = _residualize(pred, control)
    rt = _residualize(truth, control)
    if float(np.std(rp)) == 0.0 or float(np.std(rt)) == 0.0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = float(pearsonr(rp, rt).statistic)
    return value if np.isfinite(value) else None


def _partial_spearman(pred: FloatArray, truth: FloatArray, control: FloatArray) -> float | None:
    """Rank-based partial correlation: partial Pearson on rank-transformed inputs."""
    if len(truth) < _MIN_CORR_POINTS:
        return None
    return _partial_pearson(rankdata(pred), rankdata(truth), rankdata(control))


def _bootstrap_residual_delta_ci(
    prior: FloatArray,
    corr: FloatArray,
    truth: FloatArray,
    control: FloatArray,
    iterations: int,
    seed: int,
) -> tuple[float, float] | None:
    """Percentile CI of residual_spearman(corr) − residual_spearman(prior) over resampled terms."""
    n = len(truth)
    if n < _MIN_CORR_POINTS:
        return None
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        prior_s = _partial_spearman(prior[idx], truth[idx], control[idx])
        corr_s = _partial_spearman(corr[idx], truth[idx], control[idx])
        if prior_s is not None and corr_s is not None:
            samples.append(corr_s - prior_s)
    if len(samples) < iterations // 2:
        return None
    array = np.array(samples, dtype=np.float64)
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def _info_selection(scored: Sequence[ScoredVariant], budget: int, max_order: int) -> list[Variant]:
    """The frozen Gate-2 info allocation (ESM-only, label-free) — identical to gate2's selection."""
    interactions = predicted_epistasis(scored, max_order)
    graph = EpistasisFactorGraph(interactions, {item.variant: item.var_delta_g for item in scored})
    return allocate(graph, scored, budget, lambda_=0.0).selected


def _order_terms(truth_by_order: Mapping[int, Mapping[Term, float]]) -> dict[int, list[Term]]:
    return {order: sorted(truth_by_order[order]) for order, _ in _ORDERS}


def _order_cell(
    name: str,
    terms: Sequence[Term],
    truth_values: Mapping[Term, float],
    mu0: Mapping[Variant, float],
    mu_pin: Mapping[Variant, float],
    mu_star: Mapping[Variant, float],
    mu_for_lambda: Callable[[float], dict[Variant, float]],
    measured_set: frozenset[Variant],
    true_dg: Mapping[Variant, float],
    n_measured: int,
    lambda_star: float,
    bootstrap_iterations: int,
    seed: int,
) -> OrderCell:
    """Recovery of one order: calibration, raw rank, and residualized (confound-free) rank."""
    truth = np.array([truth_values[term] for term in terms], dtype=np.float64)
    prior = np.array([_epsilon(mu0, term) for term in terms], dtype=np.float64)
    pin = np.array([_epsilon(mu_pin, term) for term in terms], dtype=np.float64)
    star = np.array([_epsilon(mu_star, term) for term in terms], dtype=np.float64)

    sse_prior = float(np.sum(np.square(prior - truth)))
    prior_p, prior_s = _safe_corr(prior, truth)
    pin_p, pin_s = _safe_corr(pin, truth)
    star_p, star_s = _safe_corr(star, truth)

    def _gain(values: FloatArray) -> float | None:
        if sse_prior == 0.0:
            return None
        return 1.0 - float(np.sum(np.square(values - truth))) / sse_prior

    frontier: list[tuple[float, float | None, float | None]] = []
    for lam in _LAMBDA_GRID:
        mu_lam = mu_pin if not np.isfinite(lam) else mu_for_lambda(lam)
        eps_lam = np.array([_epsilon(mu_lam, term) for term in terms], dtype=np.float64)
        lam_s = _spearman(eps_lam, truth)
        d_s = None if (lam_s is None or prior_s is None) else lam_s - prior_s
        frontier.append((lam, _gain(eps_lam), d_s))

    star_ci = _bootstrap_delta_spearman_ci(prior, star, truth, bootstrap_iterations, seed=seed)

    # Main-effect-controlled recovery: residualize on the shared measured skeleton k(S).
    skeleton = np.array(
        [_skeleton(term, measured_set, true_dg) for term in terms], dtype=np.float64
    )
    res_prior_s = _partial_spearman(prior, truth, skeleton)
    res_corr_s = _partial_spearman(star, truth, skeleton)
    res_delta = None if (res_corr_s is None or res_prior_s is None) else res_corr_s - res_prior_s
    res_delta_ci = _bootstrap_residual_delta_ci(
        prior, star, truth, skeleton, bootstrap_iterations, seed=seed
    )
    return OrderCell(
        order=name,
        n_terms=len(terms),
        n_measured=n_measured,
        sse_prior=sse_prior,
        pin_pearson=pin_p,
        pin_spearman=pin_s,
        pin_sse_gain=_gain(pin),
        corr_lambda=lambda_star,
        corr_pearson=star_p,
        corr_spearman=star_s,
        corr_sse_gain=_gain(star),
        corr_delta_spearman=None if (star_s is None or prior_s is None) else star_s - prior_s,
        corr_delta_spearman_ci95=star_ci,
        corr_delta_pearson=None if (star_p is None or prior_p is None) else star_p - prior_p,
        skeleton_confound_spearman=_safe_corr(skeleton, truth)[1],
        residual_prior_spearman=res_prior_s,
        residual_pin_spearman=_partial_spearman(pin, truth, skeleton),
        residual_corr_spearman=res_corr_s,
        residual_corr_pearson=_partial_pearson(star, truth, skeleton),
        residual_corr_delta_spearman=res_delta,
        residual_corr_delta_spearman_ci95=res_delta_ci,
        frontier=frontier,
    )


def evaluate_budget(
    scored: Sequence[ScoredVariant],
    landscape: dict[Variant, float],
    slopes: Mapping[Variant, float],
    truth_by_order: Mapping[int, Mapping[Term, float]],
    true_dg: Mapping[Variant, float],
    budget: int,
    basis: str,
    bootstrap_iterations: int,
    max_order: int,
) -> BudgetResult:
    max_effect_order = 1 if basis == "single" else 2
    esm = {item.variant: item.delta_g for item in scored}
    mu0 = {variant: slopes[variant] * value for variant, value in esm.items()}

    selected = _info_selection(scored, budget, max_order)
    revealed, *_ = _reveal_selection(landscape, selected)
    measured = sorted(revealed, key=sorted)
    error_m = {variant: mu0[variant] - revealed[variant] for variant in measured}

    effect_index = _effect_index(measured, max_effect_order)
    design_m = _incidence(measured, effect_index, max_effect_order)
    error_vector = np.array([error_m[variant] for variant in measured], dtype=np.float64)
    lambda_star = _gcv_lambda(design_m, error_vector, _LAMBDA_GRID)
    # λ is identified only with measured redundancy; a saturated design (rows ≤ cols) is flat in λ.
    lambda_identified = design_m.shape[0] > design_m.shape[1]
    measured_set = frozenset(measured)

    terms_by_order = _order_terms(truth_by_order)
    needed: set[Variant] = set(measured)
    for terms in terms_by_order.values():
        for term in terms:
            needed.update(interaction_loop(term))
    needed_list = sorted(needed, key=lambda v: (len(v), sorted(v)))

    def mu_for_lambda(lam: float) -> dict[Variant, float]:
        if not np.isfinite(lam):
            corrected = dict(mu0)  # pin: unmeasured stay at the prior
        else:
            coefficients = _ridge_blup(design_m, error_vector, lam)
            predicted = _predicted_error(needed_list, effect_index, coefficients, max_effect_order)
            corrected = {variant: mu0[variant] - predicted[variant] for variant in needed_list}
        corrected.update({variant: revealed[variant] for variant in measured})
        return corrected

    mu_pin = mu_for_lambda(float("inf"))
    mu_star = mu_for_lambda(lambda_star)

    cells = [
        _order_cell(
            name,
            terms_by_order[order],
            truth_by_order[order],
            mu0,
            mu_pin,
            mu_star,
            mu_for_lambda,
            measured_set,
            true_dg,
            len(measured),
            lambda_star,
            bootstrap_iterations,
            budget,
        )
        for order, name in _ORDERS
        if terms_by_order[order]
    ]
    return BudgetResult(
        budget=budget,
        basis=basis,
        lambda_star=lambda_star,
        lambda_identified=lambda_identified,
        cells=cells,
    )


_OPERATIONAL_BUDGET = 96


def _decide(results: Sequence[BudgetResult]) -> tuple[str, dict[str, str]]:
    """Repair requires, at operational budgets (B≥96), sse_gain≥0 AND a *residualized* rank gain.

    The residualized (main-effect-controlled) gain separates genuine interaction recovery from the
    main-effect-sharing confound; a "repair" that merely reverts to the prior or rides the confound
    fails. B<96 is excluded because λ is unidentifiable on the saturated single-only design.
    """
    per_basis: dict[str, str] = {}
    for basis in sorted({result.basis for result in results}):
        cells = [
            cell
            for result in results
            if result.basis == basis and result.budget >= _OPERATIONAL_BUDGET
            for cell in result.cells
            if cell.order == "pairwise"
        ]
        if not cells:
            per_basis[basis] = "insufficient_operational_budget"
            continue
        sse_ok = all(cell.corr_sse_gain is not None and cell.corr_sse_gain >= 0.0 for cell in cells)
        genuine = all(
            cell.residual_corr_delta_spearman_ci95 is not None
            and cell.residual_corr_delta_spearman_ci95[0] > 0.0
            for cell in cells
        )
        if sse_ok and genuine:
            per_basis[basis] = "repair_confirmed"
        elif sse_ok:
            per_basis[basis] = "calibration_repair_rank_confounded"
        else:
            per_basis[basis] = "no_cache_only_fix"
    values = set(per_basis.values())
    if "repair_confirmed" in values:
        return "repair_confirmed", per_basis
    if "calibration_repair_rank_confounded" in values:
        return "calibration_repair_rank_confounded", per_basis
    if values == {"no_cache_only_fix"}:
        return "replace_phase2_current_model", per_basis
    return "inconclusive_zero_gpu", per_basis


def run_gate3(
    scored: Sequence[ScoredVariant],
    landscape: dict[Variant, float],
    *,
    budgets: Sequence[int] = (48, 96, 192),
    bases: Sequence[str] = ("single", "single_pair"),
    dataset: str = "gb1_wu2016",
    model_id: str = "facebook/esm2_t33_650M_UR50D",
    n_folds: int = 5,
    max_order: int = 3,
    bootstrap_iterations: int = 2000,
) -> Gate3Result:
    """Run the correlated-error inference probe on a completed score cache (CPU-only)."""
    canonical = _canonical_scored(scored)
    wild_type: Variant = frozenset()
    from epibudget.data import reveal_measured_fitness  # noqa: PLC0415

    centered_all = _center_positive(
        reveal_measured_fitness(
            landscape,
            [wild_type, *(item.variant for item in canonical if item.variant != wild_type)],
        )
    )
    shared_by_fold, _records = _shared_slopes(canonical, centered_all, n_folds)
    slopes = {
        item.variant: shared_by_fold[variant_fold(item.variant, n_folds)] for item in canonical
    }
    truth_by_order = _truth_terms(canonical, landscape, max_order)

    results: list[BudgetResult] = []
    for basis in bases:
        for budget in budgets:
            results.append(
                evaluate_budget(
                    canonical,
                    landscape,
                    slopes,
                    truth_by_order,
                    centered_all,
                    budget,
                    basis,
                    bootstrap_iterations,
                    max_order,
                )
            )
    decision, per_basis = _decide(results)
    reason = {
        "repair_confirmed": "correlated prior restores calibration and a residualized rank gain",
        "calibration_repair_rank_confounded": "correlated prior fixes L2, but the rank gain is a "
        "main-effect-sharing confound (residualized gain not > 0)",
        "replace_phase2_current_model": "no cache-only prior fixes L2 at operational budgets",
        "inconclusive_zero_gpu": "mixed or insufficient operational evidence",
    }[decision]
    return Gate3Result(
        dataset=dataset,
        model_id=model_id,
        budgets=list(budgets),
        bases=list(bases),
        public_claim_eligible=False,
        results=results,
        decision=decision,
        decision_reason=reason,
        per_basis_decision=per_basis,
    )


def result_to_dict(result: Gate3Result) -> dict[str, object]:
    return asdict(result)
