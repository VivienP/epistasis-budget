"""Post-hoc corrective Gate 2 analysis over fixed zero-shot selections."""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from math import isfinite
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import pearsonr, spearmanr

from epibudget.acquisition import allocate, fitness_greedy
from epibudget.data import (
    GB1_SITES,
    GB1_WT_AT_SITES,
    enumerate_candidates,
    reveal_measured_fitness,
)
from epibudget.epistasis import (
    ground_truth_epistasis,
    interaction_loop,
    predicted_epistasis,
    wt_centered_log_fitness,
)
from epibudget.graph import EpistasisFactorGraph, variant_variance
from epibudget.robustness import variant_fold
from epibudget.scored_cache import candidate_sha256
from epibudget.types import Mutation, ScoredVariant, Variant
from epibudget.validate import practice_heuristic, random_selection

PROTOCOL_VERSION = "gate2-v1"
RUN_TYPE = "post_hoc_corrective_gate2"
AA20 = "ACDEFGHIKLMNPQRSTVWY"
DEFAULT_BOOTSTRAP_ITERATIONS = 2000
N_BOOTSTRAP = DEFAULT_BOOTSTRAP_ITERATIONS

_PAIRWISE = 2
_THIRD = 3
_DEFAULT_BUDGETS = (48, 96, 192)
_DEFAULT_RANDOM_SEEDS = 20
_DEFAULT_STRUCTURAL_SEEDS = 100
_DEFAULT_FOLDS = 5
_DEFAULT_MAX_ORDER = 3
_EXPECTED_SELECTIONS = 372
_EXPECTED_EVALUATIONS = 1488
_EXPECTED_SLOPES = 377
_MIN_DECISIVE_BUDGETS = 2
_FROZEN_CANDIDATE_COUNT = 29678
_FROZEN_COMPOSITION = {1: 76, 2: 2166, 3: 27436}
_FROZEN_MODEL_ID = "facebook/esm2_t33_650M_UR50D"
_FROZEN_DATASET_SHA256 = "2f115d4eaf03b6083dcc22f7451b3ddfad41c9d8e519286c4e69b6d06db78f1c"
_FROZEN_WT_SHA256 = "7e859d82171047700fd3e9632f7a47eab4a39baedc8c3316d2fc62d3ce2260bb"
_MIN_CORRELATION_POINTS = 3
_INTERVAL_BOUND_COUNT = 2
_REGIMES = ("operational_method_specific", "shared_crossfit_5fold")
_ORDERS = ((_PAIRWISE, "pairwise"), (_THIRD, "third"))
_TAU_CELLS_PER_BUDGET = len(_REGIMES) * 2
_CALIBRATION_CELLS_PER_BUDGET = 3 * 2
_CORRECTED_TIE_BREAK = "canonical-json-v1"
_LEGACY_TIE_BREAK = "input-order-stable-v1"
_RANDOM_TIE_BREAK = "numpy-pcg64-canonical-v1"
_STRUCTURAL_TIE_BREAK = "exact-score-strata-canonical-pcg64-v1"
_CROSSFIT_CAVEAT = (
    "method-independent slopes fitted from full-landscape labels outside each stable variant fold; "
    "non-operational robustness evidence only"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")

FloatArray = npt.NDArray[np.float64]
Term = tuple[Mutation, ...]
EvidenceStatus = Literal["positive", "negative", "inconclusive"]


class _FiniteModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)


class Gate2Config(_FiniteModel):
    protocol_version: str = PROTOCOL_VERSION
    run_type: str = RUN_TYPE
    budgets: list[int]
    random_seeds: int = Field(default=_DEFAULT_RANDOM_SEEDS, ge=0)
    structural_seeds: int = Field(default=_DEFAULT_STRUCTURAL_SEEDS, ge=0)
    n_folds: int = Field(default=_DEFAULT_FOLDS, ge=2)
    max_order: int = Field(default=_DEFAULT_MAX_ORDER, ge=2, le=3)
    alphabet: str = AA20
    dataset: str = "gb1_wu2016"
    model_id: str = ""
    bootstrap_iterations: int = Field(default=2000, ge=1)
    public_claim_eligible: Literal[False] = False


class SelectionRecord(_FiniteModel):
    method: str
    budget: int
    seed_kind: str
    seed: int | None
    tie_break_version: str
    boundary_score: float | None
    boundary_size: int
    boundary_selected: int
    counts_by_order: dict[int, int]
    selected_ids: list[str]
    selected_sha256: str
    n_revealed: int = 0
    n_live: int = 0
    n_nonpositive: int = 0
    n_missing: int = 0
    selection_id: str

    @property
    def selected_canonical_ids(self) -> list[str]:
        return self.selected_ids


class SlopeRecord(_FiniteModel):
    regime: str
    selection_id: str | None
    method: str | None
    budget: int | None
    fold: int | None
    slope: float
    n_fit: int
    fallback: bool
    fit_sha256: str
    caveat: str | None = None


class EvaluationRecord(_FiniteModel):
    selection_id: str
    method: str
    budget: int
    regime: str
    order: str
    n_truth: int
    n_informed: int
    n_pinned: int
    n_predicted: int
    prior_pearson: float | None
    post_pearson: float | None
    prior_spearman: float | None
    post_spearman: float | None
    delta_pearson: float | None
    delta_spearman: float | None
    sse_prior: float
    sse_post: float
    sse_gain: float | None
    update_residual_pearson: float | None
    update_residual_spearman: float | None
    term_sha256: str
    truth_sha256: str
    prior_sha256: str
    post_sha256: str
    status: str
    warnings: list[str] = Field(default_factory=list)


class InferenceBudgetEvidence(_FiniteModel):
    budget: int
    n_terms: int
    bootstrap_iterations: int
    delta_spearman: float | None
    delta_spearman_ci95: tuple[float, float] | None
    delta_pearson: float | None
    delta_pearson_ci95: tuple[float, float] | None
    sse_gain: float | None
    sse_gain_ci95: tuple[float, float] | None
    status: EvidenceStatus


class Tau2BudgetEvidence(_FiniteModel):
    budget: int
    regime: str
    statistic: str
    info_value: float | None
    n_structural: int
    empirical_differences: list[float]
    q025: float | None
    q50: float | None
    q975: float | None
    status: EvidenceStatus


class CalibrationEvidence(_FiniteModel):
    budget: int
    contrast: str
    statistic: str
    operational_difference: float | None
    shared_difference: float | None
    strict_sign_reversal: bool


class Gate2Aggregates(_FiniteModel):
    inference: list[InferenceBudgetEvidence] = Field(default_factory=list)
    inference_status: EvidenceStatus = "inconclusive"
    tau2: list[Tau2BudgetEvidence] = Field(default_factory=list)
    tau2_status: EvidenceStatus = "inconclusive"
    calibration: list[CalibrationEvidence] = Field(default_factory=list)
    calibration_confounded: bool = False


class Gate2Decision(_FiniteModel):
    decision: str
    architecture_decision_eligible: bool
    reason: str


class Gate2Report(_FiniteModel):
    protocol_version: str = PROTOCOL_VERSION
    run_type: str = RUN_TYPE
    public_claim_eligible: Literal[False] = False
    status: str
    config: Gate2Config
    selections: list[SelectionRecord]
    slopes: list[SlopeRecord]
    evaluations: list[EvaluationRecord]
    aggregates: Gate2Aggregates
    architecture_decision_eligible: bool
    architecture_eligibility_reasons: list[str]
    decision: Gate2Decision
    provenance: dict[str, object] | None = None


@dataclass(frozen=True)
class _EvaluationArrays:
    truth: FloatArray
    prior: FloatArray
    post: FloatArray


def canonical_id(variant: Variant) -> str:
    """Return the order-independent compact JSON identity of a variant."""
    return json.dumps(
        [list(mutation) for mutation in sorted(variant)],
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _hash_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _float_token(value: float) -> str:
    return "0" if value == 0.0 else format(value, ".17g")


def _vector_hash(values: FloatArray) -> str:
    return _hash_json([_float_token(float(value)) for value in values])


def _seed_for(*parts: object) -> int:
    digest = hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _canonical_scored(scored: Sequence[ScoredVariant]) -> list[ScoredVariant]:
    if not scored:
        raise ValueError("scored must not be empty")
    for item in scored:
        if not isfinite(item.delta_g) or not isfinite(item.var_delta_g):
            raise ValueError(f"scored contains non-finite values for {canonical_id(item.variant)}")
    ordered = sorted(scored, key=lambda item: canonical_id(item.variant))
    identities = [canonical_id(item.variant) for item in ordered]
    duplicates = [identity for identity, count in Counter(identities).items() if count > 1]
    if duplicates:
        raise ValueError(f"scored contains duplicate variants: {duplicates[:3]}")
    return ordered


def _graphs_and_scores(
    scored: Sequence[ScoredVariant], max_order: int
) -> tuple[EpistasisFactorGraph, EpistasisFactorGraph, dict[str, float], dict[str, int]]:
    interactions = predicted_epistasis(scored, max_order)
    info_graph = EpistasisFactorGraph(interactions, variant_variance(scored, "info"))
    structural_graph = EpistasisFactorGraph(interactions, variant_variance(scored, "structural"))
    info_scores = {
        canonical_id(item.variant): info_graph.info_gain(frozenset(), item.variant)
        for item in scored
    }
    structural_scores: dict[str, int] = {}
    for item in scored:
        identity = canonical_id(item.variant)
        score = structural_graph.info_gain(frozenset(), item.variant)
        rounded = round(score)
        if score != float(rounded):
            raise ValueError(f"structural score is not exact for {identity}: {score}")
        structural_scores[identity] = int(rounded)
    return info_graph, structural_graph, info_scores, structural_scores


def _structural_scores(scored: Sequence[ScoredVariant], max_order: int = 3) -> dict[str, int]:
    canonical = _canonical_scored(scored)
    return _graphs_and_scores(canonical, max_order)[3]


def _score_strata(
    scored: Sequence[ScoredVariant], scores: Mapping[str, int]
) -> list[list[ScoredVariant]]:
    grouped: dict[int, list[ScoredVariant]] = defaultdict(list)
    for item in scored:
        grouped[scores[canonical_id(item.variant)]].append(item)
    return [
        sorted(grouped[score], key=lambda item: canonical_id(item.variant))
        for score in sorted(grouped, reverse=True)
    ]


def _permuted_strata_order(
    strata: Sequence[Sequence[ScoredVariant]], seed: int
) -> list[ScoredVariant]:
    rng = np.random.default_rng(seed)
    ordered: list[ScoredVariant] = []
    for stratum in strata:
        permutation = rng.permutation(len(stratum))
        ordered.extend(stratum[int(index)] for index in permutation)
    return ordered


def _selection_record(
    *,
    method: str,
    budget: int,
    seed_kind: str,
    seed: int | None,
    tie_break_version: str,
    selected: Sequence[Variant],
    candidate_scores: Mapping[str, float | int] | None,
) -> SelectionRecord:
    selected_ids = [canonical_id(variant) for variant in selected]
    selected_sha256 = _hash_json(selected_ids)
    boundary_score: float | None = None
    boundary_size = 0
    boundary_selected = 0
    if selected_ids and candidate_scores is not None:
        boundary_score = float(candidate_scores[selected_ids[-1]])
        boundary_size = sum(float(score) == boundary_score for score in candidate_scores.values())
        boundary_selected = sum(
            float(candidate_scores[identity]) == boundary_score for identity in selected_ids
        )
    selection_id = _selection_id(
        method=method,
        budget=budget,
        seed_kind=seed_kind,
        seed=seed,
        selected_sha256=selected_sha256,
    )
    counts = Counter(len(variant) for variant in selected)
    return SelectionRecord(
        method=method,
        budget=budget,
        seed_kind=seed_kind,
        seed=seed,
        tie_break_version=tie_break_version,
        boundary_score=boundary_score,
        boundary_size=boundary_size,
        boundary_selected=boundary_selected,
        counts_by_order=dict(sorted(counts.items())),
        selected_ids=selected_ids,
        selected_sha256=selected_sha256,
        selection_id=selection_id,
    )


def _selection_id(
    *,
    method: str,
    budget: int,
    seed_kind: str,
    seed: int | None,
    selected_sha256: str,
) -> str:
    return _hash_json(
        {
            "protocol_version": PROTOCOL_VERSION,
            "method": method,
            "budget": budget,
            "seed_kind": seed_kind,
            "seed": seed,
            "selected_sha256": selected_sha256,
        }
    )


def _build_selection_records(
    scored: Sequence[ScoredVariant],
    budgets: Sequence[int],
    *,
    random_seeds: int = 20,
    structural_seeds: int = 100,
    max_order: int = 3,
    model_id: str = "",
) -> list[SelectionRecord]:
    """Fix every Gate 2 selection without accepting or reading a measured landscape."""
    if random_seeds < 0 or structural_seeds < 0:
        raise ValueError("seed counts must be non-negative")
    original = list(scored)
    canonical = _canonical_scored(scored)
    for budget in budgets:
        if budget < 1 or budget > len(canonical):
            raise ValueError(f"budget {budget} must be in 1..{len(canonical)}")

    _info_graph, structural_graph, info_scores, structural_scores = _graphs_and_scores(
        canonical, max_order
    )
    info_order = sorted(
        canonical,
        key=lambda item: (-info_scores[canonical_id(item.variant)], canonical_id(item.variant)),
    )
    fitness_scores = {canonical_id(item.variant): item.delta_g for item in canonical}
    pair_scores = {
        canonical_id(item.variant): item.delta_g
        for item in canonical
        if len(item.variant) == _PAIRWISE
    }
    structural_strata = _score_strata(canonical, structural_scores)
    structural_orders = {
        seed: _permuted_strata_order(structural_strata, seed) for seed in range(structural_seeds)
    }
    records: list[SelectionRecord] = []
    for budget in budgets:
        info_selected = [item.variant for item in info_order[:budget]]
        fitness_selected = fitness_greedy(canonical, budget)
        practice_selected = practice_heuristic(canonical, budget)
        legacy_selected = allocate(
            structural_graph,
            original,
            budget,
            lambda_=0.0,
            model_id=model_id,
        ).selected
        records.extend(
            (
                _selection_record(
                    method="info",
                    budget=budget,
                    seed_kind="none",
                    seed=None,
                    tie_break_version=_CORRECTED_TIE_BREAK,
                    selected=info_selected,
                    candidate_scores=info_scores,
                ),
                _selection_record(
                    method="fitness",
                    budget=budget,
                    seed_kind="none",
                    seed=None,
                    tie_break_version=_CORRECTED_TIE_BREAK,
                    selected=fitness_selected,
                    candidate_scores=fitness_scores,
                ),
                _selection_record(
                    method="practice",
                    budget=budget,
                    seed_kind="none",
                    seed=None,
                    tie_break_version=_CORRECTED_TIE_BREAK,
                    selected=practice_selected,
                    candidate_scores=pair_scores,
                ),
                _selection_record(
                    method="structural_legacy_prefix",
                    budget=budget,
                    seed_kind="none",
                    seed=None,
                    tie_break_version=_LEGACY_TIE_BREAK,
                    selected=legacy_selected,
                    candidate_scores=structural_scores,
                ),
            )
        )
        records.extend(
            _selection_record(
                method="random",
                budget=budget,
                seed_kind="random",
                seed=seed,
                tie_break_version=_RANDOM_TIE_BREAK,
                selected=random_selection(canonical, budget, seed),
                candidate_scores=None,
            )
            for seed in range(random_seeds)
        )
        records.extend(
            _selection_record(
                method="structural_seeded",
                budget=budget,
                seed_kind="structural",
                seed=seed,
                tie_break_version=_STRUCTURAL_TIE_BREAK,
                selected=[item.variant for item in structural_orders[seed][:budget]],
                candidate_scores=structural_scores,
            )
            for seed in range(structural_seeds)
        )
    return records


def _center_positive(raw: Mapping[Variant, float]) -> dict[Variant, float]:
    return wt_centered_log_fitness(raw)


def _reveal_selection(
    landscape: dict[Variant, float], selected: Sequence[Variant]
) -> tuple[dict[Variant, float], int, int, int, int]:
    wild_type: Variant = frozenset()
    raw = reveal_measured_fitness(
        landscape, [wild_type, *(variant for variant in selected if variant != wild_type)]
    )
    finite_present = [variant for variant in selected if variant in raw and isfinite(raw[variant])]
    n_revealed = len(finite_present)
    n_nonpositive = sum(raw[variant] <= 0.0 for variant in finite_present)
    n_missing = len(selected) - n_revealed
    centered = _center_positive(raw)
    live = {variant: centered[variant] for variant in selected if variant and variant in centered}
    return live, n_revealed, len(live), n_nonpositive, n_missing


def _fit_slope(
    variants: Sequence[Variant],
    esm: Mapping[Variant, float],
    measured: Mapping[Variant, float],
) -> tuple[float, int, bool, str]:
    fitted = sorted(
        (variant for variant in variants if variant in esm and variant in measured),
        key=canonical_id,
    )
    x = np.array([esm[variant] for variant in fitted], dtype=np.float64)
    y = np.array([measured[variant] for variant in fitted], dtype=np.float64)
    denominator = float(np.dot(x, x))
    fallback = len(fitted) == 0 or denominator == 0.0
    slope = 1.0 if fallback else float(np.dot(x, y) / denominator)
    if not isfinite(slope):
        slope = 1.0
        fallback = True
    fit_hash = _hash_json(
        [
            [canonical_id(variant), _float_token(esm[variant]), _float_token(measured[variant])]
            for variant in fitted
        ]
    )
    return slope, len(fitted), fallback, fit_hash


def _truth_terms(
    scored: Sequence[ScoredVariant],
    landscape: dict[Variant, float],
    max_order: int,
) -> dict[int, dict[Term, float]]:
    wild_type: Variant = frozenset()
    raw = reveal_measured_fitness(
        landscape, [wild_type, *(item.variant for item in scored if item.variant != wild_type)]
    )
    centered = _center_positive(raw)
    candidate_terms = {
        tuple(sorted(item.variant))
        for item in scored
        if _PAIRWISE <= len(item.variant) <= max_order
    }
    truth = {
        interaction.mutations: interaction.epsilon_hat
        for interaction in ground_truth_epistasis(centered, max_order=max_order)
        if interaction.mutations in candidate_terms
    }
    return {
        order: {term: truth[term] for term in sorted(term for term in truth if len(term) == order)}
        for order, _ in _ORDERS
    }


def _epsilon(mu: Mapping[Variant, float], term: Term) -> float:
    order = len(term)
    value = 0.0
    for member in interaction_loop(term):
        sign = 1.0 if (order - len(member)) % 2 == 0 else -1.0
        value += sign * mu[member]
    return value


def _inference_arrays(
    terms: Sequence[Term],
    esm: Mapping[Variant, float],
    revealed: Mapping[Variant, float],
    slopes: Mapping[Variant, float],
) -> tuple[FloatArray, FloatArray]:
    prior_mu = {variant: slopes[variant] * value for variant, value in esm.items()}
    post_mu = dict(prior_mu)
    post_mu.update(revealed)
    prior = np.array([_epsilon(prior_mu, term) for term in terms], dtype=np.float64)
    post = np.array([_epsilon(post_mu, term) for term in terms], dtype=np.float64)
    return prior, post


def _safe_correlations(
    predicted: FloatArray, truth: FloatArray
) -> tuple[float | None, float | None]:
    if (
        len(truth) < _MIN_CORRELATION_POINTS
        or float(np.std(predicted)) == 0.0
        or float(np.std(truth)) == 0.0
    ):
        return None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pearson = float(pearsonr(predicted, truth).statistic)
        spearman = float(spearmanr(predicted, truth).statistic)
    return (
        pearson if isfinite(pearson) else None,
        spearman if isfinite(spearman) else None,
    )


def _difference(after: float | None, before: float | None) -> float | None:
    return None if after is None or before is None else after - before


def _relative_sse_gain(sse_prior: float, sse_post: float) -> float | None:
    return None if sse_prior == 0.0 else 1.0 - sse_post / sse_prior


def _evaluation_record(
    selection: SelectionRecord,
    regime: str,
    order_name: str,
    truth_by_term: Mapping[Term, float],
    revealed: Mapping[Variant, float],
    prior: FloatArray,
    post: FloatArray,
) -> tuple[EvaluationRecord, _EvaluationArrays]:
    terms = list(truth_by_term)
    truth = np.array([truth_by_term[term] for term in terms], dtype=np.float64)
    measured = frozenset(revealed)
    informed = [any(member in measured for member in interaction_loop(term)) for term in terms]
    pinned = [all(member in measured for member in interaction_loop(term)) for term in terms]
    predicted = [
        is_informed and not is_pinned
        for is_informed, is_pinned in zip(informed, pinned, strict=True)
    ]
    prior_pearson, prior_spearman = _safe_correlations(prior, truth)
    post_pearson, post_spearman = _safe_correlations(post, truth)
    residual_indices = np.array(predicted, dtype=np.bool_)
    update_residual_pearson: float | None = None
    update_residual_spearman: float | None = None
    if int(np.sum(residual_indices)):
        update_residual_pearson, update_residual_spearman = _safe_correlations(
            (post - prior)[residual_indices],
            (truth - prior)[residual_indices],
        )
    sse_prior = float(np.sum(np.square(prior - truth)))
    sse_post = float(np.sum(np.square(post - truth)))
    sse_gain = _relative_sse_gain(sse_prior, sse_post)
    warnings_out: list[str] = []
    required = {
        "prior_pearson": prior_pearson,
        "post_pearson": post_pearson,
        "prior_spearman": prior_spearman,
        "post_spearman": post_spearman,
        "sse_gain": sse_gain,
    }
    diagnostics = {
        "update_residual_pearson": update_residual_pearson,
        "update_residual_spearman": update_residual_spearman,
    }
    warnings_out.extend(
        f"{name}_undefined" for name, value in {**required, **diagnostics}.items() if value is None
    )
    if sse_gain is None:
        warnings_out.remove("sse_gain_undefined")
        warnings_out.append("sse_gain_undefined_zero_prior_sse")
    status = "ok" if all(value is not None for value in required.values()) else "insufficient_data"
    record = EvaluationRecord(
        selection_id=selection.selection_id,
        method=selection.method,
        budget=selection.budget,
        regime=regime,
        order=order_name,
        n_truth=len(terms),
        n_informed=sum(informed),
        n_pinned=sum(pinned),
        n_predicted=sum(predicted),
        prior_pearson=prior_pearson,
        post_pearson=post_pearson,
        prior_spearman=prior_spearman,
        post_spearman=post_spearman,
        delta_pearson=_difference(post_pearson, prior_pearson),
        delta_spearman=_difference(post_spearman, prior_spearman),
        sse_prior=sse_prior,
        sse_post=sse_post,
        sse_gain=sse_gain,
        update_residual_pearson=update_residual_pearson,
        update_residual_spearman=update_residual_spearman,
        term_sha256=_hash_json([canonical_id(frozenset(term)) for term in terms]),
        truth_sha256=_vector_hash(truth),
        prior_sha256=_vector_hash(prior),
        post_sha256=_vector_hash(post),
        status=status,
        warnings=warnings_out,
    )
    return record, _EvaluationArrays(truth=truth, prior=prior, post=post)


def _shared_slopes(
    scored: Sequence[ScoredVariant],
    centered_landscape: Mapping[Variant, float],
    n_folds: int,
) -> tuple[dict[int, float], list[SlopeRecord]]:
    esm = {item.variant: item.delta_g for item in scored}
    measurable = [item.variant for item in scored if item.variant in centered_landscape]
    slopes: dict[int, float] = {}
    records: list[SlopeRecord] = []
    for fold in range(n_folds):
        fitted = [variant for variant in measurable if variant_fold(variant, n_folds) != fold]
        slope, n_fit, fallback, fit_hash = _fit_slope(fitted, esm, centered_landscape)
        slopes[fold] = slope
        records.append(
            SlopeRecord(
                regime="shared_crossfit_5fold",
                selection_id=None,
                method=None,
                budget=None,
                fold=fold,
                slope=slope,
                n_fit=n_fit,
                fallback=fallback,
                fit_sha256=fit_hash,
                caveat=_CROSSFIT_CAVEAT,
            )
        )
    return slopes, records


def _percentile(values: Sequence[float]) -> tuple[float, float] | None:
    if len(values) < _MIN_CORRELATION_POINTS:
        return None
    lower, upper = np.percentile(np.array(values, dtype=np.float64), [2.5, 97.5])
    return float(lower), float(upper)


def _evidence_status(
    intervals: Sequence[tuple[float, float] | None],
) -> EvidenceStatus:
    if intervals and all(interval is not None and interval[0] > 0.0 for interval in intervals):
        return "positive"
    if intervals and all(interval is not None and interval[1] < 0.0 for interval in intervals):
        return "negative"
    return "inconclusive"


def _inference_evidence(
    budgets: Sequence[int],
    selections: Sequence[SelectionRecord],
    evaluations: Mapping[tuple[str, str, str], _EvaluationArrays],
) -> list[InferenceBudgetEvidence]:
    output: list[InferenceBudgetEvidence] = []
    for budget in budgets:
        info = next(
            (
                selection
                for selection in selections
                if selection.method == "info" and selection.budget == budget
            ),
            None,
        )
        arrays = (
            evaluations.get((info.selection_id, "shared_crossfit_5fold", "pairwise"))
            if info is not None
            else None
        )
        if arrays is None:
            output.append(
                InferenceBudgetEvidence(
                    budget=budget,
                    n_terms=0,
                    bootstrap_iterations=N_BOOTSTRAP,
                    delta_spearman=None,
                    delta_spearman_ci95=None,
                    delta_pearson=None,
                    delta_pearson_ci95=None,
                    sse_gain=None,
                    sse_gain_ci95=None,
                    status="inconclusive",
                )
            )
            continue
        prior_pearson, prior_spearman = _safe_correlations(arrays.prior, arrays.truth)
        post_pearson, post_spearman = _safe_correlations(arrays.post, arrays.truth)
        delta_pearson = _difference(post_pearson, prior_pearson)
        delta_spearman = _difference(post_spearman, prior_spearman)
        sse_prior = float(np.sum(np.square(arrays.prior - arrays.truth)))
        sse_post = float(np.sum(np.square(arrays.post - arrays.truth)))
        sse_gain = _relative_sse_gain(sse_prior, sse_post)
        spearman_samples: list[float] = []
        pearson_samples: list[float] = []
        sse_samples: list[float] = []
        rng = np.random.default_rng(_seed_for(PROTOCOL_VERSION, "inference", budget))
        n_terms = len(arrays.truth)
        for _ in range(N_BOOTSTRAP):
            indices = rng.integers(0, n_terms, size=n_terms) if n_terms else np.array([], dtype=int)
            truth = arrays.truth[indices]
            prior = arrays.prior[indices]
            post = arrays.post[indices]
            prior_p, prior_s = _safe_correlations(prior, truth)
            post_p, post_s = _safe_correlations(post, truth)
            sampled_pearson = _difference(post_p, prior_p)
            sampled_spearman = _difference(post_s, prior_s)
            if sampled_pearson is not None:
                pearson_samples.append(sampled_pearson)
            if sampled_spearman is not None:
                spearman_samples.append(sampled_spearman)
            sampled_sse_prior = float(np.sum(np.square(prior - truth)))
            sampled_sse_post = float(np.sum(np.square(post - truth)))
            sampled_sse_gain = _relative_sse_gain(sampled_sse_prior, sampled_sse_post)
            if sampled_sse_gain is not None:
                sse_samples.append(sampled_sse_gain)
        spearman_ci = _percentile(spearman_samples)
        pearson_ci = _percentile(pearson_samples)
        sse_ci = _percentile(sse_samples)
        output.append(
            InferenceBudgetEvidence(
                budget=budget,
                n_terms=n_terms,
                bootstrap_iterations=N_BOOTSTRAP,
                delta_spearman=delta_spearman,
                delta_spearman_ci95=spearman_ci,
                delta_pearson=delta_pearson,
                delta_pearson_ci95=pearson_ci,
                sse_gain=sse_gain,
                sse_gain_ci95=sse_ci,
                status=_evidence_status((spearman_ci, pearson_ci, sse_ci)),
            )
        )
    return output


def _overall_status(statuses: Sequence[EvidenceStatus]) -> EvidenceStatus:
    if sum(status == "positive" for status in statuses) >= _MIN_DECISIVE_BUDGETS:
        return "positive"
    if sum(status == "negative" for status in statuses) >= _MIN_DECISIVE_BUDGETS:
        return "negative"
    return "inconclusive"


def _post_statistic(record: EvaluationRecord, statistic: str) -> float | None:
    return record.post_pearson if statistic == "pearson" else record.post_spearman


def _tau2_evidence(
    budgets: Sequence[int], evaluations: Sequence[EvaluationRecord]
) -> tuple[list[Tau2BudgetEvidence], EvidenceStatus]:
    output: list[Tau2BudgetEvidence] = []
    for budget in budgets:
        for regime in _REGIMES:
            info = next(
                (
                    record
                    for record in evaluations
                    if record.budget == budget
                    and record.method == "info"
                    and record.regime == regime
                    and record.order == "pairwise"
                ),
                None,
            )
            structural = [
                record
                for record in evaluations
                if record.budget == budget
                and record.method == "structural_seeded"
                and record.regime == regime
                and record.order == "pairwise"
            ]
            for statistic in ("pearson", "spearman"):
                info_value = _post_statistic(info, statistic) if info is not None else None
                values = [
                    value
                    for record in structural
                    if (value := _post_statistic(record, statistic)) is not None
                ]
                differences = (
                    [info_value - value for value in values] if info_value is not None else []
                )
                q025: float | None
                q50: float | None
                q975: float | None
                if differences:
                    percentiles = np.percentile(
                        np.array(differences, dtype=np.float64), [2.5, 50.0, 97.5]
                    )
                    q025, q50, q975 = (float(value) for value in percentiles)
                else:
                    q025 = q50 = q975 = None
                status: EvidenceStatus = "inconclusive"
                if q025 is not None and q025 > 0.0:
                    status = "positive"
                elif q975 is not None and q975 < 0.0:
                    status = "negative"
                output.append(
                    Tau2BudgetEvidence(
                        budget=budget,
                        regime=regime,
                        statistic=statistic,
                        info_value=info_value,
                        n_structural=len(differences),
                        empirical_differences=differences,
                        q025=q025,
                        q50=q50,
                        q975=q975,
                        status=status,
                    )
                )
    budget_statuses: list[EvidenceStatus] = []
    for budget in budgets:
        cells = [record.status for record in output if record.budget == budget]
        if len(cells) == _TAU_CELLS_PER_BUDGET and all(status == "positive" for status in cells):
            budget_statuses.append("positive")
        elif len(cells) == _TAU_CELLS_PER_BUDGET and all(status == "negative" for status in cells):
            budget_statuses.append("negative")
        else:
            budget_statuses.append("inconclusive")
    return output, _overall_status(budget_statuses)


def _method_values(
    evaluations: Sequence[EvaluationRecord],
    *,
    budget: int,
    regime: str,
    method: str,
    statistic: str,
) -> list[float]:
    return [
        value
        for record in evaluations
        if record.budget == budget
        and record.regime == regime
        and record.method == method
        and record.order == "pairwise"
        and (value := _post_statistic(record, statistic)) is not None
    ]


def _contrast_difference(info: Sequence[float], baseline: Sequence[float]) -> float | None:
    if not info or not baseline:
        return None
    return float(np.median(np.array(info))) - float(np.median(np.array(baseline)))


def _calibration_evidence(
    budgets: Sequence[int], evaluations: Sequence[EvaluationRecord]
) -> list[CalibrationEvidence]:
    contrast_methods = {
        "info_fitness": "fitness",
        "info_random_median": "random",
        "info_structural_median": "structural_seeded",
    }
    output: list[CalibrationEvidence] = []
    for budget in budgets:
        for contrast, baseline_method in contrast_methods.items():
            for statistic in ("pearson", "spearman"):
                differences: dict[str, float | None] = {}
                for regime in _REGIMES:
                    info = _method_values(
                        evaluations,
                        budget=budget,
                        regime=regime,
                        method="info",
                        statistic=statistic,
                    )
                    baseline = _method_values(
                        evaluations,
                        budget=budget,
                        regime=regime,
                        method=baseline_method,
                        statistic=statistic,
                    )
                    differences[regime] = _contrast_difference(info, baseline)
                operational = differences["operational_method_specific"]
                shared = differences["shared_crossfit_5fold"]
                reversal = (
                    operational is not None
                    and shared is not None
                    and operational != 0.0
                    and shared != 0.0
                    and operational * shared < 0.0
                )
                output.append(
                    CalibrationEvidence(
                        budget=budget,
                        contrast=contrast,
                        statistic=statistic,
                        operational_difference=operational,
                        shared_difference=shared,
                        strict_sign_reversal=reversal,
                    )
                )
    return output


def _calibration_is_confounded(evidence: Sequence[CalibrationEvidence]) -> bool:
    reversal_budgets: dict[tuple[str, str], set[int]] = defaultdict(set)
    for record in evidence:
        if record.strict_sign_reversal:
            reversal_budgets[(record.contrast, record.statistic)].add(record.budget)
    return any(len(budgets) >= _MIN_DECISIVE_BUDGETS for budgets in reversal_budgets.values())


def _aggregate_gate2(
    budgets: Sequence[int],
    selections: Sequence[SelectionRecord],
    evaluations: Sequence[EvaluationRecord],
    arrays: Mapping[tuple[str, str, str], _EvaluationArrays],
) -> Gate2Aggregates:
    inference = _inference_evidence(budgets, selections, arrays)
    tau2, tau2_status = _tau2_evidence(budgets, evaluations)
    calibration = _calibration_evidence(budgets, evaluations)
    return Gate2Aggregates(
        inference=inference,
        inference_status=_overall_status([record.status for record in inference]),
        tau2=tau2,
        tau2_status=tau2_status,
        calibration=calibration,
        calibration_confounded=_calibration_is_confounded(calibration),
    )


def decide_gate2(aggregates: Gate2Aggregates, eligible: bool = True) -> Gate2Decision:
    """Apply the registered architecture decision rule, failing closed when ineligible."""
    if not eligible:
        return Gate2Decision(
            decision="inconclusive_zero_gpu",
            architecture_decision_eligible=False,
            reason=(
                "architecture decision evidence is incomplete or outside the exact default profile"
            ),
        )
    if aggregates.inference_status == "negative" or aggregates.tau2_status == "negative":
        return Gate2Decision(
            decision="replace_phase2_current_model",
            architecture_decision_eligible=True,
            reason="inference or tau2 evidence is negative at two or more budgets",
        )
    if (
        aggregates.inference_status == "positive"
        and aggregates.tau2_status == "positive"
        and not aggregates.calibration_confounded
    ):
        return Gate2Decision(
            decision="repair_current_core",
            architecture_decision_eligible=True,
            reason="inference and tau2 evidence are positive without calibration confounding",
        )
    return Gate2Decision(
        decision="inconclusive_zero_gpu",
        architecture_decision_eligible=True,
        reason="registered evidence is mixed, inconclusive, or calibration-confounded",
    )


def _finite_required_cell(record: EvaluationRecord) -> bool:
    values = (
        record.prior_pearson,
        record.post_pearson,
        record.prior_spearman,
        record.post_spearman,
        record.delta_pearson,
        record.delta_spearman,
        record.sse_prior,
        record.sse_post,
        record.sse_gain,
    )
    return record.status == "ok" and all(value is not None and isfinite(value) for value in values)


@lru_cache(maxsize=1)
def _frozen_variants() -> tuple[Variant, ...]:
    return tuple(
        enumerate_candidates(
            GB1_SITES,
            GB1_WT_AT_SITES,
            allowed_aa=AA20,
            max_order=_DEFAULT_MAX_ORDER,
        )
    )


@lru_cache(maxsize=1)
def _frozen_candidate_sha256() -> str:
    return candidate_sha256(_frozen_variants())


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _is_commit(value: object) -> bool:
    return isinstance(value, str) and _COMMIT_PATTERN.fullmatch(value) is not None


def _parse_utc_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        return None
    return parsed


def _is_timestamp(value: object) -> bool:
    return _parse_utc_timestamp(value) is not None


def _frozen_cache_identity(candidate_hash: str) -> dict[str, object]:
    return {
        "model_id": _FROZEN_MODEL_ID,
        "scorer_seed": 0,
        "n_perturbations": 16,
        "candidate_sha256": candidate_hash,
        "candidate_count": _FROZEN_CANDIDATE_COUNT,
        "candidate_alphabet": AA20,
        "max_order": _DEFAULT_MAX_ORDER,
        "wt_sha256": _FROZEN_WT_SHA256,
    }


def _provenance_reasons(  # noqa: PLR0912 - each mandatory provenance field fails closed
    provenance: Mapping[str, object] | None, candidate_hash: str
) -> list[str]:
    if provenance is None:
        return ["complete scored-cache, dataset, git, command, and timing provenance is required"]
    reasons: list[str] = []
    if not _is_sha256(provenance.get("scored_cache_sha256")):
        reasons.append("scored_cache_sha256 is missing or invalid")
    if not _is_sha256(provenance.get("scored_cache_sidecar_sha256")):
        reasons.append("scored_cache_sidecar_sha256 is missing or invalid")
    if provenance.get("dataset_sha256") != _FROZEN_DATASET_SHA256:
        reasons.append("dataset_sha256 does not identify the frozen GB1 dataset")
    if provenance.get("candidate_universe_sha256") != candidate_hash:
        reasons.append("candidate_universe_sha256 does not match the actual scored universe")
    frozen_identity = _frozen_cache_identity(candidate_hash)
    expected = provenance.get("scored_cache_identity_expected")
    observed = provenance.get("scored_cache_identity_observed")
    if (
        not isinstance(expected, Mapping)
        or not isinstance(observed, Mapping)
        or dict(expected) != frozen_identity
        or dict(observed) != frozen_identity
        or dict(expected) != dict(observed)
    ):
        reasons.append("expected and observed scored-cache identities are incomplete or mismatched")
    if provenance.get("scored_cache_validator_status") != "passed":
        reasons.append("scored_cache_validator_status does not attest a successful validator")
    if not _is_commit(provenance.get("execution_commit")):
        reasons.append("execution_commit is missing or invalid")
    code_state = provenance.get("code_state")
    if code_state not in {"clean", "dirty"}:
        reasons.append("code_state must be clean or dirty")
    if code_state == "dirty" and not _is_sha256(provenance.get("code_diff_sha256")):
        reasons.append("dirty code requires a valid code_diff_sha256")
    changed_files = provenance.get("changed_scientific_files")
    if not isinstance(changed_files, list) or not all(
        isinstance(path, str) for path in changed_files
    ):
        reasons.append("changed_scientific_files must be an explicit string list")
    exact_command = provenance.get("exact_command")
    if not isinstance(exact_command, str) or not exact_command.strip():
        reasons.append("exact_command is required")
    started_at = _parse_utc_timestamp(provenance.get("started_at_utc"))
    completed_at = _parse_utc_timestamp(provenance.get("completed_at_utc"))
    if started_at is None or completed_at is None:
        reasons.append("started_at_utc and completed_at_utc must be timezone-aware UTC timestamps")
    elif completed_at < started_at:
        reasons.append("completed_at_utc must not precede started_at_utc")
    elapsed = provenance.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, int | float)
        or not isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        reasons.append("elapsed_seconds must be finite and non-negative")
    return reasons


def _scored_universe(
    scored: Sequence[ScoredVariant],
) -> tuple[dict[str, Variant], str, list[str]]:
    reasons: list[str] = []
    try:
        canonical = _canonical_scored(scored)
    except ValueError as exc:
        return {}, "", [f"invalid scored universe: {exc}"]
    variants = [item.variant for item in canonical]
    actual_hash = candidate_sha256(variants)
    composition = dict(sorted(Counter(len(variant) for variant in variants).items()))
    if len(variants) != _FROZEN_CANDIDATE_COUNT:
        reasons.append("scored universe does not contain exactly 29,678 candidates")
    if composition != _FROZEN_COMPOSITION:
        reasons.append("scored universe order composition is not {1:76,2:2166,3:27436}")
    if set(variants) != set(_frozen_variants()) or actual_hash != _frozen_candidate_sha256():
        reasons.append("scored candidate identities do not equal the frozen GB1 AA20 universe")
    return {canonical_id(variant): variant for variant in variants}, actual_hash, reasons


def _expected_selection_metadata() -> dict[tuple[str, int, int | None], tuple[str, str]]:
    output: dict[tuple[str, int, int | None], tuple[str, str]] = {}
    for budget in _DEFAULT_BUDGETS:
        for method in ("info", "fitness", "practice"):
            output[(method, budget, None)] = ("none", _CORRECTED_TIE_BREAK)
        output[("structural_legacy_prefix", budget, None)] = (
            "none",
            _LEGACY_TIE_BREAK,
        )
        output.update(
            {
                ("random", budget, seed): ("random", _RANDOM_TIE_BREAK)
                for seed in range(_DEFAULT_RANDOM_SEEDS)
            }
        )
        output.update(
            {
                ("structural_seeded", budget, seed): (
                    "structural",
                    _STRUCTURAL_TIE_BREAK,
                )
                for seed in range(_DEFAULT_STRUCTURAL_SEEDS)
            }
        )
    return output


def _selection_integrity_reasons(
    selections: Sequence[SelectionRecord], universe: Mapping[str, Variant]
) -> list[str]:
    reasons: list[str] = []
    metadata = _expected_selection_metadata()
    keys = [(record.method, record.budget, record.seed) for record in selections]
    selection_ids = [record.selection_id for record in selections]
    if (
        len(selections) != _EXPECTED_SELECTIONS
        or len(set(keys)) != _EXPECTED_SELECTIONS
        or set(keys) != set(metadata)
        or len(set(selection_ids)) != _EXPECTED_SELECTIONS
    ):
        reasons.append(
            "selection records are missing, duplicated, or outside the required 372 cells"
        )
        return reasons
    for record, key in zip(selections, keys, strict=True):
        expected_seed_kind, expected_tie_break = metadata[key]
        if record.seed_kind != expected_seed_kind or record.tie_break_version != expected_tie_break:
            reasons.append("selection method seed metadata or tie-break version was altered")
            break
        dynamic_counts = (
            record.n_revealed,
            record.n_live,
            record.n_nonpositive,
            record.n_missing,
        )
        if (
            any(count < 0 or count > record.budget for count in dynamic_counts)
            or record.n_revealed + record.n_missing != record.budget
            or record.n_live + record.n_nonpositive != record.n_revealed
        ):
            reasons.append("selection reveal/live/nonpositive/missing counts are inconsistent")
            break
        if (
            len(record.selected_ids) != record.budget
            or len(set(record.selected_ids)) != record.budget
        ):
            reasons.append("selection is underfilled or contains duplicate candidate identities")
            break
        if any(identity not in universe for identity in record.selected_ids):
            reasons.append("selection contains a candidate outside the actual scored universe")
            break
        selected = [universe[identity] for identity in record.selected_ids]
        counts = dict(sorted(Counter(len(variant) for variant in selected).items()))
        if record.counts_by_order != counts:
            reasons.append("selection counts_by_order does not match selected identities")
            break
        selected_hash = _hash_json(record.selected_ids)
        expected_id = _selection_id(
            method=record.method,
            budget=record.budget,
            seed_kind=record.seed_kind,
            seed=record.seed,
            selected_sha256=selected_hash,
        )
        if record.selected_sha256 != selected_hash or record.selection_id != expected_id:
            reasons.append("selection SHA-256 or selection_id does not match its raw fields")
            break
    if reasons:
        return reasons
    by_seed_budget = {
        (record.seed, record.budget): record
        for record in selections
        if record.method == "structural_seeded"
    }
    if any(
        by_seed_budget[(seed, 96)].selected_ids[:48] != by_seed_budget[(seed, 48)].selected_ids
        or by_seed_budget[(seed, 192)].selected_ids[:96] != by_seed_budget[(seed, 96)].selected_ids
        for seed in range(_DEFAULT_STRUCTURAL_SEEDS)
    ):
        reasons.append("structural_seeded selections are not nested within seed")
    return reasons


def _evaluation_integrity_reasons(
    selections: Sequence[SelectionRecord], evaluations: Sequence[EvaluationRecord]
) -> list[str]:
    reasons: list[str] = []
    selection_by_id = {record.selection_id: record for record in selections}
    expected_keys = {
        (selection_id, regime, order)
        for selection_id in selection_by_id
        for regime in _REGIMES
        for _, order in _ORDERS
    }
    actual_keys = [(record.selection_id, record.regime, record.order) for record in evaluations]
    if (
        len(evaluations) != _EXPECTED_EVALUATIONS
        or len(set(actual_keys)) != _EXPECTED_EVALUATIONS
        or set(actual_keys) != expected_keys
    ):
        return ["evaluation records are missing or duplicated relative to the required 1488 cells"]
    term_hashes: dict[str, str] = {}
    truth_hashes: dict[str, str] = {}
    truth_counts: dict[str, int] = {}
    for record in evaluations:
        selection = selection_by_id[record.selection_id]
        if record.method != selection.method or record.budget != selection.budget:
            reasons.append("evaluation method or budget does not match its linked selection")
            break
        if not all(
            _is_sha256(value)
            for value in (
                record.term_sha256,
                record.truth_sha256,
                record.prior_sha256,
                record.post_sha256,
            )
        ):
            reasons.append("evaluation term/truth/prior/post hash is invalid")
            break
        if term_hashes.setdefault(record.order, record.term_sha256) != record.term_sha256:
            reasons.append("evaluation term hash is not fixed within order")
            break
        if truth_hashes.setdefault(record.order, record.truth_sha256) != record.truth_sha256:
            reasons.append("evaluation truth hash is not fixed within order")
            break
        if truth_counts.setdefault(record.order, record.n_truth) != record.n_truth:
            reasons.append("evaluation truth count is not fixed within order")
            break
        if not (
            0 <= record.n_pinned <= record.n_informed <= record.n_truth
            and record.n_predicted == record.n_informed - record.n_pinned
        ):
            reasons.append("evaluation informed/pinned/predicted counts are inconsistent")
            break
        if record.delta_pearson != _difference(record.post_pearson, record.prior_pearson) or (
            record.delta_spearman != _difference(record.post_spearman, record.prior_spearman)
        ):
            reasons.append("evaluation correlation deltas do not match prior/post values")
            break
        if record.sse_gain != _relative_sse_gain(record.sse_prior, record.sse_post):
            reasons.append("evaluation G_SSE does not match 1 - SSE_post/SSE_prior")
            break
    return reasons


def _slope_integrity_reasons(
    selections: Sequence[SelectionRecord], slopes: Sequence[SlopeRecord]
) -> list[str]:
    selection_by_id = {record.selection_id: record for record in selections}
    operational = [record for record in slopes if record.regime == "operational_method_specific"]
    shared = [record for record in slopes if record.regime == "shared_crossfit_5fold"]
    reasons: list[str] = []
    if (
        len(slopes) != _EXPECTED_SLOPES
        or len(operational) != _EXPECTED_SELECTIONS
        or len({record.selection_id for record in operational}) != _EXPECTED_SELECTIONS
        or {record.selection_id for record in operational} != set(selection_by_id)
        or len(shared) != _DEFAULT_FOLDS
        or {record.fold for record in shared} != set(range(_DEFAULT_FOLDS))
    ):
        return ["slope records are missing or duplicated relative to the required 377 cells"]
    for record in operational:
        if record.selection_id is None:
            reasons.append("operational slope has no linked selection")
            break
        selection = selection_by_id[record.selection_id]
        if (
            record.method != selection.method
            or record.budget != selection.budget
            or record.fold is not None
            or record.n_fit != selection.n_live
        ):
            reasons.append(
                "operational slope method, budget, or fit count does not match its selection"
            )
            break
    if any(
        record.selection_id is not None or record.method is not None or record.budget is not None
        for record in shared
    ):
        reasons.append("shared slope must remain method-independent")
    if any(record.caveat != _CROSSFIT_CAVEAT for record in shared):
        reasons.append("shared slope caveat does not identify non-operational cross-fit evidence")
    if any(
        not isfinite(record.slope) or record.n_fit < 0 or not _is_sha256(record.fit_sha256)
        for record in slopes
    ):
        reasons.append("slope value, fit count, or fit hash is invalid")
    if any(record.n_fit == 0 and not record.fallback for record in slopes):
        reasons.append("an empty slope fit must be marked as a fallback")
    return reasons


def _finite_ordered_interval(interval: object) -> bool:
    if not isinstance(interval, tuple) or len(interval) != _INTERVAL_BOUND_COUNT:
        return False
    lower, upper = interval
    if (
        isinstance(lower, bool)
        or isinstance(upper, bool)
        or not isinstance(lower, int | float)
        or not isinstance(upper, int | float)
    ):
        return False
    return isfinite(float(lower)) and isfinite(float(upper)) and float(lower) <= float(upper)


def _inference_integrity_reasons(
    config: Gate2Config,
    selections: Sequence[SelectionRecord],
    evaluations: Sequence[EvaluationRecord],
    aggregates: Gate2Aggregates,
) -> list[str]:
    reasons: list[str] = []
    inference = aggregates.inference
    expected_budgets = set(config.budgets)
    records_by_budget = {record.budget: record for record in inference}
    if (
        len(inference) != len(config.budgets)
        or len(records_by_budget) != len(config.budgets)
        or set(records_by_budget) != expected_budgets
    ):
        reasons.append("inference evidence is not exactly one cell per registered budget")

    info_by_budget = {record.budget: record for record in selections if record.method == "info"}
    evaluation_by_key = {
        (record.selection_id, record.regime, record.order): record for record in evaluations
    }
    invalid_cell = False
    for budget in config.budgets:
        evidence = records_by_budget.get(budget)
        selection = info_by_budget.get(budget)
        if evidence is None or selection is None:
            invalid_cell = True
            continue
        evaluation = evaluation_by_key.get(
            (selection.selection_id, "shared_crossfit_5fold", "pairwise")
        )
        intervals = (
            evidence.delta_spearman_ci95,
            evidence.delta_pearson_ci95,
            evidence.sse_gain_ci95,
        )
        if evaluation is None or not all(
            _finite_ordered_interval(interval) for interval in intervals
        ):
            invalid_cell = True
            continue
        if (
            evidence.n_terms != evaluation.n_truth
            or evidence.bootstrap_iterations != DEFAULT_BOOTSTRAP_ITERATIONS
            or evidence.delta_spearman != evaluation.delta_spearman
            or evidence.delta_pearson != evaluation.delta_pearson
            or evidence.sse_gain != evaluation.sse_gain
            or evidence.status != _evidence_status(intervals)
        ):
            invalid_cell = True
    if invalid_cell:
        reasons.append(
            "inference evidence is not linked to finite shared pairwise point and CI evidence"
        )
    expected_status = _overall_status([record.status for record in inference])
    if aggregates.inference_status != expected_status:
        reasons.append("inference_status does not match the registered per-budget statuses")
    return reasons


def _aggregate_integrity_reasons(
    config: Gate2Config,
    selections: Sequence[SelectionRecord],
    evaluations: Sequence[EvaluationRecord],
    aggregates: Gate2Aggregates,
) -> list[str]:
    reasons = _inference_integrity_reasons(config, selections, evaluations, aggregates)

    expected_tau2, expected_tau2_status = _tau2_evidence(config.budgets, evaluations)
    tau2_keys = [(record.budget, record.regime, record.statistic) for record in aggregates.tau2]
    if (
        len(aggregates.tau2) != len(config.budgets) * _TAU_CELLS_PER_BUDGET
        or len(set(tau2_keys)) != len(tau2_keys)
        or any(
            record.n_structural != _DEFAULT_STRUCTURAL_SEEDS
            or len(record.empirical_differences) != _DEFAULT_STRUCTURAL_SEEDS
            for record in aggregates.tau2
        )
    ):
        reasons.append("tau2 evidence does not contain every exact 100-seed registered cell")
    if aggregates.tau2 != expected_tau2 or aggregates.tau2_status != expected_tau2_status:
        reasons.append("tau2 evidence or status does not reproduce the evaluation records")

    expected_calibration = _calibration_evidence(config.budgets, evaluations)
    calibration_keys = [
        (record.budget, record.contrast, record.statistic) for record in aggregates.calibration
    ]
    if len(aggregates.calibration) != len(config.budgets) * _CALIBRATION_CELLS_PER_BUDGET or len(
        set(calibration_keys)
    ) != len(calibration_keys):
        reasons.append("calibration evidence does not contain every registered contrast cell")
    expected_confounded = _calibration_is_confounded(expected_calibration)
    if (
        aggregates.calibration != expected_calibration
        or aggregates.calibration_confounded != expected_confounded
    ):
        reasons.append("calibration evidence or confounding status is not reproducible")
    return reasons


def _static_selection_fields(record: SelectionRecord) -> tuple[object, ...]:
    return (
        record.method,
        record.budget,
        record.seed_kind,
        record.seed,
        record.tie_break_version,
        record.boundary_score,
        record.boundary_size,
        record.boundary_selected,
        record.counts_by_order,
        record.selected_ids,
        record.selected_sha256,
        record.selection_id,
    )


def _architecture_eligibility(
    config: Gate2Config,
    selections: Sequence[SelectionRecord],
    evaluations: Sequence[EvaluationRecord],
    slopes: Sequence[SlopeRecord],
    scored: Sequence[ScoredVariant],
    provenance: Mapping[str, object] | None,
    *,
    aggregates: Gate2Aggregates,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    profile = (
        config.protocol_version == PROTOCOL_VERSION
        and config.run_type == RUN_TYPE
        and config.public_claim_eligible is False
        and config.budgets == list(_DEFAULT_BUDGETS)
        and config.random_seeds == _DEFAULT_RANDOM_SEEDS
        and config.structural_seeds == _DEFAULT_STRUCTURAL_SEEDS
        and config.n_folds == _DEFAULT_FOLDS
        and config.max_order == _DEFAULT_MAX_ORDER
        and config.alphabet == AA20
        and config.dataset == "gb1_wu2016"
        and config.model_id == _FROZEN_MODEL_ID
        and config.bootstrap_iterations == DEFAULT_BOOTSTRAP_ITERATIONS
    )
    if not profile:
        reasons.append("configuration is not the exact default GB1 Gate 2 profile")
    universe, universe_hash, universe_reasons = _scored_universe(scored)
    reasons.extend(universe_reasons)
    reasons.extend(_provenance_reasons(provenance, universe_hash))
    reasons.extend(_selection_integrity_reasons(selections, universe))
    reasons.extend(_evaluation_integrity_reasons(selections, evaluations))
    reasons.extend(_aggregate_integrity_reasons(config, selections, evaluations, aggregates))
    pairwise = [record for record in evaluations if record.order == "pairwise"]
    if len(pairwise) != _EXPECTED_SELECTIONS * len(_REGIMES) or any(
        not _finite_required_cell(record) for record in pairwise
    ):
        reasons.append("required pairwise evaluation cells are incomplete or non-finite")
    reasons.extend(_slope_integrity_reasons(selections, slopes))
    if not reasons:
        expected = _build_selection_records(
            scored,
            config.budgets,
            random_seeds=config.random_seeds,
            structural_seeds=config.structural_seeds,
            max_order=config.max_order,
            model_id=config.model_id,
        )
        expected_by_key = {
            (record.method, record.budget, record.seed): record for record in expected
        }
        if any(
            _static_selection_fields(record)
            != _static_selection_fields(
                expected_by_key[(record.method, record.budget, record.seed)]
            )
            for record in selections
        ):
            reasons.append("selection static fields do not reproduce the registered construction")
    return not reasons, reasons


def _report_identity_reasons(report: Gate2Report) -> list[str]:
    reasons: list[str] = []
    if (
        report.protocol_version != PROTOCOL_VERSION
        or report.protocol_version != report.config.protocol_version
    ):
        reasons.append("report protocol_version does not match the registered config")
    if report.run_type != RUN_TYPE or report.run_type != report.config.run_type:
        reasons.append("report run_type does not match the registered config")
    if report.model_dump(mode="python").get("public_claim_eligible") is not False:
        reasons.append("Gate 2 report must remain ineligible for public claims")
    return reasons


def _report_status(provenance: Mapping[str, object] | None, candidate_hash: str) -> str:
    if provenance is None or provenance.get("code_state") != "clean":
        return "provisional"
    return "final" if not _provenance_reasons(provenance, candidate_hash) else "provisional"


def finalize_gate2_report(
    report: Gate2Report,
    scored: Sequence[ScoredVariant],
    provenance: dict[str, object] | None,
) -> Gate2Report:
    """Re-evaluate Gate 2 eligibility and decision against final execution provenance."""
    _universe, candidate_hash, _universe_reasons = _scored_universe(scored)
    eligible, reasons = _architecture_eligibility(
        report.config,
        report.selections,
        report.evaluations,
        report.slopes,
        scored,
        provenance,
        aggregates=report.aggregates,
    )
    identity_reasons = _report_identity_reasons(report)
    reasons.extend(identity_reasons)
    eligible = eligible and not identity_reasons
    return report.model_copy(
        update={
            "protocol_version": PROTOCOL_VERSION,
            "run_type": RUN_TYPE,
            "public_claim_eligible": False,
            "status": _report_status(provenance, candidate_hash),
            "architecture_decision_eligible": eligible,
            "architecture_eligibility_reasons": reasons,
            "decision": decide_gate2(report.aggregates, eligible=eligible),
            "provenance": provenance,
        }
    )


def gate2_report(
    scored: Sequence[ScoredVariant],
    landscape: dict[Variant, float],
    budgets: Sequence[int],
    *,
    random_seeds: int = 20,
    structural_seeds: int = 100,
    n_folds: int = 5,
    max_order: int = 3,
    alphabet: str = AA20,
    dataset: str = "gb1_wu2016",
    model_id: str = "",
    provenance: dict[str, object] | None = None,
) -> Gate2Report:
    """Run Gate 2 in memory; every measured label is read only after all selections are fixed."""
    canonical = _canonical_scored(scored)
    config = Gate2Config(
        budgets=list(budgets),
        random_seeds=random_seeds,
        structural_seeds=structural_seeds,
        n_folds=n_folds,
        max_order=max_order,
        alphabet=alphabet,
        dataset=dataset,
        model_id=model_id,
        bootstrap_iterations=N_BOOTSTRAP,
    )

    # This completes every corrected and legacy selection before the first label reveal.
    selections = _build_selection_records(
        scored,
        budgets,
        random_seeds=random_seeds,
        structural_seeds=structural_seeds,
        max_order=max_order,
        model_id=model_id,
    )
    variant_by_id = {canonical_id(item.variant): item.variant for item in canonical}
    truth_by_order = _truth_terms(canonical, landscape, max_order)
    centered_all = _center_positive(
        reveal_measured_fitness(
            landscape,
            [frozenset(), *(item.variant for item in canonical)],
        )
    )
    esm = {item.variant: item.delta_g for item in canonical}
    shared_by_fold, shared_records = _shared_slopes(canonical, centered_all, n_folds)
    shared_by_variant = {
        item.variant: shared_by_fold[variant_fold(item.variant, n_folds)] for item in canonical
    }

    updated_selections: list[SelectionRecord] = []
    slope_records: list[SlopeRecord] = []
    evaluation_records: list[EvaluationRecord] = []
    evaluation_arrays: dict[tuple[str, str, str], _EvaluationArrays] = {}
    for selection in selections:
        selected = [variant_by_id[identity] for identity in selection.selected_ids]
        revealed, n_revealed, n_live, n_nonpositive, n_missing = _reveal_selection(
            landscape, selected
        )
        updated = selection.model_copy(
            update={
                "n_revealed": n_revealed,
                "n_live": n_live,
                "n_nonpositive": n_nonpositive,
                "n_missing": n_missing,
            }
        )
        updated_selections.append(updated)
        slope, n_fit, fallback, fit_hash = _fit_slope(selected, esm, revealed)
        slope_records.append(
            SlopeRecord(
                regime="operational_method_specific",
                selection_id=updated.selection_id,
                method=updated.method,
                budget=updated.budget,
                fold=None,
                slope=slope,
                n_fit=n_fit,
                fallback=fallback,
                fit_sha256=fit_hash,
            )
        )
        operational_by_variant = {item.variant: slope for item in canonical}
        for regime, slope_by_variant in (
            ("operational_method_specific", operational_by_variant),
            ("shared_crossfit_5fold", shared_by_variant),
        ):
            for order, order_name in _ORDERS:
                truth = truth_by_order[order]
                terms = list(truth)
                prior, post = _inference_arrays(terms, esm, revealed, slope_by_variant)
                record, arrays = _evaluation_record(
                    updated,
                    regime,
                    order_name,
                    truth,
                    revealed,
                    prior,
                    post,
                )
                evaluation_records.append(record)
                if (
                    updated.method == "info"
                    and regime == "shared_crossfit_5fold"
                    and order_name == "pairwise"
                ):
                    evaluation_arrays[(updated.selection_id, regime, order_name)] = arrays
    slope_records.extend(shared_records)

    aggregates = _aggregate_gate2(
        budgets, updated_selections, evaluation_records, evaluation_arrays
    )
    report = Gate2Report(
        status="provisional",
        config=config,
        selections=updated_selections,
        slopes=slope_records,
        evaluations=evaluation_records,
        aggregates=aggregates,
        architecture_decision_eligible=False,
        architecture_eligibility_reasons=["report has not been finalized"],
        decision=decide_gate2(aggregates, eligible=False),
        provenance=provenance,
    )
    return finalize_gate2_report(report, scored, provenance)
