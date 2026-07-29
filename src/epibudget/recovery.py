"""Corrected epistasis-contrast evaluation (audit findings C-1, C-2, H-2, H-4, M-3).

WHY THIS MODULE REPLACES THE DECISION USE OF ``validate.map_recovery``
---------------------------------------------------------------------
The original metric correlated an inferred contrast against the measured one:

    eps_hat(S) = sum_{T in L(S)} c_T * mu(T),      mu(T) = DG(T) if T measured else b * esm(T)
    eps(S)     = sum_{T in L(S)} c_T * DG(T)                                   c_T = (-1)^(|S|-|T|)

Split the loop L(S) into the measured part M and the unmeasured part U:

    eps_hat(S) = k(S) + sum_{T in U} c_T * b * esm(T)
    eps(S)     = k(S) + sum_{T in U} c_T * DG(T)          with   k(S) = sum_{T in M} c_T * DG(T)

Both sides contain the SAME number k(S), built from the same measured labels. A plate that buys
lower-order loop members can therefore raise the correlation through shared measured content,
independently of whether prediction of the unmeasured terms improved. That is the shared-skeleton
confound.

WHAT IS REPORTED INSTEAD
------------------------
Four quantities, never collapsed into one number:

1. ``raw_*``          - the original correlation. DIAGNOSTIC ONLY. It contains k(S).
2. ``skeleton_*``     - the association of k(S) alone with the truth: how much of the raw number the
                        purchased lower-order component already explains, with no prediction at all.
3. ``partial_*``      - the raw association after residualising both sides on k(S). Corrective
                        diagnostic, NOT an uncontested replacement estimand: k(S) is partly
                        information the design legitimately bought, so partialling it out is
                        conservative and can remove real signal.
4. ``relative_sse_gain`` - 1 - SSE(post) / SSE(prior). Unlike a correlation this cannot be inflated
                        by a shared additive term, because a common summand cancels in the residual
                        eps_hat - eps. It is the gate on "recovery" wording.

WHAT IS NOT REPORTED, AND WHY
-----------------------------
An earlier draft of this module also scored a "held-out contrast" estimand: predict eps(S) for terms
whose loop is ENTIRELY unmeasured, so that M is empty and k(S) = 0. That estimand is **withdrawn**
(``docs/specs/prospective-amendment-2.md`` S4.1), and nothing here computes it, because it is
degenerate rather than merely hard:

* it can be empty once a plate buys loop members shared by every eligible term, so there is no term
  to score; and
* where it is non-empty it is uninformative - in the reference-coded basis a residue pair appearing
  in no training row has coefficient exactly 0, so the predicted contrast is the same constant for
  every such term.

``census.n_uninformed`` is retained precisely because it is the evidence for that withdrawal: it is
the size of the population the withdrawn estimand would have scored. Reporting the count and no
correlation is the honest form. No replacement prospective recovery estimand is currently
registered.

CALIBRATION (H-2)
-----------------
The historical policy refit a through-origin slope on each method's own revealed plate. Because
those method-specific slopes can differ in magnitude or sign, the "identical estimator for every
method" claim held in form only. Three policies are reported here, and the two decision-eligible
ones use no labels at all, so they cannot leak and cannot differ across methods:

* ``zero_prior``   - mu(T) = 0 for unmeasured T. Model-free reference.
* ``fixed_unit``   - mu(T) = esm(T). One fixed scale for every method, no fitted parameter.
* ``per_method``   - the historical slope. DIAGNOSTIC; the slope and a sign-disagreement flag are
                     recorded for every method and budget.

A shared cross-fit slope is NOT adopted as the decision policy: ``robustness.py`` fits it on the
full measurable landscape, which is far more label information than any budget buys and is not
charged to any plate, so it fails the budget-accounting condition even though it is
method-independent.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from scipy.stats import pearsonr, rankdata, spearmanr

from epibudget.epistasis import interaction_loop
from epibudget.tie_break import canonical_id
from epibudget.types import Mutation, Variant

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
Term = tuple[Mutation, ...]

# Bumped from the implicit v1 of ``validate.Report``: the field names and the estimand they denote
# both changed, so a consumer written against the old report cannot silently read this one.
SCHEMA_VERSION = 3

_PAIRWISE_ORDER = 2
_MIN_POINTS_FOR_CORR = 3
_N_BOOTSTRAP = 1000

CALIBRATION_POLICIES: tuple[str, ...] = ("zero_prior", "fixed_unit", "per_method")
DECISION_ELIGIBLE_POLICIES: frozenset[str] = frozenset({"zero_prior", "fixed_unit"})


# --------------------------------------------------------------------------- contrast algebra


def contrast_sign(order: int, member: Variant) -> float:
    """The inclusion-exclusion coefficient (-1)^(|S|-|T|) of loop member ``member``."""
    return 1.0 if (order - len(member)) % 2 == 0 else -1.0


def contrast(mu: Mapping[Variant, float], term: Term) -> float:
    """WT-referenced inclusion-exclusion contrast eps(S) = sum_T (-1)^(|S|-|T|) mu(T)."""
    order = len(term)
    total = 0.0
    for size in range(1, order + 1):
        for member in combinations(term, size):
            key = frozenset(member)
            total += contrast_sign(order, key) * mu[key]
    return total


def measured_skeleton(
    term: Term, measured: frozenset[Variant], true_dg: Mapping[Variant, float]
) -> float:
    """k(S): the part of the contrast both the estimate and the truth take from measured labels.

    Every measured loop member contributes its true value with the same sign to eps_hat and to eps,
    so k(S) is shared by construction. It is the quantity the raw correlation is confounded by.
    """
    order = len(term)
    total = 0.0
    for size in range(1, order + 1):
        for member in combinations(term, size):
            key = frozenset(member)
            if key in measured:
                total += contrast_sign(order, key) * true_dg[key]
    return total


def term_sha256(terms: Sequence[Term]) -> str:
    """Order-independent identity hash of an evaluation term set (recorded with each result)."""
    payload = json.dumps(
        sorted(canonical_id(frozenset(term)) for term in terms),
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------- statistics


# A vector is constant for correlation purposes once its spread is at floating-point noise for its
# own magnitude. An exact `std == 0.0` test does NOT catch this: an unidentifiable contrast produces
# predictions that are algebraically identical but differ in the last bits, and ranking that noise
# yields a spurious correlation (0.33 on a 20-term example) out of nothing.
_CONSTANT_RELATIVE_TOLERANCE = 1e-12


def is_effectively_constant(values: FloatArray) -> bool:
    """True when ``values`` varies only at floating-point noise for its own scale."""
    if len(values) == 0:
        return True
    scale = max(1.0, float(np.max(np.abs(values))))
    return bool(float(np.ptp(values)) <= _CONSTANT_RELATIVE_TOLERANCE * scale)


def safe_corr(a: FloatArray, b: FloatArray, kind: str) -> float | None:
    """Correlation, or None when it is not defined on this input.

    Public because every caller that correlates contrasts must use THIS constant test rather than
    an exact ``std == 0.0``. An unidentifiable contrast yields predictions that are algebraically
    identical but differ in the last bits; ranking that noise manufactures a correlation out of
    nothing. A second implementation of this guard is a second chance to reintroduce audit N-2.
    """
    if len(a) < _MIN_POINTS_FOR_CORR or is_effectively_constant(a) or is_effectively_constant(b):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat = spearmanr(a, b).statistic if kind == "spearman" else pearsonr(a, b).statistic
    value = float(stat)
    return None if not np.isfinite(value) else value


def residualise(values: FloatArray, control: FloatArray) -> FloatArray:
    """OLS residual of ``values`` on ``[1, control]``; a constant control only centres."""
    if is_effectively_constant(control):
        return np.asarray(values - float(np.mean(values)), dtype=np.float64)
    design = np.column_stack([np.ones_like(control), control])
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    return np.asarray(values - design @ beta, dtype=np.float64)


def partial_correlation(
    pred: FloatArray, truth: FloatArray, control: FloatArray, kind: str
) -> float | None:
    """Correlation of ``pred`` and ``truth`` after removing the linear effect of ``control``.

    Rank-transforms first for ``kind="spearman"`` (partial Spearman = partial Pearson on ranks).
    """
    if len(truth) < _MIN_POINTS_FOR_CORR:
        return None
    if kind == "spearman":
        pred, truth, control = rankdata(pred), rankdata(truth), rankdata(control)
    a = residualise(np.asarray(pred, dtype=np.float64), np.asarray(control, dtype=np.float64))
    b = residualise(np.asarray(truth, dtype=np.float64), np.asarray(control, dtype=np.float64))
    return safe_corr(a, b, "pearson")


def relative_sse_gain(prior: FloatArray, post: FloatArray, truth: FloatArray) -> float | None:
    """1 - SSE(post)/SSE(prior): the fraction of squared contrast error the plate removed.

    Negative means the measured plate made the contrast estimate WORSE than making no measurement at
    all. A shared additive term cancels in both residuals, so unlike a correlation this cannot be
    inflated by the skeleton.
    """
    sse_prior = float(np.sum(np.square(prior - truth)))
    if sse_prior == 0.0:
        return None
    return 1.0 - float(np.sum(np.square(post - truth))) / sse_prior


def fisher_z_mean(values: Sequence[float]) -> float | None:
    """Mean correlation via Fisher-z (audit L-2); arithmetic averaging of r is biased.

    Values at +/-1 are clipped to the nearest representable interior point so a single perfect
    correlation cannot send the mean to infinity.
    """
    finite = [v for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return None
    clipped = np.clip(np.asarray(finite, dtype=np.float64), -0.999999999999, 0.999999999999)
    return float(np.tanh(np.mean(np.arctanh(clipped))))


def paired_difference_ci(
    pred_a: FloatArray,
    pred_b: FloatArray,
    truth: FloatArray,
    kind: str,
    seed: int = 0,
    n_bootstrap: int = _N_BOOTSTRAP,
) -> tuple[float | None, tuple[float, float] | None]:
    """Bootstrap CI of corr(A) - corr(B) on the SAME resampled terms (audit H-4).

    Resampling the identical index set for both methods keeps the difference paired, which
    non-overlap of two independently-resampled marginal intervals does not. The resampling unit is
    still the evaluation term, so this interval describes term-level leverage within one landscape
    and nothing about a second protein.
    """
    n = len(truth)
    if n < _MIN_POINTS_FOR_CORR:
        return None, None
    corr_a = safe_corr(pred_a, truth, kind)
    corr_b = safe_corr(pred_b, truth, kind)
    delta = None if (corr_a is None or corr_b is None) else corr_a - corr_b
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        a = safe_corr(pred_a[idx], truth[idx], kind)
        b = safe_corr(pred_b[idx], truth[idx], kind)
        if a is not None and b is not None:
            samples.append(a - b)
    if len(samples) < _MIN_POINTS_FOR_CORR:
        return delta, None
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return delta, (float(lo), float(hi))


# --------------------------------------------------------------------------- report models


class CalibrationRecord(BaseModel):
    """The prior scale a method/budget cell used, and whether it is decision-eligible."""

    model_config = {"frozen": True}

    policy: str
    slope: float
    n_calibration_labels: int
    labels_are_method_specific: bool
    decision_eligible: bool


class TermCensus(BaseModel):
    """How the plate touches the evaluated terms; the three classes are disjoint and exhaustive."""

    model_config = {"frozen": True}

    n_terms: int
    n_pinned: int  # every loop member measured -> the contrast is read off, not predicted
    n_informed_not_pinned: int  # partly measured -> carries a non-zero skeleton k(S)
    n_uninformed: int  # no loop member measured -> k(S) = 0, a genuine prediction

    def check(self) -> None:
        total = self.n_pinned + self.n_informed_not_pinned + self.n_uninformed
        if total != self.n_terms:
            raise ValueError(f"term census sums to {total}, expected {self.n_terms}")


class OrderRecovery(BaseModel):
    """Corrected evaluation of one (method, budget, calibration policy, interaction order) cell."""

    model_config = {"frozen": True}

    order: str
    estimand: str
    n_terms: int
    term_sha256: str
    census: TermCensus
    # (1) DIAGNOSTIC: contains the purchased skeleton k(S) on both sides.
    raw_pearson_with_skeleton: float | None
    raw_spearman_with_skeleton: float | None
    # (2) how much k(S) alone explains, with no prediction whatsoever
    skeleton_pearson: float | None
    skeleton_spearman: float | None
    # (3) corrective diagnostic: association after residualising on k(S)
    partial_pearson: float | None
    partial_spearman: float | None
    # (4) the wording gate: error reduction versus making no measurement at all
    sse_prior: float
    sse_post: float
    relative_sse_gain: float | None
    recovery_wording_permitted: bool


SEED_KINDS: tuple[str, ...] = ("none", "tie", "random")


class MethodRecovery(BaseModel):
    """One realised plate under one method, budget, seed and calibration policy.

    ``seed_kind`` says what kind of draw produced this plate, because the three kinds are not
    interchangeable and collapsing them was how the original ``structural`` number came to be a
    sample of size one presented as a method:

    * ``none``   - the plate is a deterministic function of the scores (``fitness``, ``info``,
                   ``practice``). ``seed`` is None and there is no distribution to report.
    * ``tie``    - the score has an exact stratum straddling the budget cut, so the tie seed
                   decides the plate (``structural``, ``singles_zero_prior``). Every declared seed
                   is a separate record; no seed is presented as representative.
    * ``random`` - the plate is a uniform sample and ``seed`` is its RNG seed.

    ``n_selected`` may fall short of ``budget``: ``practice`` under-fills when the pool holds
    fewer valid cross-site pairs than the budget. ``n_revealed`` is smaller again, since only
    variants with a strictly positive measured fitness have a defined log-ratio.
    """

    model_config = {"frozen": True}

    method: str
    budget: int
    seed: int | None
    seed_kind: str
    selected_identity_sha256: str
    tie_stratum_crosses_budget: bool
    n_selected: int
    n_revealed: int
    calibration: CalibrationRecord
    orders: list[OrderRecovery]


TERM_SUBSETS: tuple[str, ...] = (
    "all_eligible",
    "common_informed_not_pinned",
    "common_uninformed",
)


class PairedContrastResult(BaseModel):
    """One realised A-vs-B difference on an explicitly recorded term set (M-3, H-4).

    ``delta`` is None whenever the difference is not defined on this subset, and ``reason`` says
    which case it is. That distinction is load-bearing: a null on ``common_uninformed`` under
    ``zero_prior`` is not a missing measurement but the identifiability wall itself, since both
    methods predict exactly 0 for every untouched term and a constant vector has no correlation.
    """

    model_config = {"frozen": True}

    method_a: str
    method_b: str
    seed_kind_a: str
    seed_kind_b: str
    seed_a: int | None
    seed_b: int | None
    selected_identity_sha256_a: str
    selected_identity_sha256_b: str
    budget: int
    order: str
    calibration_policy: str
    term_subset: str
    n_terms: int
    term_sha256: str
    statistic: str
    delta: float | None
    # Bootstrap over evaluation TERMS at a single realisation of each plate. It describes how much
    # the difference depends on which terms happen to be in the evaluation set -- term leverage --
    # and says nothing about how much it depends on which plate the seed drew. The old name
    # `delta_ci95` invited reading it as the latter.
    term_leverage_ci95: tuple[float, float] | None
    term_leverage_excludes_zero: bool
    reason: str  # "" when delta is defined
    interpretation: str


class SelectionVariabilitySummary(BaseModel):
    """Spread across realised plate pairs, separate from the conditional term bootstrap."""

    model_config = {"frozen": True}

    method_a: str
    method_b: str
    budget: int
    order: str
    calibration_policy: str
    term_subset: str
    n_pairs: int
    n_defined: int
    delta_mean: float | None
    delta_median: float | None
    delta_min: float | None
    delta_max: float | None
    fraction_positive: float | None


class CorrectedRecoveryReport(BaseModel):
    """Schema v3. The raw correlation survives only as a labelled diagnostic field.

    ENFORCED: ``scripts/corrected_recovery.py`` constructs this model and validates it before any
    bytes reach disk, so a file that fails these constraints is never written rather than written
    and later discovered to be wrong.

    That matters because the discipline this schema encodes -- a term-set hash on every comparison,
    an exhaustive term census, an explicit calibration record, a seed distribution wherever a plate
    is not a single draw -- is exactly what stops the corrected report from regressing into the
    confounded one. A schema nothing constructs enforces none of it.
    """

    model_config = {"extra": "forbid"}

    schema_version: int = SCHEMA_VERSION
    analysis: str
    dataset: str
    model_id: str
    alphabet: str
    budgets: list[int]
    max_order: int
    methods_evaluated: list[str]
    eligible_population: str
    calibration_policies: list[str]
    decision_eligible_policies: list[str]
    paired_contrast_policies: list[str]
    tie_seeds: int
    random_seeds: int
    tie_break_version: str
    data_path: str
    data_sha256: str
    scored_cache: str
    scored_cache_sha256: str
    n_candidates: int
    status: str
    decision_eligible: bool
    reason_not_decision_eligible: str
    generated_at_utc: str
    provenance: dict[str, object]
    methods: list[MethodRecovery]
    paired_contrasts: list[PairedContrastResult]
    selection_variability: list[SelectionVariabilitySummary]
    note: str


# --------------------------------------------------------------------------- evaluation


ELIGIBLE_POPULATION = (
    "terms of order 2..max_order over the registered candidate universe whose every loop member "
    "has a strictly positive measured fitness, so the WT-referenced log-ratio DG = log(f/f_ref) is "
    "defined for the whole loop; declared before selection and identical for every method"
)

REPORT_NOTE = (
    "raw_*_with_skeleton is a DIAGNOSTIC that contains the measured lower-order component shared "
    "with the truth; it is not epistasis-map reconstruction. relative_sse_gain gates 'recovery' "
    "wording. partial_* is a conservative corrective diagnostic, not an uncontested estimand. The "
    "held-out contrast estimand is WITHDRAWN as degenerate and is not computed; "
    "census.n_uninformed records the population it would have scored. Every number is a "
    "within-landscape retrospective "
    "corrective reanalysis, not a preregistered confirmation."
)


def prior_mu(
    esm: Mapping[Variant, float],
    revealed: Mapping[Variant, float],
    policy: str,
) -> tuple[dict[Variant, float], CalibrationRecord]:
    """The pre-measurement prior over DG under one calibration policy, plus its provenance record.

    ``zero_prior`` and ``fixed_unit`` read no label at all, so they are identical across methods by
    construction and cannot leak an evaluation label into the prior. ``per_method`` reproduces the
    historical through-origin slope and is retained only as a diagnostic.
    """
    if policy == "zero_prior":
        mu = dict.fromkeys(esm, 0.0)
        record = CalibrationRecord(
            policy=policy,
            slope=0.0,
            n_calibration_labels=0,
            labels_are_method_specific=False,
            decision_eligible=True,
        )
    elif policy == "fixed_unit":
        mu = {v: float(value) for v, value in esm.items()}
        record = CalibrationRecord(
            policy=policy,
            slope=1.0,
            n_calibration_labels=0,
            labels_are_method_specific=False,
            decision_eligible=True,
        )
    elif policy == "per_method":
        x = np.array([esm[v] for v in revealed], dtype=np.float64)
        y = np.array([revealed[v] for v in revealed], dtype=np.float64)
        denominator = float(x @ x) if len(x) else 0.0
        slope = float(x @ y / denominator) if denominator != 0.0 else 1.0
        mu = {v: slope * float(value) for v, value in esm.items()}
        record = CalibrationRecord(
            policy=policy,
            slope=slope,
            n_calibration_labels=len(revealed),
            labels_are_method_specific=True,
            decision_eligible=False,
        )
    else:
        raise ValueError(f"unknown calibration policy {policy!r}; expected {CALIBRATION_POLICIES}")
    mu[frozenset()] = 0.0
    return mu, record


def evaluate_order(
    terms: Sequence[Term],
    truth_by_term: Mapping[Term, float],
    true_dg: Mapping[Variant, float],
    prior: Mapping[Variant, float],
    revealed: Mapping[Variant, float],
    order_name: str,
) -> OrderRecovery:
    """Evaluate one interaction order for one plate under one prior."""
    measured = frozenset(revealed)
    post = dict(prior)
    post.update(revealed)

    truth = np.array([truth_by_term[t] for t in terms], dtype=np.float64)
    prior_eps = np.array([contrast(prior, t) for t in terms], dtype=np.float64)
    post_eps = np.array([contrast(post, t) for t in terms], dtype=np.float64)
    skeleton = np.array([measured_skeleton(t, measured, true_dg) for t in terms], dtype=np.float64)

    loops = [interaction_loop(t) for t in terms]
    n_pinned = sum(1 for loop in loops if all(m in measured for m in loop))
    n_touched = sum(1 for loop in loops if any(m in measured for m in loop))
    census = TermCensus(
        n_terms=len(terms),
        n_pinned=n_pinned,
        n_informed_not_pinned=n_touched - n_pinned,
        n_uninformed=len(terms) - n_touched,
    )
    census.check()

    gain = relative_sse_gain(prior_eps, post_eps, truth)
    return OrderRecovery(
        order=order_name,
        estimand=(
            "inclusion-exclusion contrast estimate vs measured contrast over the eligible "
            "population; raw fields include the purchased skeleton k(S)"
        ),
        n_terms=len(terms),
        term_sha256=term_sha256(terms),
        census=census,
        raw_pearson_with_skeleton=safe_corr(post_eps, truth, "pearson"),
        raw_spearman_with_skeleton=safe_corr(post_eps, truth, "spearman"),
        skeleton_pearson=safe_corr(skeleton, truth, "pearson"),
        skeleton_spearman=safe_corr(skeleton, truth, "spearman"),
        partial_pearson=partial_correlation(post_eps, truth, skeleton, "pearson"),
        partial_spearman=partial_correlation(post_eps, truth, skeleton, "spearman"),
        sse_prior=float(np.sum(np.square(prior_eps - truth))),
        sse_post=float(np.sum(np.square(post_eps - truth))),
        relative_sse_gain=gain,
        recovery_wording_permitted=gain is not None and gain > 0.0,
    )


def common_term_subset(
    terms: Sequence[Term],
    revealed_a: Mapping[Variant, float],
    revealed_b: Mapping[Variant, float],
    subset: str,
) -> list[Term]:
    """Terms in the SAME state for both methods, so a comparison is one estimand (audit M-3).

    The previous "precision" split correlated each method on its own informed-not-pinned set and
    compared the two numbers as though they estimated the same quantity.
    """
    a, b = frozenset(revealed_a), frozenset(revealed_b)
    out: list[Term] = []
    for term in terms:
        loop = interaction_loop(term)
        pinned_a, pinned_b = all(m in a for m in loop), all(m in b for m in loop)
        touched_a, touched_b = any(m in a for m in loop), any(m in b for m in loop)
        if subset == "common_informed_not_pinned":
            if touched_a and touched_b and not pinned_a and not pinned_b:
                out.append(term)
        elif subset == "common_uninformed":
            if not touched_a and not touched_b:
                out.append(term)
        elif subset == "all_eligible":
            out.append(term)
        else:
            raise ValueError(f"unknown term subset {subset!r}")
    return out
