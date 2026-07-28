"""Corrected epistasis-contrast reanalysis of a completed score cache (audit C-1/C-2/H-1/H-2/M-3).

Retrospective corrective reanalysis, not a preregistered confirmation: the estimands below were
chosen after the audit found the shared-skeleton confound in the original metric. It consumes only
an already-computed ESM cache and the measured landscape -- no GPU, no model, no network.

For every dataset x method x budget x calibration policy x interaction order it writes:

* the raw contrast correlation, labelled as containing the purchased skeleton (diagnostic only);
* the skeleton-alone association;
* the skeleton-controlled partial correlation;
* the relative SSE gain against the pre-measurement prior (the "recovery" wording gate);
* the pinned / informed-not-pinned / uninformed term census and the term-set hash.

It then writes paired A-vs-B contrasts restricted to terms in the SAME state for both plates, so a
comparison is one estimand rather than two (audit M-3) and its uncertainty is a paired bootstrap
rather than the non-overlap of two marginal intervals (audit H-4).

Every method whose acquisition score has an exact stratum straddling the budget cut is evaluated
over a declared distribution of seeds rather than one arbitrary draw, because for such a method the
seed *is* the plate. That test is made against the data instead of a hardcoded method list, so a
method that becomes tie-dominated is reported as a distribution automatically.

The whole report is built as a :class:`epibudget.recovery.CorrectedRecoveryReport` and validated
before any byte reaches disk.

Usage:
    python scripts/corrected_recovery.py --dataset trpb_johnston2024 \
        --scored-cache report/scored_trpb_650m_n16.jsonl --out report/remediation/
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from _console import configure_utf8_stdout

from epibudget.data import enumerate_candidates, resolve_dataset
from epibudget.epistasis import interaction_loop, wt_centered_log_fitness
from epibudget.recovery import (
    CALIBRATION_POLICIES,
    DECISION_ELIGIBLE_POLICIES,
    ELIGIBLE_POPULATION,
    REPORT_NOTE,
    SCHEMA_VERSION,
    TERM_SUBSETS,
    BoolArray,
    CalibrationRecord,
    CorrectedRecoveryReport,
    FloatArray,
    MethodRecovery,
    OrderRecovery,
    PairedContrastResult,
    SeedDistribution,
    Term,
    TermCensus,
    common_term_subset,
    contrast,
    fisher_z_mean,
    paired_difference_ci,
    partial_correlation,
    prior_mu,
    relative_sse_gain,
    safe_corr,
    term_sha256,
)
from epibudget.scored_cache import CacheMetadata, cache_metadata_path, load_cache
from epibudget.tie_break import (
    REQUIRED_TIE_SEEDS,
    TIE_BREAK_VERSION,
    canonical_id,
    loop_counts_over_universe,
    seeded_order,
    stratum_crosses_budget,
)
from epibudget.types import ScoredVariant, Variant
from epibudget.validate import practice_heuristic, random_selection

_ORDERS = ((2, "pairwise"), (3, "third"))
_CONTRAST_TOL = 1e-9

# Paired contrasts run only under the label-free policies. Under ``per_method`` the two arms
# carry differently-fitted priors -- on GB1 with opposite signs -- so the difference is not one
# estimand.
_PAIRED_POLICIES = ("zero_prior", "fixed_unit")

# Declared before the run. Each pair answers a question that was actually asked of this project.
_PAIRED_METHODS: tuple[tuple[str, str], ...] = (
    ("structural", "fitness"),  # the historical headline claim
    ("fitness", "random"),  # prospective amendment 2 S4.2 primary contrast
    ("structural", "random"),  # the original H1 second clause
    ("structural", "singles_zero_prior"),  # does ESM-free coverage beat pure mutation order?
    ("info", "structural"),  # the gate-2 question
)


@dataclass(frozen=True)
class _Plate:
    """One realised selection: what was bought, and what of it carries a usable label."""

    method: str
    budget: int
    seed: int | None
    seed_kind: str
    tie_stratum_crosses_budget: bool
    selected: list[Variant]
    mask: BoolArray  # over the universe + the WT anchor
    revealed: dict[Variant, float]


class _ContrastIndex:
    """Vectorised inclusion-exclusion over a fixed term set.

    Equivalent to calling :func:`epibudget.recovery.contrast` per term; the equivalence is asserted
    on a random sample at build time so this fast path can never silently diverge from the reference
    implementation the unit tests pin.
    """

    def __init__(self, terms: list[Term], variant_index: dict[Variant, int]) -> None:
        self.terms = terms
        width = max(len(interaction_loop(t)) for t in terms)
        self.idx = np.zeros((len(terms), width), dtype=np.int64)
        self.sign = np.zeros((len(terms), width), dtype=np.float64)
        for row, term in enumerate(terms):
            for col, member in enumerate(interaction_loop(term)):
                self.idx[row, col] = variant_index[member]
                self.sign[row, col] = 1.0 if (len(term) - len(member)) % 2 == 0 else -1.0

    def evaluate(self, mu_vector: FloatArray) -> FloatArray:
        contrasts: FloatArray = np.einsum("ij,ij->i", self.sign, mu_vector[self.idx])
        return contrasts

    def check_against_reference(self, mu: dict[Variant, float], mu_vector: FloatArray) -> None:
        fast = self.evaluate(mu_vector)
        rng = np.random.default_rng(0)
        for row in rng.choice(len(self.terms), size=min(64, len(self.terms)), replace=False):
            reference = contrast(mu, self.terms[int(row)])
            if abs(reference - float(fast[int(row)])) > _CONTRAST_TOL:
                raise AssertionError(
                    f"vectorised contrast diverged from recovery.contrast at row {row}"
                )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluate(
    index: _ContrastIndex,
    truth: FloatArray,
    true_vector: FloatArray,
    prior_vector: FloatArray,
    revealed_mask: BoolArray,
    order_name: str,
) -> tuple[OrderRecovery, FloatArray]:
    """One (method, budget, policy, order) cell, plus the posterior contrasts it was scored on.

    The contrasts are returned so a paired A-vs-B difference can be taken on the identical term
    positions without recomputing either arm.
    """
    post_vector = np.where(revealed_mask, true_vector, prior_vector)
    prior_eps = index.evaluate(prior_vector)
    post_eps = index.evaluate(post_vector)
    skeleton = index.evaluate(np.where(revealed_mask, true_vector, 0.0))

    measured_members = revealed_mask[index.idx]
    n_pinned = int(measured_members.all(axis=1).sum())
    n_touched = int(measured_members.any(axis=1).sum())
    census = TermCensus(
        n_terms=len(truth),
        n_pinned=n_pinned,
        n_informed_not_pinned=n_touched - n_pinned,
        n_uninformed=len(truth) - n_touched,
    )
    census.check()

    gain = relative_sse_gain(prior_eps, post_eps, truth)
    return (
        OrderRecovery(
            order=order_name,
            estimand=(
                "inclusion-exclusion contrast estimate vs measured contrast over the eligible "
                "population; raw fields include the purchased skeleton k(S)"
            ),
            n_terms=len(truth),
            term_sha256=term_sha256(index.terms),
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
            recovery_wording_permitted=bool(gain is not None and gain >= 0.0),
        ),
        post_eps,
    )


def _seed_distribution(seed_kind: str, cells: list[OrderRecovery]) -> SeedDistribution:
    """Spread of one order's cell over the declared seeds. Correlations average on the z scale."""
    gains = [c.relative_sse_gain for c in cells if c.relative_sse_gain is not None]
    raws = [c.raw_pearson_with_skeleton for c in cells if c.raw_pearson_with_skeleton is not None]
    return SeedDistribution(
        seed_kind=seed_kind,
        n_seeds=len(cells),
        tie_break_version=TIE_BREAK_VERSION if seed_kind == "tie" else None,
        raw_pearson_mean=fisher_z_mean(raws),
        raw_pearson_min=min(raws) if raws else None,
        raw_pearson_max=max(raws) if raws else None,
        partial_spearman_mean=fisher_z_mean(
            [c.partial_spearman for c in cells if c.partial_spearman is not None]
        ),
        relative_sse_gain_mean=float(np.mean(gains)) if gains else None,
        relative_sse_gain_min=float(np.min(gains)) if gains else None,
        recovery_wording_permitted_fraction=float(
            np.mean([c.recovery_wording_permitted for c in cells])
        ),
    )


def _interpret(subset: str, delta: float | None, ci: tuple[float, float] | None) -> str:
    if delta is None:
        return "difference not defined on this subset"
    if ci is None:
        return "point difference only; the paired bootstrap did not yield a usable interval"
    direction = "favours A" if delta > 0.0 else "favours B" if delta < 0.0 else "exactly tied"
    strength = "excludes 0" if ci[0] > 0.0 or ci[1] < 0.0 else "includes 0"
    return f"{direction}; paired 95% interval {strength} on the {subset} term set"


def main() -> None:  # noqa: PLR0915
    # A single linear analysis script: splitting it would scatter the run's provenance.
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scored-cache", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--budgets", default="48,96,192")
    parser.add_argument("--alphabet", default="ACDEFGHIKLMNPQRSTVWY")
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--random-seeds", type=int, default=20)
    parser.add_argument("--tie-seeds", type=int, default=REQUIRED_TIE_SEEDS)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("report/remediation"))
    args = parser.parse_args()

    spec = resolve_dataset(args.dataset)
    data_path = args.data or Path(spec.default_data_path)
    budgets = [int(b) for b in args.budgets.split(",")]

    landscape = spec.loader(data_path)
    enumerated = enumerate_candidates(
        spec.sites, spec.wt_at_sites, allowed_aa=args.alphabet, max_order=args.max_order
    )
    cache = load_cache(args.scored_cache)
    metadata = CacheMetadata.model_validate_json(
        cache_metadata_path(args.scored_cache).read_text(encoding="utf-8")
    )
    scored: list[ScoredVariant] = [cache[v] for v in enumerated]
    esm = {sv.variant: sv.delta_g for sv in scored}
    tau2 = {sv.variant: sv.var_delta_g for sv in scored}

    # Eligible population: declared before any selection, identical for every method.
    dg = wt_centered_log_fitness(landscape)
    universe = [sv.variant for sv in scored]
    variant_index = {v: i for i, v in enumerate(universe)}
    variant_index[frozenset()] = len(universe)
    true_vector = np.zeros(len(universe) + 1, dtype=np.float64)
    in_dg = np.zeros(len(universe) + 1, dtype=bool)
    for v, i in variant_index.items():
        if v in dg:
            true_vector[i] = dg[v]
            in_dg[i] = True
    in_dg[len(universe)] = True  # the WT reference is exactly 0 by construction

    counts = loop_counts_over_universe(universe, args.max_order)
    print(
        f"{args.dataset}: {len(universe)} candidates, n(v) values "
        f"{sorted({counts[v] for v in universe})}"
    )

    per_order: dict[str, tuple[_ContrastIndex, FloatArray]] = {}
    for order, name in _ORDERS:
        terms = [
            tuple(sorted(v))
            for v in universe
            if len(v) == order and all(m in dg for m in interaction_loop(tuple(sorted(v))))
        ]
        index = _ContrastIndex(terms, variant_index)
        # The fast path must agree with the unit-tested reference implementation.
        index.check_against_reference(
            {v: true_vector[i] for v, i in variant_index.items()}, true_vector
        )
        per_order[name] = (index, index.evaluate(true_vector))
        print(f"  {name}: {len(terms)} eligible terms")

    # ---------------------------------------------------------------- selection orderings
    scores: dict[str, Callable[[Variant], float]] = {
        "info": lambda v: tau2[v] * counts[v],
        "fitness": lambda v: esm[v],
        "structural": lambda v: float(counts[v]),
        # Model-free baseline: buy the singles first, then doubles, then triples. No ESM at all.
        "singles_zero_prior": lambda v: float(-len(v)),
    }

    def plate_of(selected: list[Variant]) -> tuple[BoolArray, dict[Variant, float]]:
        mask: BoolArray = np.zeros(len(universe) + 1, dtype=np.bool_)
        mask[len(universe)] = True  # the WT anchor is always known
        revealed: dict[Variant, float] = {}
        for v in selected:
            i = variant_index[v]
            if in_dg[i]:  # only strictly-positive-fitness rows have a log-ratio
                mask[i] = True
                revealed[v] = dg[v]
        return mask, revealed

    def make(method: str, budget: int, seed: int | None, seed_kind: str, tied: bool) -> _Plate:
        if method == "practice":
            selected = practice_heuristic(scored, budget)
        elif method == "random":
            assert seed is not None
            selected = random_selection(scored, budget, seed)
        else:
            selected = seeded_order(
                universe, scores[method], canonical_id, seed if seed is not None else 0
            )[:budget]
        mask, revealed = plate_of(selected)
        return _Plate(method, budget, seed, seed_kind, tied, selected, mask, revealed)

    def plates_for(method: str, budget: int) -> list[_Plate]:
        """Every draw of one (method, budget) cell. Length > 1 exactly when one draw is a sample.

        Whether a scored method needs a distribution is decided against the data, not from a list of
        method names: it needs one precisely when an exact score stratum straddles the budget cut.
        """
        if method == "practice":
            return [make(method, budget, None, "none", False)]
        if method == "random":
            return [
                make(method, budget, s, "random", False) for s in range(max(1, args.random_seeds))
            ]
        tied = stratum_crosses_budget(universe, scores[method], canonical_id, budget)
        if not tied:
            return [make(method, budget, None, "none", False)]
        return [make(method, budget, s, "tie", True) for s in range(max(1, args.tie_seeds))]

    # ---------------------------------------------------------------- evaluation
    # zero_prior and fixed_unit read no label, so their prior vector is the same for every plate.
    prior_cache: dict[str, tuple[FloatArray, CalibrationRecord]] = {}

    def prior_for(
        policy: str, revealed: Mapping[Variant, float]
    ) -> tuple[FloatArray, CalibrationRecord]:
        cached = prior_cache.get(policy)
        if cached is not None:
            return cached
        mu, record = prior_mu(esm, revealed, policy)
        vector = np.zeros(len(universe) + 1, dtype=np.float64)
        for v, i in variant_index.items():
            vector[i] = mu.get(v, 0.0)
        result = (vector, record)
        if policy in DECISION_ELIGIBLE_POLICIES:
            prior_cache[policy] = result
        return result

    methods = ("info", "structural", "fitness", "random", "practice", "singles_zero_prior")
    method_records: list[MethodRecovery] = []
    # (method, budget, policy, order) -> posterior contrasts of the representative plate
    representative: dict[tuple[str, int, str, str], FloatArray] = {}
    representative_plate: dict[tuple[str, int], _Plate] = {}

    for method in methods:
        for budget in budgets:
            draws = plates_for(method, budget)
            representative_plate[(method, budget)] = draws[0]
            # The model-free baseline is defined by its zero prior; other policies would not be it.
            policies = ("zero_prior",) if method == "singles_zero_prior" else CALIBRATION_POLICIES
            for policy in policies:
                per_draw: dict[str, list[OrderRecovery]] = {name: [] for _o, name in _ORDERS}
                # The calibration record must come from the SAME draw as the scalar cells below.
                # Under `per_method` the slope is fitted on that draw's own revealed labels, so
                # taking it from the last draw would pair seed 0's numbers with seed 99's slope.
                first_record: CalibrationRecord | None = None
                for draw in draws:
                    prior_vector, record = prior_for(policy, draw.revealed)
                    if first_record is None:
                        first_record = record
                    for _order, name in _ORDERS:
                        index, truth = per_order[name]
                        cell, post_eps = _evaluate(
                            index, truth, true_vector, prior_vector, draw.mask, name
                        )
                        per_draw[name].append(cell)
                        if draw is draws[0]:
                            representative[(method, budget, policy, name)] = post_eps
                assert first_record is not None
                orders = []
                for _order, name in _ORDERS:
                    cells = per_draw[name]
                    head = cells[0]
                    if len(cells) > 1:
                        head = head.model_copy(
                            update={
                                "seed_distribution": _seed_distribution(draws[0].seed_kind, cells)
                            }
                        )
                    orders.append(head)
                first = draws[0]
                method_records.append(
                    MethodRecovery(
                        method=method,
                        budget=budget,
                        seed=first.seed,
                        seed_kind=first.seed_kind,
                        tie_stratum_crosses_budget=first.tie_stratum_crosses_budget,
                        n_selected=len(first.selected),
                        n_revealed=len(first.revealed),
                        calibration=first_record,
                        orders=orders,
                    )
                )
            print(f"  {method} B={budget}: {len(draws)} draw(s), kind={draws[0].seed_kind}")

    # ---------------------------------------------------------------- paired contrasts (M-3, H-4)
    def paired() -> Iterator[PairedContrastResult]:
        for method_a, method_b in _PAIRED_METHODS:
            for budget in budgets:
                plate_a = representative_plate[(method_a, budget)]
                plate_b = representative_plate[(method_b, budget)]
                for policy in _PAIRED_POLICIES:
                    if method_b == "singles_zero_prior" and policy != "zero_prior":
                        continue  # that baseline exists only under its zero prior
                    for _order, name in _ORDERS:
                        index, truth = per_order[name]
                        pred_a = representative[(method_a, budget, policy, name)]
                        pred_b = representative[(method_b, budget, policy, name)]
                        positions = {t: i for i, t in enumerate(index.terms)}
                        for subset in TERM_SUBSETS:
                            common = common_term_subset(
                                index.terms, plate_a.revealed, plate_b.revealed, subset
                            )
                            keep = [positions[t] for t in common]
                            yield _paired_result(
                                method_a,
                                method_b,
                                plate_a,
                                plate_b,
                                budget,
                                name,
                                policy,
                                subset,
                                [index.terms[i] for i in keep],
                                pred_a[keep],
                                pred_b[keep],
                                truth[keep],
                                args.bootstrap,
                            )

    paired_contrasts = list(paired())
    print(f"  paired contrasts: {len(paired_contrasts)}")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    report = CorrectedRecoveryReport(
        schema_version=SCHEMA_VERSION,
        analysis="corrected_contrast_reanalysis",
        dataset=args.dataset,
        model_id=metadata.model_id,
        alphabet=args.alphabet,
        budgets=budgets,
        max_order=args.max_order,
        methods_evaluated=list(methods),
        eligible_population=ELIGIBLE_POPULATION,
        calibration_policies=list(CALIBRATION_POLICIES),
        decision_eligible_policies=sorted(DECISION_ELIGIBLE_POLICIES),
        paired_contrast_policies=list(_PAIRED_POLICIES),
        tie_seeds=args.tie_seeds,
        random_seeds=args.random_seeds,
        tie_break_version=TIE_BREAK_VERSION,
        data_path=str(data_path),
        data_sha256=_sha256_file(data_path),
        scored_cache=str(args.scored_cache),
        scored_cache_sha256=_sha256_file(args.scored_cache),
        n_candidates=len(universe),
        status="retrospective_corrective_reanalysis",
        decision_eligible=False,
        reason_not_decision_eligible=(
            "estimands selected after the audit; single landscape per report"
        ),
        generated_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        methods=method_records,
        paired_contrasts=paired_contrasts,
        note=REPORT_NOTE,
    )
    target = out_dir / f"corrected_recovery_{args.dataset}.json"
    target.write_text(report.model_dump_json(indent=1) + "\n", encoding="utf-8")
    print(f"wrote {target}")


def _paired_result(
    method_a: str,
    method_b: str,
    plate_a: _Plate,
    plate_b: _Plate,
    budget: int,
    order: str,
    policy: str,
    subset: str,
    terms: list[Term],
    pred_a: FloatArray,
    pred_b: FloatArray,
    truth: FloatArray,
    n_bootstrap: int,
) -> PairedContrastResult:
    """One paired difference, or an explicit statement of why it is not defined here."""
    delta, ci = paired_difference_ci(
        pred_a, pred_b, truth, "spearman", seed=budget, n_bootstrap=n_bootstrap
    )
    reason = ""
    if not terms:
        reason = "no term is in the same state for both plates at this budget"
    elif delta is None:
        constant_a = safe_corr(pred_a, truth, "spearman") is None
        constant_b = safe_corr(pred_b, truth, "spearman") is None
        if constant_a and constant_b:
            reason = (
                "both plates predict a constant on this subset, so neither correlation exists; "
                "under zero_prior on untouched terms this IS the identifiability wall, not a gap "
                "in the measurement"
            )
        else:
            side = method_a if constant_a else method_b
            reason = f"{side} predicts a constant on this subset, so its correlation is undefined"
    return PairedContrastResult(
        method_a=method_a,
        method_b=method_b,
        seed_a=plate_a.seed,
        seed_b=plate_b.seed,
        budget=budget,
        order=order,
        calibration_policy=policy,
        term_subset=subset,
        n_terms=len(terms),
        term_sha256=term_sha256(terms),
        statistic="spearman",
        delta=delta,
        delta_ci95=ci,
        excludes_zero=bool(ci is not None and (ci[0] > 0.0 or ci[1] < 0.0)),
        reason=reason,
        interpretation=_interpret(subset, delta, ci),
    )


if __name__ == "__main__":
    main()
