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
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from _console import configure_utf8_stdout

from epibudget.data import enumerate_candidates, resolve_dataset
from epibudget.epistasis import interaction_loop, wt_centered_log_fitness
from epibudget.provenance import (
    changed_scientific_files,
    workspace_code_diff_sha256,
    write_json_atomic,
)
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
    SelectionVariabilitySummary,
    Term,
    TermCensus,
    common_term_subset,
    contrast,
    paired_difference_ci,
    partial_correlation,
    prior_mu,
    relative_sse_gain,
    safe_corr,
    term_sha256,
)
from epibudget.scored_cache import (
    CacheIdentity,
    cache_metadata_path,
    validate_cache_against_universe,
)
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

    @property
    def selected_identity_sha256(self) -> str:
        digest = hashlib.sha256()
        for identity in sorted(canonical_id(variant) for variant in self.selected):
            digest.update(identity.encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()


@dataclass(frozen=True)
class _WorkspaceState:
    commit: str
    dirty: bool
    code_diff_sha256: str
    changed_scientific_files: tuple[str, ...]


def _workspace_state(repo: Path) -> _WorkspaceState:
    def git(*args: str) -> list[str]:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except (OSError, subprocess.CalledProcessError):
            return []
        return [line for line in result.stdout.splitlines() if line]

    head = git("rev-parse", "HEAD")
    commit = head[0] if len(head) == 1 else ""
    dirty = bool(git("status", "--porcelain"))
    if not (commit and dirty):
        return _WorkspaceState(commit, dirty, "", ())
    return _WorkspaceState(
        commit,
        dirty,
        workspace_code_diff_sha256(repo, commit),
        tuple(changed_scientific_files(repo, commit)),
    )


def _workspace_payload(state: _WorkspaceState) -> dict[str, object]:
    return {
        "execution_commit": state.commit,
        "code_state": "dirty" if state.dirty else "clean",
        "code_diff_sha256": state.code_diff_sha256,
        "changed_scientific_files": list(state.changed_scientific_files),
    }


def _input_hashes(
    data_path: Path, cache_path: Path, sidecar_path: Path, universe: Sequence[Variant]
) -> dict[str, str | None]:
    return {
        "dataset_sha256": _sha256_file(data_path) if data_path.is_file() else None,
        "scored_cache_sha256": _sha256_file(cache_path) if cache_path.is_file() else None,
        "scored_cache_sidecar_sha256": (
            _sha256_file(sidecar_path) if sidecar_path.is_file() else None
        ),
        "candidate_universe_sha256": hashlib.sha256(
            "\n".join(sorted(canonical_id(variant) for variant in universe)).encode("ascii")
        ).hexdigest(),
    }


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
            recovery_wording_permitted=bool(gain is not None and gain > 0.0),
        ),
        post_eps,
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
    # Expected cache identity. Declared by the caller so the check has something independent to
    # compare the sidecar against; defaults are the registered confirmatory scorer configuration.
    parser.add_argument("--model-id", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--scorer-seed", type=int, default=0)
    parser.add_argument("--n-perturbations", type=int, default=16)
    parser.add_argument("--random-seeds", type=int, default=20)
    parser.add_argument("--tie-seeds", type=int, default=REQUIRED_TIE_SEEDS)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("report/remediation"))
    args = parser.parse_args()
    for flag, value in (
        ("--tie-seeds", args.tie_seeds),
        ("--random-seeds", args.random_seeds),
        ("--bootstrap", args.bootstrap),
    ):
        if value < 1:
            parser.error(f"{flag} must be >= 1")

    spec = resolve_dataset(args.dataset)
    data_path = args.data or Path(spec.default_data_path)
    budgets = [int(b) for b in args.budgets.split(",")]
    enumerated = enumerate_candidates(
        spec.sites, spec.wt_at_sites, allowed_aa=args.alphabet, max_order=args.max_order
    )
    repo = Path(__file__).resolve().parent.parent
    sidecar_path = cache_metadata_path(args.scored_cache)
    started_at_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    argv = list(sys.argv)
    workspace_at_start = _workspace_state(repo)
    input_hashes_at_start = _input_hashes(data_path, args.scored_cache, sidecar_path, enumerated)
    landscape = spec.loader(data_path)
    # Reject a cache that is not the one this analysis claims to read. The expected identity is
    # supplied by the caller, never read back out of the sidecar: comparing a sidecar against itself
    # cannot fail, so parsing it alone would accept a cache from the wrong model, WT sequence,
    # alphabet, perturbation count, or candidate universe -- including a same-size swap.
    cache, metadata, expected_identity = validate_cache_against_universe(
        args.scored_cache,
        enumerated,
        candidate_alphabet=args.alphabet,
        max_order=args.max_order,
        model_id=args.model_id,
        scorer_seed=args.scorer_seed,
        n_perturbations=args.n_perturbations,
        wt_sequence=spec.wt_sequence,
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
            return [make(method, budget, s, "random", False) for s in range(args.random_seeds)]
        tied = stratum_crosses_budget(universe, scores[method], canonical_id, budget)
        if not tied:
            return [make(method, budget, None, "none", False)]
        return [make(method, budget, s, "tie", True) for s in range(args.tie_seeds)]

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
    plate_cells = {
        (method, budget): plates_for(method, budget) for method in methods for budget in budgets
    }
    posteriors: dict[tuple[str, int, str, int | None, str, str], FloatArray] = {}

    for method in methods:
        for budget in budgets:
            draws = plate_cells[(method, budget)]
            policies = ("zero_prior",) if method == "singles_zero_prior" else CALIBRATION_POLICIES
            for draw in draws:
                for policy in policies:
                    prior_vector, record = prior_for(policy, draw.revealed)
                    orders: list[OrderRecovery] = []
                    for _order, name in _ORDERS:
                        index, truth = per_order[name]
                        cell, post_eps = _evaluate(
                            index, truth, true_vector, prior_vector, draw.mask, name
                        )
                        orders.append(cell)
                        posteriors[(method, budget, draw.seed_kind, draw.seed, policy, name)] = (
                            post_eps
                        )
                    method_records.append(
                        MethodRecovery(
                            method=method,
                            budget=budget,
                            seed=draw.seed,
                            seed_kind=draw.seed_kind,
                            selected_identity_sha256=draw.selected_identity_sha256,
                            tie_stratum_crosses_budget=draw.tie_stratum_crosses_budget,
                            n_selected=len(draw.selected),
                            n_revealed=len(draw.revealed),
                            calibration=record,
                            orders=orders,
                        )
                    )
            print(f"  {method} B={budget}: {len(draws)} draw(s), kind={draws[0].seed_kind}")

    # ---------------------------------------------------------------- paired contrasts (M-3, H-4)
    def realised_pairs(draws_a: list[_Plate], draws_b: list[_Plate]) -> list[tuple[_Plate, _Plate]]:
        """Pair shared seeds, broadcast deterministic arms, and cross different RNG mechanisms."""
        kind_a, kind_b = draws_a[0].seed_kind, draws_b[0].seed_kind
        if kind_a == "none" or kind_b == "none":
            return [(a, b) for a in draws_a for b in draws_b]
        if kind_a == kind_b:
            by_seed_b = {draw.seed: draw for draw in draws_b}
            return [(draw, by_seed_b[draw.seed]) for draw in draws_a if draw.seed in by_seed_b]
        return [(a, b) for a in draws_a for b in draws_b]

    def paired() -> Iterator[PairedContrastResult]:
        for method_a, method_b in _PAIRED_METHODS:
            for budget in budgets:
                draw_pairs = realised_pairs(
                    plate_cells[(method_a, budget)], plate_cells[(method_b, budget)]
                )
                for policy in _PAIRED_POLICIES:
                    if method_b == "singles_zero_prior" and policy != "zero_prior":
                        continue
                    for _order, name in _ORDERS:
                        index, truth = per_order[name]
                        positions = {term: i for i, term in enumerate(index.terms)}
                        for plate_a, plate_b in draw_pairs:
                            pred_a = posteriors[
                                (method_a, budget, plate_a.seed_kind, plate_a.seed, policy, name)
                            ]
                            pred_b = posteriors[
                                (method_b, budget, plate_b.seed_kind, plate_b.seed, policy, name)
                            ]
                            for subset in TERM_SUBSETS:
                                common = common_term_subset(
                                    index.terms, plate_a.revealed, plate_b.revealed, subset
                                )
                                keep = [positions[term] for term in common]
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

    grouped: dict[tuple[str, str, int, str, str, str], list[PairedContrastResult]] = defaultdict(
        list
    )
    for result in paired_contrasts:
        grouped[
            (
                result.method_a,
                result.method_b,
                result.budget,
                result.order,
                result.calibration_policy,
                result.term_subset,
            )
        ].append(result)
    selection_variability: list[SelectionVariabilitySummary] = []
    for key, results in grouped.items():
        values = [float(result.delta) for result in results if result.delta is not None]
        selection_variability.append(
            SelectionVariabilitySummary(
                method_a=key[0],
                method_b=key[1],
                budget=key[2],
                order=key[3],
                calibration_policy=key[4],
                term_subset=key[5],
                n_pairs=len(results),
                n_defined=len(values),
                delta_mean=float(np.mean(values)) if values else None,
                delta_median=float(np.median(values)) if values else None,
                delta_min=min(values) if values else None,
                delta_max=max(values) if values else None,
                fraction_positive=(
                    float(np.mean([value > 0.0 for value in values])) if values else None
                ),
            )
        )

    workspace_at_end = _workspace_state(repo)
    input_hashes_at_end = _input_hashes(data_path, args.scored_cache, sidecar_path, enumerated)
    workspace_stable = (
        workspace_at_start == workspace_at_end and input_hashes_at_start == input_hashes_at_end
    )
    observed_identity = CacheIdentity.from_metadata(metadata)
    provenance_eligible = (
        workspace_stable
        and not workspace_at_start.dirty
        and bool(workspace_at_start.commit)
        and expected_identity == observed_identity
    )
    provenance = {
        "started_at_utc": started_at_utc,
        "completed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "argv": argv,
        "exact_command": subprocess.list2cmdline(argv),
        "workspace_at_start": _workspace_payload(workspace_at_start),
        "workspace_at_end": _workspace_payload(workspace_at_end),
        "input_hashes_at_start": input_hashes_at_start,
        "input_hashes_at_end": input_hashes_at_end,
        "workspace_stable": workspace_stable,
        "provenance_eligible": provenance_eligible,
        "cache_identity_expected": expected_identity.model_dump(mode="json"),
        "cache_identity_observed": observed_identity.model_dump(mode="json"),
    }

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
        data_sha256=str(input_hashes_at_start["dataset_sha256"]),
        scored_cache=str(args.scored_cache),
        scored_cache_sha256=str(input_hashes_at_start["scored_cache_sha256"]),
        n_candidates=len(universe),
        status="retrospective_corrective_reanalysis",
        decision_eligible=False,
        reason_not_decision_eligible=(
            "estimands selected after the audit; single landscape per report"
        ),
        generated_at_utc=str(provenance["completed_at_utc"]),
        provenance=provenance,
        methods=method_records,
        paired_contrasts=paired_contrasts,
        selection_variability=selection_variability,
        note=REPORT_NOTE,
    )
    target = out_dir / f"corrected_recovery_{args.dataset}.json"
    write_json_atomic(target, report.model_dump(mode="json"))
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
        seed_kind_a=plate_a.seed_kind,
        seed_kind_b=plate_b.seed_kind,
        seed_a=plate_a.seed,
        seed_b=plate_b.seed,
        selected_identity_sha256_a=plate_a.selected_identity_sha256,
        selected_identity_sha256_b=plate_b.selected_identity_sha256,
        budget=budget,
        order=order,
        calibration_policy=policy,
        term_subset=subset,
        n_terms=len(terms),
        term_sha256=term_sha256(terms),
        statistic="spearman",
        delta=delta,
        term_leverage_ci95=ci,
        term_leverage_excludes_zero=bool(ci is not None and (ci[0] > 0.0 or ci[1] < 0.0)),
        reason=reason,
        interpretation=_interpret(subset, delta, ci),
    )


if __name__ == "__main__":
    main()
