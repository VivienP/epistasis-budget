from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Iterator, KeysView, Sequence
from math import exp

import numpy as np
import pytest

from epibudget import gate2 as gate2_module
from epibudget.data import GB1_SITES, GB1_WT_AT_SITES, enumerate_candidates
from epibudget.gate2 import (
    AA20,
    CalibrationEvidence,
    EvaluationRecord,
    Gate2Aggregates,
    Gate2Config,
    Gate2Report,
    InferenceBudgetEvidence,
    SelectionRecord,
    SlopeRecord,
    canonical_id,
    decide_gate2,
    gate2_report,
)
from epibudget.robustness import variant_fold
from epibudget.scored_cache import candidate_sha256
from epibudget.types import ScoredVariant, Variant

_PAIRWISE_ORDER = 2
_THIRD_ORDER = 3
_DEFAULT_FOLDS = 5
_EXPECTED_SELECTIONS = 372
_EXPECTED_EVALUATIONS = 1488
_EXPECTED_SLOPES = 377
_EXPECTED_TAU_CELLS = 12
_DEFAULT_STRUCTURAL_SEEDS = 100
_FROZEN_MODEL_ID = "facebook/esm2_t33_650M_UR50D"
_FROZEN_DATASET_SHA256 = "2f115d4eaf03b6083dcc22f7451b3ddfad41c9d8e519286c4e69b6d06db78f1c"
_FROZEN_WT_SHA256 = "7e859d82171047700fd3e9632f7a47eab4a39baedc8c3316d2fc62d3ce2260bb"


def _score_value(variant: Variant) -> float:
    mutations = sorted(variant)
    additive = sum((position + 1) * (ord(mutant) - 64) / 50.0 for position, _, mutant in mutations)
    interaction = 0.0
    if len(mutations) >= _PAIRWISE_ORDER:
        interaction += 0.11 * sum(
            (left[0] + 1) * (right[0] + 1)
            for index, left in enumerate(mutations)
            for right in mutations[index + 1 :]
        )
    if len(mutations) == _THIRD_ORDER:
        interaction -= 0.07 * sum(mutation[0] + 1 for mutation in mutations)
    return additive + interaction


def _scored_pool(
    *,
    alphabet: str = "ACDE",
    positions: Sequence[int] = (0, 1, 2),
    wild_type: Sequence[str] = ("A", "C", "D"),
    tied: bool = False,
) -> list[ScoredVariant]:
    variants = enumerate_candidates(positions, wild_type, allowed_aa=alphabet, max_order=3)
    return [
        ScoredVariant(
            variant=variant,
            delta_g=0.0 if tied else _score_value(variant),
            var_delta_g=1.0 if tied else 0.2 + 0.03 * len(variant),
        )
        for variant in variants
    ]


def _landscape(
    scored: Sequence[ScoredVariant], *, scale: float = 1.7, residual: bool = True
) -> dict[Variant, float]:
    values: dict[Variant, float] = {frozenset(): 1.0}
    for index, item in enumerate(sorted(scored, key=lambda row: canonical_id(row.variant))):
        extra = 0.0
        if residual:
            extra = 0.025 * ((index % 7) - 3) + 0.04 * len(item.variant) ** 2
        values[item.variant] = exp(scale * item.delta_g + extra)
    return values


def _selection_key(record: SelectionRecord) -> tuple[object, ...]:
    return (
        record.method,
        record.budget,
        record.seed_kind,
        record.seed,
    )


class _GuardedLandscape(dict[Variant, float]):
    def __init__(self, values: dict[Variant, float], ready: Callable[[], bool]) -> None:
        super().__init__(values)
        self._ready = ready

    def _check(self) -> None:
        if not self._ready():
            raise AssertionError("landscape accessed before every selection was fixed")

    def __contains__(self, key: object) -> bool:
        self._check()
        return super().__contains__(key)

    def __getitem__(self, key: Variant) -> float:
        self._check()
        return super().__getitem__(key)

    def __iter__(self) -> Iterator[Variant]:
        self._check()
        return super().__iter__()

    def keys(self) -> KeysView[Variant]:  # type: ignore[override]
        self._check()
        return super().keys()


@pytest.fixture(scope="module")
def full_profile_scored() -> list[ScoredVariant]:
    variants = enumerate_candidates(
        GB1_SITES,
        GB1_WT_AT_SITES,
        allowed_aa=AA20,
        max_order=3,
    )
    return [ScoredVariant(variant=variant, delta_g=0.0, var_delta_g=1.0) for variant in variants]


@pytest.fixture(scope="module")
def full_profile_selections(
    full_profile_scored: list[ScoredVariant],
) -> list[SelectionRecord]:
    records = gate2_module._build_selection_records(
        full_profile_scored,
        budgets=(48, 96, 192),
        random_seeds=20,
        structural_seeds=100,
        max_order=3,
        model_id=_FROZEN_MODEL_ID,
    )
    return [
        record.model_copy(
            update={
                "n_revealed": record.budget,
                "n_live": record.budget,
                "n_nonpositive": 0,
                "n_missing": 0,
            }
        )
        for record in records
    ]


def _term_sha256(scored: Sequence[ScoredVariant], order: int) -> str:
    terms = sorted(tuple(sorted(item.variant)) for item in scored if len(item.variant) == order)
    return gate2_module._hash_json([canonical_id(frozenset(term)) for term in terms])


@pytest.fixture(scope="module")
def full_profile_evaluations(
    full_profile_scored: list[ScoredVariant],
    full_profile_selections: list[SelectionRecord],
) -> list[EvaluationRecord]:
    term_hashes = {
        "pairwise": _term_sha256(full_profile_scored, _PAIRWISE_ORDER),
        "third": _term_sha256(full_profile_scored, _THIRD_ORDER),
    }
    n_truth = {"pairwise": 2166, "third": 27436}
    evaluations: list[EvaluationRecord] = []
    for selection in full_profile_selections:
        method_post = 0.4 if selection.method == "info" else 0.2
        for regime in ("operational_method_specific", "shared_crossfit_5fold"):
            for order in ("pairwise", "third"):
                key = [selection.selection_id, regime, order]
                evaluations.append(
                    EvaluationRecord(
                        selection_id=selection.selection_id,
                        method=selection.method,
                        budget=selection.budget,
                        regime=regime,
                        order=order,
                        n_truth=n_truth[order],
                        n_informed=10,
                        n_pinned=1,
                        n_predicted=9,
                        prior_pearson=0.1,
                        post_pearson=method_post,
                        prior_spearman=0.1,
                        post_spearman=method_post,
                        delta_pearson=method_post - 0.1,
                        delta_spearman=method_post - 0.1,
                        sse_prior=2.0,
                        sse_post=1.0,
                        sse_gain=0.5,
                        update_residual_pearson=0.3,
                        update_residual_spearman=0.3,
                        term_sha256=term_hashes[order],
                        truth_sha256=gate2_module._hash_json(["truth", order]),
                        prior_sha256=gate2_module._hash_json(["prior", *key]),
                        post_sha256=gate2_module._hash_json(["post", *key]),
                        status="ok",
                    )
                )
    return evaluations


@pytest.fixture(scope="module")
def full_profile_aggregates(
    full_profile_evaluations: list[EvaluationRecord],
) -> Gate2Aggregates:
    inference: list[InferenceBudgetEvidence] = []
    for budget in (48, 96, 192):
        linked = next(
            record
            for record in full_profile_evaluations
            if record.method == "info"
            and record.budget == budget
            and record.regime == "shared_crossfit_5fold"
            and record.order == "pairwise"
        )
        inference.append(
            InferenceBudgetEvidence(
                budget=budget,
                n_terms=linked.n_truth,
                bootstrap_iterations=2000,
                delta_spearman=linked.delta_spearman,
                delta_spearman_ci95=(0.2, 0.4),
                delta_pearson=linked.delta_pearson,
                delta_pearson_ci95=(0.2, 0.4),
                sse_gain=linked.sse_gain,
                sse_gain_ci95=(0.4, 0.6),
                status="positive",
            )
        )
    tau2, tau2_status = gate2_module._tau2_evidence((48, 96, 192), full_profile_evaluations)
    calibration = gate2_module._calibration_evidence((48, 96, 192), full_profile_evaluations)
    return Gate2Aggregates(
        inference=inference,
        inference_status="positive",
        tau2=tau2,
        tau2_status=tau2_status,
        calibration=calibration,
        calibration_confounded=gate2_module._calibration_is_confounded(calibration),
    )


@pytest.fixture(scope="module")
def full_profile_slopes(
    full_profile_selections: list[SelectionRecord],
) -> list[SlopeRecord]:
    records = [
        SlopeRecord(
            regime="operational_method_specific",
            selection_id=selection.selection_id,
            method=selection.method,
            budget=selection.budget,
            fold=None,
            slope=1.0,
            n_fit=selection.budget,
            fallback=False,
            fit_sha256=gate2_module._hash_json(["fit", selection.selection_id]),
        )
        for selection in full_profile_selections
    ]
    records.extend(
        SlopeRecord(
            regime="shared_crossfit_5fold",
            selection_id=None,
            method=None,
            budget=None,
            fold=fold,
            slope=1.0,
            n_fit=100,
            fallback=False,
            fit_sha256=gate2_module._hash_json(["shared", fold]),
            caveat=gate2_module._CROSSFIT_CAVEAT,
        )
        for fold in range(_DEFAULT_FOLDS)
    )
    return records


def _full_profile_config() -> Gate2Config:
    return Gate2Config(
        budgets=[48, 96, 192],
        random_seeds=20,
        structural_seeds=100,
        n_folds=5,
        max_order=3,
        alphabet=AA20,
        dataset="gb1_wu2016",
        model_id=_FROZEN_MODEL_ID,
        bootstrap_iterations=2000,
    )


def _full_profile_provenance(scored: Sequence[ScoredVariant]) -> dict[str, object]:
    universe_sha256 = candidate_sha256([item.variant for item in scored])
    identity: dict[str, object] = {
        "model_id": _FROZEN_MODEL_ID,
        "scorer_seed": 0,
        "n_perturbations": 16,
        "candidate_sha256": universe_sha256,
        "candidate_count": 29678,
        "candidate_alphabet": AA20,
        "max_order": 3,
        "wt_sha256": _FROZEN_WT_SHA256,
    }
    return {
        "scored_cache_sha256": "a" * 64,
        "scored_cache_sidecar_sha256": "b" * 64,
        "dataset_sha256": _FROZEN_DATASET_SHA256,
        "candidate_universe_sha256": universe_sha256,
        "scored_cache_identity_expected": dict(identity),
        "scored_cache_identity_observed": dict(identity),
        "scored_cache_validator_status": "passed",
        "execution_commit": "c" * 40,
        "code_state": "dirty",
        "code_diff_sha256": "d" * 64,
        "changed_scientific_files": ["src/epibudget/gate2.py"],
        "exact_command": "epibudget gate2 --scored-cache frozen.jsonl",
        "started_at_utc": "2026-07-14T12:00:00+00:00",
        "completed_at_utc": "2026-07-14T12:00:00+00:00",
        "elapsed_seconds": 0.0,
    }


def test_canonical_id_is_compact_stable_json() -> None:
    forward = frozenset(((40, "G", "W"), (38, "V", "A")))
    reverse = frozenset(((38, "V", "A"), (40, "G", "W")))

    identity = canonical_id(forward)

    assert identity == canonical_id(reverse)
    assert identity == '[[38,"V","A"],[40,"G","W"]]'
    assert json.loads(identity) == [[38, "V", "A"], [40, "G", "W"]]


def test_corrected_selections_are_input_order_invariant_and_legacy_is_not() -> None:
    scored = _scored_pool(tied=True)
    forward = gate2_module._build_selection_records(
        scored,
        budgets=(2, 4),
        random_seeds=2,
        structural_seeds=2,
        max_order=3,
        model_id="toy",
    )
    backward = gate2_module._build_selection_records(
        list(reversed(scored)),
        budgets=(2, 4),
        random_seeds=2,
        structural_seeds=2,
        max_order=3,
        model_id="toy",
    )
    forward_by_key = {_selection_key(record): record for record in forward}
    backward_by_key = {_selection_key(record): record for record in backward}

    corrected = [key for key in forward_by_key if key[0] != "structural_legacy_prefix"]
    assert corrected
    for key in corrected:
        assert forward_by_key[key].selected_ids == backward_by_key[key].selected_ids
        assert forward_by_key[key].selected_sha256 == backward_by_key[key].selected_sha256
        assert forward_by_key[key].selection_id == backward_by_key[key].selection_id

    legacy_key = ("structural_legacy_prefix", 2, "none", None)
    assert forward_by_key[legacy_key].selected_ids != backward_by_key[legacy_key].selected_ids


def test_structural_seeded_is_nested_and_never_crosses_exact_score_strata() -> None:
    scored = _scored_pool(tied=True)
    records = gate2_module._build_selection_records(
        scored,
        budgets=(2, 8, 12),
        random_seeds=0,
        structural_seeds=3,
        max_order=3,
        model_id="toy",
    )
    scores = gate2_module._structural_scores(scored, max_order=3)

    for seed in range(3):
        seeded = sorted(
            (
                record
                for record in records
                if record.method == "structural_seeded" and record.seed == seed
            ),
            key=lambda record: record.budget,
        )
        assert seeded[1].selected_ids[: seeded[0].budget] == seeded[0].selected_ids
        assert seeded[2].selected_ids[: seeded[1].budget] == seeded[1].selected_ids
        for record in seeded:
            chosen = set(record.selected_ids)
            selected_scores = [scores[identity] for identity in chosen]
            omitted_scores = [score for identity, score in scores.items() if identity not in chosen]
            assert min(selected_scores) >= max(omitted_scores)


def test_random_selection_samples_from_canonical_candidate_order() -> None:
    scored = _scored_pool(tied=True)
    canonical = sorted(scored, key=lambda item: canonical_id(item.variant))
    expected_indices = np.random.default_rng(0).choice(len(canonical), size=4, replace=False)
    expected = [canonical[int(index)].variant for index in expected_indices]

    records = gate2_module._build_selection_records(
        list(reversed(scored)),
        budgets=(4,),
        random_seeds=1,
        structural_seeds=0,
        max_order=3,
    )
    random_record = next(record for record in records if record.method == "random")

    assert random_record.selected_ids == [canonical_id(variant) for variant in expected]


def test_selection_builder_has_no_landscape_and_label_permutation_cannot_change_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate2_module, "N_BOOTSTRAP", 8)
    assert "landscape" not in inspect.signature(gate2_module._build_selection_records).parameters
    scored = _scored_pool()
    landscape = _landscape(scored)
    candidate_ids = [
        item.variant for item in sorted(scored, key=lambda row: canonical_id(row.variant))
    ]
    labels = [landscape[variant] for variant in candidate_ids]
    permuted: dict[Variant, float] = {frozenset(): landscape[frozenset()]}
    permuted.update(dict(zip(candidate_ids, reversed(labels), strict=True)))

    first = gate2_report(
        scored,
        landscape,
        (1, 2, 3),
        random_seeds=1,
        structural_seeds=1,
        alphabet="ACDE",
    )
    second = gate2_report(
        scored,
        permuted,
        (1, 2, 3),
        random_seeds=1,
        structural_seeds=1,
        alphabet="ACDE",
    )

    assert [record.selected_sha256 for record in first.selections] == [
        record.selected_sha256 for record in second.selections
    ]
    assert [record.selection_id for record in first.selections] == [
        record.selection_id for record in second.selections
    ]


def test_landscape_is_never_accessed_before_every_selection_is_fixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate2_module, "N_BOOTSTRAP", 5)
    scored = _scored_pool()
    selections_complete = False
    real_builder = gate2_module._build_selection_records

    def guarded_builder(
        scored_arg: Sequence[ScoredVariant],
        budgets_arg: Sequence[int],
        *,
        random_seeds: int = 20,
        structural_seeds: int = 100,
        max_order: int = 3,
        model_id: str = "",
    ) -> list[SelectionRecord]:
        nonlocal selections_complete
        records = real_builder(
            scored_arg,
            budgets_arg,
            random_seeds=random_seeds,
            structural_seeds=structural_seeds,
            max_order=max_order,
            model_id=model_id,
        )
        selections_complete = True
        return records

    monkeypatch.setattr(gate2_module, "_build_selection_records", guarded_builder)
    landscape = _GuardedLandscape(_landscape(scored), lambda: selections_complete)

    report = gate2_report(
        scored,
        landscape,
        (1, 2, 3),
        random_seeds=1,
        structural_seeds=1,
        alphabet="ACDE",
    )

    assert selections_complete
    assert report.selections


def test_center_positive_rejects_nonfinite_and_drops_only_finite_nonpositive() -> None:
    single = frozenset(((0, "A", "C"),))
    with pytest.raises(ValueError, match="non-finite"):
        gate2_module._center_positive({frozenset(): 1.0, single: float("nan")})

    assert gate2_module._center_positive({frozenset(): 1.0, single: 0.0}) == {frozenset(): 0.0}


def test_full_aa20_profile_has_exact_selection_count_and_structural_composition(
    full_profile_selections: list[SelectionRecord],
) -> None:
    records = full_profile_selections

    assert len(records) == _EXPECTED_SELECTIONS
    for seed in range(100):
        seeded = {
            record.budget: record
            for record in records
            if record.method == "structural_seeded" and record.seed == seed
        }
        assert seeded[48].counts_by_order == {1: 48}
        assert seeded[96].counts_by_order == {1: 76, 2: 20}
        assert seeded[192].counts_by_order == {1: 76, 2: 116}
        assert seeded[96].selected_ids[:48] == seeded[48].selected_ids
        assert seeded[192].selected_ids[:96] == seeded[96].selected_ids


def test_tiny_three_budget_default_seed_profile_has_exact_record_counts_and_finite_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate2_module, "N_BOOTSTRAP", 20)
    captured_arrays: dict[tuple[str, str, str], gate2_module._EvaluationArrays] = {}
    real_aggregate = gate2_module._aggregate_gate2

    def capture_aggregate(
        budgets: Sequence[int],
        selections: Sequence[SelectionRecord],
        evaluations: Sequence[EvaluationRecord],
        arrays: dict[tuple[str, str, str], gate2_module._EvaluationArrays],
    ) -> Gate2Aggregates:
        captured_arrays.update(arrays)
        return real_aggregate(budgets, selections, evaluations, arrays)

    monkeypatch.setattr(gate2_module, "_aggregate_gate2", capture_aggregate)
    scored = _scored_pool()

    report = gate2_report(
        scored,
        _landscape(scored),
        (1, 2, 3),
        alphabet="ACDE",
        provenance={"code_state": "dirty", "source": "synthetic"},
    )

    assert len(report.selections) == _EXPECTED_SELECTIONS
    assert len(report.evaluations) == _EXPECTED_EVALUATIONS
    assert len(report.slopes) == _EXPECTED_SLOPES
    assert (
        sum(record.regime == "operational_method_specific" for record in report.slopes)
        == _EXPECTED_SELECTIONS
    )
    assert (
        sum(record.regime == "shared_crossfit_5fold" for record in report.slopes) == _DEFAULT_FOLDS
    )
    assert {record.order for record in report.evaluations} == {"pairwise", "third"}
    expected_arrays = {
        (record.selection_id, "shared_crossfit_5fold", "pairwise")
        for record in report.selections
        if record.method == "info"
    }
    assert len(captured_arrays) == len(expected_arrays)
    assert set(captured_arrays) == expected_arrays
    assert report.status == "provisional"
    assert report.public_claim_eligible is False
    payload = report.model_dump_json()
    assert "NaN" not in payload
    assert "Infinity" not in payload


def test_missing_provenance_is_provisional_and_architecture_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate2_module, "N_BOOTSTRAP", 5)
    scored = _scored_pool()

    report = gate2_report(
        scored,
        _landscape(scored),
        (1,),
        random_seeds=0,
        structural_seeds=0,
        alphabet="ACDE",
        provenance=None,
    )

    assert report.status == "provisional"
    assert report.architecture_decision_eligible is False
    assert any("provenance" in reason for reason in report.architecture_eligibility_reasons)


def test_operational_prior_is_exactly_the_calibrated_esm_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate2_module, "N_BOOTSTRAP", 5)
    scored = _scored_pool()

    report = gate2_report(
        scored,
        _landscape(scored, scale=2.0, residual=False),
        (3,),
        random_seeds=0,
        structural_seeds=0,
        alphabet="ACDE",
    )

    slope = next(
        record
        for record in report.slopes
        if record.regime == "operational_method_specific"
        and record.method == "info"
        and record.budget == _THIRD_ORDER
    )
    pairwise = next(
        record
        for record in report.evaluations
        if record.regime == "operational_method_specific"
        and record.method == "info"
        and record.budget == _THIRD_ORDER
        and record.order == "pairwise"
    )
    third = next(
        record
        for record in report.evaluations
        if record.regime == "operational_method_specific"
        and record.method == "info"
        and record.budget == _THIRD_ORDER
        and record.order == "third"
    )

    assert slope.slope == pytest.approx(2.0)
    assert pairwise.sse_prior == pytest.approx(0.0, abs=1e-24)
    assert third.sse_prior == pytest.approx(0.0, abs=1e-24)


def test_evaluation_sse_gain_is_relative_and_zero_denominator_is_insufficient() -> None:
    scored = _scored_pool()
    selection = gate2_module._build_selection_records(
        scored,
        budgets=(1,),
        random_seeds=0,
        structural_seeds=0,
        max_order=3,
    )[0]
    mutations = (
        (0, "A", "C"),
        (1, "C", "A"),
        (2, "D", "A"),
        (3, "E", "A"),
    )
    truth_by_term = {
        tuple(sorted((mutations[0], mutation))): truth
        for mutation, truth in zip(mutations[1:], (1.0, 2.0, 4.0), strict=True)
    }
    revealed = {frozenset((mutations[0],)): 0.5}
    prior = np.array([0.0, 0.5, 2.0], dtype=np.float64)
    post = np.array([0.5, 1.0, 3.0], dtype=np.float64)

    record, _ = gate2_module._evaluation_record(
        selection,
        "operational_method_specific",
        "pairwise",
        truth_by_term,
        revealed,
        prior,
        post,
    )

    assert record.sse_gain == pytest.approx(1.0 - record.sse_post / record.sse_prior)
    assert record.status == "ok"

    truth = np.array(list(truth_by_term.values()), dtype=np.float64)
    exact, _ = gate2_module._evaluation_record(
        selection,
        "operational_method_specific",
        "pairwise",
        truth_by_term,
        revealed,
        truth,
        truth,
    )
    assert exact.sse_prior == 0.0
    assert exact.sse_gain is None
    assert exact.status == "insufficient_data"
    assert "sse_gain_undefined_zero_prior_sse" in exact.warnings

    flat_update, _ = gate2_module._evaluation_record(
        selection,
        "operational_method_specific",
        "pairwise",
        truth_by_term,
        revealed,
        prior,
        prior + 0.5,
    )
    assert flat_update.sse_gain is not None
    assert flat_update.update_residual_pearson is None
    assert flat_update.update_residual_spearman is None
    assert flat_update.status == "ok"
    assert flat_update.warnings == [
        "update_residual_pearson_undefined",
        "update_residual_spearman_undefined",
    ]


def test_inference_bootstrap_uses_relative_sse_gain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate2_module, "N_BOOTSTRAP", 20)
    scored = _scored_pool()
    selection = gate2_module._build_selection_records(
        scored,
        budgets=(1,),
        random_seeds=0,
        structural_seeds=0,
        max_order=3,
    )[0].model_copy(update={"method": "info", "budget": 48})
    truth = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
    residual = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    arrays = gate2_module._EvaluationArrays(
        truth=truth,
        prior=truth + residual,
        post=truth + 0.5 * residual,
    )

    (evidence,) = gate2_module._inference_evidence(
        (48,),
        (selection,),
        {(selection.selection_id, "shared_crossfit_5fold", "pairwise"): arrays},
    )

    assert evidence.sse_gain == pytest.approx(0.75)
    assert evidence.sse_gain_ci95 == pytest.approx((0.75, 0.75))


def test_fully_revealed_loops_equal_truth_and_hashes_are_input_order_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate2_module, "N_BOOTSTRAP", 5)
    scored = _scored_pool()
    landscape = _landscape(scored)

    report = gate2_report(
        scored,
        landscape,
        (len(scored),),
        random_seeds=0,
        structural_seeds=0,
        alphabet="ACDE",
    )
    reversed_report = gate2_report(
        list(reversed(scored)),
        landscape,
        (len(scored),),
        random_seeds=0,
        structural_seeds=0,
        alphabet="ACDE",
    )

    for order in ("pairwise", "third"):
        evaluation = next(
            record
            for record in report.evaluations
            if record.method == "info"
            and record.regime == "operational_method_specific"
            and record.order == order
        )
        other = next(
            record
            for record in reversed_report.evaluations
            if record.method == "info"
            and record.regime == "operational_method_specific"
            and record.order == order
        )
        assert evaluation.n_pinned == evaluation.n_truth
        assert evaluation.sse_post == pytest.approx(0.0, abs=1e-24)
        assert evaluation.term_sha256 == other.term_sha256
        assert evaluation.truth_sha256 == other.truth_sha256
        assert evaluation.prior_sha256 == other.prior_sha256
        assert evaluation.post_sha256 == other.post_sha256


@pytest.mark.parametrize(
    ("inference", "tau2", "confounded", "expected"),
    [
        ("positive", "positive", False, "repair_current_core"),
        ("negative", "positive", False, "replace_phase2_current_model"),
        ("positive", "negative", False, "replace_phase2_current_model"),
        ("inconclusive", "positive", False, "inconclusive_zero_gpu"),
        ("positive", "positive", True, "inconclusive_zero_gpu"),
    ],
)
def test_decide_gate2_covers_every_outcome(
    inference: str,
    tau2: str,
    confounded: bool,
    expected: str,
) -> None:
    aggregates = Gate2Aggregates(
        inference_status=inference,
        tau2_status=tau2,
        calibration_confounded=confounded,
    )

    assert decide_gate2(aggregates).decision == expected
    ineligible = decide_gate2(aggregates, eligible=False)
    assert ineligible.decision == "inconclusive_zero_gpu"
    assert ineligible.architecture_decision_eligible is False


def test_calibration_confounding_requires_same_contrast_and_statistic_at_two_budgets() -> None:
    reversals = [
        CalibrationEvidence(
            budget=budget,
            contrast="info_fitness",
            statistic="pearson",
            operational_difference=0.2,
            shared_difference=-0.1,
            strict_sign_reversal=True,
        )
        for budget in (48, 96)
    ]
    unrelated = CalibrationEvidence(
        budget=192,
        contrast="info_random_median",
        statistic="spearman",
        operational_difference=0.1,
        shared_difference=-0.1,
        strict_sign_reversal=True,
    )

    assert gate2_module._calibration_is_confounded(reversals)
    assert not gate2_module._calibration_is_confounded([reversals[0], unrelated])


def test_tau2_records_all_four_cells_and_all_structural_seeds(
    full_profile_evaluations: list[EvaluationRecord],
) -> None:
    evidence, _ = gate2_module._tau2_evidence((48, 96, 192), full_profile_evaluations)

    assert len(evidence) == _EXPECTED_TAU_CELLS
    assert {(record.budget, record.regime, record.statistic) for record in evidence} == {
        (budget, regime, statistic)
        for budget in (48, 96, 192)
        for regime in ("operational_method_specific", "shared_crossfit_5fold")
        for statistic in ("pearson", "spearman")
    }
    for record in evidence:
        assert record.n_structural == _DEFAULT_STRUCTURAL_SEEDS
        assert len(record.empirical_differences) == _DEFAULT_STRUCTURAL_SEEDS
        assert record.q025 == pytest.approx(0.2)
        assert record.q50 == pytest.approx(0.2)
        assert record.q975 == pytest.approx(0.2)


def test_shared_slope_excludes_its_own_fold_and_hashes_fit_rows_stably() -> None:
    scored = _scored_pool()
    centered = {
        item.variant: 1.5 * item.delta_g + 0.01 * index for index, item in enumerate(scored)
    }
    slopes, records = gate2_module._shared_slopes(scored, centered, _DEFAULT_FOLDS)
    changed = dict(centered)
    changed_variant = next(
        item.variant
        for item in scored
        if variant_fold(item.variant, _DEFAULT_FOLDS) == 0 and item.delta_g != 0.0
    )
    changed[changed_variant] += 10.0
    changed_slopes, changed_records = gate2_module._shared_slopes(
        list(reversed(scored)), changed, _DEFAULT_FOLDS
    )

    assert changed_slopes[0] == slopes[0]
    assert changed_records[0].fit_sha256 == records[0].fit_sha256
    assert any(changed_slopes[fold] != slopes[fold] for fold in range(1, _DEFAULT_FOLDS))


def test_architecture_eligibility_accepts_only_coherent_frozen_profile(
    full_profile_scored: list[ScoredVariant],
    full_profile_selections: list[SelectionRecord],
    full_profile_evaluations: list[EvaluationRecord],
    full_profile_slopes: list[SlopeRecord],
    full_profile_aggregates: Gate2Aggregates,
) -> None:
    provenance = _full_profile_provenance(full_profile_scored)

    eligible, reasons = gate2_module._architecture_eligibility(
        _full_profile_config(),
        full_profile_selections,
        full_profile_evaluations,
        full_profile_slopes,
        full_profile_scored,
        provenance,
        aggregates=full_profile_aggregates,
    )

    assert eligible, reasons


def test_architecture_eligibility_allows_undefined_optional_residual_diagnostics(
    full_profile_scored: list[ScoredVariant],
    full_profile_selections: list[SelectionRecord],
    full_profile_evaluations: list[EvaluationRecord],
    full_profile_slopes: list[SlopeRecord],
    full_profile_aggregates: Gate2Aggregates,
) -> None:
    index = next(
        index for index, record in enumerate(full_profile_evaluations) if record.order == "pairwise"
    )
    evaluations = list(full_profile_evaluations)
    evaluations[index] = evaluations[index].model_copy(
        update={
            "n_informed": 1,
            "n_pinned": 0,
            "n_predicted": 1,
            "update_residual_pearson": None,
            "update_residual_spearman": None,
            "warnings": [
                "update_residual_pearson_undefined",
                "update_residual_spearman_undefined",
            ],
        }
    )

    eligible, reasons = gate2_module._architecture_eligibility(
        _full_profile_config(),
        full_profile_selections,
        evaluations,
        full_profile_slopes,
        full_profile_scored,
        _full_profile_provenance(full_profile_scored),
        aggregates=full_profile_aggregates,
    )

    assert eligible, reasons


def test_finalize_gate2_report_recomputes_provenance_eligibility_and_decision(
    full_profile_scored: list[ScoredVariant],
    full_profile_selections: list[SelectionRecord],
    full_profile_evaluations: list[EvaluationRecord],
    full_profile_slopes: list[SlopeRecord],
    full_profile_aggregates: Gate2Aggregates,
) -> None:
    base = Gate2Report(
        status="provisional",
        config=_full_profile_config(),
        selections=full_profile_selections,
        slopes=full_profile_slopes,
        evaluations=full_profile_evaluations,
        aggregates=full_profile_aggregates,
        architecture_decision_eligible=False,
        architecture_eligibility_reasons=["not finalized"],
        decision=decide_gate2(full_profile_aggregates, eligible=False),
        provenance=None,
    )
    final_provenance = _full_profile_provenance(full_profile_scored)
    placeholder = dict(final_provenance)
    placeholder.pop("scored_cache_validator_status")
    placeholder["code_state"] = "clean"

    initial = gate2_module.finalize_gate2_report(base, full_profile_scored, placeholder)
    finalized = gate2_module.finalize_gate2_report(initial, full_profile_scored, final_provenance)

    assert initial.status == "provisional"
    assert initial.architecture_decision_eligible is False
    assert initial.decision.decision == "inconclusive_zero_gpu"
    assert finalized.status == "provisional"
    assert finalized.architecture_decision_eligible is True
    assert finalized.decision.decision == "repair_current_core"
    assert finalized.selections == base.selections
    assert finalized.slopes == base.slopes
    assert finalized.evaluations == base.evaluations
    assert finalized.aggregates == base.aggregates


def test_report_status_requires_complete_clean_provenance(
    full_profile_scored: list[ScoredVariant],
) -> None:
    candidate_hash = candidate_sha256([item.variant for item in full_profile_scored])
    valid = _full_profile_provenance(full_profile_scored)
    valid["code_state"] = "clean"
    valid["code_diff_sha256"] = ""

    incomplete = dict(valid)
    incomplete.pop("scored_cache_validator_status")
    malformed = dict(valid)
    malformed["execution_commit"] = "not-a-commit"
    unknown = dict(valid)
    unknown["code_state"] = "unknown"
    dirty = dict(valid)
    dirty["code_state"] = "dirty"
    dirty["code_diff_sha256"] = "d" * 64
    naive_timestamps = dict(valid)
    naive_timestamps["started_at_utc"] = "2026-07-14T12:00:00"
    naive_timestamps["completed_at_utc"] = "2026-07-14T12:01:00"
    non_utc_timestamps = dict(valid)
    non_utc_timestamps["started_at_utc"] = "2026-07-14T12:00:00+02:00"
    non_utc_timestamps["completed_at_utc"] = "2026-07-14T12:01:00+02:00"
    reversed_timestamps = dict(valid)
    reversed_timestamps["started_at_utc"] = "2026-07-14T12:01:00+00:00"
    reversed_timestamps["completed_at_utc"] = "2026-07-14T12:00:00+00:00"

    assert gate2_module._report_status(valid, candidate_hash) == "final"
    for provenance in (
        None,
        incomplete,
        malformed,
        unknown,
        dirty,
        naive_timestamps,
        non_utc_timestamps,
        reversed_timestamps,
    ):
        assert gate2_module._report_status(provenance, candidate_hash) == "provisional"


def test_aggregate_integrity_rejects_empty_and_forged_summaries(
    full_profile_scored: list[ScoredVariant],
    full_profile_selections: list[SelectionRecord],
    full_profile_evaluations: list[EvaluationRecord],
    full_profile_slopes: list[SlopeRecord],
    full_profile_aggregates: Gate2Aggregates,
) -> None:
    empty = Gate2Aggregates(
        inference_status="positive",
        tau2_status="positive",
        calibration_confounded=False,
    )
    inference = list(full_profile_aggregates.inference)
    inference[0] = inference[0].model_copy(update={"delta_pearson": 0.99})
    forged_inference = full_profile_aggregates.model_copy(update={"inference": inference})
    tau2 = list(full_profile_aggregates.tau2)
    tau2[0] = tau2[0].model_copy(update={"q50": 999.0})
    forged_tau2 = full_profile_aggregates.model_copy(update={"tau2": tau2})
    calibration = list(full_profile_aggregates.calibration)
    calibration[0] = calibration[0].model_copy(update={"strict_sign_reversal": True})
    forged_calibration = full_profile_aggregates.model_copy(update={"calibration": calibration})
    forged_summary = full_profile_aggregates.model_copy(update={"inference_status": "negative"})

    for aggregates in (
        empty,
        forged_inference,
        forged_tau2,
        forged_calibration,
        forged_summary,
    ):
        assert gate2_module._aggregate_integrity_reasons(
            _full_profile_config(),
            full_profile_selections,
            full_profile_evaluations,
            aggregates,
        )

    report = Gate2Report(
        status="provisional",
        config=_full_profile_config(),
        selections=full_profile_selections,
        slopes=full_profile_slopes,
        evaluations=full_profile_evaluations,
        aggregates=empty,
        architecture_decision_eligible=False,
        architecture_eligibility_reasons=["not finalized"],
        decision=decide_gate2(empty, eligible=False),
        provenance=None,
    )
    finalized = gate2_module.finalize_gate2_report(
        report,
        full_profile_scored,
        _full_profile_provenance(full_profile_scored),
    )

    assert finalized.architecture_decision_eligible is False
    assert finalized.decision.decision == "inconclusive_zero_gpu"


def test_exact_profile_and_report_root_identity_fail_closed(
    full_profile_scored: list[ScoredVariant],
    full_profile_selections: list[SelectionRecord],
    full_profile_evaluations: list[EvaluationRecord],
    full_profile_slopes: list[SlopeRecord],
    full_profile_aggregates: Gate2Aggregates,
) -> None:
    provenance = _full_profile_provenance(full_profile_scored)
    for config in (
        _full_profile_config().model_copy(update={"protocol_version": "altered"}),
        _full_profile_config().model_copy(update={"run_type": "altered"}),
    ):
        assert not gate2_module._architecture_eligibility(
            config,
            full_profile_selections,
            full_profile_evaluations,
            full_profile_slopes,
            full_profile_scored,
            provenance,
            aggregates=full_profile_aggregates,
        )[0]

    base = Gate2Report(
        status="provisional",
        config=_full_profile_config(),
        selections=full_profile_selections,
        slopes=full_profile_slopes,
        evaluations=full_profile_evaluations,
        aggregates=full_profile_aggregates,
        architecture_decision_eligible=False,
        architecture_eligibility_reasons=["not finalized"],
        decision=decide_gate2(full_profile_aggregates, eligible=False),
        provenance=None,
    )
    for report in (
        base.model_copy(update={"protocol_version": "altered"}),
        base.model_copy(update={"run_type": "altered"}),
        base.model_copy(update={"public_claim_eligible": True}),
    ):
        finalized = gate2_module.finalize_gate2_report(
            report,
            full_profile_scored,
            provenance,
        )
        assert finalized.architecture_decision_eligible is False
        assert finalized.architecture_eligibility_reasons
        assert finalized.decision.decision == "inconclusive_zero_gpu"
        assert finalized.protocol_version == gate2_module.PROTOCOL_VERSION
        assert finalized.run_type == gate2_module.RUN_TYPE
        assert finalized.public_claim_eligible is False


def test_architecture_eligibility_rejects_raw_record_tampering(
    full_profile_scored: list[ScoredVariant],
    full_profile_selections: list[SelectionRecord],
    full_profile_evaluations: list[EvaluationRecord],
    full_profile_slopes: list[SlopeRecord],
    full_profile_aggregates: Gate2Aggregates,
) -> None:
    config = _full_profile_config()
    provenance = _full_profile_provenance(full_profile_scored)
    first_selection = full_profile_selections[0]
    tampered_hash = [
        first_selection.model_copy(update={"selected_sha256": "0" * 64}),
        *full_profile_selections[1:],
    ]
    underfilled = [
        first_selection.model_copy(update={"selected_ids": first_selection.selected_ids[:-1]}),
        *full_profile_selections[1:],
    ]
    inconsistent_dynamic_counts = [
        first_selection.model_copy(update={"n_missing": 1}),
        *full_profile_selections[1:],
    ]
    first_evaluation = full_profile_evaluations[0]
    wrong_link = [
        first_evaluation.model_copy(update={"method": "fitness", "budget": 96}),
        *full_profile_evaluations[1:],
    ]
    wrong_slope_link = [
        full_profile_slopes[0].model_copy(update={"method": "fitness", "budget": 96}),
        *full_profile_slopes[1:],
    ]
    wrong_operational_fit = [
        full_profile_slopes[0].model_copy(update={"n_fit": first_selection.n_live - 1}),
        *full_profile_slopes[1:],
    ]
    wrong_shared_caveat = [
        *full_profile_slopes[:-1],
        full_profile_slopes[-1].model_copy(update={"caveat": "altered"}),
    ]
    empty_shared_without_fallback = [
        *full_profile_slopes[:-1],
        full_profile_slopes[-1].model_copy(update={"n_fit": 0, "fallback": False}),
    ]

    assert not gate2_module._architecture_eligibility(
        config,
        tampered_hash,
        full_profile_evaluations,
        full_profile_slopes,
        full_profile_scored,
        provenance,
        aggregates=full_profile_aggregates,
    )[0]
    assert not gate2_module._architecture_eligibility(
        config,
        underfilled,
        full_profile_evaluations,
        full_profile_slopes,
        full_profile_scored,
        provenance,
        aggregates=full_profile_aggregates,
    )[0]
    assert not gate2_module._architecture_eligibility(
        config,
        inconsistent_dynamic_counts,
        full_profile_evaluations,
        full_profile_slopes,
        full_profile_scored,
        provenance,
        aggregates=full_profile_aggregates,
    )[0]
    assert not gate2_module._architecture_eligibility(
        config,
        full_profile_selections,
        wrong_link,
        full_profile_slopes,
        full_profile_scored,
        provenance,
        aggregates=full_profile_aggregates,
    )[0]
    assert not gate2_module._architecture_eligibility(
        config,
        full_profile_selections,
        full_profile_evaluations,
        wrong_slope_link,
        full_profile_scored,
        provenance,
        aggregates=full_profile_aggregates,
    )[0]
    for tampered_slopes in (
        wrong_operational_fit,
        wrong_shared_caveat,
        empty_shared_without_fallback,
    ):
        assert not gate2_module._architecture_eligibility(
            config,
            full_profile_selections,
            full_profile_evaluations,
            tampered_slopes,
            full_profile_scored,
            provenance,
            aggregates=full_profile_aggregates,
        )[0]


def test_architecture_eligibility_rejects_incomplete_evidence_and_identity(
    full_profile_scored: list[ScoredVariant],
    full_profile_selections: list[SelectionRecord],
    full_profile_evaluations: list[EvaluationRecord],
    full_profile_slopes: list[SlopeRecord],
    full_profile_aggregates: Gate2Aggregates,
) -> None:
    config = _full_profile_config()
    provenance = _full_profile_provenance(full_profile_scored)
    required_index = next(
        index for index, record in enumerate(full_profile_evaluations) if record.order == "pairwise"
    )
    missing_required_metric = list(full_profile_evaluations)
    missing_required_metric[required_index] = missing_required_metric[required_index].model_copy(
        update={"post_pearson": None, "delta_pearson": None, "status": "insufficient_data"}
    )
    wrong_universe = list(full_profile_scored)
    wrong_universe[0] = ScoredVariant(
        variant=frozenset(((999, "X", "Y"),)), delta_g=0.0, var_delta_g=1.0
    )
    wrong_identity = dict(provenance)
    wrong_identity["candidate_universe_sha256"] = "f" * 64

    assert not gate2_module._architecture_eligibility(
        config,
        full_profile_selections,
        full_profile_evaluations,
        full_profile_slopes,
        full_profile_scored,
        None,
        aggregates=full_profile_aggregates,
    )[0]
    assert not gate2_module._architecture_eligibility(
        config.model_copy(update={"bootstrap_iterations": 20}),
        full_profile_selections,
        full_profile_evaluations,
        full_profile_slopes,
        full_profile_scored,
        provenance,
        aggregates=full_profile_aggregates,
    )[0]
    assert not gate2_module._architecture_eligibility(
        config,
        full_profile_selections,
        missing_required_metric,
        full_profile_slopes,
        full_profile_scored,
        provenance,
        aggregates=full_profile_aggregates,
    )[0]
    assert not gate2_module._architecture_eligibility(
        config,
        full_profile_selections,
        full_profile_evaluations,
        full_profile_slopes,
        wrong_universe,
        provenance,
        aggregates=full_profile_aggregates,
    )[0]
    assert not gate2_module._architecture_eligibility(
        config,
        full_profile_selections,
        full_profile_evaluations,
        full_profile_slopes,
        full_profile_scored,
        wrong_identity,
        aggregates=full_profile_aggregates,
    )[0]
