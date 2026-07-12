"""Offline tests for the post-registration robustness analyses (no ESM, no network).

A synthetic order-1..3 pool over three sites stands in for scored GB1 candidates. A linear landscape
(fitness = exp ΔĜ) makes the cross-fitted slope exactly 1, so cross-fit must reduce to the frozen
global inference; a non-additive landscape gives the analyses something real to separate. The
determinism test re-runs the report in a subprocess under a different PYTHONHASHSEED so the
sorted-intersection cross-process guarantee is actually exercised.
"""

from __future__ import annotations

import json
import subprocess
import sys
from math import exp
from pathlib import Path

import numpy as np
import pytest

from epibudget.data import enumerate_candidates
from epibudget.robustness import (
    _CROSSFIT_CAVEAT,
    _DIFF_INTERPRETATION,
    PairDifference,
    _corr_one,
    _deterministic_selections,
    _method_state,
    _MethodState,
    _predicted_terms,
    _safe_corr,
    _truth_by_term,
    common_precision,
    crossfit_slopes,
    hierarchical_random_difference,
    infer_epistasis_crossfit,
    paired_difference,
    robustness_report,
    variant_fold,
)
from epibudget.types import ScoredVariant, Variant
from epibudget.validate import _corr

_SITES = (0, 1, 2)
_ALPHABET = "ACG"  # WT 'A' + two mutants per site
_BUDGET = 8
_N_FOLDS = 5
_MIN_COMMON = 3
_PAIRWISE = 2


def _pool() -> list[ScoredVariant]:
    variants = enumerate_candidates(_SITES, ("A", "A", "A"), allowed_aa=_ALPHABET, max_order=3)
    return [
        ScoredVariant(variant=v, delta_g=float(i) - 10.0, var_delta_g=0.05 + 0.01 * i)
        for i, v in enumerate(variants)
    ]


def _true_dg(variant: Variant) -> float:
    per_site = {0: 0.7, 1: -0.4, 2: 0.3}
    sites = {pos for pos, _, _ in variant}
    value = sum(per_site[p] for p in sites)
    if {0, 1} <= sites:
        value += 0.9  # a genuine order-2 interaction so Var[ε] > 0
    return value


def _landscape(pool: list[ScoredVariant]) -> dict[Variant, float]:
    landscape: dict[Variant, float] = {frozenset(): 1.0}
    for sv in pool:
        landscape[sv.variant] = exp(_true_dg(sv.variant))
    return landscape


def _linear_landscape(pool: list[ScoredVariant]) -> dict[Variant, float]:
    """Fitness = exp(ΔĜ): ln fitness is exactly ΔĜ, so every fold slope is 1.0."""
    landscape: dict[Variant, float] = {frozenset(): 1.0}
    for sv in pool:
        landscape[sv.variant] = exp(sv.delta_g)
    return landscape


# --- folds + cross-fit -------------------------------------------------------------------------


def test_variant_fold_is_deterministic_and_in_range() -> None:
    pool = _pool()
    for sv in pool:
        fold = variant_fold(sv.variant, _N_FOLDS)
        assert 0 <= fold < _N_FOLDS
        assert fold == variant_fold(sv.variant, _N_FOLDS)  # stable


def test_crossfit_slopes_are_all_one_on_a_linear_landscape() -> None:
    pool = _pool()
    slopes = crossfit_slopes(pool, _linear_landscape(pool), _N_FOLDS)
    assert set(slopes) == set(range(_N_FOLDS))
    for slope in slopes.values():
        assert slope == pytest.approx(1.0)


def test_crossfit_inference_reduces_to_global_when_slope_is_one() -> None:
    pool = _pool()
    landscape = _linear_landscape(pool)
    slopes = crossfit_slopes(pool, landscape, _N_FOLDS)
    truth = _truth_by_term(pool, landscape, 3)
    selected = _deterministic_selections(pool, _BUDGET, 3)["info"]
    from epibudget.validate import _measured_dg, infer_epistasis  # noqa: PLC0415

    revealed = _measured_dg(landscape, selected)
    crossfit = {
        i.mutations: i.epsilon_hat for i in infer_epistasis_crossfit(revealed, pool, slopes)
    }
    globally = {i.mutations: i.epsilon_hat for i in infer_epistasis(revealed, pool)}
    assert crossfit.keys() == globally.keys()
    for term in truth:
        assert crossfit[term] == pytest.approx(globally[term])


# --- paired difference -------------------------------------------------------------------------


def test_paired_difference_zero_for_identical_predictions() -> None:
    true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    delta, ci = paired_difference(pred, pred, true, "spearman", seed=0)
    assert delta == pytest.approx(0.0)
    assert ci is not None and ci[0] <= 0.0 <= ci[1]


def test_paired_difference_positive_when_a_tracks_truth_better() -> None:
    true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    good = true + 0.01
    bad = true[::-1].copy()  # anti-correlated
    delta, ci = paired_difference(good, bad, true, "spearman", seed=1)
    assert delta is not None and delta > 0.0
    assert ci is not None


def test_paired_difference_is_seed_reproducible() -> None:
    true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    a = np.array([1.0, 2.2, 2.9, 4.1, 4.8, 6.2])
    b = np.array([2.0, 1.0, 4.0, 3.0, 6.0, 5.0])
    assert paired_difference(a, b, true, "pearson", seed=7) == paired_difference(
        a, b, true, "pearson", seed=7
    )


# --- common precision --------------------------------------------------------------------------


def test_common_precision_empty_when_nothing_informed() -> None:
    pool = _pool()
    landscape = _landscape(pool)
    truth = _truth_by_term(pool, landscape, 3)
    truth_terms = sorted(t for t in truth if len(t) == _PAIRWISE)
    empty = _MethodState(frozenset(), {t: 0.0 for t in truth})
    result = common_precision(
        "info",
        "fitness",
        "pairwise",
        truth,
        truth_terms,
        {
            "info": empty,
            "fitness": empty,
        },
        seed=0,
    )
    assert result.n_common == 0
    assert result.spearman_a is None and result.spearman_b is None
    assert result.difference.delta is None


def test_common_precision_uses_exactly_the_informed_not_pinned_intersection() -> None:
    pool = _pool()
    landscape = _landscape(pool)
    truth = _truth_by_term(pool, landscape, 3)
    truth_terms = sorted(t for t in truth if len(t) == _PAIRWISE)
    states = {
        "info": _method_state(
            _deterministic_selections(pool, _BUDGET, 3)["info"], pool, landscape, truth, 3
        ),
        "fitness": _method_state(
            _deterministic_selections(pool, _BUDGET, 3)["fitness"], pool, landscape, truth, 3
        ),
    }
    expected = sorted(
        _predicted_terms(states["info"], truth_terms)
        & _predicted_terms(states["fitness"], truth_terms)
    )
    result = common_precision("info", "fitness", "pairwise", truth, truth_terms, states, seed=3)
    assert result.n_common == len(expected)
    if len(expected) >= _MIN_COMMON:
        pred_a = np.array([states["info"].hat[t] for t in expected])
        true = np.array([truth[t] for t in expected])
        _, spearman = _corr(pred_a, true)
        assert result.spearman_a == pytest.approx(spearman)
        assert result.mean_informed_fraction_a is not None


# --- hierarchical random difference ------------------------------------------------------------


def test_hierarchical_random_difference_is_seed_reproducible() -> None:
    pool = _pool()
    landscape = _landscape(pool)
    truth = _truth_by_term(pool, landscape, 3)
    truth_terms = sorted(t for t in truth if len(t) == _PAIRWISE)
    from epibudget.validate import random_selection  # noqa: PLC0415

    info = _method_state(
        _deterministic_selections(pool, _BUDGET, 3)["info"], pool, landscape, truth, 3
    )
    randoms = [
        _method_state(random_selection(pool, _BUDGET, s), pool, landscape, truth, 3)
        for s in range(4)
    ]
    first = hierarchical_random_difference(
        "pairwise", "spearman", truth, truth_terms, info, randoms, 4, seed=5
    )
    second = hierarchical_random_difference(
        "pairwise", "spearman", truth, truth_terms, info, randoms, 4, seed=5
    )
    assert first == second
    assert isinstance(first, PairDifference)


# --- report integration ------------------------------------------------------------------------


def _reject_constant(token: str) -> float:
    raise AssertionError(f"non-finite JSON token {token!r} written to robustness.json")


def test_report_has_serialized_caveats_and_no_pooled_order(tmp_path: Path) -> None:
    pool = _pool()
    report = robustness_report(pool, _landscape(pool), [_BUDGET], seeds=4, out_dir=tmp_path)
    raw = (tmp_path / "robustness.json").read_text(encoding="utf-8")
    # A near-constant bootstrap resample must never leak a bare NaN/Infinity token into the file:
    # undefined correlations serialize as JSON null, so a NaN-rejecting parse must succeed.
    dumped = json.loads(raw, parse_constant=_reject_constant)

    assert "does not alter the frozen decision rule" in dumped["note"]
    orders = {cp["order"] for cp in dumped["common_precision"]}
    orders |= {pd["order"] for pd in dumped["pair_differences"]}
    assert "pooled" not in orders  # per-order only
    assert all(pd["interpretation"] == _DIFF_INTERPRETATION for pd in dumped["pair_differences"])
    assert all(ss["caveat"] == _CROSSFIT_CAVEAT for ss in dumped["scale_sensitivity"])
    assert isinstance(report, type(report))
    assert report.n_candidates == len(pool)


def test_crossfit_slopes_reject_too_few_folds() -> None:
    pool = _pool()
    with pytest.raises(ValueError, match="n_folds must be >= 2"):
        crossfit_slopes(pool, _landscape(pool), 1)


def test_corr_one_is_bit_identical_to_safe_corr_components() -> None:
    """The throughput optimization must change no number: single-stat == _safe_corr's component."""
    x = np.array([1.0, 2.0, 2.0, 4.0, 5.0, 3.0])
    y = np.array([2.0, 1.0, 3.0, 5.0, 4.0, 3.5])
    pearson, spearman = _safe_corr(x, y)
    assert _corr_one(x, y, "pearson") == pearson  # exact equality, not approx
    assert _corr_one(x, y, "spearman") == spearman
    const = np.array([1.0, 1.0, 1.0, 1.0])
    assert _corr_one(const, y[:4], "spearman") is None  # degeneracy guard matches _corr


class _NanStat:
    statistic = float("nan")


def test_corr_paths_never_leak_a_nan(monkeypatch: pytest.MonkeyPatch) -> None:
    """A near-constant resample makes scipy return NaN; it must become None, never reach the CI."""
    # Both correlation paths must guard NaN: _safe_corr (via _corr) and the bootstrap (_corr_one).
    monkeypatch.setattr("epibudget.robustness._corr", lambda _p, _t: (float("nan"), float("nan")))
    monkeypatch.setattr("epibudget.robustness.spearmanr", lambda _a, _b: _NanStat())
    monkeypatch.setattr("epibudget.robustness.pearsonr", lambda _a, _b: _NanStat())
    pred = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )  # std > 0, so the degeneracy guard passes and scipy is hit
    true = np.array([1.0, 2.0, 3.0, 4.0])
    assert _safe_corr(pred, true) == (None, None)
    assert _corr_one(pred, true, "spearman") is None
    delta, ci = paired_difference(pred, pred, true, "spearman", seed=0)
    assert delta is None
    assert ci is None


_REPRO_SCRIPT = """
from math import exp

from epibudget.data import enumerate_candidates
from epibudget.robustness import robustness_report
from epibudget.types import ScoredVariant

variants = enumerate_candidates((0, 1, 2), ("A", "A", "A"), allowed_aa="ACG", max_order=3)
pool = [
    ScoredVariant(variant=v, delta_g=float(i) - 10.0, var_delta_g=0.05 + 0.01 * i)
    for i, v in enumerate(variants)
]
per = {0: 0.7, 1: -0.4, 2: 0.3}


def dg(variant):
    sites = {p for p, _, _ in variant}
    return sum(per[p] for p in sites) + (0.9 if {0, 1} <= sites else 0.0)


landscape = {frozenset(): 1.0}
for sv in pool:
    landscape[sv.variant] = exp(dg(sv.variant))
print(robustness_report(pool, landscape, [8], seeds=4).model_dump_json())
"""


def test_report_is_reproducible_across_processes(tmp_path: Path) -> None:
    """The sorted(common) guard must survive a different PYTHONHASHSEED (finding [5])."""
    import os  # noqa: PLC0415

    script = tmp_path / "repro.py"
    script.write_text(_REPRO_SCRIPT, encoding="utf-8")

    repo = Path(__file__).resolve().parent.parent

    def run(hashseed: str) -> str:
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = hashseed
        env["PYTHONPATH"] = str(repo / "src")  # src layout: package is not pip-installed
        proc = subprocess.run(
            [sys.executable, str(script)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(repo),
        )
        return proc.stdout.strip()

    assert run("0") == run("1")


def test_deterministic_selections_do_not_depend_on_labels() -> None:
    """No-leakage: selection recomputation reads only scored (signature carries no landscape)."""
    import inspect  # noqa: PLC0415

    params = set(inspect.signature(_deterministic_selections).parameters)
    assert "landscape" not in params
    assert params == {"scored", "budget", "max_order"}
