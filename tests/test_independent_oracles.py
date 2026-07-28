"""Independent mathematical oracles for estimators the suite previously checked only against itself.

Audit finding I-1: a production function must never be its own oracle. Every expected value below
comes from a closed form, a hand computation, or ``numpy``/``scipy`` primitives that share no code
path with the function under test.
"""

from __future__ import annotations

from math import log2
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from epibudget.acquisition import allocate
from epibudget.cli import app
from epibudget.data import enumerate_candidates
from epibudget.downstream import (
    FeatureSpace,
    fit_ridge,
    ndcg_at_k,
    normalized_log2_budget_auc,
)
from epibudget.graph import selection_graph
from epibudget.recovery import relative_sse_gain
from epibudget.types import ScoredVariant

_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_GB1_SITES = (38, 39, 40, 53)
_GB1_WT = ("V", "D", "G", "V")
_N_MAIN = 76
_N_PAIR = 2166
_TINY = 1e-6
_SMALL = 1e-3
_BUDGET = 5


# --------------------------------------------------------------------------- ridge


def _primal_ridge(
    design: np.ndarray, response: np.ndarray, penalties: np.ndarray
) -> tuple[np.ndarray, float]:
    """Closed-form primal solution of min ||y - Xb - c||^2 + sum_k L_k b_k^2, c unpenalised.

    Centering both X and y removes the intercept from the penalised problem, so
    b = (Xc' Xc + diag(L))^-1 Xc' yc and c = mean(y) - mean(X) b. Solved with an explicit p x p
    inverse — a completely different linear algebra path from the production n x n dual solve.
    """
    x_mean = design.mean(axis=0)
    y_mean = float(response.mean())
    xc = design - x_mean
    yc = response - y_mean
    beta = np.linalg.solve(xc.T @ xc + np.diag(penalties), xc.T @ yc)
    return beta, y_mean - float(x_mean @ beta)


def _objective(design, response, penalties, beta, intercept) -> float:
    residual = response - design @ beta - intercept
    return float(residual @ residual + (penalties * beta**2).sum())


@pytest.mark.parametrize("alpha", [1e-6, 1e-3, 1.0, 1e3, 1e8])
@pytest.mark.parametrize(
    ("n", "p", "kind"),
    [
        (20, 5, "tall"),
        (5, 30, "wide"),  # p >> n: the regime every real training plate is in
        (8, 8, "square"),
        (12, 6, "collinear"),
        (12, 5, "duplicate_rows"),
        (10, 6, "constant_target"),
        (7, 4, "zero_design"),
    ],
)
def test_dual_ridge_matches_the_closed_form_primal(n: int, p: int, kind: str, alpha: float) -> None:
    """The production dual solve reproduces the primal closed form on every degenerate shape."""
    rng = np.random.default_rng(abs(hash((n, p, kind))) % 2**32)
    design = rng.normal(size=(n, p))
    response = rng.normal(size=n)
    if kind == "collinear":
        design[:, -2:] = design[:, :2]  # exactly rank-deficient
    elif kind == "duplicate_rows":
        design[n // 2 :] = design[: n - n // 2][: n - n // 2]
        response[n // 2 :] = response[: n - n // 2][: n - n // 2]
    elif kind == "constant_target":
        response = np.full(n, 3.5)
    elif kind == "zero_design":
        design = np.zeros((n, p))

    penalties = np.full(p, alpha)
    model = fit_ridge(design, response, penalties)
    beta_ref, intercept_ref = _primal_ridge(design, response, penalties)

    assert model.coef == pytest.approx(beta_ref, abs=1e-6, rel=1e-6)
    assert model.intercept == pytest.approx(intercept_ref, abs=1e-6, rel=1e-6)
    # The dual solution is never worse on the actual penalised objective.
    obj_code = _objective(design, response, penalties, model.coef, model.intercept)
    obj_ref = _objective(design, response, penalties, beta_ref, intercept_ref)
    assert obj_code <= obj_ref + 1e-6


def test_ridge_penalty_is_per_feature_and_the_intercept_is_unpenalised() -> None:
    """A per-feature penalty vector really shrinks the two blocks differently."""
    rng = np.random.default_rng(0)
    design = rng.normal(size=(30, 4))
    response = rng.normal(size=30) + 5.0  # a large offset the unpenalised intercept must absorb
    penalties = np.array([1e-8, 1e-8, 1e8, 1e8])
    model = fit_ridge(design, response, penalties)
    assert abs(model.coef[2]) < _TINY and abs(model.coef[3]) < _TINY  # crushed
    assert abs(model.coef[0]) > _SMALL  # essentially unpenalised
    # An unpenalised intercept absorbs the offset exactly: residuals stay centred.
    assert float((response - design @ model.coef - model.intercept).mean()) == pytest.approx(0.0)


def test_ridge_on_an_empty_training_set_is_flagged_degenerate() -> None:
    model = fit_ridge(np.zeros((0, 5)), np.zeros(0), np.ones(5))
    assert model.degenerate and model.intercept == 0.0
    assert model.coef.tolist() == [0.0] * 5


def test_feature_space_counts_match_an_independent_combinatorial_derivation() -> None:
    """4 sites x 19 non-WT residues main; C(4,2) x 19^2 pairwise; no duplicate columns."""
    space = FeatureSpace(_GB1_SITES, _GB1_WT, _ALPHABET)
    n_main = int(space.penalty_is_main.sum())
    assert n_main == len(_GB1_SITES) * (len(_ALPHABET) - 1) == _N_MAIN
    assert space.n_features - n_main == 6 * (len(_ALPHABET) - 1) ** 2 == _N_PAIR
    assert len(set(space.pair_index.values())) == len(space.pair_index)
    assert len(set(space.main_index.values())) == len(space.main_index)


# --------------------------------------------------------------------------- NDCG


def test_ndcg_matches_a_hand_computed_dcg_over_idcg() -> None:
    """Hand oracle with the declared linear gain and 1/log2(rank+2) discount."""
    pred = np.array([3.0, 1.0, 2.0, 0.0])
    relevance = np.array([0.0, 1.0, 0.5, 0.25])
    ids = ["a", "b", "c", "d"]
    # Ranked by pred: a(0.0), c(0.5), b(1.0), d(0.25)
    dcg = 0.0 / log2(2) + 0.5 / log2(3) + 1.0 / log2(4) + 0.25 / log2(5)
    # Ideal by relevance: b(1.0), c(0.5), d(0.25), a(0.0)
    idcg = 1.0 / log2(2) + 0.5 / log2(3) + 0.25 / log2(4) + 0.0 / log2(5)
    assert ndcg_at_k(pred, relevance, 4, ids) == pytest.approx(dcg / idcg)


def test_ndcg_truncation_at_k_uses_only_the_top_k_positions() -> None:
    pred = np.array([3.0, 2.0, 1.0])
    relevance = np.array([0.0, 1.0, 0.5])
    ids = ["a", "b", "c"]
    dcg2 = 0.0 / log2(2) + 1.0 / log2(3)
    idcg2 = 1.0 / log2(2) + 0.5 / log2(3)
    assert ndcg_at_k(pred, relevance, 2, ids) == pytest.approx(dcg2 / idcg2)


def test_ndcg_edge_conventions_are_the_declared_ones() -> None:
    ids = ["a", "b", "c"]
    assert ndcg_at_k(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0]), 3, ids) == 1.0
    assert ndcg_at_k(np.array([]), np.array([]), 3, []) == 0.0
    assert ndcg_at_k(np.array([1.0, 2.0]), np.array([1.0, 0.0]), 0, ["a", "b"]) == 0.0


# --------------------------------------------------------------------------- AUC weights


def test_normalized_log2_budget_auc_weights_are_one_quarter_one_half_one_quarter() -> None:
    """The registered grid's implied weights, recovered by probing the linear functional."""
    basis = [
        normalized_log2_budget_auc([1.0, 0.0, 0.0]),
        normalized_log2_budget_auc([0.0, 1.0, 0.0]),
        normalized_log2_budget_auc([0.0, 0.0, 1.0]),
    ]
    assert basis == pytest.approx([0.25, 0.5, 0.25])
    # Equivalent to the normalized trapezoid on the log2(B) axis for the registered grid.
    budgets = [48, 96, 192]
    values = [0.2, 0.5, 0.9]
    x = np.log2(np.asarray(budgets, dtype=float))
    expected = float(np.trapezoid(values, x) / (x[-1] - x[0]))
    assert normalized_log2_budget_auc(values, budgets) == pytest.approx(expected)
    # ...and NOT the raw-budget-axis integral, which the old name left ambiguous.
    raw_axis = float(np.trapezoid(values, budgets) / (budgets[-1] - budgets[0]))
    assert normalized_log2_budget_auc(values, budgets) != pytest.approx(raw_axis)


def test_normalized_log2_budget_auc_rejects_a_non_doubling_grid() -> None:
    """A future grid that is not equally spaced on log2 must not inherit the equal weights."""
    with pytest.raises(ValueError, match="not equally spaced on log2"):
        normalized_log2_budget_auc([0.1, 0.2, 0.3], [48, 96, 200])
    with pytest.raises(ValueError, match="entries but"):
        normalized_log2_budget_auc([0.1, 0.2, 0.3], [48, 96])


# --------------------------------------------------------------------------- SSE gain


def test_relative_sse_gain_on_fixed_hand_arrays() -> None:
    prior = np.array([0.0, 0.0, 0.0])
    post = np.array([1.0, 2.0, 2.0])
    truth = np.array([1.0, 2.0, 3.0])
    # SSE prior = 1 + 4 + 9 = 14; SSE post = 0 + 0 + 1 = 1; gain = 1 - 1/14
    assert relative_sse_gain(prior, post, truth) == pytest.approx(1.0 - 1.0 / 14.0)
    # A perfect estimate gains exactly 1.
    assert relative_sse_gain(prior, truth, truth) == pytest.approx(1.0)
    # An estimate worse than the prior gains a negative amount.
    worse = np.array([10.0, 10.0, 10.0])
    assert relative_sse_gain(prior, worse, truth) is not None
    assert relative_sse_gain(prior, worse, truth) < 0.0
    # A zero-error prior leaves the ratio undefined rather than dividing by zero.
    assert relative_sse_gain(truth, post, truth) is None


def test_sse_gain_is_immune_to_a_shared_additive_term() -> None:
    """The property that makes SSE gain the right wording gate: a common summand cancels."""
    rng = np.random.default_rng(2)
    truth = rng.normal(size=50)
    prior = rng.normal(size=50)
    post = rng.normal(size=50)
    shared = 100.0 * rng.normal(size=50)  # a huge shared skeleton
    base = relative_sse_gain(prior, post, truth)
    shifted = relative_sse_gain(prior + shared, post + shared, truth + shared)
    assert base is not None and shifted is not None
    assert shifted == pytest.approx(base)


# --------------------------------------------------------------------------- degenerate ranking


def test_constant_acquisition_weight_is_refused_rather_than_ordered_by_enumeration() -> None:
    """Audit M-2: a constant weight makes the ranking the input order; it must fail loudly."""
    candidates = enumerate_candidates((0, 1, 2), ("M", "T", "Y"), "ACD", 3)
    # n_perturbations = 0 in the scorer produces exactly this cache: every dispersion is 0.
    scored = [
        ScoredVariant(variant=v, delta_g=float(i), var_delta_g=0.0)
        for i, v in enumerate(candidates)
    ]
    graph = selection_graph(scored, 3, "info")
    with pytest.raises(ValueError, match="seeded tie-break"):
        allocate(graph, scored, _BUDGET, lambda_=0.0, method="info")
    with pytest.raises(ValueError, match="seeded tie-break"):
        allocate(graph, scored, _BUDGET, lambda_=0.5, method="info")
    # lambda = 1 is pure fitness-greedy and never consults the degenerate weight.
    assert len(allocate(graph, scored, _BUDGET, lambda_=1.0, method="info").selected) == _BUDGET

    # A real dispersion cache varies, so the guard does not fire.
    varied = [
        ScoredVariant(variant=v, delta_g=float(i), var_delta_g=0.1 + 0.01 * i)
        for i, v in enumerate(candidates)
    ]
    assert (
        len(allocate(selection_graph(varied, 3, "info"), varied, _BUDGET, lambda_=0.0).selected)
        == _BUDGET
    )


def test_cli_rejects_info_method_with_zero_perturbations(tmp_path: Path) -> None:
    """Audit M-2: the CLI refuses the degenerate combination before loading any model.

    Only the rejected invocation is exercised: the accepted one would download and run ESM-2,
    which the offline test contract forbids. Rejection is asserted by exit code, as in
    ``tests/test_cli.py`` (typer Rich-wraps the BadParameter panel, so its text is not portable).
    """
    fasta = tmp_path / "wt.fasta"
    fasta.write_text(">toy\nADG\n", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "allocate",
            "--fasta",
            str(fasta),
            "--positions",
            "1,2,3",
            "--budget",
            "4",
            "--alphabet",
            "ACG",
            "--method",
            "info",
            "--n-perturbations",
            "0",
            "--out",
            str(tmp_path / "allocation.json"),
        ],
    )
    assert result.exit_code != 0
    # The guard fires during parameter validation, so nothing is written and no model is touched.
    assert not (tmp_path / "allocation.json").exists()
