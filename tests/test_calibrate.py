"""Offline tests for the uncertainty-prior calibration math (no ESM-2)."""

from __future__ import annotations

import numpy as np
import pytest

from epibudget.calibrate import bootstrap_ci, calibrate
from epibudget.validate import _calibrate_slope


def _abs_error(esm_dg: list[float], measured_dg: list[float]) -> list[float]:
    """The absolute calibrated error the calibration correlates σ² against (same slope)."""
    b = _calibrate_slope(esm_dg, measured_dg)
    return [abs(b * x - m) for x, m in zip(esm_dg, measured_dg, strict=True)]


_ESM = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
_MEASURED = [
    0.6,
    1.9,
    3.2,
    3.9,
    5.3,
    5.8,
]  # ~2x ESM with residuals, so |error| varies across points


def test_sigma2_monotonic_in_error_gives_spearman_one() -> None:
    err = _abs_error(_ESM, _MEASURED)
    ranks = {i: r for r, i in enumerate(sorted(range(len(err)), key=lambda i: err[i]))}
    sigma2 = [float(ranks[i]) for i in range(len(err))]  # strictly increasing in |error|
    result = calibrate(_ESM, sigma2, _MEASURED, seed=0)
    assert result.spearman == pytest.approx(1.0)
    assert result.n == len(_ESM)


def test_sigma2_reversed_in_error_gives_spearman_minus_one() -> None:
    err = _abs_error(_ESM, _MEASURED)
    ranks = {i: r for r, i in enumerate(sorted(range(len(err)), key=lambda i: err[i]))}
    sigma2 = [float(len(err) - 1 - ranks[i]) for i in range(len(err))]  # decreasing in |error|
    result = calibrate(_ESM, sigma2, _MEASURED, seed=0)
    assert result.spearman == pytest.approx(-1.0)


def test_calibration_is_deterministic() -> None:
    sigma2 = [0.1, 0.5, 0.2, 0.9, 0.3, 0.7]
    a = calibrate(_ESM, sigma2, _MEASURED, seed=0)
    b = calibrate(_ESM, sigma2, _MEASURED, seed=0)
    assert a.spearman == b.spearman
    assert a.spearman_ci95 == b.spearman_ci95  # bootstrap CI is seeded, so reproducible


def test_bootstrap_ci_needs_minimum_points() -> None:
    assert bootstrap_ci(np.array([1.0, 2.0]), np.array([1.0, 2.0]), "spearman", seed=0) is None


def test_bootstrap_ci_is_seed_reproducible() -> None:
    x = np.array([0.1, 0.5, 0.2, 0.9, 0.3, 0.7, 0.4, 0.8])
    y = np.array([0.2, 0.4, 0.3, 1.0, 0.35, 0.6, 0.5, 0.9])
    assert bootstrap_ci(x, y, "spearman", seed=7) == bootstrap_ci(x, y, "spearman", seed=7)
