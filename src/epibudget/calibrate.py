"""Uncertainty-prior calibration math (pure): does var_delta_g (σ²) track |prediction error|?

Separated from the scoring so it is offline-testable without an ESM-2 forward pass. Given a set of
per-variant ESM scores (ΔĜ), their masking-perturbation dispersions (σ²), and the measured ΔG, it
puts ΔĜ on the measured scale with the through-origin slope (reusing the validation harness's
calibration), forms the absolute prediction error, and correlates σ² against it with a bootstrap
95% CI. The interval is reported without converting a near-zero or weakly negative result into a
stronger calibration claim.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel
from scipy.stats import pearsonr, spearmanr

from epibudget.validate import _calibrate_slope

_N_BOOTSTRAP = 1000
_MIN_POINTS = 3


class CalibrationResult(BaseModel):
    """σ²-vs-|error| calibration for one model size: correlations with bootstrap CIs + raw pairs."""

    n: int
    calibration_slope_b: float
    spearman: float
    pearson: float
    spearman_ci95: tuple[float, float] | None
    pearson_ci95: tuple[float, float] | None
    sigma2: list[float]
    abs_error: list[float]


def bootstrap_ci(
    x: np.ndarray, y: np.ndarray, statistic: str, seed: int, n_boot: int = _N_BOOTSTRAP
) -> tuple[float, float] | None:
    """Percentile 95% CI of Spearman/Pearson(x, y) by resampling the pairs with replacement."""
    n = len(x)
    if n < _MIN_POINTS:
        return None
    rng = np.random.default_rng(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if float(np.std(xb)) == 0.0 or float(np.std(yb)) == 0.0:
            continue
        s = spearmanr(xb, yb).statistic if statistic == "spearman" else pearsonr(xb, yb).statistic
        if np.isfinite(s):
            stats.append(float(s))
    if len(stats) < _MIN_POINTS:
        return None
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def calibrate(
    esm_dg: Sequence[float],
    sigma2: Sequence[float],
    measured_dg: Sequence[float],
    seed: int = 0,
) -> CalibrationResult:
    """Correlate σ² against the calibrated absolute ESM error ``|b·ΔĜ − ΔG_measured|`` (+ CI).

    ``b`` is the through-origin ΔG-scale slope (``validate._calibrate_slope``). Deterministic given
    ``seed`` (the CIs bootstrap with ``seed+1`` / ``seed+2``).
    """
    x = np.asarray(esm_dg, dtype=np.float64)
    sig = np.asarray(sigma2, dtype=np.float64)
    measured = np.asarray(measured_dg, dtype=np.float64)
    slope = _calibrate_slope(x.tolist(), measured.tolist())
    abs_error = np.abs(slope * x - measured)

    # Correlation is undefined if σ² or the error is constant (e.g. n_perturbations=0 → all σ²=0):
    # report NaN explicitly rather than emit a scipy warning and a silent value.
    if float(np.std(sig)) == 0.0 or float(np.std(abs_error)) == 0.0:
        spearman = pearson = float("nan")
    else:
        spearman = float(spearmanr(sig, abs_error).statistic)
        pearson = float(pearsonr(sig, abs_error).statistic)
    return CalibrationResult(
        n=len(x),
        calibration_slope_b=slope,
        spearman=spearman,
        pearson=pearson,
        spearman_ci95=bootstrap_ci(sig, abs_error, "spearman", seed + 1),
        pearson_ci95=bootstrap_ci(sig, abs_error, "pearson", seed + 2),
        sigma2=[float(s) for s in sig],
        abs_error=[float(e) for e in abs_error],
    )
