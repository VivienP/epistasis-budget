"""Fourier-spectrum summaries for complete combinatorial landscapes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from epibudget.epistasis import _landscape_tensor, _wht_forward
from epibudget.types import Variant

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class EffectiveCounts:
    """Smallest coefficient counts carrying fixed fractions of squared energy."""

    k90: int
    k95: int
    k99: int


@dataclass(frozen=True)
class MagnitudeQuantiles:
    """Selected quantiles of absolute Fourier-coefficient magnitude."""

    q50: float
    q90: float
    q99: float
    q999: float
    maximum: float


@dataclass(frozen=True)
class OrderSpectrum:
    """Energy and effective coefficient counts for one Fourier interaction order."""

    order: int
    coefficient_count: int
    variance_contribution: float
    magnitude_quantiles: MagnitudeQuantiles
    effective_counts: EffectiveCounts


@dataclass(frozen=True)
class SpectrumSummary:
    """Non-constant Fourier-spectrum summary of a complete landscape."""

    n_cells: int
    target_variance: float
    nonconstant_variance: float
    parseval_abs_error: float
    by_order: tuple[OrderSpectrum, ...]
    overall_magnitude_quantiles: MagnitudeQuantiles
    overall_effective_counts: EffectiveCounts


def _effective_counts(squared: FloatArray) -> EffectiveCounts:
    total = float(squared.sum())
    if total == 0.0:
        return EffectiveCounts(k90=0, k95=0, k99=0)
    cumulative = np.cumsum(np.sort(squared)[::-1]) / total

    def count(fraction: float) -> int:
        return int(np.searchsorted(cumulative, fraction, side="left")) + 1

    return EffectiveCounts(k90=count(0.90), k95=count(0.95), k99=count(0.99))


def _magnitude_quantiles(coefficients: FloatArray) -> MagnitudeQuantiles:
    magnitudes = np.abs(coefficients)
    return MagnitudeQuantiles(
        q50=float(np.quantile(magnitudes, 0.50)),
        q90=float(np.quantile(magnitudes, 0.90)),
        q99=float(np.quantile(magnitudes, 0.99)),
        q999=float(np.quantile(magnitudes, 0.999)),
        maximum=float(magnitudes.max()),
    )


def summarize_spectrum(values: Mapping[Variant, float], sites: Sequence[int]) -> SpectrumSummary:
    """Summarize non-constant multiallelic Fourier energy by interaction order."""
    if any(not np.isfinite(value) for value in values.values()):
        raise ValueError("spectrum values must all be finite")
    tensor, bases = _landscape_tensor(values, sites)
    coefficients = _wht_forward(tensor, bases)
    orders = np.zeros(tensor.shape, dtype=np.int64)
    for axis, size in enumerate(tensor.shape):
        axis_order = (np.arange(size) != 0).astype(np.int64)
        shape = tuple(size if index == axis else 1 for index in range(tensor.ndim))
        orders = orders + axis_order.reshape(shape)

    n_cells = tensor.size
    target_variance = float(np.var(tensor))
    analysis_coefficients = coefficients.copy()
    if target_variance == 0.0:
        analysis_coefficients[orders > 0] = 0.0
    squared = np.square(analysis_coefficients)
    by_order = tuple(
        OrderSpectrum(
            order=order,
            coefficient_count=int(np.count_nonzero(orders == order)),
            variance_contribution=float(squared[orders == order].sum()) / n_cells,
            magnitude_quantiles=_magnitude_quantiles(analysis_coefficients[orders == order]),
            effective_counts=_effective_counts(squared[orders == order]),
        )
        for order in range(1, tensor.ndim + 1)
    )
    nonconstant_variance = float(squared[orders > 0].sum()) / n_cells
    return SpectrumSummary(
        n_cells=n_cells,
        target_variance=target_variance,
        nonconstant_variance=nonconstant_variance,
        parseval_abs_error=abs(target_variance - nonconstant_variance),
        by_order=by_order,
        overall_magnitude_quantiles=_magnitude_quantiles(analysis_coefficients[orders > 0]),
        overall_effective_counts=_effective_counts(squared[orders > 0]),
    )
