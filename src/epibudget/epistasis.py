"""WT-referenced (biochemical) epistasis terms and the Walsh-Hadamard ground truth.

The pairwise/third-order coefficients below are the wild-type sub-sampling of the multiallelic
Walsh-Hadamard transform (docs/RESEARCH_EPISTASIS.md#3). They are exact, cheap, and fully tested;
they anchor invariant #1 — if the ΔG map is additive, every coefficient here is identically zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from epibudget.types import Interaction, Mutation, ScoredVariant, Variant


def _v(*mutations: Mutation) -> Variant:
    """Build a Variant (frozenset) from point mutations."""
    return frozenset(mutations)


def epsilon_pairwise(dg: Mapping[Variant, float], i: Mutation, j: Mutation) -> float:
    """ε(i,j) = ΔG(ij) − ΔG(i) − ΔG(j). ΔG(∅) = 0 by convention."""
    return dg[_v(i, j)] - dg[_v(i)] - dg[_v(j)]


def epsilon_third(dg: Mapping[Variant, float], i: Mutation, j: Mutation, k: Mutation) -> float:
    """ε(i,j,k) = ΔG(ijk) − ΔG(ij) − ΔG(ik) − ΔG(jk) + ΔG(i) + ΔG(j) + ΔG(k)."""
    return (
        dg[_v(i, j, k)]
        - dg[_v(i, j)]
        - dg[_v(i, k)]
        - dg[_v(j, k)]
        + dg[_v(i)]
        + dg[_v(j)]
        + dg[_v(k)]
    )


def predicted_epistasis(scored: Sequence[ScoredVariant], max_order: int = 3) -> list[Interaction]:
    """Build predicted Interactions (ε_hat + seed σ²) from conjoint ESM-2 scores.

    σ² is propagated from each variant's ``var_delta_g`` through the inclusion–exclusion sum,
    assuming independent score noise (a first approximation — see docs/SPEC.md#4).
    """
    raise NotImplementedError("Seedocs/ROADMAP.md")


def ground_truth_epistasis(dg: Mapping[Variant, float], max_order: int = 3) -> list[Interaction]:
    """Compute true ε terms from measured fitness (validation only; σ²=0)."""
    raise NotImplementedError("Seedocs/ROADMAP.md")


def wht_spectrum(dg: Mapping[Variant, float], sites: Sequence[int]) -> dict[int, float]:
    """Variance-by-order from the multiallelic Walsh-Hadamard transform (context/reporting)."""
    raise NotImplementedError("Seedocs/RESEARCH_EPISTASIS.md#3")
