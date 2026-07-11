"""WT-referenced (biochemical) epistasis terms and the Walsh-Hadamard ground truth.

The pairwise/third-order coefficients below are the wild-type sub-sampling of the multiallelic
Walsh-Hadamard transform (docs/RESEARCH_EPISTASIS.md#3). They are exact, cheap, and fully tested;
they anchor invariant #1 — if the ΔG map is additive, every coefficient here is identically zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
import numpy.typing as npt

from epibudget.types import Interaction, Mutation, ScoredVariant, Variant

FloatArray = npt.NDArray[np.float64]

# Interactions are order 2 (pairwise) and up; order-1 terms are additive effects, not epistasis.
_MIN_INTERACTION_ORDER = 2


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


def interaction_loop(mutations: Sequence[Mutation]) -> list[Variant]:
    """The loop of an interaction: every non-empty sub-variant of ``mutations``.

    For a pairwise interaction {i,j} the loop is {i}, {j}, {i,j} (3 members); for a third-order
    interaction it is all 7 non-empty subsets. The empty set (wild type, ΔG≡0) is excluded. These
    are exactly the variants whose ΔG enters the inclusion–exclusion sum for ε.
    """
    n = len(mutations)
    return [frozenset(c) for r in range(1, n + 1) for c in combinations(mutations, r)]


def _epsilon(dg: Mapping[Variant, float], mutations: Sequence[Mutation]) -> float:
    """General WT-referenced inclusion–exclusion ε over ``mutations`` (matches the order-2/3 forms).

    ε(S) = Σ_{∅≠T⊆S} (−1)^{|S|−|T|} ΔG(T). Every coefficient is ±1; ΔG(∅)=0, so T=∅ is dropped.
    """
    n = len(mutations)
    total = 0.0
    for r in range(1, n + 1):
        sign = 1.0 if (n - r) % 2 == 0 else -1.0
        for combo in combinations(mutations, r):
            total += sign * dg[frozenset(combo)]
    return total


def _top_order_candidates(
    dg: Mapping[Variant, float], max_order: int
) -> list[tuple[Mutation, ...]]:
    """Every present variant of order 2..max_order, as a canonical (sorted) mutation tuple."""
    return sorted(
        (
            tuple(sorted(variant))
            for variant in dg
            if _MIN_INTERACTION_ORDER <= len(variant) <= max_order
        ),
        key=lambda muts: (len(muts), muts),
    )


def predicted_epistasis(scored: Sequence[ScoredVariant], max_order: int = 3) -> list[Interaction]:
    """Build predicted Interactions (ε_hat + seed σ²) from conjoint ESM-2 scores.

    ``scored`` must contain every candidate variant of order 1..max_order (all lower orders too, not
    only the top order — the σ² of an interaction sums over its whole loop). A missing lower-order
    member is a wiring bug, not a data gap, so it raises (never silently defaults to 0, which would
    understate σ² and bias acquisition). σ² is propagated from each variant's ``var_delta_g``:

        σ²(ε(S)) = Σ_{∅≠T⊆S} var_delta_g(T)

    ASSUMPTION (first approximation, docs/SPEC.md#4): Cov[ΔG(T), ΔG(T′)] = 0 for T ≠ T′ across all
    candidate variants — even nested/overlapping ones scored from related contexts. Because the
    inclusion–exclusion coefficients are all ±1, coefficient² = 1 and the propagated variance is the
    plain sum. The direction of the bias from real (correlated) score noise is not derivable from
    the ±1 structure, so this is a first approximation, not claimed conservative; it is checked
    empirically by the uncertainty-prior calibration (docs/VALIDATION.md), not by argument.
    """
    dg = {sv.variant: sv.delta_g for sv in scored}
    var = {sv.variant: sv.var_delta_g for sv in scored}
    interactions: list[Interaction] = []
    for mutations in _top_order_candidates(dg, max_order):
        loop = interaction_loop(mutations)
        missing = [member for member in loop if member not in dg]
        if missing:
            raise KeyError(
                f"predicted_epistasis: scored is missing lower-order sub-variant(s) "
                f"{[sorted(m) for m in missing]} needed for interaction {list(mutations)}; "
                f"pass the complete order-1..{max_order} candidate set"
            )
        sigma2 = sum(var[member] for member in loop)
        interactions.append(
            Interaction.of(mutations, epsilon_hat=_epsilon(dg, mutations), sigma2=sigma2)
        )
    return interactions


def ground_truth_epistasis(dg: Mapping[Variant, float], max_order: int = 3) -> list[Interaction]:
    """Compute true ε terms from measured fitness (validation only; σ²=0).

    ``dg`` is the measured ΔG map (e.g. ln fitness, WT-anchored so ΔG(∅)=0). Dead variants have no
    log-fitness and must be absent from ``dg`` upstream; a term is dropped (never imputed) if any of
    its up-to-seven loop members is missing (invariant #3, docs/STEP1_GATE.md).
    """
    interactions: list[Interaction] = []
    for mutations in _top_order_candidates(dg, max_order):
        loop = interaction_loop(mutations)
        if all(member in dg for member in loop):
            interactions.append(
                Interaction.of(mutations, epsilon_hat=_epsilon(dg, mutations), sigma2=0.0)
            )
    return interactions


def _orthonormal_contrast_basis(q: int) -> FloatArray:
    """A q×q orthonormal basis whose row 0 is the constant (mean) mode; rows 1..q−1 are contrasts.

    The specific contrast completion is irrelevant to the variance-by-order spectrum: only the
    per-order aggregate is basis-invariant (any orthogonal rotation within the contrast subspace
    leaves Σ of squared coefficients per order unchanged).
    """
    seed = np.eye(q, dtype=np.float64)
    seed[:, 0] = 1.0  # first column = constant vector, so QR's first basis vector is the mean mode
    basis, _ = np.linalg.qr(seed)
    return np.ascontiguousarray(basis.T, dtype=np.float64)


def _apply_along_axis(tensor: FloatArray, matrix: FloatArray, axis: int) -> FloatArray:
    """Contract ``matrix`` over ``axis``: out[...,k] = Σ_j matrix[k,j]·tensor[...,j]."""
    moved = np.moveaxis(tensor, axis, -1)
    transformed = moved @ matrix.T
    return np.moveaxis(transformed, -1, axis)


def _wht_forward(tensor: FloatArray, bases: Sequence[FloatArray]) -> FloatArray:
    """Multilinear (separable) transform of ``tensor`` by one orthonormal ``basis`` per axis."""
    out = tensor
    for axis, basis in enumerate(bases):
        out = _apply_along_axis(out, basis, axis)
    return out


def _landscape_tensor(
    dg: Mapping[Variant, float], sites: Sequence[int]
) -> tuple[FloatArray, list[FloatArray]]:
    """Build the dense (q₀,…,qₙ₋₁) ΔG tensor over ``sites`` and its per-axis orthonormal bases.

    The per-site alphabet is recovered from the mutations observed at that site (WT residue at index
    0). Requires a COMPLETE landscape: every residue combination over ``sites`` must be present in
    ``dg`` (raises otherwise — the transform cannot be run on a landscape with holes; real GB1 is
    incomplete, so this path is for complete synthetic/sub-landscape grids).
    """
    ordered_sites = sorted(sites)
    site_set = set(ordered_sites)
    wt_of: dict[int, str] = {}
    residues: dict[int, set[str]] = {s: set() for s in ordered_sites}
    for variant in dg:
        for pos, wt_aa, mut_aa in variant:
            if pos in site_set:
                if wt_of.setdefault(pos, wt_aa) != wt_aa:
                    raise ValueError(f"inconsistent WT residue at site {pos}")
                residues[pos].add(mut_aa)

    missing_sites = [s for s in ordered_sites if s not in wt_of]
    if missing_sites:
        raise ValueError(f"no mutations observed at site(s) {missing_sites}; cannot infer alphabet")

    alphabets = {s: [wt_of[s], *sorted(residues[s] - {wt_of[s]})] for s in ordered_sites}
    shape = tuple(len(alphabets[s]) for s in ordered_sites)
    tensor: FloatArray = np.empty(shape, dtype=np.float64)
    for index in np.ndindex(*shape):
        mutations = frozenset(
            (s, wt_of[s], alphabets[s][index[a]])
            for a, s in enumerate(ordered_sites)
            if index[a] != 0
        )
        if not mutations:  # the all-WT cell: ΔG(∅) ≡ 0 (WT anchor), even if absent from dg
            tensor[index] = dg.get(frozenset(), 0.0)
        elif mutations in dg:
            tensor[index] = dg[mutations]
        else:
            raise ValueError("incomplete landscape: not every residue combination is present")
    bases = [_orthonormal_contrast_basis(len(alphabets[s])) for s in ordered_sites]
    return tensor, bases


def wht_spectrum(dg: Mapping[Variant, float], sites: Sequence[int]) -> dict[int, float]:
    """Variance-by-order from the multiallelic Walsh-Hadamard transform (context/reporting).

    Returns {order → variance contribution} for order 1..n over a COMPLETE landscape at ``sites``.
    The order-0 (mean) term is excluded, so Σ of the returned values equals the population variance
    of the landscape (Parseval). An additive landscape has zero variance at every order ≥ 2.
    """
    tensor, bases = _landscape_tensor(dg, sites)
    coeffs = _wht_forward(tensor, bases)
    n = len(bases)
    shape = tensor.shape

    order = np.zeros(shape, dtype=np.int64)
    for axis, q in enumerate(shape):
        non_constant = (np.arange(q) != 0).astype(np.int64)
        broadcast_shape = tuple(q if b == axis else 1 for b in range(n))
        order = order + non_constant.reshape(broadcast_shape)

    n_cells = int(np.prod(np.array(shape, dtype=np.int64)))
    squared = np.square(coeffs)
    return {k: float(squared[order == k].sum()) / n_cells for k in range(1, n + 1)}
