"""Public-data loaders and candidate enumeration. See docs/SPEC.md and the data-engineer agent.

Public data only. Never fabricate or impute a fitness value. Provenance (source, checksum, WT
sequence, row count) is recorded by the fetch scripts; loaders assert the WT residues at target
positions.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from itertools import combinations, product
from pathlib import Path

import pandas as pd

from epibudget.types import Variant

# GB1 (Wu et al. 2016) four epistatically-coupled sites in the B1 domain of protein G.
# Positions are 0-indexed into the GB1 WT sequence; residues asserted at load time.
GB1_SITES: tuple[int, ...] = (38, 39, 40, 53)  # V39, D40, G41, V54 in 1-indexed nomenclature
GB1_WT_AT_SITES: tuple[str, ...] = ("V", "D", "G", "V")

# Wild-type sequence of the GB1 B1 domain (56 residues), the reference for the Wu-2016 landscape.
# The four target sites are V39/D40/G41/V54 (1-indexed), i.e. GB1_SITES 0-indexed above. load_gb1
# asserts this against the fetched data's reference so a wrong construct cannot slip through.
GB1_WT_SEQUENCE: str = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"


def enumerate_candidates(
    positions: Sequence[int],
    wt_at_positions: Sequence[str],
    allowed_aa: str = "ACDEFGHIKLMNPQRSTVWY",
    max_order: int = 3,
) -> list[Variant]:
    """All variants of order 1..max_order over ``positions`` (excluding the WT residue per site).

    The mutant residue at a chosen position ranges over ``allowed_aa`` minus that position's WT
    residue, so order-k variants number ``C(n, k) * (len(allowed_aa) - 1) ** k`` for n positions.
    The WT itself (order 0) is never enumerated; selection code treats it as a known reference.
    """
    if len(positions) != len(wt_at_positions):
        raise ValueError(
            f"positions and wt_at_positions length mismatch: "
            f"{len(positions)} vs {len(wt_at_positions)}"
        )
    if len(set(positions)) != len(positions):
        raise ValueError(f"positions must be unique: {positions!r}")
    if not 1 <= max_order <= len(positions):
        raise ValueError(f"max_order must be in 1..{len(positions)}, got {max_order}")

    wt_of: dict[int, str] = dict(zip(positions, wt_at_positions, strict=True))
    variants: list[Variant] = []
    for order in range(1, max_order + 1):
        for combo in combinations(positions, order):
            per_site_choices = [[aa for aa in allowed_aa if aa != wt_of[p]] for p in combo]
            for assignment in product(*per_site_choices):
                variants.append(
                    frozenset((p, wt_of[p], aa) for p, aa in zip(combo, assignment, strict=True))
                )
    return variants


def variant_from_sequence(seq: str, wt: str = GB1_WT_SEQUENCE) -> Variant:
    """Recover the genotype of a full mutant sequence by diffing it against the wild type.

    Robust to any mutant-string formatting quirk in the source: the genotype is *defined* by the
    residue differences. Raises if the sequence length differs from the WT.
    """
    if len(seq) != len(wt):
        raise ValueError(f"sequence length {len(seq)} != WT length {len(wt)}")
    return frozenset(
        (i, wt_aa, mut_aa)
        for i, (wt_aa, mut_aa) in enumerate(zip(wt, seq, strict=True))
        if wt_aa != mut_aa
    )


def load_gb1(path: Path, sites: Sequence[int] = GB1_SITES) -> dict[Variant, float]:
    """Load the measured GB1 four-site rows as {Variant -> measured fitness}.

    Expects the SaProtHub/Wu-2016 schema (columns ``protein`` = full 56-residue sequence, ``label``
    = fitness relative to WT). Genotypes are recovered by diffing each sequence against
    ``GB1_WT_SEQUENCE``. Two layers of guard, any failure meaning a wrong construct / off-by-one
    that would corrupt every ε:
      1. static self-consistency: GB1_WT_AT_SITES must match GB1_WT_SEQUENCE at ``sites``;
      2. fetched-data validation: the wild type (order-0 genotype) must be present, and no variant
         may differ from the WT outside ``sites``.
    """
    for site, expected in zip(sites, GB1_WT_AT_SITES, strict=True):
        if GB1_WT_SEQUENCE[site] != expected:
            raise ValueError(
                f"WT residue at site {site} is {GB1_WT_SEQUENCE[site]!r}, expected {expected!r}"
            )

    df = pd.read_csv(path)
    if not {"protein", "label"} <= set(df.columns):
        raise ValueError(
            f"expected columns 'protein' and 'label' in {path}, got {list(df.columns)}"
        )

    site_set = set(sites)
    landscape: dict[Variant, float] = {}
    for seq, label in zip(df["protein"].astype(str), df["label"].astype(float), strict=True):
        variant = variant_from_sequence(seq)
        off_site = [pos for pos, _, _ in variant if pos not in site_set]
        if off_site:
            raise ValueError(
                f"variant mutates off-target position(s) {off_site}: not scoped to {sites}"
            )
        landscape[variant] = float(label)

    if frozenset() not in landscape:
        raise ValueError(f"wild type (order-0 genotype) absent from {path}")
    return landscape


def variant_order_composition(landscape: dict[Variant, float]) -> dict[int, int]:
    """Count genotypes by mutation order (0=WT, 1=single, ...); for provenance and sanity checks."""
    counts = Counter(len(v) for v in landscape)
    return dict(sorted(counts.items()))


def reveal_measured_fitness(
    landscape: dict[Variant, float], selected: Sequence[Variant]
) -> dict[Variant, float]:
    """Simulated wet-lab readout: look up true fitness for exactly the selected variants.

    This is the ONLY place labels enter selection-adjacent code — keep the boundary tight so
    selection can never leak labels (docs/VALIDATION.md threats table).
    """
    return {v: landscape[v] for v in selected if v in landscape}


def apply_mutations(wt: str, variant: Variant) -> str:
    """Return ``wt`` with every mutation in ``variant`` applied (used by conjoint scoring)."""
    seq = list(wt)
    for pos, wt_aa, mut_aa in variant:
        if seq[pos] != wt_aa:
            raise ValueError(f"WT mismatch at {pos}: expected {wt_aa}, found {seq[pos]}")
        seq[pos] = mut_aa
    return "".join(seq)
