"""Public-data loaders and candidate enumeration. See docs/SPEC.md and the data-engineer agent.

Public data only. Never fabricate or impute a fitness value. Provenance (source, checksum, WT sequence,
row count) is recorded by the fetch scripts; loaders assert the WT residues at target positions.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from epibudget.types import Mutation, Variant

# GB1 (Wu et al. 2016) four epistatically-coupled sites in the B1 domain of protein G.
# Positions are 0-indexed into the GB1 WT sequence; residues asserted at load time.
GB1_SITES: tuple[int, ...] = (38, 39, 40, 53)  # V39, D40, G41, V54 in 1-indexed nomenclature
GB1_WT_AT_SITES: tuple[str, ...] = ("V", "D", "G", "V")


def enumerate_candidates(
    positions: Sequence[int],
    wt_at_positions: Sequence[str],
    allowed_aa: str = "ACDEFGHIKLMNPQRSTVWY",
    max_order: int = 3,
) -> list[Variant]:
    """All variants of order 1..max_order over ``positions`` (excluding the WT residue per site)."""
    raise NotImplementedError("Week 0 — see docs/ROADMAP.md")


def load_gb1(path: Path) -> dict[Variant, float]:
    """Load the complete GB1 four-site landscape as {Variant -> measured fitness}.

    Asserts row count and that WT residues at GB1_SITES match GB1_WT_AT_SITES (guards off-by-one).
    """
    raise NotImplementedError("Week 0 — run scripts/fetch_gb1.py first; see docs/VALIDATION.md")


def reveal_measured_fitness(
    landscape: dict[Variant, float], selected: Sequence[Variant]
) -> dict[Variant, float]:
    """Simulated wet-lab readout: look up true fitness for exactly the selected variants.

    This is the ONLY place labels enter selection-adjacent code — keep the boundary tight so selection
    can never leak labels (docs/VALIDATION.md threats table).
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


def _mut(pos: int, wt_aa: str, mut_aa: str) -> Mutation:
    return (pos, wt_aa, mut_aa)
