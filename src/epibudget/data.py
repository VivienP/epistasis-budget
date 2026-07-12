"""Public-data loaders and candidate enumeration. See docs/SPEC.md and the data-engineer agent.

Public data only. Never fabricate or impute a fitness value. Provenance (source, checksum, WT
sequence, row count) is recorded by the fetch scripts; loaders assert the WT residues at target
positions.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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

# TrpB (Johnston et al. 2024, PNAS 2400439121): four-site combinatorially complete landscape at the
# active site of the thermostable β-subunit of tryptophan synthase. The parent enzyme is Tm9D8*
# ("VFVS") — residues V/F/V/S at target positions 183/184/227/228 (1-indexed; 0-indexed below). ε is
# referenced to this parent, as GB1's is to its WT. A SECOND, independently-motivated landscape
# (enzyme catalysis vs GB1's binding) — see docs/VALIDATION.md for the deferred protocol.
TRPB_SITES: tuple[int, ...] = (182, 183, 226, 227)  # 183/184/227/228 in 1-indexed paper numbering
TRPB_WT_AT_SITES: tuple[str, ...] = ("V", "F", "V", "S")  # Tm9D8* parent = VFVS
TRPB_WT_SEQUENCE: str = (
    "MKGYFGPYGGQYVPEILMGALEELEAAYEGIMKDESFWKEFNDLLRDYAGRPTPLYFARRLSEKYGARVYLKREDLLHTGAHKINNAIGQVL"
    "LAKLMGKTRIIAETGAGQHGVATATAAALFGMECVIYMGEEDTIRQKLNVERMKLLGAKVVPVKSGSRTLKDAIDEALRDWITNLQTTYYVF"
    "GSVVGPHPYPIIVRNFQKVIGEETKKQIPEKEGRLPDYIVACVSGGSNAAGIFYPFIDSGVKLIGVEAGGEGLETGKHAASLLKGKIGYLHGS"
    "KTFVLQDDWGQVQVSHSVSAGLDYSGVGPEHAYWRETGKVLYDAVTDEEALDAFIELSRLEGIIPALESSHALAYLKKINIKGKVVVVNLSGR"
    "GDKDLESVLNHPYVRERIRLEHHHHHH"
)


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


def _load_landscape(
    path: Path, wt_sequence: str, sites: Sequence[int], wt_at_sites: Sequence[str]
) -> dict[Variant, float]:
    """Load a four-site DMS landscape as {Variant -> fitness}, ε-referenced to ``wt_sequence``.

    Expects the ProteinGym/SaProtHub schema (column ``protein`` = full-length sequence, ``label`` =
    fitness; extra columns like ``stage`` are ignored). Genotypes are recovered by diffing each
    sequence against ``wt_sequence``. Two layers of guard, any failure meaning a wrong construct /
    off-by-one that would corrupt every ε:
      1. static self-consistency: ``wt_at_sites`` must match ``wt_sequence`` at ``sites``;
      2. fetched-data validation: the reference (order-0 genotype) must be present, and no variant
         may differ from the reference outside ``sites``.
    """
    for site, expected in zip(sites, wt_at_sites, strict=True):
        if wt_sequence[site] != expected:
            raise ValueError(
                f"reference residue at site {site} is {wt_sequence[site]!r}, expected {expected!r}"
            )

    df = pd.read_csv(path)
    if not {"protein", "label"} <= set(df.columns):
        raise ValueError(
            f"expected columns 'protein' and 'label' in {path}, got {list(df.columns)}"
        )

    site_set = set(sites)
    landscape: dict[Variant, float] = {}
    for seq, label in zip(df["protein"].astype(str), df["label"].astype(float), strict=True):
        variant = variant_from_sequence(seq, wt_sequence)
        off_site = [pos for pos, _, _ in variant if pos not in site_set]
        if off_site:
            raise ValueError(
                f"variant mutates off-target position(s) {off_site}: not scoped to {sites}"
            )
        landscape[variant] = float(label)

    if frozenset() not in landscape:
        raise ValueError(f"reference (order-0 genotype) absent from {path}")
    return landscape


def load_gb1(path: Path, sites: Sequence[int] = GB1_SITES) -> dict[Variant, float]:
    """Load the measured GB1 four-site rows (Wu 2016), ε-referenced to ``GB1_WT_SEQUENCE``."""
    return _load_landscape(path, GB1_WT_SEQUENCE, sites, GB1_WT_AT_SITES)


def load_trpb(path: Path, sites: Sequence[int] = TRPB_SITES) -> dict[Variant, float]:
    """Load the measured TrpB four-site rows (Johnston 2024), ε-referenced to the Tm9D8* parent."""
    return _load_landscape(path, TRPB_WT_SEQUENCE, sites, TRPB_WT_AT_SITES)


@dataclass(frozen=True)
class DatasetSpec:
    """Everything the validation harness needs to run one four-site landscape.

    Binds a dataset identifier to its loader and its reference construct — the target ``sites``,
    the WT residues expected there, and the full ``wt_sequence`` the ε anchor is defined against —
    so the CLI never hardcodes a single landscape's constants. ``default_data_path`` is the
    conventional on-disk CSV the matching fetch script writes.
    """

    identifier: str
    loader: Callable[[Path], dict[Variant, float]]
    sites: tuple[int, ...]
    wt_at_sites: tuple[str, ...]
    wt_sequence: str
    default_data_path: str


# Registered validation datasets. Each maps a stable identifier to its loader and reference
# construct; a caller resolves one via :func:`resolve_dataset`, which rejects anything unregistered
# rather than silently defaulting to GB1.
DATASETS: dict[str, DatasetSpec] = {
    "gb1_wu2016": DatasetSpec(
        identifier="gb1_wu2016",
        loader=load_gb1,
        sites=GB1_SITES,
        wt_at_sites=GB1_WT_AT_SITES,
        wt_sequence=GB1_WT_SEQUENCE,
        default_data_path="data/proteingym/gb1_wu2016.csv",
    ),
    "trpb_johnston2024": DatasetSpec(
        identifier="trpb_johnston2024",
        loader=load_trpb,
        sites=TRPB_SITES,
        wt_at_sites=TRPB_WT_AT_SITES,
        wt_sequence=TRPB_WT_SEQUENCE,
        default_data_path="data/proteingym/trpb_johnston2024.csv",
    ),
}


def resolve_dataset(identifier: str) -> DatasetSpec:
    """Return the registered :class:`DatasetSpec`, rejecting an unknown identifier explicitly."""
    try:
        return DATASETS[identifier]
    except KeyError:
        supported = ", ".join(sorted(DATASETS))
        raise ValueError(
            f"unknown dataset {identifier!r}; supported identifiers: {supported}"
        ) from None


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
