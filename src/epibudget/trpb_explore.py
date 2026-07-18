"""Exploratory, non-decision-eligible profiler for the TrpB second landscape.

Phase 1.5 transfer check (docs/specs/trpb-exploratory.md). Answers one question: can the static
``epibudget`` allocation/evaluation abstractions operate on the TrpB landscape (Johnston et al.
2024, PNAS 121(32) e2400439121) without GB1-specific assumptions? It does **not** compute any
benchmark number — every output is explicitly ``exploratory_non_decision_eligible`` and is not part
of the frozen GB1 claim (docs/VALIDATION.md).

This module is complementary to :func:`epibudget.data.load_trpb`, never a replacement. ``load_trpb``
is the strict production loader: it returns a ``{Variant -> fitness}`` dict and therefore silently
collapses duplicate genotype rows and cannot report missing or invalid rows. The profiler here reads
the **raw** rows before any collapse, classifies each one, and reports the duplicate/conflict/
missing/invalid structure a second landscape must be audited for before it can be trusted. It
imports the frozen TrpB constants from :mod:`epibudget.data` and never mutates them.

No ESM, no torch, no network: the profiler reads a CSV and the candidate enumeration only. It never
imputes, invents, or drops a malformed row silently.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from epibudget.data import (
    TRPB_SITES,
    TRPB_WT_AT_SITES,
    TRPB_WT_SEQUENCE,
    enumerate_candidates,
)
from epibudget.types import Variant

# Every artifact this module produces carries this label; nothing here is decision-eligible.
RUN_TYPE = "exploratory_non_decision_eligible"

# The 20 canonical amino acids. A mutant (or reference) residue outside this set is flagged, never
# silently accepted. Identical to the alphabet the frozen candidate enumeration uses.
CANONICAL_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# epibudget selects and infers over interaction orders 1..3 only (max_order <= 3 is permanent v1
# scope, docs/ROADMAP.md). TrpB is combinatorially complete and so also contains order-4 genotypes;
# the profiler reports their mass so a full landscape is not mistaken for a coverage gap.
MAX_SELECTION_ORDER = 3

# Registered TrpB provenance (docs/VALIDATION.md "Second landscape — TrpB"; scripts/fetch_trpb.py).
# Used only as a fallback label when the fetch script's provenance_trpb.json is not on disk; the
# authoritative checksum is always computed from the file itself, never assumed.
TRPB_SOURCE = (
    "SeprotHub/Dataset-TrpB_fitness_landsacpe (Johnston et al. 2024, PNAS 121(32) e2400439121)"
)

# A genotype seen on fewer than this many rows cannot be a duplicate.
_MIN_DUPLICATE_ROWS = 2

# Per-row classification. A row is usable (carries a genotype) for status "ok" or "missing_label";
# the other three statuses mark a row with no recoverable genotype.
RecordStatus = Literal[
    "ok",
    "missing_label",
    "wrong_length",
    "invalid_amino_acid",
    "off_site_mutation",
]


@dataclass(frozen=True)
class RawRecord:
    """One classified CSV row. ``variant``/``order`` are ``None`` for an unrecoverable genotype."""

    row_index: int
    label: float | None
    variant: Variant | None
    order: int | None
    status: RecordStatus
    detail: str


class FitnessSummary(BaseModel):
    """Distribution of the measured labels (rows with a present, numeric label)."""

    n: int
    n_positive: int  # label > 0 (log-transformable, enters ground-truth epsilon / calibration)
    n_nonpositive: int  # label <= 0 (TrpB "inactive"; GB1's dead rows are exactly 0)
    min: float | None
    max: float | None
    mean: float | None
    median: float | None
    std: float | None
    q05: float | None
    q25: float | None
    q75: float | None
    q95: float | None


class DuplicateSummary(BaseModel):
    """Duplicate genotype structure — the check ``load_trpb``'s dict collapse cannot report."""

    n_genotype_rows: int  # rows carrying a genotype (ok + missing_label)
    n_unique_variants: int
    n_duplicate_variants: int  # distinct genotypes appearing on >1 row
    n_identical_duplicate_variants: int  # every copy carries the same label
    n_conflicting_duplicate_variants: int  # copies disagree on the label
    conflicting_samples: list[str]  # canonical ids, canonically sorted, bounded


class CoverageSummary(BaseModel):
    """Coverage of the order-1..max candidate universe, and the higher-order mass outside it."""

    max_selection_order: int  # min(3, n_sites); enumeration cannot exceed the number of sites
    universe_size: int  # |order-1..max candidates| over the alphabet (excludes WT)
    n_universe_measured: int  # universe genotypes present with a measured label
    coverage_fraction: float
    n_universe_measured_positive: int  # present with label > 0 (log-transformable)
    positive_coverage_fraction: float
    # rows of order > max_selection_order that carry a measured label: in the dataset but outside
    # epibudget's selection/inference scope (order 4 for the 4-site TrpB landscape).
    n_beyond_selection_order_measured: int


class BudgetRecommendation(BaseModel):
    """Exploratory vs confirmatory budgets. Confirmatory matches the frozen GB1 grid by design."""

    exploratory_budgets: list[int]
    exploratory_partitions: int
    confirmatory_budgets: list[int]
    rationale: str


class TrpbProfile(BaseModel):
    """Full exploratory profile of a TrpB CSV. JSON-serializable; non-decision-eligible."""

    model_config = {"frozen": True}

    run_type: str = RUN_TYPE
    decision_eligible: bool = False

    path: str
    dataset_checksum_sha256: str
    source: str

    sites_0indexed: list[int]
    wt_at_sites: list[str]
    alphabet: str

    n_rows: int
    status_counts: dict[str, int]
    n_measured: int
    n_missing_label: int
    n_invalid_records: int

    wt_present: bool
    wt_label: float | None
    wt_canonical_id: str

    order_distribution: dict[int, int]
    n_singles: int
    n_doubles: int
    n_triples: int
    n_quadruples: int

    aa_coverage_by_site: dict[int, str]  # site -> sorted distinct observed mutant residues
    aa_coverage_counts: dict[int, int]
    all_sites_fully_covered: bool  # every site observes all 19 non-WT residues

    fitness: FitnessSummary
    duplicates: DuplicateSummary
    coverage: CoverageSummary
    budget_recommendation: BudgetRecommendation
    gb1_incompatibilities: list[str]


def canonical_variant_id(variant: Variant) -> str:
    """Deterministic, order-independent id for a genotype. The wild type (order 0) is ``"WT"``."""
    if not variant:
        return "WT"
    return ",".join(f"{pos}:{wt}>{mut}" for pos, wt, mut in sorted(variant))


def sha256_file(path: Path) -> str:
    """SHA-256 of the file's bytes (same convention as scripts/fetch_trpb.py)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_label(raw: object) -> float | None:
    """Return the numeric label, or ``None`` for a missing/blank/unparseable/NaN value."""
    if isinstance(raw, bool):  # bool is an int subclass, never a fitness label
        return None
    value: object = raw
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            value = float(stripped)
        except ValueError:
            return None
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return None if math.isnan(value) else value
    return None


def _classify(
    row_index: int,
    seq_raw: object,
    label_raw: object,
    wt_sequence: str,
    site_set: frozenset[int],
    alphabet: frozenset[str],
) -> RawRecord:
    """Classify one raw row against the reference, without ever raising on malformed content."""
    label = _parse_label(label_raw)

    if not isinstance(seq_raw, str) or len(seq_raw) != len(wt_sequence):
        got = len(seq_raw) if isinstance(seq_raw, str) else "non-string"
        return RawRecord(
            row_index=row_index,
            label=label,
            variant=None,
            order=None,
            status="wrong_length",
            detail=f"sequence length {got} != reference length {len(wt_sequence)}",
        )

    diffs = [
        (i, wt_aa, mut_aa)
        for i, (wt_aa, mut_aa) in enumerate(zip(wt_sequence, seq_raw, strict=True))
        if wt_aa != mut_aa
    ]

    bad_aa = sorted({mut_aa for _, _, mut_aa in diffs if mut_aa not in alphabet})
    if bad_aa:
        return RawRecord(
            row_index=row_index,
            label=label,
            variant=None,
            order=None,
            status="invalid_amino_acid",
            detail=f"non-standard residue(s) {bad_aa} in the mutant sequence",
        )

    off_site = sorted({pos for pos, _, _ in diffs if pos not in site_set})
    if off_site:
        return RawRecord(
            row_index=row_index,
            label=label,
            variant=None,
            order=None,
            status="off_site_mutation",
            detail=f"mutation(s) at off-target position(s) {off_site}",
        )

    variant: Variant = frozenset(diffs)
    status: RecordStatus = "ok" if label is not None else "missing_label"
    return RawRecord(
        row_index=row_index,
        label=label,
        variant=variant,
        order=len(variant),
        status=status,
        detail="" if label is not None else "genotype recovered but label is missing",
    )


def read_raw_records(
    path: Path,
    wt_sequence: str = TRPB_WT_SEQUENCE,
    sites: Sequence[int] = TRPB_SITES,
    wt_at_sites: Sequence[str] = TRPB_WT_AT_SITES,
    alphabet: str = CANONICAL_ALPHABET,
) -> list[RawRecord]:
    """Read and classify every CSV row, in file order.

    Fails explicitly (raises ``ValueError``) on a structural problem that invalidates the whole
    file: a missing required column, or a reference construct that disagrees with ``wt_at_sites`` at
    ``sites`` (the same static guard as ``epibudget.data._load_landscape``). Per-row malformations
    (wrong length, non-standard residue, off-target mutation, missing label) are classified into an
    explicit :class:`RawRecord` status and never hidden.
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

    seqs: list[object] = df["protein"].tolist()
    labels: list[object] = df["label"].tolist()
    site_set = frozenset(sites)
    alpha_set = frozenset(alphabet)
    return [
        _classify(i, seq, lab, wt_sequence, site_set, alpha_set)
        for i, (seq, lab) in enumerate(zip(seqs, labels, strict=True))
    ]


def _fitness_summary(labels: list[float]) -> FitnessSummary:
    n = len(labels)
    n_positive = sum(1 for x in labels if x > 0.0)
    if n == 0:
        return FitnessSummary(
            n=0,
            n_positive=0,
            n_nonpositive=0,
            min=None,
            max=None,
            mean=None,
            median=None,
            std=None,
            q05=None,
            q25=None,
            q75=None,
            q95=None,
        )
    # Sort before reducing so mean/std have a fixed summation order: the profile is then
    # byte-identical under any input row permutation, not merely equal up to rounding.
    arr = np.sort(np.asarray(labels, dtype=float))
    q05, q25, q75, q95 = (float(v) for v in np.quantile(arr, [0.05, 0.25, 0.75, 0.95]))
    return FitnessSummary(
        n=n,
        n_positive=n_positive,
        n_nonpositive=n - n_positive,
        min=float(arr.min()),
        max=float(arr.max()),
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        std=float(arr.std()),
        q05=q05,
        q25=q25,
        q75=q75,
        q95=q95,
    )


def _duplicate_summary(records: Sequence[RawRecord], sample_cap: int = 20) -> DuplicateSummary:
    """Group genotype-bearing rows by identity; split duplicates into identical vs conflicting."""
    groups: dict[str, list[float | None]] = {}
    for r in records:
        if r.variant is None:
            continue
        groups.setdefault(canonical_variant_id(r.variant), []).append(r.label)

    identical: list[str] = []
    conflicting: list[str] = []
    for cid, labs in groups.items():
        if len(labs) < _MIN_DUPLICATE_ROWS:
            continue
        present = [x for x in labs if x is not None]
        # More than one distinct present label (or a mix of present and missing) is a conflict;
        # otherwise the copies agree (all identical present values, or all missing).
        if len(set(present)) > 1 or (present and len(present) != len(labs)):
            conflicting.append(cid)
        else:
            identical.append(cid)

    n_genotype_rows = sum(1 for r in records if r.variant is not None)
    return DuplicateSummary(
        n_genotype_rows=n_genotype_rows,
        n_unique_variants=len(groups),
        n_duplicate_variants=len(identical) + len(conflicting),
        n_identical_duplicate_variants=len(identical),
        n_conflicting_duplicate_variants=len(conflicting),
        conflicting_samples=sorted(conflicting)[:sample_cap],
    )


def _coverage_summary(
    records: Sequence[RawRecord],
    sites: Sequence[int],
    wt_at_sites: Sequence[str],
    alphabet: str,
) -> CoverageSummary:
    eff_max_order = min(MAX_SELECTION_ORDER, len(sites))
    universe = {
        canonical_variant_id(v)
        for v in enumerate_candidates(sites, wt_at_sites, alphabet, max_order=eff_max_order)
    }
    measured: set[str] = set()
    measured_positive: set[str] = set()
    n_beyond = 0
    for r in records:
        if r.variant is None or r.order is None:
            continue
        if r.order > eff_max_order:
            if r.label is not None:
                n_beyond += 1
            continue
        if r.label is None:
            continue
        cid = canonical_variant_id(r.variant)
        if cid in universe:
            measured.add(cid)
            if r.label > 0.0:
                measured_positive.add(cid)

    size = len(universe)
    return CoverageSummary(
        max_selection_order=eff_max_order,
        universe_size=size,
        n_universe_measured=len(measured),
        coverage_fraction=len(measured) / size if size else 0.0,
        n_universe_measured_positive=len(measured_positive),
        positive_coverage_fraction=len(measured_positive) / size if size else 0.0,
        n_beyond_selection_order_measured=n_beyond,
    )


def _budget_recommendation(coverage: CoverageSummary) -> BudgetRecommendation:
    """Exploratory (small) vs confirmatory budgets, justified by the actual universe scale."""
    return BudgetRecommendation(
        exploratory_budgets=[24, 48],
        exploratory_partitions=5,
        confirmatory_budgets=[48, 96, 192],
        rationale=(
            f"The order-1..3 candidate universe is {coverage.universe_size} variants (4 sites x "
            "20-letter alphabet), identical in shape to GB1's full-alphabet universe, so pool >> B "
            "holds at every GB1 budget and the frozen (48, 96, 192) grid transfers directly for a "
            "confirmatory run (docs/VALIDATION.md). Exploratory smoke uses smaller budgets and 5 "
            "partitions to keep CPU cost low; every exploratory number stays non-decision-eligible."
        ),
    )


def _gb1_incompatibilities(coverage: CoverageSummary, fitness: FitnessSummary) -> list[str]:
    """Concrete, data-grounded assumptions/differences a TrpB transfer must respect."""
    n_beyond = coverage.n_beyond_selection_order_measured
    return [
        (
            f"Order-4 saturation: {n_beyond} measured genotypes of order > 3 exist in the TrpB "
            "landscape, outside epibudget's order-1..3 selection/inference universe (max_order<=3 "
            "is permanent v1 scope). The exploratory run must restrict to the order-1..3 subset, "
            "as the GB1 harness already does; enumerate_candidates/_load_landscape need no change."
        ),
        (
            f"Inactive semantics differ: TrpB marks inactivity as label <= 0 "
            f"({fitness.n_nonpositive} of {fitness.n} measured rows), which may be negative; GB1 "
            "dead rows are exactly 0. The 'positive, log-transformable' conditioning is generic "
            "and handles both, but the non-positive fraction must be reported so it stays visible."
        ),
        (
            "Imputation: ~871 of 160,000 TrpB labels (~0.5%) are imputed, not measured, and the "
            "public mirror does not flag which (see scripts/fetch_trpb.py). The profiler cannot "
            "separate them and never invents the flag; every TrpB number carries that caveat."
        ),
        (
            "GB1-default WT argument: data.variant_from_sequence defaults wt=GB1_WT_SEQUENCE. "
            "load_trpb and this profiler pass TRPB_WT_SEQUENCE explicitly; any new TrpB code must "
            "do the same, never relying on the GB1 default."
        ),
    ]


def profile_trpb(
    path: Path,
    sites: Sequence[int] = TRPB_SITES,
    wt_at_sites: Sequence[str] = TRPB_WT_AT_SITES,
    wt_sequence: str = TRPB_WT_SEQUENCE,
    alphabet: str = CANONICAL_ALPHABET,
    source: str = TRPB_SOURCE,
) -> TrpbProfile:
    """Profile a TrpB CSV end-to-end. Deterministic and independent of input row order."""
    records = read_raw_records(path, wt_sequence, sites, wt_at_sites, alphabet)
    return build_profile(records, path, sites, wt_at_sites, alphabet, source)


def build_profile(
    records: Sequence[RawRecord],
    path: Path,
    sites: Sequence[int],
    wt_at_sites: Sequence[str],
    alphabet: str,
    source: str = TRPB_SOURCE,
) -> TrpbProfile:
    """Aggregate classified records into a profile. Split out from I/O so tests need no CSV file."""
    status_counts = {
        s: 0
        for s in ("ok", "missing_label", "wrong_length", "invalid_amino_acid", "off_site_mutation")
    }
    for r in records:
        status_counts[r.status] += 1

    order_dist = Counter(r.order for r in records if r.order is not None)
    n_non_wt = len(set(alphabet)) - 1  # residues a site can mutate to (alphabet minus its WT)

    aa_by_site: dict[int, set[str]] = {p: set() for p in sites}
    for r in records:
        if r.variant is None:
            continue
        for pos, _, mut in r.variant:
            if pos in aa_by_site:
                aa_by_site[pos].add(mut)

    wt_row = next((r for r in records if r.variant is not None and len(r.variant) == 0), None)
    measured_labels = [r.label for r in records if r.status == "ok" and r.label is not None]

    fitness = _fitness_summary(measured_labels)
    coverage = _coverage_summary(records, sites, wt_at_sites, alphabet)

    aa_counts = {p: len(v) for p, v in aa_by_site.items()}
    all_full = bool(aa_counts) and all(c == n_non_wt for c in aa_counts.values())

    return TrpbProfile(
        path=str(path),
        dataset_checksum_sha256=sha256_file(path) if path.exists() else "unavailable",
        source=source,
        sites_0indexed=list(sites),
        wt_at_sites=list(wt_at_sites),
        alphabet=alphabet,
        n_rows=len(records),
        status_counts=status_counts,
        n_measured=status_counts["ok"],
        n_missing_label=status_counts["missing_label"],
        n_invalid_records=(
            status_counts["wrong_length"]
            + status_counts["invalid_amino_acid"]
            + status_counts["off_site_mutation"]
        ),
        wt_present=wt_row is not None,
        wt_label=wt_row.label if wt_row is not None else None,
        wt_canonical_id="WT",
        order_distribution=dict(sorted(order_dist.items())),
        n_singles=order_dist.get(1, 0),
        n_doubles=order_dist.get(2, 0),
        n_triples=order_dist.get(3, 0),
        n_quadruples=order_dist.get(4, 0),
        aa_coverage_by_site={p: "".join(sorted(v)) for p, v in aa_by_site.items()},
        aa_coverage_counts=aa_counts,
        all_sites_fully_covered=all_full,
        fitness=fitness,
        duplicates=_duplicate_summary(records),
        coverage=coverage,
        budget_recommendation=_budget_recommendation(coverage),
        gb1_incompatibilities=_gb1_incompatibilities(coverage, fitness),
    )
