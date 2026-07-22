# Historical design record: exploratory TrpB transfer

Status: **superseded, exploratory, non-confirmatory, and never decision-eligible**.

This record preserves the plan frozen at commit `4077f55` before the TrpB smoke run. The run itself and
its later interpretation are documented in [`trpb-smoke-20260713.md`](trpb-smoke-20260713.md). Current
confirmatory settings and run status live in [`VALIDATION.md`](../VALIDATION.md).

## Original question

Could the static allocation and evaluation abstractions operate on TrpB without GB1-specific assumptions?

The work was a compatibility and data-profiling check. It was not the confirmatory second-landscape
benchmark and could not update a README claim or registered artifact.

## Reused implementation seam

The plan reused:

- `data.load_trpb` and the generic landscape loader;
- `TRPB_SITES`, `TRPB_WT_AT_SITES`, and `TRPB_WT_SEQUENCE`;
- `scripts/fetch_trpb.py` for data retrieval and provenance;
- the existing candidate enumeration, scoring, epistasis, and validation machinery.

The added `trpb_explore` profiler inspected raw rows before dictionary construction so duplicates,
conflicts, missing labels, invalid sequences, and off-target mutations could not disappear silently. It
performed no ESM inference and no network access.

## Dataset contract

- Source: the SeprotHub redistribution of Johnston et al. 2024.
- Reference: the assayed Tm9D8* parent `VFVS` at positions 183, 184, 227, and 228 (1-indexed).
- Genotypes: derived by comparing each full sequence with the parent.
- Fitness: aggregated catalytic fitness, higher is better.
- Missing labels: preserved and counted, never imputed by the profiler.
- Duplicate rows: classified as identical or conflicting; no arbitrary row selected.
- Invalid rows: wrong length, noncanonical residues, and off-target mutations classified explicitly.
- Provenance: CSV checksum recomputed rather than hard-coded.

The source paper reports 871 imputed fitness values, while the public mirror does not identify them row
by row. The plan therefore treated imputation as an unavoidable dataset-level caveat.

## Candidate and evaluation scope

The four-site, 20-letter order-1 through order-3 universe contains 76 singles, 2,166 doubles, and 27,436
triples. Order-4 genotypes were outside v1 scope and reported separately so their absence from selection
was not mistaken for missing data.

Ground-truth interactions required positive, log-transformable, complete loops. Every exploratory metric
was descriptive and used the same order-specific recovery definitions as GB1; no metric was rescaled or
promoted into the frozen decision rule.

The planned smoke profile used B=(24, 48) and five seeds. The confirmatory profile remained B=(48, 96,
192), at least 20 seeds, full alphabet, ESM-2 650M, and `n_perturbations=16`. The smoke scale could not
become confirmatory regardless of its direction.

## Acceptance criteria

The exploratory seam was acceptable only if:

1. loading required no GB1-specific assumptions;
2. the reference and mutation encoding were unambiguous;
3. raw-data problems and missingness were explicit;
4. enumeration and checksums were deterministic;
5. order-1 through order-3 metrics were computable without redefining them;
6. no architecture rewrite or public result claim was required.

## Verification contract

Offline synthetic tests covered reference identification, row-order invariance, invalid and off-site
sequences, missing labels, duplicate conflicts, order counts, checksum sensitivity, datasets with fewer
than four sites, and repository hygiene. The plan itself authorized no expensive ESM run.

This record is historical. It must not be used to infer the status or result of the current TrpB 650M
`n_perturbations=16` scoring run.
