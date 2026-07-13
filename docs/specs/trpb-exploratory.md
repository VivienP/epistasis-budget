# Spec: exploratory TrpB transfer profile (`src/epibudget/trpb_explore.py`)

Status: **exploratory · non-confirmatory · `decision_eligible = false` · not part of the frozen GB1
claim.** This is a Phase 1.5 transfer check produced in an isolated worktree (`explore/trpb-port`), not a
roadmap Step and not the frozen TrpB second-landscape benchmark. It pairs with `docs/SPEC.md` (the
frozen design), `docs/VALIDATION.md` §"Second landscape — TrpB" (the **deferred** confirmatory protocol,
which this document does not modify), and `docs/ROADMAP.md` Ambition-layer A (generalisation). Per-feature
specs are exempt from the `no-ai-narration` rule (they describe a planned change at a point in time).

## Purpose

One question, no benchmark number:

> Can the static `epibudget` allocation and evaluation abstractions operate on a second protein
> landscape (TrpB) without GB1-specific assumptions?

The frozen confirmatory TrpB run is deliberately deferred until the GB1 headline is interpreted
(`docs/VALIDATION.md`; running a second landscape and only then choosing how to report it would be
landscape cherry-picking, invariant #2). This exploratory work is the compatibility/profiling step that
precedes that run: it audits the dataset and the code seam, and produces **zero** decision-eligible
numbers. Every artifact it emits is stamped `run_type = exploratory_non_decision_eligible`.

## Relationship to what already exists (do not duplicate)

The committed baseline (`e75535d`) already ships the production TrpB seam, and this work reuses it
unchanged:

- `epibudget.data.load_trpb` / `_load_landscape` — the strict `{Variant -> fitness}` loader (reference
  asserts, on-site guard). Frozen; not modified here.
- `epibudget.data.TRPB_SITES / TRPB_WT_AT_SITES / TRPB_WT_SEQUENCE` — the registered constants.
- `scripts/fetch_trpb.py` — the provenance-recording fetch.
- `docs/VALIDATION.md` §"Second landscape — TrpB" — the frozen deferred protocol.

The **only** new production code is `epibudget.trpb_explore`, a profiler that is complementary to
`load_trpb`: `load_trpb` returns a dict and therefore silently collapses duplicate genotype rows and
cannot report missing/invalid rows; the profiler reads the **raw** rows before any collapse and reports
the duplicate/conflict/missing/invalid structure a second landscape must be audited for. It performs no
ESM inference, imports no torch, and touches no network.

## Dataset

All fields below are registered provenance (`docs/VALIDATION.md`, `data.py`, `scripts/fetch_trpb.py`),
not invented here.

- **Source.** `SeprotHub/Dataset-TrpB_fitness_landsacpe` on the Hugging Face Hub (the source's literal
  spelling), redistributing Johnston et al. 2024, *PNAS* 121(32) e2400439121.
- **Version / retrieval.** `python scripts/fetch_trpb.py` downloads `dataset.csv` and writes
  `data/proteingym/trpb_johnston2024.csv` plus `provenance_trpb.json` (source URL, UTC download date,
  sha256, byte size, row count, order composition, label semantics). Git-ignored; never committed.
- **Wild-type / reference.** The ε anchor is the assayed parent **Tm9D8\* = VFVS** — residues V/F/V/S at
  positions 183/184/227/228 (1-indexed; `TRPB_SITES = (182, 183, 226, 227)` 0-indexed). The reference is
  the order-0 genotype (empty `Variant`), never literal TmTrpB. `TRPB_WT_SEQUENCE` is the 397-residue
  parent (C-terminal His6 tag included).
- **Sequence/mutation encoding.** Rows are full-length sequences in a `protein` column; the genotype is
  recovered by diffing each sequence against the parent (`variant_from_sequence`), so any mutant-string
  formatting quirk in the mirror is irrelevant — the genotype is defined by the residue differences.
- **Measured target / direction.** `label` = an aggregated catalytic-fitness score (higher = fitter).
- **Missing-value semantics.** A blank/NaN/unparseable `label` is preserved as a genotype with a missing
  measurement (`status = "missing_label"`), never imputed and never dropped.
- **Duplicate-handling rule.** Rows are grouped by canonical genotype id. A genotype on more than one row
  is a duplicate; **identical** (all copies carry the same label) is distinguished from **conflicting**
  (copies disagree, or mix present and missing). The profiler reports both counts and bounded samples;
  it never silently keeps one copy the way the `load_trpb` dict does.
- **Invalid-record rule.** Wrong sequence length, a non-standard residue (outside the 20-letter
  alphabet), or an off-target mutation is classified into an explicit status and counted, never hidden.
  A missing required column or a reference construct that disagrees with `TRPB_WT_AT_SITES` raises.
- **Checksum.** sha256 of the CSV bytes, computed at fetch time (`provenance_trpb.json`) and recomputed
  by the profiler; never hardcoded in code or docs.

## Candidate universe

- **Allowed positions.** The four TrpB sites above. Nothing in the profiler assumes exactly four; the
  order cap is `min(3, n_sites)`.
- **Alphabet.** The 20 canonical amino acids (`ACDEFGHIKLMNPQRSTVWY`), same as the frozen enumeration.
- **Maximum supported order.** 1–3. `max_order <= 3` is permanent v1 scope (`docs/ROADMAP.md`).
- **Observed vs enumerable.** TrpB is combinatorially complete (20⁴ = 160,000). The order-1..3 candidate
  universe is `enumerate_candidates(...)` = 76 singles + 2,166 doubles + 27,436 triples = **29,678**
  variants — identical in shape to GB1's full-alphabet universe. The remaining **130,321 order-4**
  genotypes (81% of the landscape) are complete in TrpB but sit **outside** epibudget's
  selection/inference scope. The profiler reports this order-4 mass so a complete landscape is not
  mistaken for a coverage gap; the exploratory run restricts to the order-1..3 subset exactly as the GB1
  harness already does.
- **Unavailable variants.** GB1's artifact is incomplete (~149k of 160k measured); TrpB's order-1..3
  subset is expected complete, but the profiler reports observed coverage rather than assuming it.

## Exploratory metrics

Reuse the existing benchmark machinery where it is scientifically valid, with **no** redefinition of the
frozen GB1 primary claim:

- The frozen `validate.py` map-recovery statistics (pairwise/third-order Spearman & Pearson of inferred
  vs ground-truth ε) apply unchanged, because the ε machinery, the factor graph, and `infer_epistasis`
  are position-count- and length-agnostic and consume only the order-1..3 universe.
- **TrpB-specific adaptation.** Ground-truth ε is conditioned on positive, log-transformable, complete
  loops (identical to GB1). Because TrpB inactivity is `label <= 0` (possibly negative) rather than
  GB1's exactly-0 dead rows, the conditioning drops more rows on TrpB; the profiler's non-positive count
  makes that visible. No metric is silently rescaled.
- Any metric emitted in this workstream is descriptive and non-decision-eligible; it never updates a
  README claim, a manifest, or a frozen artifact.

## Exploratory budgets

- **Exploratory smoke:** `B ∈ {24, 48}`, 5 partitions — small, CPU-cheap, non-decisional.
- **Confirmatory (deferred, unchanged):** `B ∈ {48, 96, 192}`. This matches the frozen GB1 grid **by
  construction**: the order-1..3 universe is 29,678 for both landscapes (4 sites × 20 letters), so
  `pool ≫ B` holds identically and the grid transfers directly. This is a justification, not a blind
  copy — the two universes are the same size.

## Run profile

- CPU is the default execution path; no GPU, no second PLM, no learned surrogate, no interaction order
  above 3, no multi-round loop.
- The profiler itself needs neither ESM nor the network — it runs on the CSV alone in seconds.
- A methods/metrics run additionally needs the ESM scored cache; that is expensive and is **not** run in
  this workstream (the exact future command is given in `scripts/explore_trpb.py` and the README-less
  script docstring). Every generated report remains non-decision-eligible.

## Go / no-go criteria for Phase 2

Phase 2 (the deferred confirmatory TrpB run and, separately, any active-learning work) is unblocked when:

1. the loader works on TrpB without GB1-specific hacks (satisfied: `load_trpb` reuses the generic
   `_load_landscape`; no four-position or 56-length assumption remains in the path);
2. the wild type and mutations are unambiguous (the Tm9D8\* parent is the order-0 genotype; genotypes are
   diff-defined);
3. the candidate and measured universes are reproducible (deterministic enumeration + a file checksum);
4. metrics are computable on the order-1..3 subset;
5. missingness is explicit (missing labels preserved and counted; imputation caveated, never separated);
6. method behaviour is interpretable (the profiler surfaces order distribution, coverage, and the
   order-4 mass so a run's shape is legible);
7. no full architecture rewrite is required (the only new module is a thin sibling profiler).

The exploratory methods need **not** reproduce the GB1 ranking. A per-protein split is itself an honest
finding (`docs/ROADMAP.md`).

## Test plan (`tests/test_trpb_explore.py`, offline, synthetic)

A tiny TrpB-shaped CSV built from `TRPB_WT_SEQUENCE` + `apply_mutations` (no network, no ESM, no copy of
the external dataset): valid loading; stable WT identification regardless of row position; wrong-length,
invalid-amino-acid, and off-site classification (not raised); missing labels preserved not dropped;
mutation-order counting and the order-4-beyond-universe count; canonical-id order-independence; identical
vs conflicting duplicates (including present-vs-missing); `build_profile` byte-identical under a row
permutation; checksum stable and content-sensitive; missing-column and wrong-reference guards raise; a
two-site landscape profiled correctly (no four-position assumption); and a repo-hygiene test asserting
`data/proteingym/` stays git-ignored.

## Verification

`ruff format --check .`, `ruff check .`, `mypy --strict src/`, `pytest -q` (offline). No public number is
added, so no artifact/manifest validator applies. A methods smoke is **not** run: it would require ESM
scoring; the exact future command is documented in `scripts/explore_trpb.py`.
