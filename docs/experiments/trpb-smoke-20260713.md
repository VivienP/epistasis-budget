# TrpB exploratory validate run — 20260713T135240Z

Status: **exploratory · non-confirmatory · not decision-eligible · recovery invalidated.**

**Correction after WT-centred reanalysis design.** This artifact predates the current
ΔG(v) = log(f(v)/f(reference)) path. Its recovery coefficients, correlations and truth-map variance are
not interpretable. Its selection identities, attempted/revealed counts, coverage, hit-rate and run
configuration remain valid descriptive outputs. The serialized `var_epsilon` is truth variance, not the
predicted-epistasis invariant used by the current CLI.

This run executes the production `epibudget validate` harness on the TrpB landscape at reduced settings
(`B ∈ {24, 48}`, 5 seeds), per the archived
[`trpb-exploratory-plan.md`](trpb-exploratory-plan.md). It is **not** the frozen second-landscape
benchmark of `docs/VALIDATION.md` §"Second landscape — TrpB", which freezes `B ∈ {48, 96, 192}` and ≥ 20
seeds and states the run is deferred. It therefore violates its own pre-registered budget grid and seed
count, and no number in it may be reported as *the* TrpB result regardless of its quality.

The run is a **technical success and a scientific non-result**. The TrpB code path works end to end, but
three defects make the historical recovery comparison unreadable. They are documented in §6.

---

## 1. Artifact and provenance

| Field | Value | Verification |
|---|---|---|
| Report | `report/20260713T135240Z/metrics.json` (19,751 bytes) | sha256 `c85a9abc051d00521037cc50c3078f186f3cb1cee1ef1fa05cdaf9e9f2bd4ca7` |
| Archive | `trpb-smoke-20260713T135240Z.zip` | inner `metrics.json` byte-identical to the report |
| Dataset | `trpb_johnston2024` | sha256 `e94e2bed…a2e205f`, matches the local CSV exactly |
| Reference | Tm9D8\* (VFVS), sites 182/183/226/227 (0-indexed) | `wt_sha256` `c0964e6d…1af4cb` |
| Model | `esm2_t33_650M`, `scorer_seed = 0`, `n_perturbations = 16` | serialized |
| Alphabet | `ACDEFGHIKLMNPQRSTVWY` (full 20-letter) | serialized |
| Budgets / seeds | `24, 48` / `5` | serialized — **off-protocol** (frozen: 48/96/192, ≥ 20) |
| Device | `cuda` | serialized |
| Candidates | 29,678 (76 + 2,166 + 27,436) | reproduced by enumeration |
| Truth terms | 17,709 (1,784 pairwise + 15,925 third) | **reproduced exactly** from the local CSV |
| `var_epsilon` | 4.844110 | reproduced historical truth variance; anchor-contaminated and not invariant #1 |

**The repository SHA is not verifiable from the artifact.** The `validate.Report` schema carries no commit
field, so the expected run SHA `42ef311f391fa2f95a3a109f427fb45995c57679` cannot be confirmed against the
JSON. This is the same gap `docs/LIMITATIONS.md` §6 records for the GB1 headline, which works around it
with a wrapper field; the TrpB report has no wrapper. `42ef311` is an ancestor of the current tip, and the
only `src/` change since it is a docstring cross-reference in `epistasis.py` with no runtime effect — so
the artifact is reproducible at HEAD, but that is established from Git, not from the artifact.

Also absent: `run_type` / `decision_eligible`, timestamps, elapsed time. The report is
**schema-indistinguishable from a confirmatory run**. The run also did not export its ESM scored cache, so
no TrpB post-hoc diagnostic can be run without re-paying full GPU scoring. And
`data/proteingym/provenance.json` documents **GB1 only** — there is no machine-readable provenance record
for the TrpB CSV.

## 2. Cell completeness

All 10 cells present (5 methods × 2 budgets), no duplicates, no `null`/NaN/out-of-range correlations.
`ci_method` is `bootstrap-over-terms` for the four deterministic methods, `bootstrap-over-seeds` for
`random`.

**There is no seed × budget × method grid, by construction.** `run_validation` executes `info`, `fitness`,
`structural` and `practice` **once per budget** — they are deterministic and consume no seed. `seeds = 5`
replicates only `random`, and `_random_result` averages those replicates before serialization. **No
per-seed value is recoverable for any method.** Seed-level paired contrasts, sign counts and per-seed
dispersion are therefore not computable — not merely absent from this run, but outside what the schema can
express.

## 3. Results as serialized

`n_pinned = 0` in every cell: no ε loop is fully measured at these budgets, so every ε̂ retains at least
one ESM-prior loop member. (GB1 at B=48 also has `n_pinned = 0` for all five methods; structural first
pins terms at B=96. This is a budget property, not a TrpB difference.)

### Pairwise (1,784 terms)

| Method | B | ρ | ρ CI95 | r | informed | coverage |
|---|---|---|---|---|---|---|
| info | 24 | 0.0308 | [−0.0137, 0.0786] | 0.0498 | 1010 | 56.61% |
| structural | 24 | 0.0646 | [0.0204, 0.1113] | 0.0161 | 1088 | 60.99% |
| practice | 24 | 0.1365 | [0.0888, 0.1784] | 0.1054 | 24 | 1.35% |
| fitness | 24 | 0.1279 | [0.0815, 0.1713] | 0.1156 | 4 | 0.22% |
| random | 24 | −0.1271 | [−0.1282, −0.1260] | −0.1161 | 2 | 0.10% |
| info | 48 | 0.0434 | [−0.0056, 0.0875] | 0.0603 | 1622 | 90.92% |
| structural | 48 | −0.0174 | [−0.0665, 0.0284] | −0.0504 | 1657 | 92.88% |
| practice | 48 | 0.1424 | [0.0976, 0.1874] | 0.1115 | 48 | 2.69% |
| fitness | 48 | 0.1305 | [0.0866, 0.1765] | 0.1123 | 8 | 0.45% |
| random | 48 | −0.1264 | [−0.1275, −0.1257] | −0.1046 | 12 | 0.65% |

### Third order (15,925 terms)

| Method | B | ρ | ρ CI95 | r | informed | coverage |
|---|---|---|---|---|---|---|
| info | 24 | 0.1061 | [0.0908, 0.1209] | 0.1197 | 11560 | 72.59% |
| structural | 24 | 0.0849 | [0.0704, 0.0998] | 0.0648 | 13213 | 82.97% |
| practice | 24 | 0.0309 | [0.0162, 0.0456] | 0.0289 | 658 | 4.13% |
| fitness | 24 | 0.0166 | [0.0019, 0.0317] | 0.0009 | 142 | 0.89% |
| random | 24 | −0.0169 | [−0.0184, −0.0154] | −0.0080 | 60 | 0.37% |
| info | 48 | 0.1321 | [0.1176, 0.1468] | 0.1387 | 15645 | 98.24% |
| structural | 48 | 0.0589 | [0.0436, 0.0737] | 0.0408 | 15925 | 100.00% |
| practice | 48 | 0.0398 | [0.0247, 0.0546] | 0.0408 | 1306 | 8.20% |
| fitness | 48 | 0.0205 | [0.0057, 0.0349] | 0.0143 | 263 | 1.65% |
| random | 48 | −0.0203 | [−0.0282, −0.0154] | −0.0053 | 198 | 1.24% |

### Pooled (17,709 terms) — invalid, see §6.1

| Method | B | ρ | ρ CI95 | r | informed | coverage |
|---|---|---|---|---|---|---|
| info | 24 | 0.2060 | [0.1926, 0.2204] | 0.2356 | 12570 | 70.98% |
| structural | 24 | 0.1557 | [0.1427, 0.1696] | 0.1642 | 14301 | 80.76% |
| practice | 24 | 0.0539 | [0.0390, 0.0678] | 0.0152 | 682 | 3.85% |
| fitness | 24 | 0.0481 | [0.0324, 0.0622] | 0.0264 | 146 | 0.82% |
| random | 24 | −0.0461 | [−0.0473, −0.0450] | −0.0256 | 61 | 0.35% |
| info | 48 | 0.2891 | [0.2750, 0.3040] | 0.3134 | 17267 | 97.50% |
| structural | 48 | 0.2432 | [0.2291, 0.2578] | 0.2581 | 17582 | 99.28% |
| practice | 48 | 0.0553 | [0.0411, 0.0695] | 0.0119 | 1354 | 7.65% |
| fitness | 48 | 0.0496 | [0.0345, 0.0646] | 0.0232 | 271 | 1.53% |
| random | 48 | −0.0450 | [−0.0461, −0.0439] | −0.0077 | 210 | 1.18% |

Note the arithmetic tell: `info` at B=48 has **pooled 0.2891 above both of its own sub-orders** (pairwise
0.0434, third 0.1321). A pooled correlation can only exceed both its parts through between-group
separation. §6.1 shows that separation is manufactured by a bug.

### Hit-rate

Zero for every method at B=24. At B=48: `fitness` 0.0208, `practice` 0.0208, `random` 0.0042, `info` and
`structural` 0. The TrpB parent has fitness 0.408074 and 0.88% of live variants exceed it (GB1: 3.04%
exceed WT), so fitness discovery is a thin axis here.

## 4. Contrasts

Each deterministic method has exactly **one** observation per budget, so these are single differences with
**no dispersion estimate**. Sign counts across seeds cannot be formed (§2).

| Contrast | Order | Δρ @ B=24 | Δρ @ B=48 |
|---|---|---|---|
| info − fitness | pairwise | −0.0970 | −0.0871 |
| info − fitness | third | 0.0895 | 0.1115 |
| info − random | pairwise | 0.1579 | 0.1698 |
| structural − info | pairwise | 0.0338 | −0.0608 |
| structural − info | third | −0.0212 | −0.0732 |
| structural − fitness | pairwise | −0.0633 | −0.1479 |
| structural − fitness | third | 0.0682 | 0.0384 |
| practice − fitness | pairwise | 0.0086 | 0.0118 |

Read naively this is contradictory: `info` beats `fitness` and `random` on third order at both budgets but
**loses to both `fitness` and `practice` on pairwise** — the order `docs/VALIDATION.md` registers as the
decisive statistic. `structural − info` changes sign between budgets on pairwise. §6 explains why none of
these differences is interpretable.

## 5. Landscape comparison — TrpB vs GB1

Both are four-site, 20-letter landscapes with an identical 29,678-variant order-1..3 universe, so the
candidate pool and budget grid transfer by construction. They differ materially everywhere else.

| | GB1 (Wu 2016) | TrpB (Johnston 2024) |
|---|---|---|
| Assay | IgG-Fc binding | enzyme catalysis |
| Reference | WT GB1 (VDGV), **f = 1.0** | Tm9D8\* parent (VFVS), **f = 0.408074** |
| Rows | 149,361 of 160,000 (10,639 absent) | 160,000 (complete — but ~871 imputed, not measured) |
| Order-1..3 present | 28,186 / 29,678 (94.97%) | 29,678 / 29,678 (100%) |
| Inactivity encoded as | **exactly 0.0** (29,477 rows, 19.74%) | **negative** (35,643 rows, 22.28%); **no zeros** |
| Fitness range | 0.0 – 8.762 | −0.164 – 1.0 (max-normalised) |
| Live above reference | 3.04% | 0.88% |
| `ln f` on positives | mean −4.89, sd 1.72, min −8.85 | mean −4.15, sd 1.14, **min −13.76** |
| Truth terms | 17,782 (1,822 pairwise + 15,960 third) | 17,709 (1,784 pairwise + 15,925 third) |
| Terms dropped | 11,820 (2,476 missing row + 9,344 dead member) | 11,893 (0 missing + 11,893 dead member) |
| Dead singles | 0 | **4** — (226,V,D), (227,S,E), (227,S,H), (227,S,W) |
| `Var[ε_true]` | 2.616 | 4.844 as-run / **3.930 re-anchored** (§6.1) |
| Raw zero-shot ESM ε̂ pairwise ρ | **+0.302** (`artifacts/signal_650m.json`) | ‖ρ‖ ≈ 0.13; **sign unknown** (§6.3) |
| Seeds / budgets | 20 / 48, 96, 192 | 5 / 24, 48 |

**Two structural differences drive everything below.**

1. **GB1's WT fitness is exactly 1.0; TrpB's parent is 0.408074.** The ε machinery assumes ΔG(∅) = 0 —
   see §6.1. GB1 satisfies this by luck of normalisation; TrpB does not.
2. **Inactivity encoding.** GB1's dead variants are exactly `0.0` and are cleanly dropped by the `f > 0`
   rule. TrpB has no zeros: its dead mass is a noisy continuum straddling zero. The rule drops the negative
   half but **retains the near-zero positive half**, whose `ln f` reaches −13.76 — assay noise amplified
   into large ΔG outliers, which inclusion–exclusion propagates into every ε term touching them.

Only B=48 is shared. Even there, comparisons remain confounded by assay, activity distribution, seed count
(5 vs 20) and the defects in §6.

## 6. Three historical defects and their current status

### 6.1 The historical TrpB WT anchor was broken; the current code centres it

`interaction_loop` (`epistasis.py`) iterates `range(1, n+1)` — it **excludes the empty set** — so
`epsilon_pairwise` / `epsilon_third` structurally assume **ΔG(∅) = 0**. `docs/SIGNAL_GATE.md` states this
as an invariant: *"Measured ΔG(v) = ln(fitness(v)); WT = 1 ⇒ ΔG(∅) = 0."* And `scoring.py` sets
`delta_g = 0.0` for the order-0 variant with no forward pass, so the ESM side satisfies ΔĜ(∅) = 0 exactly.

The historical `validate.py` path built `landscape_dg = {v: log(f) ...}` with no normalisation by
f(reference). On GB1, f(WT) = 1.0, so ln f(WT) = 0. **On TrpB, f(parent) = 0.408074, so
ln f(∅) = −0.896307 ≠ 0.** The current validation and robustness paths instead use
`wt_centered_log_fitness`, which sets the reference exactly to zero.

Consequences, all recomputed from the CSV:

- A **constant** offset per order: every pairwise ε is shifted by **+0.896307**, every third-order ε by
  **−0.896307**. (GB1: exactly 0.0 in both orders.)
- `var_epsilon` as-run 4.844110 vs **re-anchored 3.930211** — the reported figure is **+23.3% inflated**.
- η², the between-order share of pooled ε variance: GB1 0.079; TrpB as-run **0.257** (3.2× GB1); TrpB
  re-anchored **0.085** — landing on GB1's value.

A constant within-order offset leaves the **truth-only** Pearson and Spearman ranks invariant. Recovery is
different because the through-origin calibration slope is refitted after centring. If g₀ = log f(reference)
and x denotes the ESM scores used for calibration, then
`b_centered = b_raw − g₀ Σx / Σx²`. Fully measured loops shift with the truth, but partially measured loops
retain prior members scaled by the changed slope; their ε̂ changes non-uniformly. Pairwise and third-order
recovery can therefore change even when evaluated separately. Pooled recovery is additionally confounded
by the opposite between-order offsets. No historical TrpB recovery metric survives this correction.

Caveat: re-anchoring fixes the ΔG(∅) = 0 *convention*. It does not make TrpB's `ln f` a valid free energy —
the label is max-normalised, goes negative, and has no exact zeros. 3.930 is the **anchor-consistent**
`var_epsilon`, not an "honest" one.

### 6.2 `structural-only` is an arbitrary tie-break, not a structural method (both landscapes)

`structural_graph` sets τ² ≡ 1.0, so its greedy weight is exactly `n(v)`, the number of interaction loops
containing `v`. In a four-site, 20-letter universe `n(v)` is **constant within each order**:

| Order | variants | distinct `n(v)` |
|---|---|---|
| 1 | 76 | `{1140}` |
| 2 | 2,166 | `{39}` |
| 3 | 27,436 | `{1}` |

`structural-only`'s weight takes **three distinct values in total** — it carries *zero* within-order
information. `acquisition.allocate` ranks with Python's **stable** `sorted`, so the within-order ordering
is an exact tie resolved by the input order of `scored`, which `enumerate_candidates` emits **site-major,
residue-alphabetical**. At any `B ≤ 76`, `structural-only` is literally *"take the first B singles in
enumeration order"*.

This is confirmed against the frozen artifacts by predicting `n_informed` from the tie-break rule alone,
with no ESM input — exact integer matches on five-digit quantities:

| | predicted | artifact |
|---|---|---|
| GB1 structural B=48 (pooled `n_informed`) | 17,700 | **17,700** |
| GB1 structural B=96 / B=192 | 17,782 | **17,782** |
| GB1 structural `n_pinned` B=96 / B=192 | 20 / 116 | **20 / 116** |
| TrpB structural B=24 | 14,301 | **14,301** |
| TrpB structural B=48 | 17,582 | **17,582** |

At GB1 B=48 it bought 19 singles at site 38, 19 at site 39, 10 at site 40 and **zero at site 53**. At
B=192, all 116 of its doubles come from the single site-pair (38,39).

**Replaying the frozen GB1 scored cache under other, equally valid tie-breaks of the same tie** (every draw
is 48 singles, all tied at n(v) = 1140) — pooled Spearman, B=48:

| structural tie-break | pooled ρ |
|---|---|
| as-run (enumeration order) | **0.2470** |
| reversed enumeration order | **0.0355** |
| balanced 12-per-site | 0.1736 |
| 20 random draws | mean 0.1449, sd 0.045, min 0.0567, max 0.2168 |
| *(info-optimal, for reference)* | *0.1997* |

Only **4 of 20** random tie-breaks beat info-optimal; the as-run value sits **above the maximum** of all 20
draws. **Under the reversed enumeration order, info-optimal wins.** At B=96 and B=192 — where all 76
singles are forced and only the doubles are tied — structural beats info under **20/20** tie-breaks, so the
ablation verdict is robust there.

The conclusion "the ESM uncertainty prior earns no credit" therefore **survives at B=96/192 but is
tie-break-dependent at B=48**, where the frozen headline reports it as holding at *every* budget. (These
sensitivity figures are **pooled**; the equivalent on the registered *pairwise* statistic has not been
computed and must be, before the claim is restated.)

The same `structural_graph` is the control in three places: the frozen GB1 ablation verdict, this TrpB run,
and — per `downstream.py:812` — the **primary contrast (`structural − fitness`) of the not-yet-run
downstream benchmark**.

### 6.3 The per-method calibration slope sets the sign of a low-coverage method's recovery (both landscapes)

`esm_prior_mu` sets `mu[v] = revealed[v]` if measured, else `b · esm[v]`, where `b` is a through-origin
slope fit **on that method's own revealed set**. Since ε is a linear inclusion–exclusion sum, for a term
with **no** measured loop member:

```
ε̂(term) = b · ε̂_ESM(term)        exactly
```

Spearman flips sign exactly under negative scaling, so a method that informs almost nothing reports
`ρ = sign(b) · ρ_prior`, with `ρ_prior` a **method-independent constant**. Its recovery measures the sign
of a nuisance parameter, not the quality of its selection.

The TrpB artifact shows exactly this — the three ~0%-coverage methods report near-identical magnitudes with
opposite signs:

| Order | B | fitness | practice | random | ‖fitness‖ − ‖random‖ |
|---|---|---|---|---|---|
| pairwise | 24 | 0.1279 | 0.1365 | −0.1271 | 0.0008 |
| pairwise | 48 | 0.1305 | 0.1424 | −0.1264 | 0.0041 |
| third | 24 | 0.0166 | 0.0309 | −0.0169 | −0.0002 |
| third | 48 | 0.0205 | 0.0398 | −0.0203 | 0.0002 |

**Already confirmed on GB1 by the repository's own artifacts.** `signal_650m.json` gives the raw ESM ε̂
pairwise ρ as **+0.302**; the GB1 headline's low-coverage methods report −0.259 (fitness), −0.271
(practice) and +0.279 (random) — all ‖ρ‖ ≈ 0.26–0.28, with `random` (lowest coverage, 0.52%) closest to the
prior. And the A2 cross-fit probe in `robustness_650m.json`, which shares **one** slope across methods,
flips fitness-greedy's GB1 pairwise recovery positive:

| B (pairwise) | fitness, per-method slope | fitness, **shared** slope |
|---|---|---|
| 48 | −0.2590 | **+0.2706** |
| 96 | −0.2472 | **+0.2728** |
| 192 | −0.1342 | **+0.3489** |

The GB1 *ranking* survives the shared slope (`structural > info > fitness` everywhere), so H1's direction is
unaffected. The **effect size is not**: `info − fitness` at B=48 collapses from **+0.667** (headline) to
**+0.160** (shared slope). The negative fitness-greedy values quoted in the README are calibration
artifacts, not measured anti-recovery. A2 was never run for `random` or `practice`, and never for TrpB.

`b` is not serialized. On TrpB it is additionally biased by §6.1 (the slope is fit on unanchored `ln f`),
so the **sign of TrpB's ρ_prior cannot be determined from this artifact** — only that its magnitude is
≈ 0.13, roughly half GB1's +0.302.

## 7. Verdicts

### Technical transfer — PASS

The native TrpB path worked end to end. The dataset checksum matches; the Tm9D8\* reference, sites and
alphabet are correct; ESM-2 650M scoring completed on GPU over all 29,678 candidates; selection,
inference, recovery and bootstrap CIs ran for all five methods at both budgets; all 10 cells are present,
finite and well-formed. The 17,709 historical truth terms and `var_epsilon` 4.844110 are reproducible from
the old uncentred transform, but they do not certify the current predicted-epistasis invariant.

Qualified by: the historical run did not catch the WT-anchor bug (§6.1); the
report records no repository SHA or `run_type`; no scored cache was exported; and no machine-readable
provenance exists for the TrpB CSV.

### Scientific transfer — UNINTERPRETABLE (metric defect)

The method ranking cannot be read off this artifact.

- **Pooled is invalid** — the between-order separation it rewards is manufactured by the broken anchor
  (§6.1), and the contamination is concentrated in exactly the high-coverage arms whose pooled numbers look
  best.
- **`info − fitness` and `info − random` are invalid** — at 0.2–1.7% coverage, `fitness`, `practice` and
  `random` all report `sign(b) · ρ_prior` (§6.3). Taking them at face value would say "fitness-greedy beats
  info-optimal on TrpB pairwise", which is false: fitness-greedy informed 8 of 1,784 pairwise loops.
- **`structural − info` is not a clean ablation** (§6.2).
- **No contrast has a dispersion estimate** (§2), and the run is off-protocol on both budgets and seeds.

The following descriptive outputs survive:

1. Candidate enumeration, selection identities and run configuration.
2. Attempted/revealed counts, coverage and hit-rate, which do not depend on the ΔG anchor.

Five seeds at B ∈ {24, 48} are **not** a confirmatory second-landscape benchmark, and this run is evidence
neither for nor against the frozen GB1 claim.

## 8. Corrective status and remaining recommendation

**Partly overtaken on 2026-07-16.** The downstream-impact benchmark has since run: the confirmatory GB1
run is decision-eligible with `structural_downstream_supported = true`, and an exploratory TrpB downstream
replication ran at `n_perturbations = 0` (non-decision-eligible). See
[`trpb-downstream-generalization-20260716.md`](trpb-downstream-generalization-20260716.md). The confirmatory
TrpB recovery re-run of item 3 below and Phase 2 remain blocked; the rest of this section stands.

The WT anchor and zero-GPU GB1 corrective gate are complete. The gate separates seeded structural
tie-breaks from method-specific and shared-slope regimes, but its registered decision is inconclusive and
ineligible for public claims. `structural − fitness` remains the primary downstream contrast, so
confirmatory downstream, TrpB and Phase 2 runs stay blocked pending an architecture decision.

The two shared defects can be quantified on the **existing GB1 650M scored cache**
(`report/scored_650m.jsonl`) at **zero GPU cost**.

1. **Completed:** WT-centred log fitness and regression tests. Historical TrpB recovery, ε coefficients
   and truth-map variance are invalidated; selections, coverage, hit-rate and configuration remain valid.
2. **Completed:** 100 seeded structural tie-breaks and shared-slope attribution for every Gate 2 selection.
   The outcome is inconclusive, so no replacement public claim is made.
3. **After an architecture decision, re-run TrpB** under the frozen protocol (`B ∈ {48, 96, 192}`, ≥ 20 seeds, anchor fixed,
   scored cache exported, shared-slope reporting, TrpB provenance JSON written).

Phase 2 (multi-round sequential design) stays behind milestone 2: its acquisition baselines inherit the
same `structural_graph`.

## 9. Limitations of this analysis

- TrpB's ε̂ could not be recomputed: the run exported no scored cache, and recomputing it needs ESM. §6.1's
  ε̂-side contamination is established from the code plus the coverage gradient in the artifact, not by
  direct numerical evaluation; §6.3's slope-sign claim for TrpB is established algebraically and by the GB1
  analogue, not by recomputing TrpB's `b`.
- The §6.2 pilot tie-break figures are pooled. The separate pairwise corrective analysis is recorded in
  Gate 2 and is not a public TrpB result.
- No paired seed-level statistic is reported because none exists in the artifact (§2).
- Per `docs/VALIDATION.md`, 871 of 160,000 TrpB fitness values (~0.5%) are imputed rather than measured and
  the public mirror does not flag which; every TrpB number inherits that caveat.
- GB1 and TrpB are never pooled.
