# Validation protocol

The credibility of `epibudget` rests entirely on one honest benchmark. This document freezes the
protocol **before** any result exists, so the outcome cannot be reverse-engineered. Changing anything
here after seeing results requires an explicit note in the report and Vivien's sign-off.

## The claim under test

> **H1.** At equal budget *B*, variants selected by information-optimal allocation (`--lambda 0`)
> recover the ground-truth epistasis map of GB1 better than the same budget spent fitness-greedily
> (`--lambda 1`), and better than random.

Null hypothesis **H0**: information-optimal is indistinguishable from (or worse than) fitness-greedy.
**We report H0 as the headline if that is what the data show.** A clean negative — "information-optimal
DMS design does *not* beat fitness-greedy for epistasis recovery on GB1, here is the evidence" — is a
legitimate, publishable audit and a perfectly good portfolio artifact. It is *not* a failure to hide.

## Dataset

- **GB1 four-site landscape**, Wu, Olson, Fowler & Sun 2016, *eLife* — the complete 20⁴ = 160,000
  combinatorial landscape at positions **V39, D40, G41, V54** of protein G domain B1. Contains every
  single, double, triple and quadruple mutant with a measured fitness (binding/stability enrichment).
- Access via **ProteinGym** (substitution DMS assays include GB1) and/or the original supplementary
  data. Fetching is explicit and lives in `scripts/fetch_gb1.py`; data is **never committed** (see
  `.gitignore`). The script records a checksum of the downloaded file.
- Why this dataset: it is the only public landscape with *complete* higher-order ground truth, so we can
  compute true ε terms and simulate any budgeted experiment exactly (§Simulation).

## Ground truth

`ground_truth_epistasis(full_landscape)` computes, from the measured fitnesses:

- all pairwise ε(i,j) and third-order ε(i,j,k) terms (WT-referenced, inclusion–exclusion), and
- the multiallelic Walsh–Hadamard spectrum (variance explained by order) for context.

These are the target coefficients the selected experiments must recover.

## Simulation of a budgeted experiment

1. A method selects `B` variants **zero-shot** — using only ESM-2 scores and the factor graph. It never
   sees any measured fitness during selection.
2. `reveal_measured_fitness(selected)` looks up the true GB1 fitness of exactly those `B` variants
   (this is the simulated wet-lab readout — the only place labels enter).
3. `infer_epistasis(revealed)` fits the epistasis coefficients from just those `B` measurements. This
   precisifies "regularised least squares over the interaction basis" as the closed-form **posterior
   mean** of the graph.py linear-Gaussian model: a measured variant pins its ΔG; every unmeasured loop
   member keeps its unit-calibrated ESM prior mean (a Tikhonov estimator with prior mean = the
   calibrated ESM ΔĜ and precision = 1/`var_delta_g`). This is chosen over a zero-shrinkage ridge on
   the measured data alone because the frozen `info_gain` weight `τ²·n(v)` front-loads low-order
   variants (a single sits in ~1140 interaction loops); a zero-shrinkage fit would grade info-optimal's
   own top picks as recovering nothing (an all-zero, undefined-correlation map). The posterior mean
   keeps selection and grading on **one coherent model**, and real GB1 fitness stays the external
   referee, so info-optimal can still lose. The same estimator runs on every method; only `revealed`
   differs.
4. `map_recovery(inferred, truth)` = correlation between inferred and true ε over all pairwise +
   third-order terms.

## Metrics

- **Primary (frozen):** Spearman and Pearson correlation between inferred and ground-truth ε
  coefficients, reported per order (pairwise, third) and pooled, at **B ∈ {48, 96, 192}**. Both
  correlations are always reported (never cherry-pick one). This is the number the decision rule reads.
- **Companion diagnostics (pre-registered, additive — never a replacement for the primary, never in the
  decision rule):** at each B, ~0.3–0.6% of third-order terms are informed at these budgets, so the
  full-set number is real but low-power. Alongside it we report (a) the **informed-union-subset**
  Spearman/Pearson — the same correlation restricted to terms with `loop(T) ∩ measured ≠ ∅` for at
  least one of the four methods, a set fixed from the *selections* only (no ground-truth peeking, no
  per-method cherry-picking, leakage-free); and (b) **`coverage_fraction`** per method/order, so an
  info-optimal win is auditable as breadth vs precision. We do **not** restrict by `|ε| > threshold`:
  that builds the evaluation set from the answer key (a p-hacking vector).
- **Pre-registered expectation:** pairwise (~1,822 terms; each order-2/3 measurement touches ≤3
  pairwise loop members) is the better-powered, decisive comparison; a third-order null at B ∈ {48, 96}
  is to be read as *underpowered at this order/budget*, not as "H1 false" (invariant #2).
- **Secondary:** hit-rate@B (fraction of the true top-fitness variants captured) — to demonstrate that
  chasing epistasis information does not catastrophically forfeit fitness discovery.
- **Effect size + uncertainty:** for each B, bootstrap the correlation (≥ 1000 resamples) and report the
  95% CI (`ci_method = "bootstrap-over-terms"` for the deterministic methods). The random baseline is
  averaged over ≥ 20 seeds with its own CI (`ci_method = "bootstrap-over-seeds"`, its genuine
  selection variance).

## Decision rule (frozen)

The decision reads **one fixed statistic**, named before any result exists: the **pairwise-order
Spearman AND pairwise-order Pearson** map-recovery correlation. (Pairwise is the better-powered order at
these budgets; pooling orders can be distorted by between-order separation, so pooled is a companion,
not the headline.) For H1 to be reported as **supported**, on that statistic, at a majority of the
tested budgets:

- `recovery(info) − recovery(fitness) > 0` with non-overlapping bootstrap 95% CIs, **and**
- `recovery(info) > recovery(random)` with non-overlapping CIs.

Both correlations must move the same way; a split (one supports, one does not) is reported as partial.
Otherwise the report headline is the observed relationship (partial, null, or negative), stated plainly,
with the same figures.

## Mandatory baselines

Every figure and table shows **info-optimal**, **fitness-greedy**, and **random** together. Dropping a
baseline to flatter a curve is a CLAUDE.md hard-limit violation. From v1.1, also report the real-practice
heuristic (top beneficial singles → all pairwise, cf. MULTI-evolve) as a fourth comparison; the frozen
decision rule below still concerns info vs fitness vs random.

## Reproducibility

- One command: `epibudget validate --dataset gb1_wu2016 --budgets 48,96,192 --seeds 20 --out report/`.
  For a tractable fast-model first pass, `--alphabet` restricts the per-site candidate set (recorded in
  the report's provenance); the full 20-letter alphabet is the headline.
- The run writes `report/<run_id>/metrics.json` (one row per method × budget, with per-order
  correlations, CIs, and coverage) and prints a rich summary; the figures are rendered by
  `notebooks/gb1_demo.ipynb` from that JSON. Every claim in the README or docs must trace to a
  `metrics.json` that exists — this is the artifact.
- Every run embeds `(model_id, seed, config, data checksum)` in the report.
- CI runs the same pipeline on the **35M** model over a reduced budget grid as a smoke test; the
  headline figure uses **650M**. A reproducible Jupyter notebook (`notebooks/gb1_demo.ipynb`) renders the
  headline figure from the saved report.

## Threats to validity (and mitigations)

| Threat | Mitigation |
|--------|------------|
| ε ≡ 0 from additive scoring | invariant #1 + `test_epsilon_not_identically_zero` |
| Selection leaks labels | selection code has no access to the DMS frame; enforced by module boundaries and a test |
| GB1 has only 4 positions | claim framed as *principle validation*; power comes from 20³ AA instantiations per triplet, not from many positions (see RESEARCH §4) |
| Overfitting the metric to one B | report all B; decision rule requires a majority |
| Inference step does the work, not selection | same `infer_epistasis` used for all three methods; only the *selected set* differs |
