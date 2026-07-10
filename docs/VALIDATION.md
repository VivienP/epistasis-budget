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

- **Factual correction (2026-07-10; the decision rule is unchanged).** The four-site genotype space at
  **V39, D40, G41, V54** contains 20⁴ = 160,000 theoretical genotypes. The local public-data artifact
  contains 149,361 measured rows: 119,884 with positive fitness, 29,477 dead rows with fitness zero,
  and 10,639 genotypes absent from the artifact.
- Access via **ProteinGym** (substitution DMS assays include GB1) and/or the original supplementary
  data. Fetching is explicit and lives in `scripts/fetch_gb1.py`; data is **never committed** (see
  `.gitignore`). The script records a checksum of the downloaded file.
- Ground-truth ε is computed only when every required loop member is present and has positive,
  log-transformable fitness. Results are therefore conditional on a measurable positive-fitness subset,
  not representative of the entire theoretical genotype space.

## Ground truth

`ground_truth_epistasis(measured_live_landscape)` computes, from positive measured fitnesses:

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
- **Breadth vs precision (pre-registered, additive — never a replacement for the primary, never in the
  decision rule):** a method can score well on the full-set correlation for two very different reasons —
  it *measured* many terms directly (breadth), or it *predicted* the unmeasured ones well (precision).
  Info-optimal front-loads low-order variants, so it directly measures (pins) a large fraction of the
  pairwise terms; that inflates full-set recovery for a boring reason. To separate the two we report, per
  method/order, all fixed from the *selections* only (leakage-free, no `|ε|>threshold` answer-key
  restriction): (a) **`coverage_fraction`** and **`n_informed`** — terms this method's selection touches;
  (b) **`n_pinned`** — terms whose *entire* loop is measured, hence recovered *exactly* (the pure breadth
  count); and (c) **precision correlation** (`pearson_predicted` / `spearman_predicted`) — Spearman/Pearson
  over the terms this method *informs but does not fully pin*, i.e. where it must genuinely predict the
  ε from the ESM prior plus partial measurements. Breadth is `n_pinned`; precision is the predicted-term
  correlation. A real, non-tautological info-optimal advantage must show up in **precision**, not only in
  breadth.
- **Pre-registered expectation:** pairwise (~1,822 terms at the full alphabet; each order-2/3 measurement
  touches ≤3 pairwise loop members) is the better-powered, decisive comparison; a third-order null at
  B ∈ {48, 96} is to be read as *underpowered at this order/budget*, not as "H1 false" (invariant #2).
  Because the modular `info_gain = τ²·n(v)` weight is dominated by `n(v)` (a single sits in ~1140 loops),
  info-optimal will structurally front-load singles then doubles and may measure *no triples* at these
  budgets — an honest, expected property to report, not hide.
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
baseline to flatter a curve is a CLAUDE.md hard-limit violation. Also reported at every budget:

- **practice** — the real-practice heuristic (top beneficial singles → all pairwise, cf. MULTI-evolve).
- **structural-only** — the ablation that isolates what the ESM uncertainty prior actually contributes:
  the same modular sort with `τ² ≡ const`, so selection ranks purely by `n(v)` (how many loops a variant
  braces) and the ESM masking-perturbation dispersion plays no role. **If info-optimal ≈ structural-only,
  the ESM uncertainty prior does nothing to the allocation and must be dropped from the claims; if
  info-optimal > structural-only, that gap is the contribution.** This must be run and reported before any
  headline framing.

The frozen decision rule still concerns info vs fitness vs random; practice and structural-only are
reported companions that determine how the result is framed.

## Headline regime (pre-registered)

The reduced-`--alphabet` fast pass runs in an *exhaustion regime* (small pool, so info-optimal directly
measures most doubles) and is a smoke test only. The **headline run is frozen to the full 20-letter
alphabet** (`pool ≫ B`: ~76 singles, ~2,166 doubles, ~27,436 triples), so at B ∈ {48, 96, 192} no method
can trivially measure the whole map and a win cannot be an artefact of pool exhaustion. The alphabet,
model (650M), budgets, seed count, and the baseline set above are fixed here **before** the headline
result exists.

## Reproducibility

- The frozen run requires all scientific settings explicitly:
  `epibudget validate --dataset gb1_wu2016 --model esm2_t33_650M --alphabet
  ACDEFGHIKLMNPQRSTVWY --budgets 48,96,192 --seeds 20 --n-perturbations 16 --device cuda --out
  report/`. A reduced model, alphabet, budget grid, perturbation count, or baseline set is a smoke or
  supplementary run, never the headline.
- The run writes `report/<run_id>/metrics.json` (one row per method × budget, with per-order
  correlations, CIs, and coverage) and prints a rich summary; the figures are rendered by
  `notebooks/gb1_demo.ipynb` from that JSON. Every claim in the README or docs must trace to a
  `metrics.json` that exists — this is the artifact.
- Every run embeds `(model_id, seed, config, data checksum)` in the report.
- CI runs the same pipeline on the **35M** model over a reduced budget grid as a smoke test; the
  headline figure uses **650M**. A reproducible Jupyter notebook (`notebooks/gb1_demo.ipynb`) renders the
  headline figure from the saved report.

## Post-registration robustness analyses — 2026-07-10

**Status: implemented in `src/epibudget/robustness.py` (run via `epibudget robustness` on a completed
scored cache; spec in `docs/specs/robustness.md`). No number from these analyses exists anywhere in this
repo yet — they wire into public artifacts only after the real 650M run.** They do not alter or replace
the frozen statistic or decision rule above. This section is written after the Step-1 signal, the
650M masking-variance calibration, and the 650M deterministic supplementary recovery were already
computed and committed (`docs/LIMITATIONS.md` §1, §5) — the qualitative shape of those results (the
uncertainty prior looking unhelpful; structural-only beating random and fitness-greedy) was visible when
these three devices were chosen. So this section cannot claim the bias-protection of a blind
pre-registration; it only fixes the method before its own numbers exist, which still blocks tuning the
method to a specific number once computed.

- **Common identities.** Full-set correlations use the same complete-loop truth terms for every method.
  Method-specific precision remains descriptive because its eligible terms differ. Direct precision
  comparisons use only the intersection of terms for which both methods produce an informed,
  non-pinned prediction. Terms without an informed prediction contribute to coverage counts only; no
  Pearson or Spearman correlation is computed for them. Caveat: this intersection is not a neutral
  subsample — a term is informed more easily the larger its loop (7 members at third order vs. 3 at
  pairwise, `interaction_loop`), and info-optimal's `n(v)`-driven hub bias and fitness-greedy/practice's
  high-ΔG bias both concentrate coverage on the same popular positions, so "both methods inform it"
  correlates with a term's loop size and structural popularity, not only with predictive skill.
- **Method-independent scale sensitivity.** A deterministic five-fold partition is defined from variant
  identities before any labels enter analysis. Fold-specific through-origin slopes are fit from positive,
  log-transformable fitness values outside the held-out fold and reused identically for all methods; a
  multi-member term (e.g. a third-order loop) converts each member with that member's own fold's slope,
  not one slope for the whole loop. These labels and live/dead states are post-selection analysis inputs
  and are never available to a selector. Results using them are explicitly conditional on positive
  measurable fitness. Constraint this assumes (stated, not argued): sharing one slope across methods
  removes a per-method small-sample calibration confound, but assumes the ESM-to-measured-fitness linear
  relationship is homogeneous across whatever subpopulation each method's selection happens to leave
  unmeasured — the same kind of assumption `_calibrate_slope` already states explicitly as a constraint
  for its own through-origin convention.
- **Paired differences.** Correlation differences are bootstrapped on identical terms. Random comparisons
  report term resampling, seed resampling, and a hierarchical bootstrap that draws a fresh term-resample
  for each resampled seed (seed variance nested outside term variance). This inherits the leverage-
  concentration caveat already stated for term bootstraps (`docs/LIMITATIONS.md`, "Confidence intervals
  measure two different things"): nesting seeds and terms does not make the terms independent. Separate
  confidence intervals are never treated as a test of a
  direct difference — a stricter standard than the frozen decision rule's own non-overlapping-CI
  criterion above; that rule remains the frozen bar for H1, and this stricter standard applies only to
  these companion analyses, not to it.

## Threats to validity (and mitigations)

| Threat | Mitigation |
|--------|------------|
| ε ≡ 0 from additive scoring | invariant #1 + `test_epsilon_not_identically_zero` |
| Selection leaks labels | selection code has no access to the DMS frame; enforced by module boundaries and a test |
| GB1 has only 4 positions | claim framed as *principle validation*; power comes from 20³ AA instantiations per triplet, not from many positions (see RESEARCH §4) |
| Overfitting the metric to one B | report all B; decision rule requires a majority |
| Inference step does the work, not selection | same `infer_epistasis` used for all three methods; only the *selected set* differs |
