# Constraints and limitations

This document states the boundaries that affect interpretation of `epibudget` results. The current
protocol and result status live in [`VALIDATION.md`](VALIDATION.md); tracked evidence is indexed by
[`artifacts/manifest.json`](../artifacts/manifest.json).

## Compute and execution

- **CPU is supported; large variance-inclusive runs are GPU workloads.** Tests cover CPU execution and
  the CLI supports `--device cpu|cuda|auto`. The complete 650M GB1 headline used a GPU and is recorded in
  [`headline_650m.json`](../artifacts/headline_650m.json). No complete CPU duration is published.

- **Scoring performance depends on batching and deduplication.** The scorer batches masked forwards and
  reuses identical masked inputs. These optimizations must remain score-equivalent to the per-variant
  reference; the measured benchmark scope is recorded in
  [`bench_35m.json`](../artifacts/bench_35m.json) and
  [`bench_650m.json`](../artifacts/bench_650m.json).

- **Small-model and reduced-alphabet runs are smoke tests.** They exercise the pipeline quickly but do
  not replace the full-alphabet 650M profile. At small candidate-pool sizes, the evaluated budgets can
  approach pool exhaustion and overstate the advantage of broad low-order coverage.

## Data

- **GB1 is one four-site landscape.** Its local artifact contains 149,361 measured genotypes from a
  theoretical grid of 160,000. Replication comes from amino-acid combinations, not from many independent
  protein positions, so conclusions do not establish whole-protein positional generality.

- **Dead and missing genotypes are not imputed.** Fitness-zero rows cannot be log-transformed, and some
  genotypes are absent. Any interaction whose inclusion-exclusion loop touches an unavailable value has
  no recoverable ground truth and is excluded. This restricts evaluation to viable, complete loops.

- **The Walsh-Hadamard spectrum requires a complete tensor.** The real GB1 grid is incomplete, so the
  implementation rejects it for this calculation. Spectrum tests use complete synthetic grids.

- **TrpB has a different evidence boundary.** The source paper reports 871 imputed fitness values, but
  the public mirror does not identify them row by row. They therefore cannot be excluded individually;
  every TrpB result must retain this conditioning caveat.

## Model

- **The v1 acquisition score is modular.** With independent variant noise,
  `info_gain(v) = var_delta_g(v) * n(v)` does not depend on previous selections. Allocation is therefore
  a fixed ranking, not a general diminishing-returns or correlated-posterior design.

- **Loop coverage strongly favors low mutation orders.** The number of interaction loops braced by a
  single or double mutant is much larger than for a triple. This structural factor dominates the
  across-order ranking; masking dispersion mainly changes rankings within an order.

- **Epistasis is WT-referenced.** Background-averaged epistasis and a MoCHI handoff are not implemented.
  Results should not be interpreted as estimates of ensemble epistasis across genetic backgrounds.

- **Uncertainty propagation assumes independent score errors.** Interaction variance sums the component
  `var_delta_g` values and omits covariance between related mutant contexts. The direction of the error
  is unknown; the model is not claimed to be conservative.

- **Recovery retains the ESM prior.** Revealed variants are pinned to measured WT-centred log fitness;
  unmeasured loop members retain a through-origin-calibrated ESM estimate. Recovery therefore measures
  a measurement-plus-prior estimator, not measurements alone.

## Metrics and inference

- **Map recovery combines breadth and prediction.** An interaction is exact when its complete loop has
  been measured. Reports must separate pinned-loop breadth from correlation on informed but unpinned
  terms so that direct measurement is not presented as predictive accuracy.

- **Pairwise and third-order results have different power.** They are reported separately. The pooled
  correlation is diagnostic only and cannot replace an order-specific decision.

- **Precision sets differ by method.** Direct method comparisons use the intersection of terms informed
  by both methods. Uninformed terms contribute to coverage, not to an invented correlation value.

- **Confidence intervals have different sampling meanings.** Random-baseline intervals include
  selection variability across seeds. Deterministic-method bootstrap intervals resample interaction
  terms and measure leverage in the correlation, not repeated-budget variability.

- **Calibration slopes can dominate sparse selections.** When no loop member is measured, the inferred
  interaction is a scaled ESM prior. A method-specific slope can therefore determine the sign of a
  low-coverage correlation. Shared cross-fit slopes are attribution diagnostics, not an operational
  selection method.

- **One downstream circularity diagnostic uses the wrong calibration scale.** Its `log1p(fitness)` labels
  do not match the WT-centred log-fitness contract of `esm_prior_mu`. It is excluded from decision use
  until corrected; the downstream prediction target itself is unchanged.

## Current evidence boundary

- **Conjoint scoring produces non-additive signal; masking dispersion is not validated as an error
  proxy.** The tracked signal and calibration artifacts support these as separate conclusions:
  [`signal_650m.json`](../artifacts/signal_650m.json) and
  [`calibration_650m.json`](../artifacts/calibration_650m.json). At 650M, Spearman is −0.113 with a 95%
  interval [−0.220, −0.002], while Pearson is −0.100 with [−0.198, 0.003]. This is weak negative rank
  association, not evidence of positive calibration or a general anti-calibration claim.

- **The corrective GB1 recovery decision is inconclusive.** The relevant result remains
  `inconclusive_zero_gpu` with `public_claim_eligible=false`. Neither an advantage nor a disadvantage of
  masking dispersion is a public claim.

- **The downstream evidence supports structural allocation, not masking dispersion.** The registered
  gates support structural selection over fitness-greedy on GB1 and TrpB. Neither landscape supports the
  added ESM masking-dispersion weight. The compact result is tracked in
  [`structural_allocation_650m.json`](../artifacts/structural_allocation_650m.json) and remains provisional.

- **The earlier TrpB downstream run is exploratory.** It used `n_perturbations=0`, so it cannot evaluate
  masking dispersion and is not decision-eligible. Its historical interpretation is recorded in
  [`trpb-downstream-generalization-20260716.md`](experiments/trpb-downstream-generalization-20260716.md).

- **The TrpB 650M `n_perturbations=16` profile is complete but provisional.** Its map-recovery result
  supports `info` over fitness-greedy and random, while the structural ablation shows that masking
  dispersion does not carry the gain. See
  [`trpb-650m-n16-20260723.md`](experiments/trpb-650m-n16-20260723.md).

- **The earlier TrpB recovery smoke run is not confirmatory.** Its old WT anchoring invalidates its
  recovery coefficients and truth-map summaries. Selection identities, coverage, hit rate, and run
  configuration remain descriptive only; see
  [`trpb-smoke-20260713.md`](experiments/trpb-smoke-20260713.md).

## Out of scope

- interaction orders above three;
- background-averaged epistasis;
- multi-round or sequential experimental design;
- a second PLM or learned surrogate;
- distributed or multi-GPU execution;
- a hosted API or web service.
