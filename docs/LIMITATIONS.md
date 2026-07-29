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

- **Non-positive scores are training data downstream, and excluded from the log-ratio upstream.**
  TrpB has 35,643 negative labels and GB1 has 29,477 exact zeros. These are score-sign classes, not
  biological activity calls. All TrpB values are `> -1`, so `log1p` is defined and they are valid
  downstream training rows; the earlier filtering silently discarded them. The map-recovery
  log-ratio is undefined at `f <= 0`, so its eligible population is declared separately and before
  selection.

- **Non-positive and missing genotypes have different roles.** Non-positive rows remain downstream
  training data. The recovery log-ratio requires strictly positive scores, and some genotypes are
  absent; an interaction whose inclusion-exclusion loop touches either case has no eligible recovery
  ground truth. This restriction is mathematical and does not classify biological activity.

- **The Walsh-Hadamard spectrum requires a complete tensor.** The real GB1 grid is incomplete, so the
  implementation rejects it for this calculation. Spectrum tests use complete synthetic grids.

- **TrpB has a different evidence boundary.** The source paper reports 871 imputed fitness values, but
  the public mirror does not identify them row by row. They therefore cannot be excluded individually;
  every TrpB result must retain this conditioning caveat.

## Model

- **The acquisition score is a prior trace reduction, not an information gain.** With independent
  variant noise, `weight(v) = var_delta_g(v) * n(v)` does not depend on previous selections, so
  allocation is a fixed ranking. It is an A-optimal trace reduction under a diagonal, uncalibrated
  masking-dispersion prior — not an expected information gain, and it contains **no loop-closure
  term**: at B=48 on the four-site universe the objective-optimal plate identifies exactly zero
  interaction contrasts, and a plate that fully identifies 16 of them scores strictly lower.

- **The coverage score is three-valued on a four-site landscape.** `n(v)` is exactly 1140 for every
  single, 39 for every double and 1 for every triple, so `structural` reduces to "singles, then
  doubles, then triples" and every within-order comparison is a tie. A declared
  `tie_seed` now makes a selection reproducible, but the tracked artifacts do not estimate a
  distribution over tie seeds.

- **Epistasis is WT-referenced.** Background-averaged epistasis and a MoCHI handoff are not implemented.
  Results should not be interpreted as estimates of ensemble epistasis across genetic backgrounds.

- **Uncertainty propagation assumes independent score errors.** Interaction variance sums the component
  `var_delta_g` values and omits covariance between related mutant contexts. The direction of the error
  is unknown; the model is not claimed to be conservative.

- **Recovery retains the ESM prior.** Revealed variants are pinned to measured WT-centred log fitness;
  unmeasured loop members retain a through-origin-calibrated ESM estimate. Recovery therefore measures
  a measurement-plus-prior estimator, not measurements alone.

## Metrics and inference

- **Map recovery is confounded by the purchased contrast component.** The inferred and the
  true contrast both contain `k(S)`, the signed sum of the measured loop members' true values. The
  breadth/precision split does *not* remove it: once all singles are bought, essentially every term is
  informed-not-pinned and still carries `k(S)`. The corrected schema uses `relative_sse_gain` as the
  wording gate and the skeleton-controlled association as a diagnostic, but no corrected-run artifact
  is tracked yet; see the [evaluation notes](AUDIT_REMEDIATION_20260728.md).

- **Pairwise and third-order results have different power.** They are reported separately. The pooled
  correlation is diagnostic only and cannot replace an order-specific decision.

- **Precision sets differ by method.** The decision-bearing `validate.map_recovery` split computed each
  method's precision on *its own* informed-not-pinned terms and compared the two as if they estimated
  the same quantity. Only the post-hoc robustness suite intersected them.
  `recovery.common_term_subset` now evaluates a comparison on terms in the same state for both
  methods, with the size and SHA-256 recorded.

- **Confidence intervals have different sampling meanings.** A bootstrap over interaction terms
  measures term leverage conditional on the selected plates; it does not measure selection
  variability. Recovery schema v3 emits per-realisation paired differences and reports term-leverage
  intervals separately from summaries across realised selections.

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

- **The historical downstream evidence favored structural allocation, not masking dispersion.** The
  v1 gates favored the particular structural plates over fitness-greedy on GB1 and TrpB, while neither
  landscape passed the incremental masking-dispersion gate. These observations do not estimate the
  structural method over tie seeds. The compact result is tracked in
  [`structural_allocation_650m.json`](../artifacts/structural_allocation_650m.json), remains provisional,
  and is not used as a promotional result; see the
  [evaluation notes](AUDIT_REMEDIATION_20260728.md).

- **The earlier TrpB downstream run is exploratory.** It used `n_perturbations=0`, so it cannot evaluate
  masking dispersion and is not decision-eligible. Its historical interpretation is recorded in
  [`trpb-downstream-generalization-20260716.md`](experiments/trpb-downstream-generalization-20260716.md).

- **The TrpB 650M `n_perturbations=16` profile is complete but its map-recovery interpretation is
  withdrawn.** The frozen gate returned a positive comparison against fitness-greedy and random, but
  the shared-skeleton confound prevents that result from supporting recovery. See
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
