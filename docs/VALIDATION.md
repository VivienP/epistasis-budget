# Validation protocol

This document is the normative protocol for `epibudget` validation. It defines the claim, data
conditioning, label boundary, metrics, decision rules, baselines, and confirmatory profiles. Run
narratives and numerical evidence belong in the linked records.

## GB1 map-recovery claim

> **H1.** At equal budget *B*, variants selected by information-optimal allocation (`--lambda 0`)
> recover the ground-truth epistasis map of GB1 better than the same budget spent fitness-greedily
> (`--lambda 1`), and better than random.

The null is that information-optimal is indistinguishable from, or worse than, fitness-greedy. In this
protocol, "information-optimal" denotes the v1 modular loop-bracing heuristic: an A-optimal trace
reduction under a diagonal, uncalibrated prior. It is not a claim of globally optimal design.

## Dataset and conditioning

- The GB1 landscape covers positions **V39, D40, G41, V54**. Its theoretical space contains `20^4 =
  160,000` genotypes.
- The public-data artifact contains 149,361 measured rows: 119,884 positive-fitness rows, 29,477 zero-fitness
  rows, and 10,639 theoretical genotypes absent from the artifact.
- `python scripts/fetch_gb1.py` fetches the data explicitly, records its checksum, and leaves it git-ignored.
- Fitness becomes `log(fitness / fitness(reference))`, making the assayed reference the zero anchor.
- A ground-truth term requires every loop member to be present with positive, log-transformable fitness.
- GB1 recovery is conditional on that subset, not the entire 160,000-genotype theoretical landscape.
- Pairwise and third-order terms remain separate for every decision. Cross-order pooling is diagnostic.

## Ground truth and budget simulation

`ground_truth_epistasis(wt_centered_log_fitness(landscape))` computes WT-referenced pairwise and third-order
inclusion-exclusion coefficients. The Walsh-Hadamard spectrum is contextual, not a replacement target.

For each method and budget:

1. Enumerate the registered order-1 through order-3 candidate universe.
2. Score conjointly: apply every mutation before reading conditional log-likelihoods.
3. Select `B` variants from ESM-derived scores and the factor graph without measured fitness.
4. Call `data.reveal_measured_fitness` only after the selected identities are fixed.
5. WT-centre the revealed values and run the same `infer_epistasis` estimator for every method.
6. Compare inferred and ground-truth coefficients with `map_recovery`.

`infer_epistasis` is the posterior mean of the linear-Gaussian graph model. A measured variant pins its
WT-centred value; an unmeasured member retains the calibrated ESM prior mean. It is identical across methods.

## Label boundary

Measured fitness labels may enter only through `data.reveal_measured_fitness`, after selection. Enumeration,
scoring, graph construction, acquisition, folds, and random sampling must not read labels or their statistics.

Post-selection evaluation may use revealed labels for recovery, hit rate, calibration, and downstream
training. Label-derived parameters and grading subsets must not feed back into selected identities.

## Map-recovery metrics

- **Primary:** pairwise-order Spearman and Pearson correlation between inferred and ground-truth epsilon
  at `B in {48, 96, 192}`.
- **Required order split:** report Spearman and Pearson separately for pairwise and third-order terms.
  Pooled recovery is a compatibility diagnostic only.
- **Breadth:** `coverage_fraction`, `n_informed`, and `n_pinned` report terms touched or fully measured by
  the selected plate.
- **Precision:** `pearson_predicted` and `spearman_predicted` are computed over informed but unpinned
  terms. They distinguish prediction of unmeasured loop members from direct recovery by coverage.
- **Fitness companion:** `hit_rate@B` is computed after selection over the candidate universe.
- **Uncertainty:** deterministic methods use 1,000 term-resampling bootstrap replicates for 95% intervals;
  random uses at least 20 seeds and bootstraps the per-seed recovery values.
- Undefined correlations remain `None`; they are never converted to zero.

Pairwise recovery is decision-bearing because it is better powered at these budgets. A third-order null,
especially at `B=48` or `B=96`, is reported as an order-and-budget limitation, not substituted into the
pairwise decision.

## Frozen map-recovery decision rule

H1 is **supported** only when both pairwise Spearman and pairwise Pearson satisfy, at a majority of the
three budgets:

- `recovery(info) - recovery(fitness) > 0` with non-overlapping bootstrap 95% intervals; and
- `recovery(info) > recovery(random)` with non-overlapping bootstrap 95% intervals.

Both correlations must move in the same direction. A split between Pearson and Spearman is partial.
Failure to meet every support condition is reported as the observed partial, null, or negative result.
The decision rule is not changed by companion analyses.

## Mandatory map-recovery baselines

Every map-recovery table and figure includes all five methods at every budget:

- **info:** ESM masking-perturbation dispersion weighted by the number of interaction loops braced;
- **fitness:** fitness-greedy ranking from the zero-shot ESM score;
- **random:** seeded random selection over the same candidate pool;
- **practice:** predicted beneficial singles followed by their cross-site pairwise combinations; and
- **structural:** the same modular allocation with constant variance, isolating loop coverage from ESM
  masking dispersion.

The frozen H1 comparison uses info, fitness, and random. Practice and structural are mandatory framing
companions. The ESM uncertainty prior earns a contribution claim only if info exceeds structural under the
registered comparison; otherwise that contribution is omitted.

## Confirmatory 650M map-recovery profile

The confirmatory universe uses the full 20-letter alphabet: 76 singles, 2,166 doubles, 27,436 triples,
and 29,678 candidates in total. A reduced alphabet, model, budget grid, perturbation count, seed count, or
baseline set is exploratory or supplementary.

The exact GB1 command is:

```bash
epibudget validate --dataset gb1_wu2016 --model esm2_t33_650M --alphabet ACDEFGHIKLMNPQRSTVWY --budgets 48,96,192 --seeds 20 --n-perturbations 16 --device cuda --out report/
```

The command writes `report/<run_id>/metrics.json` and records the dataset, model, scorer seed,
`n_perturbations`, device, alphabet, candidate universe, and data checksum. A completed scored cache may
be supplied with `--scored-cache`; its sidecar identity must match the requested universe.

The TrpB map-recovery profile uses `trpb_johnston2024` with the same model, alphabet, budgets, seeds,
perturbation count, baselines, metrics, and decision rule. Its reference is the assayed Tm9D8* parent
`VFVS` at positions 183, 184, 227, and 228. TrpB fitness values are conditioned the same way as GB1;
871 of 160,000 source values are imputed and unflagged in the public mirror, so any result must retain
that limitation.

**TrpB status: IN PROGRESS.** TrpB 650M scoring with `n_perturbations=16` is in progress outside the
repository; it is not yet a map-recovery or downstream result.

## Current GB1 map-recovery decision

The corrective GB1 decision is `inconclusive_zero_gpu`, with `public_claim_eligible=false`. Pairwise
Pearson and Spearman improve at every registered budget, while relative squared-error gain is negative at
every budget. The ESM-dispersion contribution against seeded structural ties is inconclusive, and no
registered calibration contrast reverses sign at the required number of budgets.

This mixed result supports no current public map-recovery winner. The frozen headline artifact remains an
historical record; it does not override the corrective decision. See [R&D record](RND_RECORD.md) and the
[registered artifacts](../artifacts/README.md) for historical evidence and provenance.

## Downstream-impact benchmark

The downstream benchmark is separate from map recovery. Its authoritative protocol is
[`docs/specs/downstream.md`](specs/downstream.md); this summary does not amend it.

The primary question is whether a method's selected plate trains a fixed pairwise-ridge learner to rank
held-out double and triple mutants. The primary score is
`S_macro = 0.5 * (Spearman_doubles + Spearman_triples)`. The learner reads revealed training labels but no
held-out ESM score and no prior-inclusive `infer_epistasis` output.

The confirmatory downstream profile is `epibudget-downstream-v1`: 20 partitions, 5 outer folds, budgets
`(48, 96, 192)` in that order, full alphabet, `max_order=3`, `n_perturbations=16`, random seeds 0 through
19, 3 inner folds, both registered estimands and missingness regimes, and all five methods.

For the structural-minus-fitness learning-curve AUC gate, all 20 partition means must be valid, at least
16 must be strictly positive, and the global mean, median partition mean, and effect-size check must be
positive. The frozen minimum effect size is `0.0`. The info-minus-structural gate at `B=192` uses the same
coverage and sign requirements. Corrected-CV intervals are sensitivity companions, not decision gates.

Missing, duplicate, unexpected, wrongly versioned, or divergent raw-record cells fail closed under the
status precedence defined in the authoritative spec. A nonconforming recipe cannot become
decision-eligible from favorable descriptive values.

The current amended GB1 downstream report is decision-eligible and sets
`structural_downstream_supported=true`; the ESM uncertainty contribution is not supported. The report is
provisional and local, so this status does not change the public map-recovery verdict. The TrpB
`n_perturbations=16` downstream result is pending that scoring cache; no result is claimed. Run evidence is
indexed in [the downstream experiment record](experiments/trpb-downstream-generalization-20260716.md).

## Threats to validity

| Threat | Required mitigation |
|---|---|
| Additive scoring makes predicted epsilon identically zero | Score every mutation conjointly and retain `test_epsilon_not_identically_zero` |
| Selection leaks measured labels | Enforce the `reveal_measured_fitness` boundary and label-substitution tests |
| Missing or dead loop members bias ground truth | Require complete positive-fitness loops and state the conditioning |
| Pool exhaustion inflates recovery | Use the full 29,678-candidate universe for confirmatory runs |
| One budget drives the conclusion | Report all three budgets and require a majority |
| Coverage is mistaken for prediction | Report breadth and unpinned-term precision separately |
| Calibration or tie-breaking drives the result | Use corrective seeded ties and method-independent calibration companions |
| A baseline is dropped after inspection | Report all five methods at every budget |
| One landscape is generalized broadly | Decide GB1 and TrpB separately; never pool them |
| Downstream diagnostics feed the primary learner | Keep ESM diagnostics outside primary features and decisions |

## Evidence and routing policy

- `docs/VALIDATION.md` defines normative map-recovery settings and summarizes the current decision only.
- `docs/specs/downstream.md` is authoritative for every downstream profile, gate, integrity rule, and status.
- `docs/RND_RECORD.md` stores design rationale, amendments, audits, and historical evidence.
- `docs/experiments/` stores run-specific analyses, including exploratory and non-decision-use work.
- `artifacts/` stores registered public evidence; `scripts/validate_artifacts.py` checks its claim mapping.
- `report/` stores generated and often git-ignored run outputs. Presence there does not make a claim public.
- `README.md` states only the current public-facing result and links here for the protocol.

Any empirical number must route to a reproducible artifact or a cited experiment record. A provisional,
ignored, exploratory, smoke, or in-progress result must retain that status wherever it is mentioned.
