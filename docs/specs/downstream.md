# Downstream-impact benchmark

Implementation: `src/epibudget/downstream.py`. This is the effective protocol after amendment 1.

The original protocol was frozen on 2026-07-11. A confirmatory process started under that version was
stopped before it wrote an artifact; its observed smoke direction is non-decision-use. Amendment 1 froze
the integrity rules, complete profile, and partition-level gate before the confirmatory rerun. Historical
rationale and review chronology remain in [`RND_RECORD.md`](../RND_RECORD.md).

This benchmark is separate from the GB1 map-recovery decision in
[`VALIDATION.md`](../VALIDATION.md). It does not amend that decision.

## Question and claim boundary

At equal initial budget, does a method's selected plate train a fixed supervised learner to rank held-out
double and triple mutants better than fitness-greedy selection?

The primary predictor uses only labels revealed by the selected plate. It does not consume a held-out
variant's ESM score, masking variance, or `infer_epistasis` output. A positive result is limited to the
landscape named in the report, the fixed learner, and the registered one-step retrospective protocol. It
does not establish active learning, general wet-lab utility, or transfer to another protein.

## Inputs and output

Inputs are:

- the complete order-1 through order-3 `ScoredVariant` universe and its sidecar;
- a landscape mapping variants to raw fitness, including zero-fitness rows;
- the budgets, random seeds, outer folds, and partition count.

The engine canonicalizes scored variants by `canonical_id` before selection or sampling. It writes one
`DownstreamReport` atomically and exclusively to `<out_dir>/downstream.json`; an existing final path is
never replaced.

The five methods are recomputed from the scored cache with the same allocation utilities as validation:
`info`, `structural`, `fitness`, `random`, and `practice`.

## Confirmatory profile

`CONFIRMATORY_PROFILE` is checked at the CLI boundary and again inside the decision engine.

```text
protocol_version   = "epibudget-downstream-v1"
partitions         = 20
outer_folds        = 5
budgets            = (48, 96, 192)          # exact order; AUC uses this order
alphabet           = "ACDEFGHIKLMNPQRSTVWY"
max_order          = 3
n_perturbations    = 16
random_seeds       = tuple(range(20))
inner_folds        = 3
estimands          = {"target_blind", "target_aware"}
missingness_regimes = {"attempted_budget", "measured_available"}
methods            = {"info", "structural", "fitness", "random", "practice"}
```

Budgets are compared as an ordered list; the other sequence-valued dimensions are compared as sets.
No value is coerced toward the profile. `n_perturbations` is an identity field: any value other than 16
is nonconforming, including zero. The profile is landscape-blind, so eligibility applies only to the
dataset recorded by that report.

## Held-out design

### Outer folds

For partition `i`, the salt is `sha256("epibudget-downstream-v1:" + str(i))`. Double and triple identities
are stratified by mutation order, sorted by `(sha256(salt + ":" + canonical_id), canonical_id)`, and assigned
by `rank % outer_folds`. Singles are never held out. Assignment is balanced within one item per order,
input-order invariant, and label-free.

For outer fold `j`, `E_j` is the held-out double/triple set and `pool_j` is the remaining universe.

### Estimands

- **target-blind (primary):** remove interactions keyed by `E_j` before building the selection graph;
- **target-aware (mandatory companion):** keep those interactions, while still withholding their labels
  and excluding their variants from the selectable pool.

### Missingness regimes

- **attempted-budget (primary):** select from the entire `pool_j`;
- **measured-available (mandatory oracle sensitivity):** select only identities with an available label.

Both estimands and both regimes run for every method and budget. They are never substituted for one
another.

## Inner cross-validation

Each training set is sorted by `(sha256(INNER_SALT + ":" + canonical_id), canonical_id)`, where
`INNER_SALT = sha256("epibudget-downstream-inner:v1")`, and assigned by `rank % 3`. All three folds must
be non-empty. Fewer than three training identities fall back to the
strongest-shrinkage grid corner with `training_set_too_small`; any other failure to produce three folds
uses `insufficient_distinct_inner_folds`. Two-fold CV is never substituted.

Frozen grids:

```text
alpha_main = (0.1, 1.0, 10.0)
alpha_pair = (1.0, 10.0, 100.0)
```

The criterion is mean held-out `log1p` MSE. Ties prefer larger `alpha_main`, then larger `alpha_pair`.
The main-only model searches only `alpha_main` and records `alpha_pair=null`.

## Learners and diagnostics

The primary learner is deterministic NumPy ridge on `y = log1p(raw_fitness)`, with a fixed reference-coded
dictionary of 76 main-effect and 2,166 pairwise columns. It contains no third-order or ESM feature.

Four regimes tune their own hyperparameters on their own training rows:

1. main plus pairwise effects on all revealed labels (primary);
2. main effects only (epistasis-uplift companion);
3. main plus pairwise effects without selected triples (triple-transfer companion);
4. supervised residuals around an ESM offset (diagnostic only).

Mandatory non-decision diagnostics are:

- `esm_circular_diagnostic`, which may reintroduce a held-out variant's own ESM prior;
- `esm_zero_shot_no_budget`, a raw zero-shot control that consumes no budget;
- `esm_offset_supervised`, the residual learner above.

The circular diagnostic currently has the calibration-scale limitation recorded in
[`LIMITATIONS.md`](../LIMITATIONS.md). None of these fields may enter primary summaries, gates, or
decisions. At a fixed plate and labels, changing ESM values leaves the clean learner unchanged. At fixed
raw primary records, changing only diagnostic fields leaves every aggregate and decision unchanged.

## Metrics

Primary statistic:

```text
S_macro = 0.5 * (Spearman_doubles + Spearman_triples)
```

Either undefined order makes `S_macro` undefined. A correlation is `None` with fewer than three paired
observations, a constant input, or a NaN result. Undefined values are never replaced by zero.

Required companions include the two order-specific Spearman values, pooled-order Spearman (diagnostic),
Pearson, `log1p` RMSE, `NDCG@B`, hit rate, best true fitness in the predicted top B, regret, live fraction,
order and identity diversity, epistasis uplift, and no-triples-to-triples transfer. The triple-transfer
statistic is decision-relevant only at B=96 and B=192 and warns when fewer than three training doubles
remain.

Random-seed variability remains separate from partition/fold variability. Raw per-seed records are never
pre-averaged before serialization.

## Registered decision gate

The registered seven-step construction is:

1. compute the five paired fold deltas within each partition;
2. average the valid fold deltas into one mean per partition;
3. require all 20 partitions to be present and usable;
4. require at least 16 partition means to be strictly positive;
5. require the global mean over valid fold deltas to be positive;
6. require the median partition mean to be positive;
7. require the global mean to exceed `MIN_STRUCTURAL_EFFECT_SIZE = 0.0`.

The structural decision applies this gate to the learning-curve AUC of `structural - fitness` across
B=(48, 96, 192). The masking-dispersion decision applies the same gate to `info - structural` at B=192.
The B=48 and B=96 masking-dispersion contrasts are descriptive.

`MIN_STRUCTURAL_EFFECT_SIZE` remains zero because no external minimum meaningful effect was available
when the amendment was frozen. It is a strict-positive gate, not a magnitude claim.

Corrected-CV intervals are sensitivity companions only. Two separately labelled conventions are emitted:

- held-out measured count divided by selectable-pool size;
- held-out measured count divided by effective revealed-training size.

Each records its counts, ratio, variance, degrees of freedom, t critical value, and interval. Neither can
override the partition gate or reduce its 20-partition denominator.

## Raw records and fail-closed integrity

The report serializes one deterministic record per estimand, regime, partition, fold, deterministic
method, and budget; and one random record per estimand, regime, partition, fold, budget, and random seed.
Every scientific summary is reconstructed from these records plus the protocol profile and coverage rule.

Expected keys are generated independently of observed records. Multiplicity-aware coverage detects
missing, unexpected, duplicated, wrongly versioned, out-of-register, and replacement-seed cells.
`registered_records()` is the only input to scientific aggregation.

- Exact duplicate payloads collapse to one representative for descriptive aggregation, but the report
  remains nonconforming.
- Divergent duplicates select no representative. All scientific summaries become unavailable and the
  report records `divergent_duplicate_raw_record`.
- Unexpected records remain serialized for audit but never enter a scientific quantity.
- Extra partitions never enter partition aggregates or global deltas.

Raw records and forensic samples are canonically ordered, so input order cannot change the report.

Status precedence is:

1. `nonconforming_protocol_profile` for identity mismatches, unexpected/wrong/replacement cells, any
   duplicate, or an incompletely covered declared smoke profile;
2. `insufficient_valid_partitions` for an exact declared profile with otherwise clean coverage but
   missing or wholly degenerate required cells;
3. `smoke_or_exploratory_profile` for an exactly covered run whose only reductions are fewer partitions
   and/or fewer random seeds;
4. the registered gate result.

A nonconforming or incomplete report has `decision_eligible=false` and `supported=null`. Descriptive
values may remain available unless a divergent duplicate makes them ambiguous.

## Leakage, cache, and provenance

- Selection reads only scored variants and seeds. Labels enter through `reveal_measured_fitness` after
  identities are fixed.
- Fold assignment, feature indexing, and seeded operations use canonical sorted inputs.
- The primary feature path cannot call `infer_epistasis` or consume `delta_g`/`var_delta_g`.
- Cache loading rejects malformed lines, duplicate identities, missing sidecars, and mismatches in
  candidate hash/count/alphabet, max order, model, scorer seed, perturbation count, or WT hash.
- The full-alphabet cache must contain 76 singles, 2,166 doubles, and 27,436 triples.
- Provenance records protocol/amendment versions, execution and base commits, scientific diff hash,
  cache and dataset identities, salts, alpha policy and choices, estimands, regimes, command, timestamps,
  and `provisional | final | invalidated` status.

## Verification contract

Offline tests cover outer and inner fold balance, label substitution, pool/holdout separation, all methods,
estimands and regimes, ridge solvability, hyperparameter isolation, missing/dead/live accounting, metric
degeneracy, NDCG ties, triple transfer, all gate boundary cases, corrected-CV formulas, raw-record
reconstruction, profile mismatches, duplicate handling, cross-process determinism, cache completeness,
diagnostic isolation, and exclusive atomic writes.

Run:

```bash
pytest -q tests/test_downstream.py tests/test_cli.py
python scripts/validate_artifacts.py
```

A reduced-scale smoke must report `smoke_or_exploratory_profile`; its biological direction is not a
decision result.
