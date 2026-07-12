# Spec: downstream-impact benchmark (`src/epibudget/downstream.py`)

Status: **amended**. The original spec was frozen 2026-07-11, before any downstream number existed. A
confirmatory `R=20 x K=5 x 20-seed` process started under that original spec on 2026-07-12 produced a
favorable smoke-direction signal on the `structural-fitness` contrast before it was stopped without writing
any artifact — no `downstream.json` exists from it. Review of that process surfaced implementation and
protocol deviations from the original spec (missing-partition handling that silently lowered the 16/20 sign
gate, no raw per-fold record trail, inner-CV fold count and alpha grid not actually present in the frozen
spec text, shared hyperparameters across regimes, missing mandatory ESM diagnostics, an unenforced
cache/provenance contract, and an order-dependent report). **Protocol amendment 1 below is frozen before
any confirmatory downstream number is read or interpreted.** The favorable smoke direction observed before
this amendment is explicitly non-decision-use (see below); this amendment does not select, adjust, or tune
any frozen value based on that direction.

This document does **not** replace the frozen historical GB1 map-recovery decision rule (`docs/VALIDATION.md`
§Outcome). It only concerns the separate, independently-decided downstream-impact benchmark.

## Protocol amendment 1 — frozen before confirmatory rerun

**Disposition of prior executions.** Every downstream execution before this amendment — the exploratory
smoke run(s) already on disk and the `R=20/K=5/20-seed` process stopped before writing any artifact — is exploratory
and non-decision-use. None of their numbers, including the favorable `structural-fitness` smoke direction,
informed any value frozen in this amendment. The next `downstream` run performed under this amended
protocol is a **confirmatory rerun**, not an untouched first test; it is reported as such in every artifact
it produces (`provenance.protocol_version` / `provenance.amendment_version`).

### Inner cross-validation (frozen)

- `n_inner_folds = 3`, fixed for every regime (full, main-only, no-triples, ESM-offset).
- **Fold-assignment algorithm** (identity-based, deterministic, order-independent input): for a training
  set of `n` identities under a fixed inner-fold salt
  `INNER_SALT = sha256(b"epibudget-downstream-inner:v1")`,
  1. compute `h(v) = sha256(f"{INNER_SALT}:{canonical_id(v)}")` for every training identity `v`;
  2. sort the training identities by `(h(v), canonical_id(v))`;
  3. assign `fold(v) = rank(v) % 3` where `rank` is the 0-indexed position in that sorted order.
  This is the same balancing mechanism as the outer folds (`assign_outer_folds`), applied without the
  outer per-order stratification (an inner training set mixes whatever orders the outer selection
  contains, so there is nothing to stratify by). The result is invariant to the input order of the
  training set (tested).
- **Fallback policy.** Inner CV requires all 3 folds to be non-empty (a genuine 3-way held-out split). If
  the training set has fewer than 3 members, or the balanced assignment above produces fewer than 3
  distinct non-empty fold labels, inner CV is not run: the fit falls back to the frozen strongest-shrinkage
  grid corner (`max(grid)` on every penalized axis) and the record's `alpha_fallback_reason` is set to
  `"training_set_too_small"` (n < 3) or `"insufficient_distinct_inner_folds"` (n >= 3 but < 3 distinct
  non-empty labels — only possible for a training set with fewer than 3 distinct identities, since the
  hash-sorted rank assignment is otherwise balanced within ±1). Two-fold CV is never silently substituted.
- **Alpha grids (frozen, chosen a priori, not from any observed result):** `alpha_main = [0.1, 1.0,
  10.0]`, `alpha_pair = [1.0, 10.0, 100.0]`. Both are 3-point log-spaced grids spanning two orders of
  magnitude of shrinkage. `alpha_pair`'s range
  starts one decade above `alpha_main`'s because the pairwise dictionary (2,166 columns) is far larger and
  far sparser per training point than the main-effect dictionary (76 columns) at the frozen budgets
  (B <= 192), so the same raw penalty value shrinks a pairwise coefficient's effective per-observation
  support far less than a main coefficient's; a higher starting penalty keeps the pairwise block in a
  comparable working range. These are the same values already used by the pre-audit code; they are kept
  unchanged here specifically because there is no record of them having been touched after any result was
  observed — freezing them in writing here (rather than continuing to treat them as an implementation
  default) is the amendment.
- **Main-effects-only regime.** Uses a **dedicated one-dimensional grid**: `grid_main` above, no
  `alpha_pair` search. The model is fit with `include_pairs=False` (no pairwise column is ever active), so
  a pairwise penalty has no effect on any prediction; searching it would waste computation without changing
  the fitted model. `AlphaChoice.alpha_pair` is `null` for this regime's records, marked
  `applicable=False`.
- **Tie-breaking rule.** Among grid points with equal held-out inner-fold `log1p`-MSE, prefer the point
  with the larger `alpha_main` first, then the larger `alpha_pair` (stronger shrinkage wins ties) —
  unchanged from the pre-audit code.
- **Criterion.** Mean squared error on the *held-out* inner fold only (never the training-fit error, which
  is monotone decreasing in penalty strength and would always pick the weakest shrinkage).

### Degenerate-metric policy (frozen)

- A Spearman/Pearson correlation is undefined (`None`) when fewer than 3 paired observations exist, or
  either side is constant (zero variance), or the computed statistic is `NaN`. `None` propagates through
  `S_macro` (either order undefined -> `S_macro` undefined) and is excluded from every mean and every
  corrected-CV/sign-consistency computation — it is never treated as zero.
- **A missing or degenerate required partition invalidates the registered decision gate; it never lowers
  the gate's denominator.** The sign-consistency and AUC-contrast gates require all `EXPECTED_PARTITIONS =
  20` partitions to be present and valid (`complete_partition_coverage = True`); if any of the 20 is
  missing or every fold within it degenerate, `decision_eligible = False` and the decision fields are
  `null` with `status = "insufficient_valid_partitions"`. This replaces the pre-audit behavior that
  recomputed the 80% sign threshold over however many partitions happened to survive.

### Primary robustness gate (frozen; replaces the corrected-CV interval as the primary gate)

The scientific claim under test is stability of the `structural - fitness` (and separately
`info - structural`) effect **across the 20 frozen GB1 partitions**, not frequentist generalization to
independent proteins — GB1's four sites and one assay cannot license the latter, and the corrected-CV
interval's train/test ratio is not naturally identified in this selection-then-training design (see
below). The primary gate is therefore reframed as partition-level robustness:

1. Compute the 5 paired within-partition fold effects (`K=5`) for the contrast.
2. Compute one mean paired effect per partition (average the up-to-5 fold deltas within it; a partition
   with a degenerate fold drops that fold from its own mean but is otherwise still usable — see below).
3. Require all 20 partitions to be represented in the aggregate (`complete_partition_coverage`).
4. Require at least 16 of 20 partition means to be strictly positive (`sign_positive >= 16`, `>` not `>=`
   0.0 — exact zero is not positive).
5. Require the global mean effect (over all valid fold-level deltas) to be positive.
6. Require the median partition-mean effect to be positive.
7. Require the global mean effect to exceed `MIN_STRUCTURAL_EFFECT_SIZE`.

**Minimum effect size: `MIN_STRUCTURAL_EFFECT_SIZE = 0.0`.** No prior study establishes what magnitude of
`S_macro`-AUC delta corresponds to a practically meaningful downstream improvement on this benchmark, and
freezing a non-zero number now — after having already seen a favorable exploratory smoke direction — would
be exactly the kind of after-the-fact threshold-picking this amendment exists to prevent. The gate is
therefore the weakest non-trivial bar (strictly positive), not a magnitude bar. A future amendment may
freeze a non-zero value if it is derived from something other than this benchmark's own results (e.g. a
pre-registered minimum-detectable-effect power calculation).

This structural gate is applied to the pre-specified learning-curve AUC contrast `structural - fitness`
(primary) over `B in {48, 96, 192}`. The ESM-prior gate (`info - structural` at `B=192` only) uses the same
`complete_partition_coverage`, sign (`>=16/20`), global-mean-positive, and median-positive requirements.

### Corrected-CV interval (demoted to a labelled sensitivity-only companion)

The Nadeau-Bengio-style corrected interval is retained **only** as a companion sensitivity analysis; it is
never the support/not-supported gate. Its train/test ratio (`n_test/n_train`) is not naturally identified
in this selection-then-training protocol (the "test" set is the outer held-out fold's measured members;
the "train" pool is a budget-limited *selected* subset of a much larger candidate universe, not an
independent sample from the same population the test set is drawn from), so two explicit, separately
labelled conventions are emitted instead of one authoritative ratio:

- `pool_ratio = |E_j measured| / |pool_j|` — outer held-out candidate count over the *selectable* pool
  size for that fold (identity-level, ignores the budget actually spent).
- `effective_label_ratio = |E_j measured| / effective_train_size` — held-out measured count over the
  *actual* number of labels the method's budget produced (post-missingness).

Each is reported with its own `n_test`, `n_train`, ratio, valid-effect count, sample variance, degrees of
freedom, Student-t critical value, and interval; either is `"unavailable"` if its denominator is undefined
(e.g. zero effective training size). Partitions/folds salted from the same universe are never described as
independent biological datasets, and the number of surviving observations is never used to lower the
registered 16/20 sign threshold (that threshold is fixed at 20 partitions regardless of how many produced
a valid effect).

## Protocol amendment 1 addendum — confirmatory protocol profile (frozen)

Without an explicit recipe check, `decision_eligible` could become `True`
merely because 20 partitions were present, even if the executed run used the wrong alphabet, budgets,
`K`, `max_order`, or seed count — 20 partitions of the wrong recipe is not a confirmatory result. A
single authoritative `ConfirmatoryProfile` (`downstream.py: CONFIRMATORY_PROFILE`) is frozen below and
checked **twice**: once at the CLI boundary (an early, operator-facing, non-blocking signal recorded in
`provenance.cli_protocol_profile_conforming`/`cli_protocol_profile_mismatches`) and again, defensively
and authoritatively, inside `_decision_summary` itself — so a direct `downstream_report()` call that
bypasses the CLI is protected identically.

**Frozen confirmatory profile:**

```text
protocol_version   = "epibudget-downstream-v1"
partitions         = 20
outer_folds        = 5
budgets             = (48, 96, 192)   # order-sensitive: learning_curve_auc integrates in this order
alphabet            = "ACDEFGHIKLMNPQRSTVWY"
max_order           = 3
random_seeds        = tuple(range(20))   # the seeds parameter is a count; the set is {0, ..., 19}
inner_folds         = 3
estimands           = {"target_blind", "target_aware"}
missingness_regimes = {"attempted_budget", "measured_available"}
methods             = {"info", "structural", "fitness", "random", "practice"}
```

`budgets` is compared by exact ordered-list equality (a reordering that preserves the set is still
flagged, since the AUC contrast trapezoidal-integrates over `budgets` in the given order). The other
sequence fields are compared as sets. No value is ever coerced toward the profile — a mismatch is
always reported, never silently accepted or truncated.

**Three distinct non-decision statuses**, replacing the single generic
`status = "insufficient_valid_partitions"` for every non-eligible case:

- `status = "nonconforming_protocol_profile"` — the executed configuration does not match the frozen
  profile in some dimension other than partition count alone (wrong alphabet, budgets, `K`, `max_order`,
  seed count, or missing estimand/regime/method coverage — e.g. a regime silently skipped because its
  pool was smaller than the max budget). `decision_eligible = False`, `supported = None`.
- `status = "smoke_or_exploratory_profile"` — every profile dimension matches **except**
  `partitions` and/or `random_seeds`, both below (never above) the frozen register — a deliberate
  exploratory/smoke execution that reduces scale but keeps the same recipe (e.g. `R=1, K=5, one
  seed`, full alphabet, real budgets). Any mismatch on an identity field (alphabet, budgets, `K`,
  `max_order`, ...) or an *oversized* `partitions`/`random_seeds` is `nonconforming_protocol_profile`
  instead. `decision_eligible = False`, `supported = None`.
- `status = "insufficient_valid_partitions"` — the declared profile conforms exactly (including
  `partitions = 20`) and the raw records contain no unexpected, duplicated, or wrongly-versioned cell, but
  one or more required raw-record cells are missing, or the partitions built from what is present show
  wholly degenerate coverage within the valid `{0, ..., 19}` register (the raw-record coverage rule below extends
  this from partition-aggregate-level degeneracy to individual missing raw-record cells).
  `decision_eligible = False`, `supported = None`.

A profile mismatch always overrides whatever `robustness_gate` independently computed from partition
coverage — even a coincidentally-complete 20/20 coverage under the wrong recipe never reads as
confirmatory. Descriptive fields (`sign_positive`, `global_mean_delta`, `median_partition_delta`, ...)
are left untouched by the override, so a nonconforming or smoke run still reports honest descriptive
metrics; only `decision_eligible`/`supported`/`status` change.

**Extra partitions (`--partitions > 20`).** `partition_aggregates_for` already only ever builds a
`PartitionAggregate` for `p in range(EXPECTED_PARTITIONS)`, so an extra partition was already excluded
from `sign_positive`/`median_partition_delta`/`complete_partition_coverage`. This exclusion did **not** extend to
`robustness_gate`'s `all_deltas` argument (used for
`global_mean_delta`/`effect_size_pass`), which read every fold-instance delta regardless of partition
index — an extra partition's data would leak into the global mean while being excluded from the sign
count. `_global_deltas_within_expected_partitions` now restricts `all_deltas` to the same
`{0, ..., 19}` register before it reaches `robustness_gate`, for both the structural and ESM gates. A
run with `partitions > 20` is additionally flagged `nonconforming_protocol_profile` by the profile
check above (never `smoke_or_exploratory_profile`, which is reserved for `partitions < 20`).

`DecisionSummary` now also serializes `expected_protocol_profile`, `observed_protocol_profile`,
`protocol_profile_mismatches`, and `protocol_profile_conforming`, so an auditor can see exactly what
was compared without recomputing it.

## Protocol amendment 1 addendum — raw-record coverage validation (frozen)

The profile check above compares *declared* configuration
(the `partitions`/`seeds` arguments, `PROTOCOL_VERSION`) against the frozen register, but never validates
that the *raw records actually present* fully and exactly cover that declaration. A missing random-seed
record, a duplicated record, a record carrying the wrong `protocol_version`, or an extra out-of-register
partition record could all leave `protocol_profile_conforming = True`, because nothing independently
reconstructed the expected set of raw-record cells and checked observed coverage against it.

`raw_record_coverage` (`downstream.py: RawRecordCoverage`) closes this gap. It independently builds the
expected deterministic and random record-key sets from the frozen/declared profile alone — never from the
observed records — using key tuples `(protocol_version, partition_index, estimand, missingness_regime,
method, budget, fold_index)` for deterministic records and the same plus `random_seed` for random records.
Comparison is multiplicity-aware (`collections.Counter`, not a bare `set`), so a duplicate record is
detected even when the unique-key set is unchanged. `registered_records()` is the single canonical scope
(unexpected-key records dropped, duplicates deduped) that every scientific summary — `method_budget`,
partition aggregates, corrected-CV companions, and both robustness gates — now consumes; no summary
function independently re-filters records with its own logic. Unexpected records remain visible in the raw
`deterministic_records`/`random_records` collections and in `raw_record_coverage`'s counts/samples for
forensic auditability; they are simply excluded from every registered scientific quantity.

`protocol_profile_conforming` is now `declared_protocol_profile_conforming AND
raw_record_coverage_conforming`, both serialized on `DecisionSummary` alongside `raw_record_coverage`
itself (expected/observed deterministic and random counts, missing/duplicate/unexpected cell counts,
bounded canonical-order samples of the offending keys, and the observed protocol-version/partition/seed
sets). Status precedence: `nonconforming_protocol_profile` (any unexpected, duplicated, or wrongly-versioned
raw cell; a declared-smoke profile whose raw records do not exactly cover that smaller declaration; or any
declared-identity mismatch) takes precedence over `insufficient_valid_partitions` (declared profile exact,
raw coverage otherwise clean, but cells are missing), which takes precedence over `smoke_or_exploratory_profile`
(declared profile is a clean, smaller-only reduction and the raw records exactly cover it), which takes
precedence over the gate's own `ok` status. A replacement seed (wrong seed, correct total count) is always
`nonconforming_protocol_profile`, never `insufficient_valid_partitions` — the two are deliberately
distinguished, since one is silent substitution and the other is honest absence.

## Protocol amendment 1 addendum — divergent vs exact duplicates (frozen)

A duplicated registered cell (two or more raw records sharing a `_det_key`/`_rand_key`) is now split into
two cases by comparing the **complete canonical record payload**, not the identity key alone and never a
single metric such as `s_macro`:

- **Exact duplicate** — every copy is byte-identical. It collapses deterministically to one representative
  (`_collapse_exact_duplicates`), so multiplicity never alters any scientific summary. The cell remains
  `nonconforming_protocol_profile` under the frozen duplicate policy above; identical duplicate collapse is
  not permitted to read as conforming.
- **Divergent duplicate** — the copies differ in any scientific or provenance-bearing field. This cell is
  scientifically ambiguous: no independent trusted identity says which record is authoritative, so **no
  arbitrary record is ever selected** from the group (the pre-fix lexical-`min` rule could publish an
  appended extreme record and depended on order/sign). Whenever any divergent duplicate is present the run
  fails closed: `status = nonconforming_protocol_profile`, `decision_eligible = false`, `supported = null`,
  and **every registered scientific summary is unavailable** — `method_budget`, partition aggregates,
  corrected-CV companions, both robustness gates' descriptives, and the report's
  `scientific_summaries_available = false` / `scientific_summaries_unavailable_reason =
  "divergent_duplicate_raw_record"`. Divergent-duplicate precedence overrides a co-occurring missing cell
  (which alone would be `insufficient_valid_partitions`). The raw records and the coverage diagnostics
  (`exact_duplicate_*_key_count`, `divergent_duplicate_*_key_count`, bounded divergent-key samples,
  `has_divergent_duplicate`) remain for forensic auditing.

The deterministic and random forensic raw-record arrays are canonically ordered before serialization.
Their registered record key defines the primary order, with the complete canonical record payload as the
tie-breaker for records sharing a key. This ordering does not select an authoritative record and is never
used for scientific aggregation.

This is a fail-closed integrity rule, not an additional statistical estimand: it prevents an ambiguous
record set from producing a misleading scientific summary; it never adds, adjusts, or reinterprets any
frozen decision quantity. The result is byte-identical under any input-record ordering.

## Why

The frozen headline grades each method on map-recovery correlation over epsilon terms. That metric is
partly tautological (`docs/LIMITATIONS.md` par.4): a method can score well by *measuring* many terms
(breadth) and the inferrer keeps the ESM prior for unmeasured terms (`validate.py`,
`mu[v] = revealed[v] if measured else b*esm[v]`). So the recovery number does not show that a
structure-aware plate leads to a **better downstream experimental decision**. This benchmark asks a
different, non-tautological question:

> At equal initial budget B on GB1, does a method's selected plate provide a better training set for a
> fixed supervised learner to rank **held-out** double/triple mutants?

The primary predictor is trained **only** from the fitness labels the budget reveals, and consumes neither
the held-out variant's own ESM score nor the prior-inclusive `infer_epistasis` output — so it cannot
recover the ESM prior algebraically (the new, less-visible tautology this design must avoid).

## Claimable scope

A positive result may state: *"At equal initial budget on GB1, structure-aware selection provides a better
training set for a fixed pairwise ridge learner to rank held-out double and triple mutants than
fitness-greedy, across 20 salted partitions of this one landscape."* It may **not** state that epibudget is
an active-learning system, that the plate yields generally better wet-lab decisions, or that the result
generalizes beyond GB1. The benchmark is retrospective, single-landscape, single-assay,
single-primary-learner, one-step, with no sequential selector update — even a fully positive confirmatory
result remains all of these (see the final report's "Remaining limitations").

## Non-goals

- Not a new selection method, not a change to `infer_epistasis`/`map_recovery`/`run_validation`'s primary
  role (the ESM-circular diagnostic below reuses `infer_epistasis` but never feeds it back into selection).
- Not a formal frequentist CI over future wet-lab campaigns — the corrected-CV object is a labelled
  sensitivity companion, not a claim of generalization.
- Not GPU work: runs on the already-computed `ScoredVariant`s (`load_cache`) — no torch, no model.
- No new heavyweight dependency: the estimator is a pure-numpy ridge.

## Data flow

Inputs (all post-run): `scored: Sequence[ScoredVariant]` (the complete order-1..3 universe, from
`scored_cache.load_cache`, validated against its sidecar), `landscape: Mapping[Variant, float]`
(full GB1 `{variant -> fitness}` from `data.load_gb1`, including dead-0 rows and the WT), `budgets`,
`seeds`, `n_folds`, `partitions`. `scored` is canonicalized (sorted by `canonical_id`) at the engine
boundary before any selection, graph construction, or random sampling reads it, so an arbitrary input
order never changes a selection, a record, or the report. Output: a `DownstreamReport` (pydantic)
written **atomically** to `<out_dir>/downstream.json`, plus a printed summary.

## Reused utilities (import, do not duplicate)

From `acquisition.py`: `allocate`. From `validate.py`: `fitness_greedy`, `structural_graph`,
`random_selection`, `practice_heuristic`, `infer_epistasis` (diagnostic only). From `epistasis.py`:
`predicted_epistasis`. From `graph.py`: `EpistasisFactorGraph`. From `data.py`: `reveal_measured_fitness`,
`GB1_SITES`, `GB1_WT_AT_SITES`. From `scored_cache.py`: `load_cache`, `candidate_sha256`,
`validate_cache_against_universe`. From `provenance.py`: `write_json_atomic`, `workspace_code_diff_sha256`.
The five selection methods are recomputed exactly as `run_validation` does; only their `candidates`
argument is restricted (see pool_j).

## Held-out protocol

### Outer folds (deterministic, order-stratified, SHA-256, label-free)

`canonical_id(v) = json.dumps(sorted([list(m) for m in v]), separators=(",",":"))`.
`partition_salt(i) = sha256("epibudget-downstream-v1:" + str(i))`, for i in 0..partitions-1 (frozen). For a
salt, over the **entire order-2/3 candidate universe** (29,602 variants; singles are never held out),
group by mutation order; within each order sort by `(sha256(f"{salt}:{canonical_id(v)}"), canonical_id(v))`;
assign `fold = rank % n_folds`. Balanced +/-1 per order, reorder-stable, and derived **only** from
identity — never from a fitness value, live/dead status, or missingness.

For fold j: `E_j` = the fold-j order-2/3 identities; `pool_j` = universe \ E_j.

### Estimands (both run; never interchangeable)

- **target-blind (PRIMARY).** `predicted_epistasis` over the full universe, then drop interactions keyed
  by an `E_j` identity before constructing the factor graph.
- **target-aware (MANDATORY companion).** Keep all interactions; `E_j` labels are never accessible and
  `E_j` is still excluded from the selectable pool.

### Missingness regimes (both run)

- **attempted-budget (PRIMARY).** Selectable = `pool_j` over the full universe.
- **measured-available oracle (MANDATORY sensitivity).** Selectable = `pool_j intersect measured
  identities`.

## Primary predictor (global fixed feature space) — regime-separated hyperparameters

Pure-numpy deterministic ridge. Response `y = log1p(raw fitness)`. Features: a single global fixed
dictionary (76 main + 2,166 pairwise reference-coded indicators; no third-order; no ESM feature). Solve via
the generalized dual (`beta = Lambda^-1 X^T (I_n + X Lambda^-1 X^T)^-1 y_c`, an n x n solve). See
Protocol amendment 1 above for the frozen grids/fold mechanism.

**Regime-separated hyperparameter tuning.** Four independent models are fit per
(method, fold, budget), each with its **own** inner-CV-selected alpha(s) on its **own** training rows —
none reuses another regime's alpha:

1. **Full** (main + pairwise, all revealed labels) — the primary `S_macro` predictor.
2. **Main-effects-only** (main only, all revealed labels, `include_pairs=False`) — used for the
   epistasis-uplift diagnostic (`S_macro`(full) - `S_macro`(main-only)).
3. **No-triples-training** (main + pairwise, revealed labels restricted to selected order <= 2) — used
   for the no-triples-to-triples transfer test.
4. **ESM-offset supervised** (main + pairwise, response `y - b * scaled_esm`, same revealed labels as (1))
   — an always-on diagnostic, never decisional.

## Secondary predictors / controls (mandatory; never in the decision rule)

1. **`esm_circular_diagnostic`** — the prior-inclusive `infer_epistasis`-derived prediction; for a held-out
   variant it collapses to `b*esm[v]` (b fit on training labels), demonstrating the tautology the primary
   predictor is designed to avoid. The report and this spec state explicitly: predictions for held-out
   variants under this diagnostic may reintroduce their own ESM prior and therefore cannot establish
   learned downstream information.
2. **`esm_zero_shot_no_budget`** — raw zero-shot ESM `delta_g` ranking over `E_j`-measured; a no-budget
   control, never associated with or repeated as if it consumed budget `B`.
3. **`esm_offset_supervised`** — regime (4) above; answers "can a little supervision correct the ESM prior
   better than the pure ridge?" Reported, never decisional.

None of the three diagnostics may enter the structural claim, the ESM-uncertainty-prior claim, the primary
robustness gate, sign consistency, or the primary `S_macro`/AUC.

**Two invariants, replacing an earlier overstated claim.** The former text asserted
that changing ESM values changes only the three diagnostic fields — false: `delta_g`/`var_delta_g` are
legitimate acquisition inputs for several ESM-dependent selection methods, so changing them may legitimately
change which identities are selected, the resulting training plate, that method's downstream primary
metrics, and the final cross-method comparison. That is intended behavior, not a leak.

- **Invariant A (clean-predictor isolation).** At a fixed selected plate and fixed revealed labels, the
  clean supervised ridge predictor (regime 1 above) never consumes any ESM-derived feature, prior,
  variance, or offset; replacing the ESM values used elsewhere leaves its predictions and metrics
  bit-for-bit identical. Verified by `test_clean_predictor_is_invariant_to_esm_at_fixed_selection_and_labels`,
  which fixes the plate/labels, evaluates twice under the real and an extreme synthetic ESM mapping without
  re-running acquisition, and asserts every clean-model field is unchanged while at least one ESM diagnostic
  moves.
- **Invariant B (serialized diagnostic isolation).** At fixed raw primary records, changing only the
  serialized ESM diagnostic fields (`esm_circular_s_macro`, `esm_zero_shot_s_macro`, `esm_offset_s_macro`)
  never changes primary summaries, partition aggregates, robustness gates, or decision summaries. Proven by
  `test_esm_diagnostic_fields_never_feed_the_decision_pipeline`.

Neither invariant claims acquisition-time ESM inputs are inert: methods whose selection logic reads
`delta_g`/`var_delta_g` may legitimately select a different plate under different ESM values.

## Metrics

- **Primary statistic** `S_macro = 1/2 * (rho_doubles + rho_triples)`.
- **Mandatory components** `rho_doubles`, `rho_triples`, reported separately.
- **Companion** pooled-order Spearman (never decisional).
- Pearson and `log1p`-RMSE.
- **Decision-relevant secondary (frozen definitions):**
  - `NDCG@B` (`K=B`): relevance = deterministic percentile rank of raw held-out fitness within
    `E_j`-measured (ties averaged, zeros retained); gain = relevance; discount = `1/log2(rank_0indexed+2)`
    (standard DCG discount, rank 0-indexed at the top); ranking and the ideal ordering both broken by
    `canonical_id` ascending on a predicted-value tie; an all-tied-relevance evaluation set yields
    `NDCG = 1.0` by convention (every ordering is equally ideal, `IDCG == DCG` for any permutation of ties).
  - `hit_rate@B`, `best_true@topB` (max true fitness within the predicted top-B), `regret@B`
    (`best_available - best_true@topB`), `live_fraction@topB`.
  - `top_B_order_diversity` (count of distinct mutation orders present in the predicted top-B) and
    `top_B_identity_diversity` (count of distinct `canonical_id` in the predicted top-B; equals `min(B,
    n_eval)` unless predictions are degenerate/duplicated, in which case ties can still yield fewer than B
    distinct ranks under a shared discount — reported to detect that case, not to reject it).
- **Epistasis-uplift:** `S_macro`(full) - `S_macro`(main-only), each with its own regime-separated alpha.
- **No-triples training -> held-out-triples transfer:** train on the method's selected singles
  and doubles **only** (every selected triple is excluded from this sub-test's training rows; singles and
  doubles are retained), fit the same main+pairwise model with its own inner-CV alpha, evaluate
  `rho_triples` on `E_j` triples. Reports `train_singles_count`, `train_doubles_count`, and a
  `degenerate_double_coverage` warning when too few doubles remain to fit meaningfully
  (`train_doubles_count < 3`). Decisional only at `B in {96, 192}`; always reported at every budget.

## Statistical plan

See Protocol amendment 1 (primary gate) above; corrected-CV is a companion only.

Repeated salted partitions: `R` partitions (default 20) x `K` folds (default 5). Random baseline: seed
variance kept separate from partition/fold variance (raw per-seed records preserved — never
pre-averaged before serialization).

Reported pairs: `structural-fitness` (primary), `info-structural` (ablation), `structural-random`,
`practice-structural` (companion), on the target-blind primary estimand and the attempted-budget primary
regime; the same pairs are reported for the target-aware estimand and measured-available regime.

## Decision rule (see Protocol amendment 1 "Primary robustness gate")

- **Structural downstream supported** iff `complete_partition_coverage` and the 7-point partition-level
  robustness gate above all pass for `structural-fitness` on the `S_macro` learning-curve AUC over
  `B in {48,96,192}`.
- **ESM-uncertainty contribution supported** iff the same gate (all 7 points) passes for `info-structural`
  on `S_macro` at `B=192` only (`B in {48,96}` reported, non-decisional — underpowered by construction).
- If `decision_eligible` is `False` (any of the 20 partitions missing/degenerate), both fields are `null`
  with `status = "insufficient_valid_partitions"` — never computed over a reduced denominator.
- Otherwise the observed partial/null/negative is reported. All three narrative outcomes are preserved:
  (1) structural beats fitness downstream; (2) info does not beat structural; (3) nothing beats fitness.
  Secondary metrics and the three ESM diagnostics may explain but never override this rule. This does not
  modify the frozen historical GB1 map-recovery rule.

## Raw record schema

The report serializes immutable raw records **before** any aggregation; every summary, effect, partition
aggregate, and decision field is a pure function of these records (tested by exact reconstruction).
Reconstruction requires three serialized pieces together, not raw records alone: the raw records, the
protocol profile, and the raw-record coverage rule (`raw_record_coverage` below)
that determines which records are in the registered scope a given summary is a pure function of.

- `DeterministicFoldRecord` — exactly one row per
  `estimand x missingness_regime x partition x fold x method x budget`
  for `{info, structural, fitness, practice}`. Expected count per (estimand, regime):
  `R x K x 4 methods x 3 budgets`.
- `RandomFoldSeedRecord` — exactly one row per
  `estimand x missingness_regime x partition x fold x budget x random_seed`.
  Expected count per (estimand, regime): `R x K x len(seeds) x 3 budgets`. Deterministic methods are never
  repeated across seeds as if they were independent records.
- `PartitionAggregate` — one row per `(estimand, regime, method_a, method_b, statistic, partition)`: the
  mean paired fold-level delta within that partition, valid-fold count, and whether the partition is
  usable for the primary gate.
- `DecisionSummary` — `expected_partitions`, `observed_valid_partitions`, `complete_partition_coverage`,
  `decision_eligible`, the 7-point gate's individual booleans, and the final `supported` (`bool | None`)
  with `status`.

Required fields per fold record (deterministic or random): protocol version, estimand, missingness regime,
partition index and salt, fold index, fold identity hash, method, budget, random seed or null, selected
identity count, selected-identity hash, revealed count, live count, dead-zero count, missing count,
unusable count, effective training size, training live fraction, selected/training singles/doubles/triples
counts, fallback status and reason per regime, `alpha_main`/`alpha_pair` per regime (full, main-only,
no-triples, esm-offset), inner-fold count actually used, primary `S_macro`, `rho_doubles`, `rho_triples`,
pooled Spearman, Pearson, `log1p` RMSE, NDCG and the secondary metrics, uplift, no-triples-transfer
metrics, the three ESM diagnostics, execution status, and warnings.

## No leakage / determinism (enforced at the engine boundary)

- Selection recomputation reads only `scored` (ESM) + seeds — never a label; `pool_j` excludes `E_j`; the
  primary predictor's feature builder consumes no ESM field and never calls `infer_epistasis` (AST-guarded,
  scoped to the primary-path functions listed in the test — the ESM diagnostics are a distinct,
  explicitly-labelled, non-primary path that is allowed to call it).
- `scored` is canonicalized by `canonical_id` at the top of `downstream_report` before any selection, graph
  construction, or random sampling reads it; an arbitrary permutation of the input `scored` sequence
  produces a byte-identical report (timestamps and other explicitly-permitted runtime fields excepted).
- Fold assignment, feature indexing, and any seeded step take canonical, sorted inputs (subprocess
  `PYTHONHASHSEED` test).
- Labels enter only via `reveal_measured_fitness`, after selection.

## Cache integrity

`scored_cache.load_cache`/the CLI gate reject (never warn) on: unparseable lines, duplicate canonical
identities within the JSONL (detected explicitly, not silently collapsed by dict insertion order), a
missing sidecar, or a sidecar whose `candidate_sha256`, `candidate_count`, `candidate_alphabet`,
`max_order`, `model_id`, `scorer_seed`, `n_perturbations`, or WT hash does not match the CLI's requested
universe. Expected counts for the frozen alphabet: 76 singles, 2,166 doubles, 27,436 triples, 29,678 total.
Expected values for all eight checked fields come from `validate_cache_against_universe`'s own
independently-computed `CacheIdentity` (never read back from the sidecar under check);
both the expected and observed `CacheIdentity` are serialized in full in `downstream` provenance
(`scored_cache_identity_expected`/`scored_cache_identity_observed`) and alongside `robustness.json` as
`robustness_cache_identity.json`, so an auditor can see every checked field, not only three of them.

## Provenance

`protocol_version`, `amendment_version`, `execution_commit`, `base_commit`, `workspace_code_diff_sha256`,
the list of changed scientific files covered by that hash, cache/sidecar/dataset/universe/headline
SHA-256, every partition salt, per-fold identity hashes, the exact alpha grids, the exact inner-fold
policy, the selected alpha per raw record, estimands, missingness regimes, budgets, seeds, the exact
command including every option, start/completion timestamps, and a `status` of
`provisional | final | invalidated`. The final report is written atomically: a fully-written, flushed,
fsynced temp file in the same directory is published by creating a hard link at the final path
(`os.link`, create-only on both POSIX and Windows) rather than by renaming — `rename` would silently
replace an existing destination on POSIX, which a check-then-rename sequence cannot prevent under
concurrent writers. An interrupted write never leaves a partial `downstream.json` at
the final path, an existing final path is never replaced, and of two concurrent writers exactly one
succeeds while the other receives `FileExistsError`.

## Module API (signatures)

See `src/epibudget/downstream.py` docstrings for the authoritative signatures; this section is
intentionally not duplicated in prose to avoid drifting from the implementation (the pre-amendment spec's
literal signature block had already drifted from the shipped code).

## Test plan (`tests/test_downstream.py`, offline, synthetic — no ESM, no network)

Deterministic + order-stratified + reorder-stable outer folds; balanced (not modulo) inner folds,
order-independent, with the frozen 3-fold-or-fallback policy; sentinel-label substitution changes neither
folds, pool, selection, nor predictions; no `pool_j intersect E_j` overlap; identical `E_j` and feature
space across methods; five methods retained in both estimands and regimes, never repeated across random
seeds as independent deterministic records; primary predictor consumes no `delta_g`/`var_delta_g` and
never calls `infer_epistasis` (AST guard, scoped to primary-path functions); the ESM diagnostics
positively call `infer_epistasis`; at a fixed selected plate and fixed revealed labels, extreme ESM values
leave every clean-model field identical while moving at least one diagnostic field (Invariant A); at fixed
raw records, corrupting only the serialized diagnostic fields leaves every primary summary, aggregate, and
decision quantity identical (Invariant B); generalized-dual ridge is PD/solvable on an all-singles design; inner CV selects a non-trivial
alpha on a held-out criterion; regime-separated alpha selection produces different optima on a constructed
fixture where full/main-only/no-triples optima genuinely differ; dead-0 retained through `log1p`;
missing/dead/live accounting; degenerate-training fallback recorded with a reason; NDCG@B correctness and
tie handling and the all-tied convention; no-triples transfer excludes triples from training while
retaining singles/doubles; missing-partition adversarial cases (4 positive and 16 missing gives not
eligible; 16 positive and 4 zero gives eligible with sign count exactly 16; 15 positive and 5 negative
fails the sign gate; 20 positive passes); corrected-CV exact formula on a hand fixture, now reported as a
`sensitivity_only` companion with
both ratio conventions; frozen-salt + sign-rule determinism (subprocess `PYTHONHASHSEED`); report is
byte-identical (except timestamps) under an arbitrary permutation of `scored`; raw records reconstruct
every aggregate and decision field exactly; report schema + provenance fields present. The default suite
stays offline.

## Verification

`pytest -q tests/test_downstream.py` green (offline); `pytest -q tests/test_cli.py` green (offline);
`mypy --strict src/` clean; `ruff format --check .` and `ruff check .` clean.
`scripts/validate_artifacts.py` still green (no public number added by this amendment). A permitted minimal
smoke (`R=1, K=5`, one seed, full alphabet, budgets 48/96/192) validates schema, raw-record counts,
aggregate reconstruction, and provenance only; it must report `decision_eligible = false` /
`status = "smoke_or_exploratory_profile"` (`partitions` and `random_seeds` are the only profile
mismatches, both below the frozen register) and its biological direction is not interpreted or
prominently reported. The confirmatory `R=20` run is out of
scope for this amendment and requires separate authorization.
