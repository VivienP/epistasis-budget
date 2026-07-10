# Spec: Phase B robustness analyses (`src/epibudget/robustness.py`)

Status: spec for implementation, revised after an adversarial design-review panel (epistasis-theorist,
scientific-validator, reviewer — all three: *sound with fixes*, no redesign). Implements the three
pre-registered analyses named in `docs/VALIDATION.md` §"Post-registration robustness analyses —
2026-07-10". These are **post-hoc**: they run on a completed run's inputs (the ESM-scored candidates +
the full measured landscape) and never feed selection. They do not alter or replace the frozen decision
rule. Review findings are folded in below and tagged `[n]`: serialized (not comment-only) disclaimers
[1], A1 within-common depth reporting [3], `sorted(common)` cross-process reproducibility [5], paired-row
alignment [7], CLI cache completeness/order guard [8][9], pinned hierarchical procedure [6]; the sole
BLOCKING (A3 "wrong estimand") was refuted — both readings centre on `delta`, only CI width differs.

## Why

The frozen headline reports, per method, a recovery correlation over that method's own evaluated terms.
Three questions the frozen statistic leaves open, each a companion analysis here:

1. **Coverage vs precision.** Method A can score a higher full-set correlation just by *informing* more
   terms (breadth), not by predicting unseen ε better (precision). A fair precision comparison must use
   the **same terms** for both methods.
2. **Scale-fitting confound.** `infer_epistasis` fits one through-origin slope `b` on each method's own
   revealed set (`validate.py:144`). Methods that reveal differently get different `b`; a recovery gap
   could be a slope-fitting artifact rather than a selection difference.
3. **CI overlap ≠ difference test.** The frozen rule reads non-overlapping per-method CIs. That is a
   conservative proxy; a **paired** bootstrap of the difference on identical terms measures it directly.

None of this changes any frozen number; it characterises what the frozen number does and does not show.

## Non-goals

- Not a new selection method, not a change to `infer_epistasis`/`map_recovery`/`run_validation`.
- Not a formal hypothesis test. Difference CIs are descriptive (documented as such).
- Not GPU work: the module imports no torch and needs no model — only already-computed `ScoredVariant`s.

## Data flow

Inputs (all already available post-run):
- `scored: Sequence[ScoredVariant]` — the candidate universe with ESM `delta_g`, `var_delta_g`. In
  practice loaded from the completed Colab cache via `scored_cache.load_cache(path)` (read-only; no
  metadata needed to read).
- `landscape: Mapping[Variant, float]` — full GB1 `{variant → fitness}` from `data.load_gb1`.
- `budgets`, `seeds`, `max_order` — the frozen grid.

The module recomputes each analysed method's selection deterministically from `scored` (reusing
`allocate`, `fitness_greedy`, `structural_graph`, `random_selection`) — identical to `run_validation`,
so the analysed selections are the frozen ones. The pairs analysed are info/fitness/structural/random;
`practice_heuristic` is a `run_validation` companion not carried into Phase B. Truth terms come from
`ground_truth_epistasis` restricted to `_candidate_terms(scored, max_order)`, exactly as
`run_validation` builds `truth_by_term`.

Output: a `RobustnessReport` (pydantic) written to `<out_dir>/robustness.json`, plus a printed summary.

## Reused utilities (import, do not duplicate)

From `validate.py`: `_calibrate_slope`, `infer_epistasis`, `_informed`, `_pinned`, `_corr`,
`_measured_dg`, `_candidate_terms`, `structural_graph`, `_MIN_POINTS_FOR_CORR`, `_N_BOOTSTRAP`.
(`map_recovery` is intentionally NOT reused: A1/A2/A3 need paired / cross-fitted / per-order-intersection
statistics that `map_recovery`'s per-method aggregate does not expose, so the per-order correlation is
computed directly via `_corr`.) From `acquisition.py`: `allocate`, `fitness_greedy`. From
`validate.py` baselines: `random_selection`. From `epistasis.py`:
`interaction_loop`, `epsilon_pairwise`, `epsilon_third`, `ground_truth_epistasis`,
`predicted_epistasis`. From `scoring_plan.py`: `variant_key` (the single stable, salt-free integer key
— reused for deterministic fold assignment so fold membership is reproducible across processes). From
`graph.py`: `EpistasisFactorGraph`. From `scored_cache.py`: `load_cache`.

> Some of these are underscore-prefixed module-internals of `validate.py`. **Resolved (design review):**
> import the privates directly (`from epibudget.validate import _calibrate_slope, _informed, _pinned, …`)
> — same-package internal reuse. Do NOT promote/rename: `_calibrate_slope` has three external importers
> (`calibrate.py`, `test_calibrate.py`, `test_validate.py`), so a rename risks breaking them, and
> promotion would add new public functions needing their own test coverage. Direct private import is the
> lowest-risk choice and adds no new public surface.

## Analyses

Every analysis is computed **per order** (pairwise and third separately), never pooled — the pairwise
order is the frozen decision order and pooling mixes order-composition differences into the comparison
(the exact confound the intersection is meant to avoid). Pairwise is the headline; third is reported as
underpowered companion.

### A1. Common-predicted-term precision

For an ordered method pair `(A, B)` at a budget and order:
- `predicted(M)` = terms of that order that method `M`'s selection *informs but does not pin*
  (`_informed and not _pinned` over `M`'s measured set) — the terms where `M` had to predict ε, not read
  it off. Same definition as `_order_metric`'s `predicted` split.
- `common = predicted(A) ∩ predicted(B)` (term-identity intersection).
- On `common`, compute each method's precision correlation (Pearson & Spearman of ε̂ vs true ε), using
  each method's own `infer_epistasis` ε̂ on that shared term set.
- Report: `n_common`, the two methods' precision correlations on `common`, and their paired difference
  (via A3). Gate on `len(common) >= _MIN_POINTS_FOR_CORR`; else correlations are `None`.

**Bias to state explicitly in output + docs (theorist finding):** `common` is not a neutral subsample.
A term is "informed" more easily the larger its loop, and both info-optimal's `n(v)` hub bias and
fitness/practice's high-ΔG bias concentrate on the same popular positions, so intersection membership
correlates with a term's loop size and structural popularity. Computing per-order removes the
cross-order part of this; the within-order popularity bias remains and is reported (not silently
absorbed): the output carries `n_common` and the intersection's term list so a reader sees the sample.

### A2. Cross-fitted, method-independent scale sensitivity

Goal: re-evaluate recovery with a **single, method-independent, cross-fitted** slope in place of each
method's own revealed-set `b`, to test whether the method ranking survives removing per-method
slope-fitting noise.

- **Folds.** Assign every candidate variant `v` to fold `variant_key(sorted(v)) % n_folds`
  (`n_folds=5`, deterministic, identity-based, label-free).
- **Fold slopes.** For each fold `f`, `slope[f] = _calibrate_slope(esm, measured)` over the candidates
  that are (i) **not** in fold `f`, (ii) measurable (`landscape[v] > 0`), using `esm=delta_g` and
  `measured=ln landscape[v]`. Out-of-fold fitting avoids a member's own fold leaking into its slope. The
  slope is fit on the **full measurable candidate set** (method-independent), not on any method's reveal.
- **Cross-fitted inference.** `infer_epistasis_crossfit(revealed, scored, folds, slopes, max_order)`:
  identical to `infer_epistasis` except an unmeasured loop member `m` uses `μ[m] = slope[fold[m]]·esm[m]`
  (its own fold's slope) instead of one global `b`; measured members stay pinned to their true ΔG.
- Recompute `map_recovery` per method with the cross-fitted ε̂ and compare the method ranking (pairwise
  order) to the frozen (global-slope) ranking. Report both rankings and whether they agree.

**Assumption to state (theorist finding):** one shared slope per fold assumes the ESM→measured-fitness
relation is homogeneous across the subpopulations different methods leave unmeasured. This is a
robustness probe, not a claim of homogeneity; using more label information than an operational run would
have is acceptable because this never feeds selection and never replaces the headline. Documented in the
module docstring and the output.

### A3. Paired / hierarchical difference CIs

- **Deterministic pairs** (`info vs fitness`, `info vs structural`): on the shared evaluated term set at
  an order, resample term indices with replacement (one shared index vector per bootstrap draw), compute
  `corr(A) − corr(B)` on each resample, take the percentile 95% CI. Reuses the `_bootstrap_ci` resampling
  idiom; skips degenerate (constant) resamples like `_bootstrap_ci` does.
- **Random pair** (`info vs random`): hierarchical bootstrap, pinned procedure (finding [6], so the CI
  brackets its own point estimate — the random arm's reported recovery is the mean over the `seeds`
  seeds, per `_mean_metric`/`_seed_ci`). Per bootstrap iteration: resample `seeds` seed-labels with
  replacement (same cardinality as `_seed_ci`); give each drawn label an **independent fresh**
  term-resample; on that label's resample compute both the random arm's correlation (that seed's
  selection) and the info arm's correlation (info's fixed selection); average the `seeds` per-label
  correlations within each arm; the iteration's delta is `mean_info − mean_random`. Percentile 95% CI
  over iterations. Seed variance sits outside term variance, matching the documented nesting.
- Output per pair: `delta` (point difference of the full-set correlation), `delta_ci95`, and
  `excludes_zero: bool` — explicitly labelled "descriptive, not a hypothesis test" in the schema doc and
  the printed summary.

## Module API (signatures)

Every disclaimer is a **serialized string field** (`Field(description=...)` and `#` comments do not
appear in `model_dump(mode="json")`, so the JSON artifact must carry the caveat as real data):

```python
_DIFF_INTERPRETATION = "descriptive difference on matched terms; NOT a hypothesis test"
_CROSSFIT_CAVEAT = (
    "cross-fitted on full-landscape labels (more label information than an operational run); a "
    "robustness probe of the frozen ranking, NOT an operational recovery number; never quote as a "
    "headline figure and never adopt crossfit_ranking as the reported method order"
)

class PairDifference(BaseModel):
    method_a: str
    method_b: str
    order: str                       # "pairwise" | "third"
    statistic: str                   # "spearman" | "pearson"
    delta: float | None              # corr(A) − corr(B) on the shared, index-aligned full-set terms
    delta_ci95: tuple[float, float] | None
    excludes_zero: bool
    interpretation: str = _DIFF_INTERPRETATION   # serialized; see finding [1]

class CommonPrecision(BaseModel):
    method_a: str
    method_b: str
    order: str
    n_common: int
    spearman_a: float | None
    spearman_b: float | None
    pearson_a: float | None
    pearson_b: float | None
    # Depth of informedness on the common set (finding [3]): mean over common terms of
    # (#loop members measured by the method / loop size). Unequal depth means the precision
    # comparison still partly reflects coverage depth, not only prediction skill — reported, not hidden.
    mean_informed_fraction_a: float | None
    mean_informed_fraction_b: float | None
    difference: PairDifference       # paired diff on the common set (built from sorted(common))

class ScaleSensitivity(BaseModel):
    order: str
    n_folds: int
    global_ranking: list[str]        # method order by spearman, frozen (global-slope) inference
    crossfit_ranking: list[str]      # robustness only — NEVER the reported headline order
    ranking_agrees: bool
    per_method_spearman_global: dict[str, float | None]
    per_method_spearman_crossfit: dict[str, float | None]
    caveat: str = _CROSSFIT_CAVEAT   # serialized; see findings [1],[2],[4]

class RobustnessReport(BaseModel):
    dataset: str
    model_id: str
    budgets: list[int]
    seeds: int
    max_order: int
    n_candidates: int
    n_folds: int
    note: str  # post-hoc, descriptive; does not alter the frozen decision rule; difference CIs are
               # not hypothesis tests; A2 crossfit numbers are non-operational robustness probes
    common_precision: list[CommonPrecision]     # per (pair, budget, order)
    scale_sensitivity: list[ScaleSensitivity]   # per (budget, order)
    pair_differences: list[PairDifference]      # per (pair, budget, order, statistic)

def crossfit_slopes(scored, landscape, n_folds=5) -> dict[int, float]: ...
def infer_epistasis_crossfit(revealed, scored, slopes, max_order=3, n_folds=5) -> list[Interaction]: ...
def common_precision(inferred_a, measured_a, inferred_b, measured_b, truth_by_term, order, seed) -> CommonPrecision: ...
def paired_difference_ci(rows_a, rows_b, statistic, seed) -> tuple[float | None, tuple[float,float] | None]: ...
def hierarchical_random_difference_ci(info_rows, random_per_seed_rows, statistic, seed) -> ...: ...
def robustness_report(scored, landscape, budgets, seeds, *, max_order=3, n_folds=5, dataset, model_id, out_dir) -> RobustnessReport: ...
```

Method pairs analysed (pairwise order is the headline; third reported too): `info vs fitness`,
`info vs structural`, `info vs random`.

## CLI

`epibudget robustness` (new command in `cli.py`):
- `--scored-cache PATH` (required) — the completed JSONL cache; loaded via `load_cache`.
- `--data data/proteingym/gb1_wu2016.csv`, `--alphabet ACDEFGHIKLMNPQRSTVWY`, `--budgets 48,96,192`,
  `--seeds 20`, `--max-order 3`, `--n-folds 5`, `--out report/`.
- **Completeness + ordering guard (findings [8],[9]).** Re-enumerate `enumerate_candidates(GB1_SITES,
  GB1_WT_AT_SITES, alphabet, max_order)` (the exact universe `run_validation` used) and assert the cache
  covers it exactly (`set(cache) == set(enumerated)`); a truncated/partial cache (a timed-out Colab run
  drops top-order triples first, silently) is **rejected**, not silently analysed on a smaller universe.
  Build `scored = [cache[v] for v in enumerated]` in enumeration order, so the stable-sort tie-breaks in
  `allocate`/`structural` reproduce the frozen selections exactly (structural ranks by the integer
  `n(v)` with large tie classes — order matters).
- Loads the landscape via `load_gb1`, runs `robustness_report`, writes `<run_id>/robustness.json` via
  `write_json_exclusive`, prints a summary. No `--device`/`--model`: the scores already exist, so this
  runs on any CPU in seconds–minutes.

## No leakage / determinism

- Selection recomputation reads only `scored` (ESM) + seeds — never a label — identical to
  `run_validation`. Labels enter only after selection, via `_measured_dg` (reveal) and the truth terms —
  post-hoc, exactly as the frozen harness already does.
- A2's cross-fitted slope uses full-landscape labels, but only to re-score recovery post-hoc; it never
  touches selection and never replaces a frozen number. Stated in the docstring + output `note`.
- Fold assignment via `variant_key` is deterministic and salt-free; all bootstraps take an explicit
  `seed` (derived deterministically per `(pair, budget, order, statistic)`, mirroring `run_validation`'s
  `seed=budget` convention).
- **Canonical ordering for cross-process reproducibility (findings [5],[7]).** Any term collection that
  feeds an ordered array or a seeded bootstrap must be sorted into a canonical order before use — in
  particular `common = sorted(predicted(A) & predicted(B))` (a Python `set` of `Term` tuples iterates in
  `PYTHONHASHSEED`-salted order otherwise, so the seeded index bootstrap would differ across processes
  and `robustness.json` would not be byte-identical). The paired bootstrap builds `rows_a[i]` and
  `rows_b[i]` from the **same** `common[i]` term (alignment invariant), each looking up that method's own
  `(epsilon_hat, true)` — never two independently-ordered lists.

## Test plan (`tests/test_robustness.py`, offline, synthetic — no ESM, no network)

- **Folds:** `variant_key(...) % 5` is deterministic and partitions the pool; every fold's slope is fit
  on out-of-fold variants only (assert a variant's own fold is excluded from its slope's inputs).
- **Cross-fit reduces to global** when all folds share one slope: if the landscape is exactly linear
  (`ln f = c·ΔĜ`), every fold slope equals `c` and `infer_epistasis_crossfit == infer_epistasis`.
- **Common precision:** on a synthetic pool with two hand-built selections, assert `common` is exactly
  the intersection of their informed-not-pinned term sets, and `n_common`/correlations match a direct
  computation; empty intersection → `None` correlations, `n_common=0`.
- **Paired difference:** identical rows for A and B → `delta=0`, CI brackets 0, `excludes_zero=False`; a
  constructed A≫B → `delta>0` and (with enough terms) `excludes_zero=True`. Determinism: same seed →
  identical CI.
- **Hierarchical random CI:** seed reproducibility; nesting draws a fresh term-resample per seed draw
  (assert via a counting stub or a fixed-seed value pin).
- **Report determinism:** `robustness_report` on a synthetic `scored`+`landscape` is byte-identical
  across two runs; the second pass runs in a **subprocess under a different `PYTHONHASHSEED`** so the
  `sorted(common)` cross-process guarantee (finding [5]) is actually exercised, not masked by
  same-process set order. `note`/`caveat`/`interpretation` fields present in the dumped JSON; no pooled
  entries (only pairwise/third).
- **Paired alignment (finding [7]):** permuting one method's internal term order leaves
  `common_precision` output unchanged (the sorted-common alignment holds).
- **Cross-fit reduces to global** uses `pytest.approx` (not `==`) and small rational fixtures, since the
  per-fold and global summations differ in float association (finding, theorist MINOR).
- **Cache completeness (finding [8]):** the CLI rejects a cache missing any enumerated candidate (build a
  cache with a top-order triple removed → expect an error, not a smaller-universe result).
- **No-leakage guard:** selection sets recomputed here equal those `run_validation` produces for the same
  `scored` order (selection is label-free and identical), including the structural tie-break (finding [9]).

## Open design decisions (surface to the design review before coding)

1. **Promote `_informed`/`_pinned`/`_calibrate_slope` to public** in `validate.py` vs. import privates.
   Recommended: promote (shared across a module boundary).
2. **A2 uses full-landscape labels for the cross-fitted slope.** Confirm this is acceptable as a
   post-hoc robustness probe (it is more label info than an operational run has). Recommended: yes, with
   the explicit `note` and docstring caveat; it never feeds selection or the headline.
3. **Third-order inclusion.** Report third-order companion analyses (underpowered) or pairwise only?
   Recommended: compute both, headline pairwise, mark third underpowered — consistent with the frozen
   protocol's own order treatment.

## Verification

- `pytest -q tests/test_robustness.py` green (offline); `mypy --strict src/` clean; `ruff` clean.
- `scripts/validate_artifacts.py` still green (no public-number claims added yet — Phase B numbers wire
  into artifacts only after the real run, a separate task).
- End-to-end once the Colab cache lands: `epibudget robustness --scored-cache scored_650m.jsonl` writes a
  `robustness.json` whose pairwise `info vs structural` common-precision difference answers the coverage-
  vs-precision question the frozen headline leaves open.
