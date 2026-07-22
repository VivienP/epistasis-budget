# Post-hoc robustness analyses

Implementation: `src/epibudget/robustness.py`. Status: implemented companion analysis.

The module analyzes completed selections and measured landscapes. It never feeds selection and does not
alter the registered map-recovery decision in [`VALIDATION.md`](../VALIDATION.md). Every difference
interval is descriptive, not a hypothesis test.

## Questions

The analysis addresses three confounds in recovery metrics:

1. methods can differ in how many interaction terms they inform, so precision must also be compared on
   common term identities;
2. each method's operational estimator fits its own calibration slope, which can affect sparse recovery;
3. separate confidence intervals do not estimate a paired difference.

Analyses are order-specific. Pairwise is the decision-bearing order; third-order is an underpowered
companion. Orders are never pooled here.

## Inputs and output

Inputs are a complete scored candidate universe, the full measured landscape, budgets, random seeds,
`max_order`, and five cross-fit folds. The module recomputes the `info`, `fitness`, `structural`, and
`random` selections with the same utilities used by validation. `practice` is not part of this analysis.

Truth is WT-centred and restricted to complete, positive-fitness interaction loops present in the scored
candidate universe. The CLI writes `<out_dir>/robustness.json` and prints a summary.

## A1: common-term precision

For a method `M`, `predicted(M)` contains terms that its selection informs but does not fully pin. For an
ordered pair `(A, B)`:

```text
common = sorted(predicted(A) intersection predicted(B))
```

On this shared identity set, report each method's Pearson and Spearman precision, their paired difference,
`n_common`, the term identities, and each method's mean measured-loop fraction. Fewer than three usable
terms or a constant input yields `None`, never zero.

The common set is not a neutral sample: structural popularity and selection biases influence which terms
both methods inform. Order stratification removes only the cross-order component. The serialized report
must retain this caveat and the coverage-depth fields.

## A2: method-independent cross-fit scale

Assign each candidate to `variant_key(sorted(v)) % 5`. For fold `f`, fit a through-origin slope on all
positive, measurable candidates outside `f`. The folds are identity-based and label-free; the slope fit is
post-selection.

For an unmeasured loop member `m`, cross-fit inference uses:

```text
mu[m] = slope[fold[m]] * esm[m]
```

Measured members remain pinned to WT-centred measured values. Report the operational method-specific-slope
ranking, the cross-fit ranking, per-method correlations, and whether the rankings agree.

This probe uses more labels than an operational run and assumes one fold slope is meaningful across the
subpopulations different methods leave unmeasured. It is attribution evidence only; the cross-fit ranking
cannot replace the registered result.

## A3: paired difference intervals

For deterministic pairs, bootstrap shared term indices and compute `corr(A) - corr(B)` on aligned rows.
Degenerate bootstrap draws are skipped.

For `info - random`, use a hierarchical bootstrap: resample seed labels, draw an independent term resample
for each sampled seed, compute both arms on that resample, average within arm, then take their difference.
This keeps seed variation outside term variation and brackets the same mean-over-seeds estimand used by
validation.

Each result contains `delta`, percentile `delta_ci95`, `excludes_zero`, and the serialized interpretation
`descriptive difference on matched terms; NOT a hypothesis test`.

Pairs are `info - fitness`, `info - structural`, and `info - random`, for Pearson and Spearman at every
budget and order.

## Report contract

`RobustnessReport` includes dataset and model identity, budgets, seed and fold counts, candidate count,
the three analysis collections, and serialized caveats. Comments or schema descriptions do not substitute
for caveat fields in the JSON artifact.

The report preserves:

- the common-term identities and informed-depth values for A1;
- operational and cross-fit rankings plus the non-operational caveat for A2;
- aligned point differences, intervals, and interpretation for A3;
- only pairwise and third-order entries, never pooled entries.

Implementation signatures and field types are authoritative in `robustness.py`; this spec does not copy
them.

## CLI and cache integrity

```bash
epibudget robustness --scored-cache PATH --data data/proteingym/gb1_wu2016.csv \
  --alphabet ACDEFGHIKLMNPQRSTVWY --budgets 48,96,192 --seeds 20 \
  --max-order 3 --n-folds 5 --out report/
```

The command re-enumerates the requested universe and requires exact cache coverage. It rejects malformed
or duplicate cache records, a missing sidecar, a partial universe, or identity mismatches. The resulting
sequence follows canonical candidate order before any tied allocation is recomputed.

The command performs no model inference and accepts no device option; it runs on an existing cache.

## Leakage and determinism

- Selection recomputation reads only scored variants and explicit seeds.
- Measured labels enter after selection for reveal, truth, and post-hoc cross-fit slopes.
- Cross-fit folds are identity-based; all bootstraps use deterministically derived seeds.
- Shared term sets are sorted once, and both methods' rows use the same indexed identity.
- Reordering inputs or changing `PYTHONHASHSEED` does not change the report.
- A2's full-landscape labels and cross-fit ranking never feed selection or the registered decision.

## Verification contract

Offline tests cover fold exclusion, the cross-fit/global identity case, common-set construction and empty
sets, paired alignment, deterministic and hierarchical bootstrap behavior, serialized caveats, absence of
pooled rows, cache completeness, selection equivalence with validation, and cross-process determinism.

Run:

```bash
pytest -q tests/test_robustness.py tests/test_cli.py
python scripts/validate_artifacts.py
```

Numerical results and their status belong in registered artifacts or experiment records, not in this
protocol.
