# Audit remediation — 2026-07-28

An independent mathematical audit of this repository found two critical and four high-severity
defects in the map-recovery evaluation, the `structural` baseline definition, the calibration policy,
and the TrpB label accounting. This document records the disposition of every finding: the original
defect, the corrected estimand or implementation, why the correction is mathematically appropriate,
the tests added, the artifacts affected, old versus corrected results, the current claim status, and
what remains limited.

This is a correction of an **estimand and its interpretation**, not a retraction of a computation.
Every original number reproduces exactly; the audit's own independent reimplementation matched all 36
reported TrpB correlations to a maximum absolute difference of 1.6e-7, and both downstream gates to
full double precision. What changed is what those numbers are evidence *for*.

## Status of this remediation

Every analysis below is a **retrospective corrective reanalysis** of GB1 and TrpB. The corrected
estimands were chosen *after* seeing the defect, so none of them is preregistered or confirmatory for
these two landscapes. The prospective rule for the next independent landscape is separated into
[`docs/specs/prospective-amendment-2.md`](specs/prospective-amendment-2.md).

No artifact produced by this remediation is `primary`. All of it comes from an uncommitted working
tree; see "Provenance and what remains blocked".

## Baseline before any edit

| Gate | Result |
|---|---|
| `ruff format --check src/ tests/ scripts/` | 51 files already formatted |
| `ruff check src/ tests/ scripts/` | All checks passed |
| `mypy --strict src/` | Success, 20 source files |
| `pytest -q -m "not slow and not data"` | 429 passed |
| `python scripts/validate_artifacts.py` | exit 0 |

All six of the audit's decisive counterexamples reproduced against this baseline. The findings are
not artefacts of a stale checkout.

---

## C-1 — the map-recovery correlation is confounded by the purchased skeleton

**Original defect.** `validate.infer_epistasis` estimates a contrast by pinning measured loop members
to their true value and filling the rest from a calibrated ESM prior. Splitting the loop L(S) into
its measured part M and unmeasured part U:

```
eps_hat(S) = k(S) + sum_{T in U} c_T * b * esm(T)
eps(S)     = k(S) + sum_{T in U} c_T * DG(T)          k(S) = sum_{T in M} c_T * DG(T)
```

Both sides contain the identical number `k(S)`, built from the same measured labels with the same
signs. A plate that buys all 76 singles inserts a two-term `k(S)` into every pairwise contrast, and
on TrpB `sd(k) = 2.33` exceeds `sd(eps) = 1.71`. The reported correlation therefore rises with how
much of the contrast was purchased outright, whether or not anything was predicted.

**Why this inflates on real landscapes and not on arbitrary ones.** The confound is not an algebraic
identity — it needs `eps` to be associated with `k(S)`. Under nonspecific ("global") epistasis
`DG = g(a)` with `a` the summed latent additive effects and `g` concave, the pairwise contrast is
`eps(ij) = g(a_i + a_j) - g(a_i) - g(a_j)`, which for `g(a) = a - c a^2` is exactly `-2c a_i a_j` — a
deterministic function of the very latent effects that make up `k(S)`. GB1 and TrpB both show strong
global epistasis, so the association is large. A synthetic landscape with independent couplings shows
no confound at all, which is itself the useful statement of when the metric misleads. This is encoded
as the fixture `_global_epistasis_dg` in `tests/test_recovery.py`.

**Corrected implementation.** New module `src/epibudget/recovery.py` (schema v2) reports four
quantities and never collapses them:

1. `raw_*_with_skeleton` — the original correlation, named so that it cannot be read as anything else;
2. `skeleton_*` — the association of `k(S)` alone with the truth, with no prediction whatsoever;
3. `partial_*` — the association after residualising both sides on `k(S)`;
4. `relative_sse_gain` — `1 - SSE(post)/SSE(prior)`, which gates any wording equivalent to "recovery".

Plus the term census (`n_pinned`, `n_informed_not_pinned`, `n_uninformed`), the term-set SHA-256, and
the `singles_zero_prior` model-free baseline that buys the singles and assigns prior 0 to everything
unmeasured.

**Why `partial_*` is a diagnostic and not a replacement estimand.** `k(S)` is partly information the
design legitimately purchased. Residualising on it is conservative and can remove real signal, so it
is reported as a corrective diagnostic, never as the primary quantity. The prospective primary
estimand is instead held-out contrast prediction over terms whose loop is *entirely* unmeasured —
there `M` is empty, `k(S) = 0` by construction, and no observed value can be copied into both
prediction and truth.

**Tests added** (`tests/test_recovery.py`, 20 tests; none calls the production contrast as its own
oracle): purely additive four-site landscape → contrast exactly 0 at every order; known pairwise
couplings → contrast equals the injected coupling exactly; known third-order term → contrast equals
it exactly; all-singles-measured with zero prior → large raw correlation, `skeleton == raw` to 1e-9,
negligible partial, `recovery_wording_permitted == False`; a hand case where correlation rises while
SSE worsens; partial correlation against the closed-form three-correlation formula.

**Old vs corrected (TrpB, pairwise, B=192).** Computed by `scripts/corrected_recovery.py` over 100
declared tie seeds; artifact `report/remediation/corrected_recovery_trpb_johnston2024.json`.

| Method | Calibration | raw Pearson (diagnostic) | skeleton alone (Spearman) | skeleton-controlled partial Spearman | relative SSE gain | "recovery" wording |
|---|---|---:|---:|---:|---:|:--|
| `info` | `per_method` (historical) | **+0.698** | +0.606 | +0.236 | +0.482 | permitted |
| `info` | `zero_prior` | +0.722 | +0.606 | **-0.019** | **-0.704** | refused |
| `structural` | `per_method` | +0.709 [0.683, 0.740] | +0.606 | +0.226 | +0.483 | permitted |
| `structural` | `zero_prior` | +0.782 [0.774, 0.791] | +0.606 | **-0.005** | **-0.912** | refused (0/100 seeds) |
| `singles_zero_prior` (model-free) | `zero_prior` | **+0.782 [0.774, 0.791]** | +0.606 | -0.005 | -0.912 | refused |
| `fitness` | `per_method` | +0.101 | +0.104 | +0.125 | -0.000 | refused |

Three things follow.

1. **The model-free baseline matches `structural` bit for bit** under the label-free prior. A plate
   that buys the singles and predicts every unmeasured double as literally zero reaches +0.782,
   above the originally reported +0.774. `structural` contributes nothing beyond mutation order —
   an independent confirmation of H-1 from a completely different code path.
2. **Once the skeleton is controlled, the association is ~0** under a label-free prior (-0.019 and
   -0.005). The residual +0.236 under `per_method` comes with a fitted, method-specific slope.
3. **The plate increases contrast error** under both decision-eligible policies (-0.70, -0.91), so
   "recovery" wording is refused at 0 of 100 tie seeds. It is permitted only under `per_method`, the
   policy that is not comparable across methods.

The originally reported single draw (+0.774 for `structural`) lies **outside** the 100-seed range
under both policies — below [0.774, 0.791] under `zero_prior` and above [0.683, 0.740] under
`per_method`.

**GB1, pairwise, B=192** (artifact `report/remediation/corrected_recovery_gb1_wu2016.json`, 100 tie
seeds) shows the same structure:

| Method | Calibration | raw Pearson | skeleton (Spearman) | partial Spearman | relative SSE gain |
|---|---|---:|---:|---:|---:|
| `structural` | `zero_prior` | +0.563 [0.545, 0.576] | +0.471 | **-0.009** | **-3.403** |
| `singles_zero_prior` | `zero_prior` | **+0.563 [0.545, 0.576]** | +0.471 | -0.009 | -3.403 |
| `info` | `zero_prior` | +0.564 | +0.471 | **-0.000** | -3.263 |
| `info` | `per_method` (historical) | +0.504 | +0.471 | +0.165 | -1.083 |

Again `structural` and the model-free baseline are identical, the skeleton-controlled association is
zero to three decimals under a label-free prior, and the SSE gain is negative everywhere — including
under the historical calibration, which is why the GB1 decision was already `inconclusive`. GB1
third-order is worse still: raw +0.194 / +0.318 / +0.309 at B = 48 / 96 / 192 against a
skeleton-controlled partial Spearman of +0.001 / -0.001 / -0.001 and SSE gains of -5.03 / -9.59 /
-8.62. **No GB1 cell permits "recovery" wording.**

---

## C-2 — the plate increases contrast error while the correlation rises

**Original defect.** `relative_sse_gain` existed in `gate2` but was computed only for GB1 and only
for `info`. It was never reported for TrpB, whose decision was declared satisfied.

**Corrected implementation.** `recovery.relative_sse_gain` is computed for **every** dataset × method
× budget × calibration policy × interaction order, and `recovery_wording_permitted` is false whenever
it is negative.

**Why SSE is the right gate.** A shared additive term cancels in the residual `eps_hat - eps`, so
unlike a correlation the SSE gain cannot be inflated by the skeleton. This invariance is asserted
directly in `tests/test_independent_oracles.py::test_sse_gain_is_immune_to_a_shared_additive_term`.

**Corrected result (TrpB pairwise, 100 tie seeds).** Under both decision-eligible,
method-independent calibration policies the gain is negative for every method at every budget
(`structural`/`zero_prior`: -0.269, -0.997, -0.912 at B = 48/96/192), so `recovery` wording is
refused across the board. Under the operational `per_method` policy `info` does achieve a positive
gain (+0.093 / +0.291 / +0.482) — but that is precisely the policy that is not comparable across
methods (see H-2). GB1 remains in flight; the audit's independent computation put its pairwise gain
at -1.05 and its third-order gain at -2.71.

---

## H-1 — `structural` was an undocumented arbitrary tie-break

**Original defect.** On a four-site landscape the loop-coverage weight takes exactly three values.
Closed form, derived independently in `tie_break.analytic_loop_counts`:

```
n(k) = sum_{j=0}^{max_order-k} C(n_sites - k, j) * (alphabet_size - 1)^j
     = 1140 (singles), 39 (doubles), 1 (triples)   for 4 sites, 20 letters, max_order 3
```

`structural` is therefore not an ordering at all: it is "singles, then doubles, then triples" with
every within-order comparison an exact tie. `validate` inherited the enumeration order (all 116
doubles of a TrpB B=192 plate from the single site pair 182/183); `downstream` pre-sorted by
`canonical_id` (the same 116 spread over three site pairs). The same named method selected different
plates in the two pipelines, and the as-run values fell **outside** the range of 100 seeded tie
resolutions at TrpB B=48 and B=192 and at GB1 B=192.

**Corrected implementation.** New module `src/epibudget/tie_break.py`: one algorithm
(`TIE_BREAK_VERSION = "canonical-strata-pcg64-v1"`) that groups candidates into exact-score strata,
sorts each stratum by canonical identity — which is what removes the caller's input order — and
permutes within a stratum from an explicit `tie_seed`. `stratum_crosses_budget` detects when a plate
is seed-dependent, i.e. when a single-seed table is a sample of size one.
`scripts/corrected_recovery.py` reports `structural` as a distribution over 100 declared seeds.

**`structural` is the model-free baseline, not a comparator for it.** Under `zero_prior` the
corrected reanalysis gives `structural` and `singles_zero_prior` **bit-identical** results at every
budget and every seed. This is not two methods agreeing: `n(v)` is a strictly decreasing function of
`|v|` on this universe (1140 / 39 / 1), so ranking by loop coverage and ranking by `-|v|` produce
the same strata in the same order, and the seeded permutation within each stratum consumes the RNG
identically. The two orderings are equal element-for-element, verified at seeds 0, 1 and 7.

The identity is the finding, and it is stronger than an agreement would be: at unit τ²,
"dispersion-weighted loop-coverage allocation" *is* "buy low-order variants first", carrying no
information beyond mutation order. The paired contrast measures this as exactly 0.0 with a
zero-width interval rather than asserting it.

**Tests added** (`tests/test_tie_break.py`, 12 tests): analytic counts 1140/39/1; closed form versus
brute-force enumeration on a 3-site/3-letter universe; agreement with the factor graph's own weight;
fixed-seed invariance to reversed and shuffled input; different seeds give different plates;
`stratum_crosses_budget` true at B=48 and B=192, false at B=76 (a stratum boundary); identical
canonicalisation across `tie_break`, `downstream` and `gate2`.

No diversity-aware or loop-closure tie-break was added: that would be a different method.

---

## H-2 — method-specific calibration was a hidden source of separation

**Original defect.** `validate._calibrate_slope` refits a through-origin slope on each method's own
revealed plate. The slopes come out with opposite signs across methods:

| Dataset, B=192 | `info` | `structural` | `fitness` |
|---|---:|---:|---:|
| TrpB | +0.281 | +0.225 | **-0.061** |
| GB1 | +1.107 | +0.727 | **-1.178** |

For `fitness`, which informs only 108 of 1784 pairwise loops, nearly every inferred contrast was the
**negated** ESM contrast. `docs/VALIDATION.md` claimed the estimator was "identical across methods";
it was identical in form only.

**Assessment of the existing shared cross-fit (as the prompt requires, before adopting it).** The
`robustness.py` cross-fit slope is method-independent, but it is fitted on the full measurable
landscape — far more label information than any budget purchases, and charged to no plate. It fails
the budget-accounting condition, so it was **not** adopted as the decision policy.

**Corrected implementation.** `recovery.prior_mu` implements three declared policies, and records the
slope, the label count, and `labels_are_method_specific` for every cell:

* `zero_prior` (mu = 0) and `fixed_unit` (mu = esm) read **no labels at all**, so they are identical
  across methods by construction and cannot leak. Decision-eligible.
* `per_method` reproduces the historical slope. Operationally reasonable — a real experimenter does
  calibrate on their own plate — but not comparable across methods. Marked `decision_eligible=False`.

**Tests added**: `test_label_free_policies_are_identical_across_methods_and_read_no_labels` (same
prior from two different plates, zero calibration labels) and
`test_per_method_policy_is_method_specific_and_not_decision_eligible` (hand-computed slopes +2.0 and
-2.0, opposite signs, from the closed form `b = <x,y>/<x,x>`).

**Documentation corrected** from "identical across methods" to "identical in form; the former
implementation refit a method-specific slope".

---

## H-3 — TrpB negative labels vanished from training and from every count

**Original defect.** `_evaluate_selection` bucketed revealed labels as positive / exactly-zero /
non-finite and trained on the first two. A finite **negative** label matched none of them: it was
dropped from training and counted nowhere. TrpB has 35,643 negative values (22.3%) and **no** exact
zeros, so up to 17% of a random plate silently disappeared while `train_live_fraction` reported
1.000.

**Scientific semantics, from the committed dataset documentation** (not assumed from the sign).
`scripts/fetch_trpb.py:16` records: *"Label = an aggregated catalytic-fitness score (Kowalsky et al.);
<= 0 is inactive (like a dead row)."* So a non-positive TrpB value is a **valid measurement of an
inactive variant** — the same biological category as a GB1 fitness-zero row, recorded by an assay
whose inactive readout scatters slightly below zero instead of resting exactly at it. These are
training data, not missing data. Every observed value satisfies `f > -1` (minimum -0.164), so `log1p`
is defined and strictly increasing across the whole class, and the ranking evaluation is unaffected.
The values are therefore **included**, and not clipped.

**Corrected implementation.** New module `src/epibudget/labels.py` with six exhaustive, mutually
exclusive buckets — `valid_positive`, `valid_zero`, `valid_negative_in_domain`,
`outside_transform_domain` (f <= -1, where `log1p` has no finite image), `non_finite`, `missing` —
and `LabelAccounting.check()`, which raises `LabelAccountingError` unless the buckets sum exactly to
the plate size. The misleading `train_live_fraction` is replaced by `train_active_fraction`, defined
over the real training denominator.

The map-recovery log-ratio is a different transform, undefined at `f <= 0`. Its eligible population
is now declared explicitly in `recovery.ELIGIBLE_POPULATION` and recorded in every corrected report:
terms whose every loop member has strictly positive fitness. No logarithm of a non-positive value is
taken anywhere.

**Tests added** (`tests/test_labels.py`, 21 tests, plus one end-to-end test in
`tests/test_downstream.py`): a bucket oracle derived from the value alone, parameterised over finite
positive, exact zero, finite negative, the observed TrpB minimum, -1, below -1, NaN, +/-inf and
missing; the accounting identity over a mixed plate; `active_fraction < 1` on an inactive-heavy plate
(the regression the old code could not fail); `training_target` strictly increasing across the
inactive boundary and refusing out-of-domain values; and, end to end, that every selected row in a
downstream run lands in exactly one bucket.

**Artifacts affected and recomputed.** The full 20-partition TrpB downstream rerun completed under
the corrected accounting (`report/remediation/20260728T074331Z/downstream.json`). The primary gate
**survives and strengthens**:

| Contrast (target_blind / attempted_budget) | Before H-3 | **After H-3** |
|---|---:|---:|
| `structural - fitness`, log2-budget AUC of S_macro | +0.286 (20/20) | **+0.313 (20/20)**, median +0.312, min partition +0.304 |
| `info - structural` at B=192 | -0.025 (0/20) | **-0.028 (0/20)** |

Mean S_macro after correction (B = 48 / 96 / 192): `structural` 0.364 / 0.456 / 0.465, `info`
0.289 / 0.400 / 0.437, `random` 0.219 / 0.296 / 0.380, `practice` 0.098 / 0.149 / 0.268, `fitness`
0.083 / 0.128 / 0.149. The zero-budget ESM zero-shot control is 0.323, so a 48-variant plate barely
exceeds ranking for free and a 192-variant plate buys about +0.14 Spearman over it.

I predicted before the rerun that `structural - random` would shrink, because `random` recovers the
most discarded rows. It did not: +0.135 to +0.137. The inactive rows a triple-heavy plate recovers
are mostly uninformative about main effects, so `random` gained less than the arithmetic suggested.
The prediction was wrong and is recorded as such.

GB1 was rerun for consistency and its gates came back **bit-identical** to the historical values —
`structural - fitness` = 0.3422903257144479 (20/20) and `info - structural` at B=192 =
0.007390820815059291 (15/20), matching `structural_allocation_650m.json` to full double precision.
That is the expected outcome and a useful check on the scope of the fix: GB1 has no negative labels,
so H-3 changes none of its training rows, and the NDCG and AUC changes do not enter `S_macro`. The
correction is confined to the landscape it was supposed to affect.

**Both downstream artifacts are `epibudget-downstream-v1`.** They were produced before the tie-break
was wired into `allocate` and before the version bump. Wiring the seeded tie-break changes which
doubles `structural` buys (the 76 singles are common to both orderings at B=192; the 116 doubles are
not), so a v2 rerun can shift these numbers. They stand as the corrected v1 result and must be
reproduced under v2 from a clean commit before promotion. See "Provenance and what remains blocked".
GB1 has no negative labels, so its training rows are unchanged by this fix.

---

## H-4 — uncertainty and multiplicity

**Original defect.** Deterministic methods received a bootstrap over evaluation *terms*; the random
baseline received a bootstrap over *seeds*; the frozen rule compared the two by non-overlap. The term
bootstrap does not cover the dominant variance source: for TrpB `structural` at B=192 the reported CI
[0.749, 0.795] is **disjoint from** the 100-seed tie-break range [0.676, 0.736]. Separately,
`effect_size_pass` was defined as `global_mean > 0.0`, bit-identical to `global_mean_positive`, so
the advertised 7-point gate had 6 distinct conditions.

**Corrected implementation.**
* `recovery.paired_difference_ci` resamples the **same** term indices for both methods, so the
  difference is paired; the resampling unit is stated to be the evaluation term within one landscape.
* Tie-seed variability is represented wherever a stratum crosses the budget boundary
  (`stratum_crosses_budget`, and the 100-seed distribution in the corrected reports).
* `recovery.fisher_z_mean` replaces arithmetic averaging of correlations (L-2), with clipping so a
  single perfect correlation cannot diverge.
* The redundant condition is removed from the `supported` conjunction — a numerical no-op at
  threshold 0.0 — and retained as a reported field with `effect_size_is_redundant`, so a future
  amendment can set a preregistered non-zero practical-effect threshold that actually binds.

**Multiplicity structure, stated in full.** The reported grid spans 2 datasets × 5–6 methods × 3
budgets × 2 interaction orders × 2 correlation statistics × 3 calibration policies for map recovery,
and 2 estimands × 2 missingness regimes × 3 budgets × 5 methods for downstream. No multiplicity
adjustment is applied, and none of these cells is preregistered for GB1/TrpB. One prospective primary
cell is defined in the amendment.

**The 20 downstream partitions are within-landscape rerandomisations** of a single measured
landscape, not 20 independent biological replicates. Their spread is correspondingly tiny (TrpB range
0.276–0.303 on a mean of 0.286), which is a consistency check, not evidence of generalisation. No
term bootstrap or fold rerandomisation is used to claim cross-protein generality.

---

## Findings discovered during remediation, not in the original audit

### N-1 — a contrast over an unassayed residue pair is exactly unidentifiable

Raised while testing the prospective held-out estimand, which turned out to be infeasible: after a
plate buys the 76 singles, `n_uninformed` is **0** for `structural` at B=96 and B=192, so the
estimand has no evaluable term at the primary budget. Investigating why produced a stronger result.

In the reference-coded main-effects-plus-pairwise basis, the column for a specific residue pair is
active only in variants containing exactly that pair. If no training row does, the centered column is
identically zero and the fitted coefficient is exactly 0 — not poorly estimated, *zero*. The predicted
contrast then collapses to one constant for every such term:

```
eps_hat(ij) = (c + m_i + m_j + 0) - (c + m_i) - (c + m_j) = -c
```

Measured on a 4-site/2-letter landscape with a plate covering one site pair: contrasts over the
assayed pair are recovered to 1e-4; contrasts over unassayed pairs have predicted standard deviation
**3.5e-17** (every value equal to `-intercept`) against a true standard deviation of 1.67.

**Consequences.** (a) The mutation-level holdout suggested as a fix for the infeasible estimand is
also degenerate — restricting to entirely-unmeasured loops makes the prediction constant *by
construction*. (b) It explains why the skeleton-controlled association is ~0 in the corrected
reports: once k(S) is removed there is nothing left, because the model holds no information about
unobserved pairs. (c) It bounds the project: with 2,166 pairwise terms and a budget of 192, at most
~9% of pairwise contrasts can ever be identified; the other ~91% are unidentifiable, and any number
reported over them reflects the prior or the skeleton, never learning. (d) It explains why the
downstream benchmark is sound while map recovery is not: a *variant's* fitness is predictable from
main effects even when its pair coefficient is unknown; a *contrast* is not.

Test: `test_a_contrast_over_an_unassayed_residue_pair_is_exactly_unidentifiable`.

### N-2 — the degeneracy guard manufactured correlation from floating-point noise

`np.std(x) == 0.0` does not catch a vector that is constant up to rounding. The unidentifiable
contrasts of N-1 are algebraically identical but differ in their last bits, and ranking those bits
produced a spurious Spearman of **+0.33** over 20 terms — precisely in the case where the correct
answer is "undefined", and that case is the large majority of the term universe.

Replaced by `is_effectively_constant`, which compares the range against floating-point noise scaled
to the data's own magnitude. Test:
`test_a_near_constant_vector_yields_no_correlation_instead_of_rounding_noise`. The same exact-zero
guard remains in `validate._corr` and `downstream._corr` and should be migrated; downstream's risk is
lower because it correlates variant predictions, which are rarely numerically constant.

## Medium and low findings

| ID | Disposition |
|---|---|
| M-1 | Terminology corrected in user-facing prose: "prior trace reduction" for the A-optimal objective, "masking dispersion" for `var_delta_g` (calibration is not demonstrated: Spearman -0.113, 95% CI [-0.220, -0.002] against real error), and wording that does not imply loop closure, because the modular objective contains no closure term. `info` is retained as a CLI method name for compatibility. |
| M-2 | `--method info --n-perturbations 0` is rejected with an explicit parameter error; `acquisition.allocate` raises whenever all weights are equal and the ranking would otherwise be the enumeration order. Regression tests in `tests/test_independent_oracles.py`. |
| M-3 | `recovery.common_term_subset` evaluates a comparison on terms in the same state for **both** methods, with the term count and SHA-256 recorded. The previous per-method "precision" split compared 1669 terms against 107 as if they were one estimand. `docs/LIMITATIONS.md` claimed an intersection that only the post-hoc robustness module implemented. |
| M-4 | `transfer_rho_triples` keeps recording its effective singles/doubles counts and is labelled diagnostic-only. It is **not** budget-matched (structural trains on 192 rows, fitness on 29), so no claim about transfer quality is made from it. A size-matched sensitivity requires a separate designed run and is listed as a remaining limitation rather than invented here. |
| M-5 | `learning_curve_auc` renamed `normalized_log2_budget_auc`. Budgets 48/96/192 are equally spaced on log2, so the equal-weight trapezoid *is* the exact normalized integral on that axis, with weights (1/4, 1/2, 1/4); a non-doubling grid now raises. Oracle test recovers the weights by probing the linear functional and compares against `np.trapezoid` on both axes. |
| L-1 | NDCG relevance: inactive variants (f <= 0) receive relevance exactly 0; actives receive their percentile rank *within the actives*. Previously each of GB1's 29,477 dead genotypes carried relevance ~0.10 and a ranker was rewarded for retrieving them. Hand-computed DCG/IDCG oracle added. |
| L-2 | Fisher-z averaging (`recovery.fisher_z_mean`), with an exact hand oracle. |
| L-3 | `ddof=0` for the masking dispersion documented in `docs/SPEC.md` §3.2 and in `scoring.py`. It is a uniform factor and cannot change a ranking. |
| L-4 | Figures are generated only by the committed renderer `scripts/render_figures.py`; `scripts/validate_artifacts.py` fails if a headline figure lacks a renderer, a manifest entry, or claim-map coverage. Superseded artifacts are marked, not deleted. |
| I-1 | `tests/test_independent_oracles.py` (47 tests): closed-form primal ridge versus the production dual across p>>n, exact collinearity, duplicate rows, constant target, all-zero design and alpha from 1e-6 to 1e8; hand DCG/IDCG/NDCG; AUC weights; SSE gain on fixed arrays; partial correlation against the three-correlation formula. |

---

## Current claim status

**Withdrawn.** The former map-recovery claim does not survive the audit in its original form. The
TrpB H1 result is non-decision-eligible: the metric it satisfied is dominated by lower-order
measurements shared algebraically with the ground-truth contrast, and is sensitive to method-specific
calibration. The original correlations remain reproducible and are retained as labelled diagnostics.

**Retained, and kept separate from map recovery.** The downstream benchmark asks a different
question — does a plate train a fixed learner to rank held-out variants — and its primary predictor
never sees a held-out ESM score and never calls the prior-inclusive estimator. Its GB1 numbers are
unaffected by H-3. Its TrpB numbers are being recomputed.

**Explicitly not established.** No result establishes that protein language models are generally
ineffective for experimental design; the evidence is about one zero-shot dispersion prior on two
landscapes. No result establishes cross-protein generalisation: GB1 and TrpB are two biological case
studies and their internal partitions are not replicates. The method performs static one-plate
allocation, not sequential closed-loop active learning.

## Remaining limitations

* The corrected estimands are retrospective for GB1 and TrpB.
* `partial_*` is conservative: it can remove signal a design legitimately purchased.
* The held-out contrast estimand is defined and unit-tested but has not yet been run at scale on the
  two landscapes.
* No size-matched transfer sensitivity exists.
* Both landscapes are four-site and complete; `n(v)` is degenerate in exactly that setting, so the
  `structural` degeneracy may not carry over to a larger design space.
* 871 of 160,000 TrpB values are imputed and unflagged row-wise in the public mirror.

## Integration debt (confirmed by counter-audit; all four closed)

An independent counter-audit of this remediation confirmed four integration gaps that the corrected
primitives do not by themselves close.

**Closed (tie-break wiring).** `acquisition.allocate` and `acquisition.fitness_greedy` now resolve
exact ties through
the shared `tie_break` algorithm under an explicit `tie_seed`, and `downstream_report` threads that
seed to every selection and records it (with `tie_break_version`) on the report. A tie-free score
takes a fast path that never builds the canonical identities, so the cost is bounded: on the full
29,678-candidate TrpB universe a `structural` allocation takes 0.29 s and is verified invariant to
input order, while a different seed yields a different plate. `PROTOCOL_VERSION` is bumped to
**`epibudget-downstream-v2`**, with `SUPERSEDED_PROTOCOL_VERSIONS` naming v1; the version is part of
the raw-record identity key, so a stale v1 record is simultaneously an unexpected and a missing cell
and can never be pooled into a v2 summary.

Tests added: `test_allocate_is_input_order_invariant_for_a_fixed_tie_seed`,
`test_validate_and_downstream_select_the_same_plate_for_the_same_inputs_and_seed`,
`test_fitness_greedy_still_equals_allocate_at_lambda_one`,
`test_v1_records_are_not_pooled_with_v2_records`.

**Consequence for the artifacts.** Wiring the tie-break changes `structural`'s plate, so both
corrected downstream artifacts — the completed TrpB run and the GB1 run that was still executing —
were produced by pre-integration code and are now **v1 artifacts superseded by v2**. Their numbers
remain valid as a record of what the pre-integration code did; they are not the v2 result. Both
landscapes must be rerun under v2 from a clean commit before anything is promoted.

**Closed (validating v2 emitter).** `scripts/corrected_recovery.py` now constructs a
`CorrectedRecoveryReport` and validates it before any byte reaches disk, so a field the emitter
forgets or invents fails immediately instead of after a long run. The model gained
`model_config = {"extra": "forbid"}`, and `test_emitter_writes_a_file_that_validates_against_its_own_schema`
runs the real script end to end over a 64-candidate synthetic landscape and parses the result back
through the model.

Three substantive changes came out of closing it, none of them mechanical:

* **The withdrawn estimand is gone from the schema.** `OrderRecovery` still carried
  `held_out_contrast_pearson`, `held_out_contrast_spearman` and `n_held_out_terms` — the
  §4.1 estimand that this audit withdrew as degenerate. Emitting them would have written `null` on
  every plate, which reads as *measured, found nothing* rather than *withdrawn as degenerate*. They
  are removed; `census.n_uninformed` is retained because it is the evidence for the withdrawal.
* **`safe_corr` is now public and single-sourced.** The emitter carried its own `_corr` helper still
  using the pre-N-2 exact `std == 0.0` constant test. It was latent at full-term-set scale, but the
  paired contrasts added here evaluate exactly the constant-prediction case, where it would have
  manufactured a correlation out of rounding noise. A second implementation of that guard is a
  second chance to reintroduce N-2.
* **Whether a method needs a seed distribution is now decided from the data**, via
  `stratum_crosses_budget`, not from a hardcoded method list. `MethodRecovery` records
  `seed_kind ∈ {none, tie, random}`, because a tie seed, an RNG seed, and no seed at all are not
  interchangeable — collapsing them is how the original `structural` number became a sample of size
  one presented as a method.

**Closed (complete methods and paired contrasts).** The emitter now evaluates `random` (20 seeds)
and `practice` alongside `info`, `structural`, `fitness` and `singles_zero_prior`, and emits paired
A-vs-B contrasts through `common_term_subset` + `paired_difference_ci` over all three term subsets,
under the two label-free calibration policies only. `per_method` is excluded from the paired
contrasts on purpose: its two arms carry differently-fitted priors — on GB1 with opposite signs — so
their difference is not one estimand.

Two results fall straight out of that wiring, both recorded in the artifacts:

* On `common_uninformed` under `zero_prior`, `delta` is `null` for **every** pair, with an explicit
  `reason`: both plates predict the same constant, so neither correlation exists. That is the
  identifiability wall appearing as a first-class field rather than as prose.
* `structural` minus `singles_zero_prior` is **exactly 0.0 with a zero-width interval** on every
  subset and budget. The two are the same plate: on this universe `n(v)` is a strictly decreasing
  function of `|v|`, so "dispersion-weighted loop coverage" at unit τ² reduces exactly to "buy
  low-order variants first". The paired contrast now measures that identity instead of asserting it.

## Provenance and what remains blocked

Every artifact in `report/remediation/` is produced from an **uncommitted working tree** and is
marked `decision_eligible: false` with `status: retrospective_corrective_reanalysis`. No new primary
result is declared. After authorisation to commit, the affected analyses must be rerun from the
identified clean commit before any artifact is promoted.
