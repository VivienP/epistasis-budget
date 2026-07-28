# Protocol amendment 2 — prospective contrast-recovery estimand

Status: **prospective**. Registered 2026-07-28, after the independent mathematical audit and before
any run on a third landscape. It governs the **next independent landscape only**. It does not
retroactively license any GB1 or TrpB number.

## 1. What was originally preregistered

`docs/VALIDATION.md` H1: at equal budget *B*, variants selected by dispersion-weighted loop-coverage
allocation recover the ground-truth epistasis map of GB1 better than the same budget spent
fitness-greedily, and better than random. The decision statistic was the Pearson and Spearman
correlation between the inferred and the measured inclusion–exclusion contrast, over the eligible
term population, with support requiring non-overlapping bootstrap 95% intervals at a majority of the
three budgets.

## 2. What the audit found (the defect)

The decision statistic is confounded by construction. Writing the loop L(S) as its measured part M
and unmeasured part U,

```
eps_hat(S) = k(S) + sum_{T in U} c_T * mu(T)
eps(S)     = k(S) + sum_{T in U} c_T * DG(T)          k(S) = sum_{T in M} c_T * DG(T)
```

`k(S)` is the *same number* on both sides. It is not an estimate of anything; it is the part of the
contrast the plate bought outright. Under global epistasis `eps` is a function of the same latent
effects that make up `k(S)`, so the correlation rises with coverage even when nothing is predicted.
On TrpB a plate that buys the 76 singles and assigns prior 0 to every unmeasured variant reaches
Pearson 0.798, above every method the original table reported.

Two further defects bore on the same statistic: the calibration slope was refit per method and could
flip sign, and the `structural` comparator was an undocumented tie-break whose as-run draw fell
outside the range of 100 seeded resolutions.

## 3. Retrospective corrective analysis on GB1 and TrpB

Reported in [`../AUDIT_REMEDIATION_20260728.md`](../AUDIT_REMEDIATION_20260728.md) and computed by
`scripts/corrected_recovery.py`. Four separated quantities, a model-free `singles_zero_prior`
baseline, three declared calibration policies, a 100-seed tie distribution for tie-dominated methods,
and a term-set hash on every comparison.

These analyses are **retrospective**. The estimands were selected after seeing the defect on these
two landscapes, so they carry no preregistered inferential status here. They exist to bound what the
existing evidence can support, not to establish a new claim.

## 4. Prospective rule for the next independent landscape

### 4.1 WITHDRAWN: held-out contrast prediction is degenerate, not merely hard

The first version of this amendment proposed evaluating only terms whose entire loop is unmeasured.
That is **withdrawn**. Two things kill it, both established in
[`../AUDIT_REMEDIATION_20260728.md`](../AUDIT_REMEDIATION_20260728.md) finding N-1:

1. It is infeasible for the primary method at the primary budget: once a plate buys the 76 singles,
   `n_uninformed` is **0** for `structural` at B=96 and B=192. There is nothing to evaluate.
2. Where it *is* feasible it is degenerate. In the reference-coded basis, the coefficient of a
   residue pair that appears in no training row is exactly 0, so the predicted contrast collapses to
   the same constant `-intercept` for every such term (measured predicted sd 3.5e-17 against a true
   sd of 1.67). A constant prediction supports no correlation at all.

A mutation-level holdout does not rescue this; it makes the degeneracy worse, since the held-out
mutations' main effects become unidentifiable too.

The general statement: **a contrast is identifiable only if the plate assays its specific residue
pair.** With 2,166 pairwise terms and a budget of 192, at most ~9% of pairwise contrasts can ever be
identified. Landscape reconstruction is not a budget-allocation problem at these budgets; it is an
identifiability wall.

### 4.1b Replacement primary estimand: out-of-context contrast prediction

The one non-degenerate, non-confounded task available is to predict a contrast whose residue pair the
plate observed **in a different genetic context** from the one being evaluated. Concretely: the pair
(i, j) appears inside at least one measured higher-order variant, while the double {i,j} and the
singles {i}, {j} are all unmeasured. Then

* the pair coefficient is identifiable (it was observed in the triple), so the prediction is not
  constant; and
* no loop member of the evaluated term was measured, so `k(S) = 0` and nothing is shared.

This is well-posed and measurable. It is also **asymmetric across methods by construction**: a plate
that buys every single (`structural`) makes `k(S)` non-zero for every pairwise term and therefore has
no evaluable term at all, whereas triple-heavy plates (`fitness`, `random`) do. That asymmetry is a
real scientific statement — coverage-driven design and non-confounded contrast prediction are in
tension at a fixed budget — and it must be reported rather than engineered away by picking whichever
method the estimand happens to favour.

Any future run must therefore report, per method, how many terms are evaluable under this estimand
*before* reporting any correlation over them. A method with zero evaluable terms is recorded as
"not evaluable", never as a zero score.

### 4.1c (historical) the withdrawn formulation

Evaluate only terms whose **entire loop is unmeasured** by the plate. For such a term `M` is empty,
so `k(S) = 0` and the shared component vanishes identically:

```
eps_hat(S) = sum_{T in L(S)} c_T * f(T)        f = a learner trained ONLY on the plate's labels
eps(S)     = sum_{T in L(S)} c_T * DG(T)
```

No observed value appears in both prediction and truth. This is the property the original statistic
lacked, and it is why this estimand is preferred to skeleton-controlled partial correlation: the
partial correlation removes `k(S)` *after the fact* and therefore also removes information the design
legitimately purchased, whereas restricting to uninformed terms removes the confound *by construction*
without discarding anything the estimate was entitled to use.

`f` is the fixed main-effects-plus-pairwise ridge already used by the downstream benchmark, trained on
the WT-centred log-ratio of the plate's revealed rows. It never sees a held-out label, a held-out ESM
score, or the prior-inclusive estimator.

Primary statistic: Spearman correlation of predicted against measured contrast, pairwise order, at
the largest registered budget.

### 4.2 The single prospective primary cell

To keep multiplicity controlled, exactly one cell is primary:

| Field | Value |
|---|---|
| Estimand | out-of-context contrast prediction (S4.1b): pair observed in a higher-order variant, loop of the evaluated term entirely unmeasured |
| Interaction order | pairwise |
| Budget | the largest registered budget |
| Calibration policy | `zero_prior` (no fitted parameter, identical across methods) |
| Statistic | Spearman |
| Contrast | `fitness` minus `random`, both 100-seed where tied |
| Sampling unit | the landscape; the tie seed within it |

**`structural` is deliberately not in the primary contrast, and this is not an oversight.** Under
S4.1b it has zero evaluable terms at every registered budget: buying the singles makes `k(S)` non-zero
for every pairwise term. Naming it here would force either a fabricated score or a silent fallback to
the confounded statistic. The two questions are genuinely different, and the amendment records that
rather than papering over it:

* **"Which plate best reconstructs contrasts?"** — S4.1b. `structural` cannot answer it.
* **"Which plate best trains a learner to rank variants?"** — the downstream benchmark. `structural`
  wins it on TrpB (+0.313 AUC, 20/20 partitions).

A future design that wants to compete on both would need to spend part of its budget on higher-order
variants specifically to make contrasts identifiable out of context. That is a **new method**, not a
tie-break or a re-weighting of `structural`, and it must be registered and evaluated as one.

Every other cell — third order, other budgets, other policies, raw and partial correlations, SSE gain
— is a declared secondary or diagnostic report, not a decision.

### 4.3 Mandatory reporting conditions

A run is decision-eligible only if all of the following hold and are recorded:

1. `relative_sse_gain >= 0` for the primary cell. A negative gain means the plate made the contrast
   estimate worse than making no measurement; no wording equivalent to "recovery" may be used.
2. The `singles_zero_prior` model-free baseline is reported in the same table.
2b. The count of evaluable terms per method is reported before any correlation over them; a method
   with zero evaluable terms is "not evaluable", never a zero score.
3. The evaluation term set is identical for every compared method, with its size and SHA-256 recorded.
4. Any method whose score stratum crosses the budget boundary is reported as a distribution over at
   least 100 declared tie seeds, never as one draw.
5. The calibration policy is label-free and identical across methods; per-method slopes may be
   reported only as a diagnostic, with a sign-disagreement flag.
6. The label accounting buckets sum exactly to the plate size.
7. Comparative uncertainty is a paired difference on identical units, never the non-overlap of two
   marginal intervals.

### 4.4 What a positive result would and would not establish

A positive primary cell on a third landscape would establish that, on that landscape, a plate chosen
by mutation-order coverage predicts unmeasured pairwise contrasts better than a fitness-ranked plate
of the same size, under a fixed learner and a label-free prior. It would **not** establish
cross-protein generality (that needs several independent landscapes analysed as replicates, with the
landscape as the sampling unit), nor anything about protein language models in general, nor anything
about sequential closed-loop design, which this method does not perform.

## 5. Relationship to the downstream benchmark

Unchanged. The downstream-impact benchmark (`docs/specs/downstream.md`) asks a different question —
whether a plate trains a fixed learner to rank held-out *variants* — and was not implicated in the
C-1 confound. Its GB1 result is unaffected by the label-accounting correction; its TrpB result is
being recomputed under corrected accounting.
