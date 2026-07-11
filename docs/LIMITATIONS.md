# Constraints & limitations

An honest register of the constraints that bound this project and the limitations they impose on what
can be claimed. It exists because the credibility of `epibudget` rests on a rigorous null-tolerant audit
(invariant #2): a reader must be able to see exactly where the walls are. Pairs with
[`VALIDATION.md`](VALIDATION.md), which contains the frozen protocol and the separately labelled
post-registration robustness analyses.

Each item states the constraint, its consequence, and how the code/docs handle it honestly.

---

## 1. Compute & environment

- **CPU-capable, GPU-capable.** CPU is the default and remains covered by tests. GPU execution is
  supported with `--device auto|cuda`, and the resolved device is recorded in provenance. The complete
  650M variance-inclusive benchmark is not claimed to be practically CPU-tractable.

- **Most of the compute cost is implementation, not the method.** `ConjointScorer.score_batch`
  (+ `scoring_plan.py`) batches masked forwards across variants, de-duplicates identical masked inputs,
  and tunes `torch` threads — all throughput-only and **bit-exact** to the per-variant reference (measured
  `delta_g` and `var_delta_g` gap = 0.0 at both 35M and 650M — `report/bench_35M.json`,
  `report/bench_650m.json`; guarded by `test_optimized_batch_matches_reference`):
  - **Masked-row de-duplication.** Masking a site erases its residue, so one masked forward yields the
    conditional distribution for all 19 substitutions there. The full 20-letter deterministic pass is
    **29,678 → 4,564 unique forwards** (≈6.5× fewer forward *calls*, ≈19× fewer masked *rows*; verified in
    `tests/test_scoring_plan.py`).
  - **Measured benchmark scope.** The provisional 650M benchmark uses alphabet `AC`, 64 variants,
    four perturbations, batch size 32 and 12 CPU threads. It reports 0.127 reference variants/s,
    0.171 optimized variants/s and a 1.341× speed-up with zero recorded score gap
    ([artifact](../artifacts/bench_650m.json)). It does not support the earlier undocumented batch-1
    comparison, which is excluded from public claims.

- **The frozen headline has now run on a GPU.** It was executed on a Colab T4 (`device=cuda`, artifact
  [`headline_650m.json`](../artifacts/headline_650m.json)) over all 29,678 candidates with `var_delta_g`
  from 16 perturbations each. A complete CPU duration is still not published — the GPU is the recommended
  execution path — and [`headline_650m_colab.md`](headline_650m_colab.md) remains the reproducible recipe.

- **Additional in-session CPU runs** (both write `report/<run_id>/metrics.json` with full provenance,
  including `device`): (a) the **supplementary 650M full-alphabet deterministic-only** recovery giving the
  var-independent methods (fitness / random / practice / structural-only) at `B ∈ {48, 96, 192}`,
  `pool ≫ B` (`scripts/headline_650m_supplementary.py`); and (b) the **650M uncertainty-prior calibration**
  (`scripts/calibrate_uncertainty.py`, see §5). The supplementary deterministic run omits info-optimal and
  does not evaluate the decision rule; the frozen variance-inclusive headline (with info-optimal) is the
  `headline_650m.json` artifact above.

- **The reduced-alphabet 35M runs remain a smoke test, not the headline.** They restrict the per-site
  alphabet (e.g. `ADEF`, ~307 candidates) so the pool scores in minutes; recorded as provenance in every
  `metrics.json`. See the exhaustion caveat in §4.

## 2. Data

- **GB1/Wu-2016 has only four positions and an incomplete measured grid.** The theoretical space contains
  160,000 genotypes; the local artifact contains 149,361 measured rows. Statistical replication comes
  from amino-acid instantiations, not from many independently sampled positions. The claim is therefore
  scoped to this measured four-site system, never a whole-protein positional map.

- **~25% of the ΔG grid is unusable.** Of the 160,000 possible four-site genotypes, 149,361 are present;
  29,477 of those are dead (fitness 0 → `ln` undefined), and ~10,639 are genuinely absent. Dead/missing
  constituents are **dropped, never imputed** (invariant #3), so any interaction whose loop touches one
  is simply unrecoverable ground truth. Dropping the dead (strong negative-epistasis) cases biases the
  tested domain toward all-viable interactions — if anything this *deflates* the measured signal.

- **The Walsh–Hadamard spectrum needs a complete dense tensor**, which the incomplete real GB1 landscape
  does not provide. `wht_spectrum` therefore raises on incomplete input and is validated on synthetic
  complete grids only; on real GB1 it is context/reporting, not a clean identity check.

## 3. Modeling

- **`info_gain` is modular, not strictly submodular.** Under the v1 independent-noise model,
  `info_gain(M, v) = var_delta_g(v)·n(v)` is independent of the measured set, so greedy selection reduces
  to a single sort by that fixed weight. The "geodetic loop-closure / diminishing-returns" intuition is
  **not realized** in v1 — it would require correlated priors across variants, which are out of scope.
  This is stated wherever the objective is described (never sold as generic submodularity).

- **The modular weight structurally front-loads low-order variants.** Because `n(single) ≈ 1140`,
  `n(pair) ≈ 39`, `n(triple) = 1`, the ranking is dominated by `n(v)`: info-optimal measures ~all singles
  then doubles and, at small budgets, **may measure no triples at all**. The ESM uncertainty (`var_delta_g`)
  only breaks ties *within* an order; it plays no role in the across-order allocation that drives most of
  the recovery number.

- **WT-referenced (biochemical) ε only.** Background-averaged (ensemble) ε — the basis inference tools
  like MoCHI consume — is out of scope for v1. The "feeds MoCHI" story is not yet real.

- **Independent-noise error propagation is a documented first approximation.** `σ²(ε(S))` sums
  `var_delta_g` over the loop assuming `Cov[ΔG(T), ΔG(T′)] = 0` for distinct variants — almost certainly
  false for nested/overlapping variants scored from related contexts. The direction of the resulting bias
  is **not** derivable from the ±1 structure, so it is not claimed to be conservative; it is checked
  empirically by the uncertainty-prior calibration, not by argument.

- **The recovery inferrer rides on the ESM prior.** `infer_epistasis` is the posterior mean of the
  factor-graph model: unmeasured loop members keep their (through-origin-calibrated) ESM prior, so
  recovery on unmeasured terms is partly a restatement of the zero-shot prior, not of the measurements.
  This is deliberate (it keeps selection and grading on one coherent model) but means the metric is not a
  pure "what did the measurements teach" quantity.

## 4. Metric & evaluation

- **Map-recovery is partly tautological.** A method can score high on the full-set correlation because it
  *measured* many terms directly (breadth), not because it *predicted* the unmeasured ones (precision).
  Pairwise recovery ≈ "measure the pairs"; third-order recovery ≈ "measure the triples" — order-matched
  and close to trivial. The report therefore separates **breadth** (`n_pinned`: terms whose full loop is
  measured, recovered exactly) from **precision** (correlation over informed-but-not-pinned terms), per
  method and per order. A non-tautological advantage must show up in precision.

- **The reduced-alphabet fast run is an *exhaustion regime*.** With a small pool, a budget of 48–96
  measures a large fraction of it, so "measure the low-order scaffold broadly" wins trivially (e.g. the
  structural baseline pins 57/58 pairwise terms at B=96). The exhaustion regime **cannot** distinguish
  breadth from precision; only a `pool ≫ B` regime (full alphabet, §1) can — which is exactly the run
  needed for a meaningful comparison.

- **Confidence intervals measure two different things.** The deterministic methods' CI is bootstrapped
  over the evaluated ε terms — it reflects leverage concentration of the correlation, **not** "would a
  repeated wet-lab budget give a similar number." Only the random baseline's over-seeds CI captures
  genuine selection variance. Both are labelled with `ci_method` so they are not conflated.

- **Method-specific precision correlations are not directly comparable.** They use different informed,
  non-pinned term identities. Post-registration direct comparisons therefore restrict precision to the
  intersection of terms predicted by both methods. Non-informed terms are reported only through coverage;
  they are not assigned a precision correlation.

## 5. Empirical evidence and unresolved interpretation

- **Conjoint scores and masking variance are separate claims.** The 650M conjoint-score spike contains
  positive epistatic signal. The masking-variance calibration does not show positive association with
  absolute error: Spearman is −0.113 with 95% CI [−0.220, −0.002], while Pearson is −0.100 with 95% CI
  [−0.198, 0.003] ([artifact](../artifacts/calibration_650m.json)). This supports a weak negative rank
  association but not a general anti-calibration claim. At 35M, both intervals include zero
  ([artifact](../artifacts/calibration_35m.json)).
- **Structural-only outperforms information-optimal.** In the frozen variance-inclusive 650M headline the
  prior-free `structural-only` sort has higher full-set pairwise recovery than info-optimal at every
  budget, and the post-hoc paired common-predicted-term analysis puts it ahead on matched precision too
  (descriptive Δ excludes zero at all three budgets; [artifact](../artifacts/robustness_650m.json)). So the
  ESM masking-variance uncertainty prior earns no credit and is dropped from the claims: the recovery is
  carried by direct loop coverage `n(v)`, not the uncertainty prior. The difference CIs are descriptive
  (post-hoc), not hypothesis tests, and do not alter the frozen decision rule.

## 6. Statistical power & protocol scope

- **Third-order is underpowered at B ∈ {48, 96}.** Only ~0.3–0.6% of third-order terms are informed at
  these budgets, so a third-order null is a *power* limitation, not evidence that no effect exists.
  Pairwise is the better-powered, decisive order.

- **The frozen protocol has now been exercised.** The headline requires 650M, the full alphabet,
  B ∈ {48, 96, 192}, and ≥ 20 seeds with bootstrap CIs — all met by the variance-inclusive run on a Colab
  T4 ([artifact](../artifacts/headline_650m.json)). It was executed on the GitHub branch tip (`3ba75eb`),
  an ancestor of the current manifest base; the metrics schema does not itself record that commit, so it is
  stamped in the artifact `configuration.colab_commit`.

## 7. Not yet done (scope, not failure)

- **No downstream-impact demonstration** — the one test that escapes the recovery tautology (does a
  structure-aware budget's map support a *better decision* on held-out mutants?). This is the highest-value
  next step precisely because §4 makes the recovery headline thin on its own.
- **No second landscape** (generalisation) — premature until the mechanism on GB1 is settled.
- **No background-averaged ε, no MoCHI handoff, no multi-round sequential design** — all deliberately out
  of scope for v1.
