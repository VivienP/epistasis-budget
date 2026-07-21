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

- **Conjoint scores and masking variance are separate claims.** The 650M conjoint scores contain
  positive epistatic signal. The masking-variance calibration does not show positive association with
  absolute error: Spearman is −0.113 with 95% CI [−0.220, −0.002], while Pearson is −0.100 with 95% CI
  [−0.198, 0.003] ([artifact](../artifacts/calibration_650m.json)). This supports a weak negative rank
  association but not a general anti-calibration claim. At 35M, both intervals include zero
  ([artifact](../artifacts/calibration_35m.json)).
- **The historical structural-only comparison is not a current claim.** The frozen 650M report used one
  deterministic enumeration-order tie-break for a score that is exactly tied within mutation order. Its
  "structural wins at every budget" interpretation is therefore withdrawn. The corrective seeded analysis
  is itself inconclusive under its registered rule and is ineligible for public claims, so it does not
  support the converse claim that ESM masking dispersion improves allocation either.

## 6. Statistical power & protocol scope

- **Third-order is underpowered at B ∈ {48, 96}.** Only ~0.3–0.6% of third-order terms are informed at
  these budgets, so a third-order null is a *power* limitation, not evidence that no effect exists.
  Pairwise is the better-powered, decisive order.

- **The frozen protocol has now been exercised.** The headline requires 650M, the full alphabet,
  B ∈ {48, 96, 192}, and ≥ 20 seeds with bootstrap CIs — all met by the variance-inclusive run on a Colab
  T4 ([artifact](../artifacts/headline_650m.json)). It was executed on the GitHub branch tip (`3ba75eb`),
  an ancestor of the current manifest base; the metrics schema does not itself record that commit, so it is
  stamped in the artifact `configuration.colab_commit`.

## 6b. Metric defects and corrective status

Surfaced by the exploratory TrpB run
([`experiments/trpb-smoke-20260713.md`](experiments/trpb-smoke-20260713.md)). The WT bug is fixed in the
current validation and robustness paths. The structural tie and slope estimands are separated in the
corrective Gate 2 report but remain constraints on interpreting the historical artifacts.

- **The historical TrpB recovery used an uncentred reference; the current code does not.** Epistasis now
  uses ΔG(v) = log f(v) − log f(reference), with exact ΔG(∅) = 0, in validation truth, revealed calibration
  labels, robustness truth and cross-fit slopes. GB1 remains bit-exact because f(WT) = 1.0. TrpB's parent
  has f = 0.408074, so the old recovery coefficients, correlations and truth-map variance require
  regeneration. The old selection identities, attempted/revealed counts, coverage, hit-rate and run
  configuration do not depend on that centring and remain valid descriptive outputs.

- **`structural-only` has no within-order signal.** `n(v)` is constant per order (1140 singles / 39 doubles
  / 1 triples), so with τ² ≡ 1 the greedy weight takes three distinct values and the within-order ranking is
  an exact tie broken by `enumerate_candidates`' site-major order. `structural-only` is a single,
  unreplicated draw with no variance over its tie-break. Gate 2 retains that legacy prefix as diagnostic
  only and evaluates 100 seeded permutations of each exact score stratum. Its registered τ²-contribution
  decision is inconclusive, so neither direction is promoted to a public claim. The same structural graph
  remains the control in the downstream benchmark's primary contrast; that benchmark has since run on GB1
  (confirmatory, decision-eligible) and TrpB (exploratory) — see
  [`experiments/trpb-downstream-generalization-20260716.md`](experiments/trpb-downstream-generalization-20260716.md).

- **A per-method calibration slope can set low-coverage recovery signs.** With no measured loop member,
  ε̂ = b · ε̂_ESM exactly, so a near-zero-coverage method reports `sign(b) · ρ_prior` — the sign of a
  nuisance parameter, not a property of its selection. Gate 2 reports the operational method-specific
  slope and a method-independent five-fold cross-fit attribution regime for every selection. No registered
  contrast has a strict sign reversal at two budgets, but effect sizes differ materially; the shared slope
  remains post-hoc attribution evidence, not an operational selection method.

- **The downstream ESM-circular diagnostic mixes scales.** `_esm_circular` supplies `log1p(fitness)` labels
  to `esm_prior_mu`, whose slope contract is WT-centred log fitness. This is confined to a non-decision-use
  diagnostic. It must use a downstream-specific calibration scale before any downstream rerun; no global
  replacement of the downstream `log1p` prediction target is warranted.

## 7. Not yet done (scope, not failure)

- **The confirmatory downstream-impact result is GB1-only, unregistered, and does not support the
  masking-variance prior.** The R=20 GB1 run (`report/20260715T111312Z/downstream.json`, 650M,
  `n_perturbations = 16`, `B ∈ {48, 96, 192}`, 20 partitions × 20 seeds) matches the frozen confirmatory
  profile exactly and passes the 7-point robustness gate on `structural − fitness` S_macro-AUC (20/20
  partitions positive, mean +0.342), so `structural_downstream_supported = true`. This is the one test that
  escapes the recovery tautology §4 describes. Three gaps remain: the artifact is `status = provisional`
  under the git-ignored `report/` and is not registered in `artifacts/`; the `info − structural` gate on
  S_macro at B = 192 fails (15/20, below the 16/20 sign threshold), so `esm_uncertainty_supported = false`
  and the masking-variance prior stays unsupported; and the second landscape is exploratory only. See
  [`experiments/trpb-downstream-generalization-20260716.md`](experiments/trpb-downstream-generalization-20260716.md).
- **No second-landscape *recovery* result; the downstream replication is corroborating only.** The
  pre-registered TrpB recovery run ([`VALIDATION.md`](VALIDATION.md) §"Second landscape — TrpB", frozen at
  `B ∈ {48, 96, 192}`, ≥ 20 seeds) remains deferred; the only recovery run executed is off-protocol
  (`B ∈ {24, 48}`, 5 seeds — [`experiments/trpb-smoke-20260713.md`](experiments/trpb-smoke-20260713.md))
  and uninterpretable. Separately, the downstream-impact benchmark *has* run on TrpB at its own full
  protocol scale (R=20 × K=5 × 20 seeds, budgets {48, 96, 192}) and reproduces the GB1 direction —
  structural − random 20/20 positive (+0.135), structural − fitness 20/20 (+0.286), with a permutation
  null leaving ~90% of the effect as real signal. That run is scored at `n_perturbations = 0` (its only
  protocol mismatch), so it is `decision_eligible = false`: TrpB transfer is **corroborated, not
  established**, and no `info`/masking-variance conclusion is available from it at n=0. See
  [`experiments/trpb-downstream-generalization-20260716.md`](experiments/trpb-downstream-generalization-20260716.md).
  Its historical recovery and truth-map summaries are invalidated by the old anchor, while its selections,
  coverage, hit-rate and configuration remain descriptive. The old `var_epsilon` field is truth variance
  and cannot establish the predicted-epistasis CLI invariant.
- **No background-averaged ε, no MoCHI handoff, no multi-round sequential design** — all deliberately out
  of scope for v1.
