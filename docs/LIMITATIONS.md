# Constraints & limitations

An honest register of the constraints that bound this project and the limitations they impose on what
can be claimed. It exists because the credibility of `epibudget` rests on a rigorous null-tolerant audit
(invariant #2): a reader must be able to see exactly where the walls are. Pairs with
[`VALIDATION.md`](VALIDATION.md) (the frozen protocol) and [`ROADMAP.md`](ROADMAP.md) (the plan).

Each item states the constraint, its consequence, and how the code/docs handle it honestly.

---

## 1. Compute & environment

- **Prefer CPU / free tiers, but use available compute.** The default is CPU (`$0`, reproducible by
  anyone). The tool now also runs on a GPU (`--device auto|cuda`, CPU-fallback, `device` recorded in
  provenance); GPU changes throughput only, never the numbers (see below).

- **Most of the compute cost is implementation, not the method.** `ConjointScorer.score_batch`
  (+ `scoring_plan.py`) batches masked forwards across variants, de-duplicates identical masked inputs,
  and tunes `torch` threads — all throughput-only and **bit-exact** to the per-variant reference (measured
  `delta_g` and `var_delta_g` gap = 0.0 at both 35M and 650M — `report/bench_35M.json`,
  `report/bench_650m.json`; guarded by `test_optimized_batch_matches_reference`):
  - **Masked-row de-duplication.** Masking a site erases its residue, so one masked forward yields the
    conditional distribution for all 19 substitutions there. The full 20-letter deterministic pass is
    **29,678 → 4,564 unique forwards** (≈6.5× fewer forward *calls*, ≈19× fewer masked *rows*; verified in
    `tests/test_scoring_plan.py`).
  - **Cross-variant batching + thread tuning.** `esm2_t33_650M` throughput measured in-session on this
    12-core CPU (`inference_mode`, 12 threads; reproducible with `scripts/bench_scoring.py` — `report/`
    is git-ignored, so these figures live on the machine, not in git): **0.66 forward-rows/s at batch 1,
    1.84 at batch 32** (≈2.8×); the batched 650M path in `report/bench_650m.json` runs at 1.83 rows/s.

- **The residual wall is fundamental to GPU-less hardware, not to the method.** The frozen headline needs
  `var_delta_g` (16 masking perturbations) on **every** candidate for info-optimal, and those background
  masks are essentially non-de-dupable — **≈1.39M short forward passes**. At the measured 1.84 rows/s that
  is **~8–9 days of CPU**. De-dup/batching/threads do not change that (the var pass dominates and de-dups
  little); only a GPU does. On a free Colab **T4** the same pass is **~1–4 h** (see
  [`headline_650m_colab.md`](headline_650m_colab.md)). This host has **no GPU** (`torch` CPU build,
  `cuda_available False`), so the full frozen headline is **deferred to a GPU**, run by that recipe.

- **What was run in-session instead** (both write `report/<run_id>/metrics.json` with full provenance,
  including `device`): (a) the **supplementary 650M full-alphabet deterministic-only** recovery — the
  4,564-forward regime above, ~40 min, giving the var-independent methods (fitness / random / practice /
  structural-only) at `B ∈ {48, 96, 192}`, `pool ≫ B` (`scripts/headline_650m_supplementary.py`); and
  (b) the **650M uncertainty-prior calibration** (`scripts/calibrate_uncertainty.py`, see §5). The
  supplementary run is **explicitly not the frozen headline**: info-optimal (which needs the var pass) is
  omitted and the `VALIDATION.md` decision rule is not evaluated there.

- **The reduced-alphabet 35M runs remain a smoke test, not the headline.** They restrict the per-site
  alphabet (e.g. `ADEF`, ~307 candidates) so the pool scores in minutes; recorded as provenance in every
  `metrics.json`. See the exhaustion caveat in §4.

## 2. Data

- **GB1/Wu-2016 has only four positions.** Statistical power comes from the ~20³ amino-acid
  instantiations per position-triplet, not from many positions. The claim is framed as *principle
  validation on a complete higher-order landscape*, never as a whole-protein positional map — no public
  higher-order dataset supports the latter.

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
  that is compute-bound here.

- **Confidence intervals measure two different things.** The deterministic methods' CI is bootstrapped
  over the evaluated ε terms — it reflects leverage concentration of the correlation, **not** "would a
  repeated wet-lab budget give a similar number." Only the random baseline's over-seeds CI captures
  genuine selection variance. Both are labelled with `ci_method` so they are not conflated.

- **A shared "informed-union" subset is deliberately not used.** Restricting the correlation to terms
  informed by *any* method collapses to the full term set — the union over the random baseline's seeds
  touches every term — so it cannot separate breadth from precision. The per-method breadth/precision
  split above is used instead.

## 5. Empirical (the current null)

- **The ESM zero-shot uncertainty prior is a confirmed null at both model sizes, now with backing
  code.** Two consistent pieces of evidence. (a) The structure-aware baseline recovers the pairwise map
  at least as well as the ESM-uncertainty ranking: on the 35M smoke the structural-only baseline
  (`τ²≡const`, rank by `n(v)`) *beats* info-optimal (`var_delta_g·n(v)`) at B=96 Spearman 0.97 vs 0.76,
  info-optimal never winning on precision; and on the 650M full-alphabet `pool ≫ B` supplementary run
  (§1) structural-only recovers pairwise at Spearman 0.48 / 0.46 / 0.50 (B ∈ {48, 96, 192}), above random
  (~0.28) and fitness-greedy (negative). (b) The uncertainty-prior calibration
  (`scripts/calibrate_uncertainty.py`) shows `var_delta_g` does **not** positively track the model's real
  per-variant prediction error `|b·ΔĜ − ΔG_measured|`: Spearman(σ², |error|) = **+0.042 (95% CI
  [−0.078, +0.157]) at 35M and −0.113 (95% CI [−0.220, −0.002]) at 650M**, both n=300 (reproducible from
  the raw pairs in `report/calibration_*/metrics.json`). Both are indistinguishable from — or slightly
  below — zero: the masking-perturbation dispersion is not larger where the model is more wrong, so it
  cannot be a useful acquisition signal — the mechanistic root of the ablation. **Moving to 650M does not
  rescue the prior** (if anything the correlation is marginally negative there). These are pre-registered
  ablations plus a mechanistic calibration with reported CIs, not post-hoc excuses. Remaining gap: the
  var-inclusive info-optimal-vs-structural-only comparison at 650M full alphabet is deferred to a GPU
  (§1, `headline_650m_colab.md`); the calibration already gives the reason it is expected to match, not
  beat, structural-only.

## 6. Statistical power & protocol scope

- **Third-order is underpowered at B ∈ {48, 96}.** Only ~0.3–0.6% of third-order terms are informed at
  these budgets, so a third-order null is a *power* limitation, not evidence that no effect exists.
  Pairwise is the better-powered, decisive order.

- **The frozen protocol is only partially exercised.** The headline requires 650M, the full alphabet,
  B ∈ {48, 96, 192}, and ≥ 20 seeds with bootstrap CIs. What has run is the 35M reduced-alphabet smoke and
  the structural-only ablation; the 650M headline and the uncertainty-prior calibration at scale remain
  compute-bound.

## 7. Not yet done (scope, not failure)

- **No 650M headline run** (compute, §1).
- **No downstream-impact demonstration** — the one test that escapes the recovery tautology (does a
  structure-aware budget's map support a *better decision* on held-out mutants?). This is the highest-value
  next step precisely because §4 makes the recovery headline thin on its own.
- **No second landscape** (generalisation) — premature until the mechanism on GB1 is settled.
- **No background-averaged ε, no MoCHI handoff, no multi-round sequential design** — all deliberately out
  of scope for v1.
