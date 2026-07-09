# Constraints & limitations

An honest register of the constraints that bound this project and the limitations they impose on what
can be claimed. It exists because the credibility of `epibudget` rests on a rigorous null-tolerant audit
(invariant #2): a reader must be able to see exactly where the walls are. Pairs with
[`VALIDATION.md`](VALIDATION.md) (the frozen protocol) and [`ROADMAP.md`](ROADMAP.md) (the plan).

Each item states the constraint, its consequence, and how the code/docs handle it honestly.

---

## 1. Compute & environment

- **CPU-only, `$0` compute (a design constraint, not an accident).** No GPU, no paid API. Everything
  runs on a laptop CPU. This is deliberate — it keeps the artifact reproducible by anyone — but it caps
  the scale of any single run.

- **650M scoring is ~20–30× slower than 35M on CPU.** Measured on this machine: `esm2_t12_35M` scores a
  variant in ~0.4 s (deterministic ΔG) to ~1.1 s (with 16 masking perturbations); `esm2_t33_650M` takes
  **22 s/variant at 8 perturbations and 32 s/variant at 16**. Consequence: scoring the full four-site,
  20-letter candidate pool (~29,678 variants) at 650M is **9–20+ hours**, and even a modest `pool ≫ B`
  pool (~1,500 variants) is **9–13 hours**. The headline 650M run in the frozen protocol therefore
  **could not be executed in-session**; it needs a machine that can hold a multi-hour job, or a GPU.

- **The fast-model runs use a reduced candidate alphabet for tractability.** The end-to-end and ablation
  runs on `esm2_t12_35M` restrict the per-site alphabet (e.g. `ADEF`, ~307 candidates) so the pool is
  scored in minutes. This is recorded as provenance in every `metrics.json` (`candidate_alphabet`,
  `n_candidates`, `scorer_seed`, `n_perturbations`, `data_sha256`). It is a **smoke test, not the
  headline** — see the exhaustion caveat in §4.

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

- **35M's uncertainty signal is near-noise.** The de-risk gate showed 35M's ε point estimates are weak
  (pairwise Spearman ≈ 0.085 vs 0.30 at 650M). The *uncertainty* (`var_delta_g`) is a separate quantity,
  but there is no reason to expect a weak-signal model to produce well-calibrated uncertainty. So a null
  on the uncertainty prior at 35M may not transfer to 650M — which is why the uncertainty-prior
  calibration is run at both sizes.

- **On the 35M smoke, the structural-only baseline beats info-optimal.** Ranking by `n(v)` alone (τ²≡const)
  recovers the pairwise map *better* than ranking by `var_delta_g·n(v)` (B=96 Spearman 0.97 vs 0.76), and
  info-optimal never wins on precision. Read literally, **the ESM zero-shot uncertainty prior does not
  help the allocation — it slightly hurts.** This is a pre-registered ablation result, not a
  post-hoc excuse. It is likely (not certain) to hold at 650M; confirming it there is compute-bound here,
  so the mechanistic proxy (does σ² correlate with real prediction error?) stands in for the full run.

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
