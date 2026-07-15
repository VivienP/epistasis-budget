# Step 6A — Compressed-sensing (Fourier) epistasis-recovery baseline (cache-only, zero-GPU)

## Question

Gate 3 found our ESM inclusion-exclusion pipeline recovers the epistasis map weakly (residualized
pairwise rank ≈0.20–0.29, order-3 ≈0). The established method for budgeted epistasis recovery is
**compressed sensing** — sparse L1 recovery of the Walsh-Hadamard (Fourier) spectrum (Aghazadeh
"Epistatic Net"; Poelwijk 2019; Brookes 2022), with GB1 as the canonical testbed. Unlike the indicator
basis `1[S ⊆ v]` (sparse rows, cannot extrapolate), the multiallelic **Fourier basis is dense per
variant**, so sparsity lets it predict **unmeasured** ε. How does it position against our pipeline?

This is a **diagnostic baseline** (like `gate2`/`gate3`, and the sanctioned downstream ridge,
`docs/specs/downstream.md`): pure-numpy, it **never feeds selection**, and its report is
`public_claim_eligible = false`. It is not a production replacement for `infer_epistasis`.

## Model

For each budget B ∈ {48,96,192} and each selection method (`info` = the frozen Gate-2 allocation,
label-free; `random` = uniform, averaged over seeds), fit on the WT-centered ΔG of the measured set.

**Fourier design.** Per GB1 site s: alphabet `A_s = [WT_s, *sorted(non-WT)]` (q=20); per-site
orthonormal contrast basis `B_s = _orthonormal_contrast_basis(20)` (row 0 = mean mode). A coefficient
is a mode tuple `m = (m_0,…,m_3)`, order = #{s : m_s ≠ 0}; keep orders 1..3 (drop the order-0 mean).
Character `χ_m(v) = ∏_s B_s[m_s, idx_s(v)]`, `idx_s(v)` = residue index of v at site s (0 = WT). Design
`X[v, m] = χ_m(v)` (dense, p ≈ 29,678).

**Estimators (no ESM).**
- `fourier_lasso` — `min_β ‖y_M − X_M β − c‖² + λ‖β‖₁`, coordinate descent with soft-thresholding,
  unpenalized intercept `c`, λ chosen by K-fold CV on measured rows only.
- `fourier_ridge` — L2, single λ by the same CV. Companion.

**Sanity invariant (tested):** on a *complete* small landscape, the full-design least-squares fit
reconstructs ΔG exactly and its squared coefficients per order match `wht_spectrum` (Parseval).

**Recover ε (do not invert the basis).** Reconstruct `ΔĜ(v) = X_U[v]·β̂ + c` for every loop member of
the evaluation terms, then `ε̂(S) = _epsilon(ΔĜ_map, S)` — the same ε operator as the pipeline. This
sidesteps the WT-referenced-vs-background-averaged distinction (the Fourier β are background-averaged;
the recovered ε̂ are WT-referenced by construction).

## Leakage barrier

Fit and λ-CV use only measured labels (revealed via `data.reveal_measured_fitness`). Selections are
ESM-only (`info`) or label-free (`random`). Truth ε enters only the final scoring. No measured label
reaches selection (invariant #3). Dead/non-positive variants are dropped by `wt_centered_log_fitness`.

## Outputs (per budget, per order ∈ {pairwise, third}, per estimator)

Raw Pearson/Spearman and SSE of ε̂ vs truth; the **residualized** Spearman controlling for the
true-main-effect skeleton `k(S)` (Gate-3's `_skeleton`/`_partial_spearman`/`_residualize`) with a
bootstrap-over-terms CI; the count of estimable coefficients (support). The ESM pipeline's Gate-3
residualized recovery is the reference line.

## Decision rule

At operational budgets (B ≥ 96), per order:
- `compressed_sensing_competitive` — Fourier-LASSO residualized recovery ≥ the ESM pipeline's
  (CI-overlapping or higher): a standard sparse method suffices; the ESM prior is not decisive.
- `esm_pipeline_ahead` — the ESM pipeline's residualized recovery is strictly higher (CI-separated).
- `both_weak` — both residualized recoveries are ≈0 (map not recoverable at this budget by either →
  the bottleneck is acquisition/data, motivating Step 6B).
- `inconclusive` — otherwise.

`public_claim_eligible = false`; the report drives only the architecture decision.
