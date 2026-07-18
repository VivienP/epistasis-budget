# Gate 3 — correlated-error inference probe (cache-only, zero-GPU)

## Question

Gate 2 found that revealing info-optimal measurements improves the **rank** of recovered epistasis
(Δρ > 0) but nearly **doubles the squared error** (`sse_gain ≈ −0.9 … −1.0`). The diagnosed cause is
that ε is an inclusion–exclusion *difference* that cancels positively-correlated nested ESM error;
hard-pinning true measured ΔG on some loop members breaks that cancellation. This probe asks, on the
**same 650M cache** (no GPU): does a correlated-error prior over ΔG close the SSE gap without
destroying the rank gain? The answer decides `repair_current_core` vs `replace_phase2` for the
*inference* model only.

## Model

Work in ΔG space. Prior `z ~ N(μ₀, Σ_e)` with `μ₀(v) = b·esm(v)` (`b` the leakage-safe shared
cross-fit slope, identical to Gate 2). The ESM prior error `e = μ₀ − true` is modelled as an
**additive random effect over shared sub-mutations** plus independent residual:

    e(v) = Σ_{effect ⊆ v} a_effect + r_v ,   a ~ N(0, τ_a² I),  r ~ N(0, σ_r² I)

so `Σ_e = τ_a² G Gᵀ + σ_r² I`, where `G[v, effect] = 1[effect ⊆ v]`. The effect basis is either
`single` (size-1 sub-mutations) or `single+pair`.

Conditioning on the exactly-measured set `M` (Gaussian conditioning, equivalently a ridge BLUP via
Woodbury with `λ = σ_r²/τ_a²`):

    â = (Gₘᵀ Gₘ + λ I)⁻¹ Gₘᵀ eₘ ,          eₘ(m) = μ₀(m) − true(m)   (m ∈ M, revealed)
    μ_post(v) = true(v)            if v ∈ M
              = μ₀(v) − G_v â      otherwise
    ε̂_post(S) = Σ_{T ⊆ S} (−1)^{|S|−|T|} μ_post(T)

`λ → ∞` (τ_a²→0) gives `â = 0` ⇒ **exactly the Gate-2 pin baseline** (sanity invariant). `λ → 0`
gives the full additive correction.

## Leakage barrier

`λ`/`(τ_a², σ_r²)` are fit **only on the measured errors eₘ** (revealed labels, which inference may
use) by generalized cross-validation. Evaluation terms are unmeasured pair/third-order ε and never
enter selection, slope fitting, or hyper-parameter fitting. Selection is the frozen Gate-2 info
allocation (ESM-only). GB1 anchor is a bit-exact no-op (`f_WT = 1`).

## Outputs (per budget ∈ {48,96,192}, per order)

`sse_prior`, `sse_gain_pin`, `sse_gain_corr`, `Δspearman`/`Δpearson` for pin and correlated, the full
`λ`-frontier `(sse_gain, Δspearman)`, and bootstrap-over-terms CIs for the headline cells.

## Decision rule

- `repair_current_core` — some cache-only Σ (basis, λ*) yields `sse_gain_corr ≥ 0` at **all** budgets
  while keeping `Δspearman > 0` (CI excludes 0): only the inference needs the correlated posterior.
- `replace_phase2` — no cache-only Σ reaches `sse_gain ≥ 0` without collapsing the rank gain at ≥2
  budgets: the L2 defect is intrinsic to the independent-noise objective.
- `inconclusive_zero_gpu` — otherwise.

The probe is `public_claim_eligible = false`; it drives only the architecture decision.
