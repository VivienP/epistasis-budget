# Step 1 — de-risk gate result

Records the outcome of the Step 1 gate from [`ROADMAP.md`](ROADMAP.md): *does ESM-2 conjoint scoring
carry usable GB1 epistasis signal, before anything is built on top?* Reproduce with
`python scripts/fetch_gb1.py` then `python scripts/spike_gb1_epistasis.py --model <checkpoint>`.

## The gate (both must hold)

1. **`Var[ε_pred] > 0`** on a real GB1 slice — conjoint scoring is genuinely non-additive (invariant #1).
   Additive per-site scoring would make every ε identically zero.
2. **ESM-predicted ε correlates with measured ε**, Spearman ≳ 0.2 — judged **per interaction order**
   (pairwise, third), not on the pooled number (pooling a 3-term and a 7-term ε into one Spearman
   distorts the estimate and overstates significance through shared sub-terms).

## Data

`SaProtHub/Dataset-GB1-fitness` — the complete four-site GB1 landscape of Wu et al. 2016
(eLife 5:e16965), 149,361 measured genotypes at V39/D40/G41/V54 (0-indexed 38/39/40/53 = V/D/G/V),
`label` = fitness relative to the wild type (WT = 1.0). Verified on download: WT present, every variant
mutates only the four target sites, and all orders are represented — 76 singles, 2,091 doubles, 26,019
triples, 121,174 quadruples. 29,477 variants are dead (fitness 0). Checksum and composition in
`data/proteingym/provenance.json`. (The official `OATML-Markslab/ProteinGym` Hugging Face mirror ships
GB1's Olson-2014 pairwise set only, not this four-site assay.)

## Method

- **Measured** ΔG(v) = ln(fitness(v)); WT = 1 ⇒ ΔG(∅) = 0. Dead variants (fitness 0) have no
  log-fitness, so any interaction with a dead or missing constituent is **dropped, never imputed**
  (invariant #3).
- **Predicted** ΔG(v) is the conjoint ESM-2 conditional log-likelihood ratio (`ConjointScorer`,
  deterministic, `n_perturbations=0`).
- ε for both maps is the same WT-referenced inclusion–exclusion term (`epsilon_pairwise` /
  `epsilon_third`), evaluated over the same sampled amino-acid instances (50 combos per position-pair,
  40 per position-triple; seed 0). Because the single-mutant terms are subtracted with coefficient −1
  on both sides, the correlation isolates *interaction* agreement — it is not carried by ESM's known
  single-mutant fitness signal.

## Result

Spearman between predicted and measured ε, and the non-additivity variance, by ESM-2 size (seed 0,
n = 257 pairwise instances, 97 third-order):

| ESM-2 | pairwise ρ | third-order ρ | Var[ε_pred] |
|-------|-----------:|--------------:|------------:|
| 35M (`t12`)  | 0.085 | 0.108 | 0.361 |
| 150M (`t30`) | 0.167 | 0.131 | 0.530 |
| **650M (`t33`)** | **0.302** | **0.249** | **0.777** |

(Pooled Spearman, reported for context only: 0.114 / 0.120 / 0.316.)

## Verdict: **PASS** (at the 650M headline model)

- **Gate #1** holds at every size — `Var[ε_pred] > 0`, conjoint scoring is non-additive.
- **Gate #2** holds at 650M: both orders clear ≈0.2 independently (pairwise 0.30, third 0.25). The
  signal rises monotonically with model capacity, matching the literature finding that ~650M is the
  best regime for GB1 epistasis (`RESEARCH_EPISTASIS.md` §5). At 35M the correlation is weak (0.09/0.11):
  the fast model is a CI smoke-test, not the headline.

The signal is real and non-trivial — the epistasis factor graph and acquisition (Steps 2–4) rest on a
measured effect, not an assumption. **Proceed to Step 2.**

## Caveats (carried into Step 4)

- **Two seeds, indicative not final.** The gate is a go/no-go; the frozen Step 4 protocol adds ≥20
  seeds and bootstrap 95% CIs. A second seed at 650M reproduces the effect closely — seed 0:
  pairwise 0.302 / third 0.249; seed 1: pairwise 0.305 / third 0.231 — so the pass is not a
  sampling fluke.
- **Dead-variant exclusion** removes the strongest negative-epistasis (live→dead) cases, biasing the
  tested domain toward all-viable-constituent interactions; if anything this deflates the estimate.
- **Instance non-independence.** Sampled interactions share sub-ΔG terms, so an exact p-value/CI from
  the reported n would be optimistic — the point estimates stand, significance is treated as indicative.
- **35M vs 650M.** CI runs the pipeline on 35M as a smoke test; empirical claims use 650M.
