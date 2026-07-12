# epibudget

**Spend your wet-lab budget on the variants that teach you the most about epistasis — not the ones with the highest predicted fitness.**

`epibudget` allocates a fixed experimental budget of *B* wells across candidate protein variants
(singles, doubles, triples) to **maximally reduce uncertainty about the epistatic structure** of the
fitness landscape. It is zero-shot (ESM-2), CPU-first and GPU-capable, and is evaluated on the measured,
viable subset of the public GB1 four-site landscape.

---

## The idea in one picture

Standard practice ranks variants by **predicted fitness** and tests the top *B*. But the top-*B* by
fitness are correlated and cluster near the wild type, so they re-measure epistasis you already
half-know. The question *"which combinations most sharpen my map of the interactions?"* has a
different answer.

The design principle is borrowed from **geodetic triangulation surveys**: when you chain measurements
across a network, the weakest link is never the least-precise instrument — it is the *poorly-braced
triangle*, the loop that fails to close. Surveyors don't add better instruments; they measure the
**redundant loops** that localise and cancel error. Transposed to a protein: the most informative
variants to synthesise are the ones that *close epistatic loops* — the double and triple mutants whose
measured effect most tightens the model's uncertainty about the higher-order interaction terms.

`epibudget` builds a factor graph over candidate mutations, seeds each interaction with an
ESM-2-derived uncertainty, and ranks the *B* variants by a fixed weight — each variant's ESM
uncertainty times the number of epistatic loops it braces. A single slider (`--lambda`) trades this
exploration against plain fitness exploitation.

## Where it sits (and where it doesn't)

| Tool | Question it answers | Stage |
|------|---------------------|-------|
| ALDE / BO-EVO | which variants maximise **fitness** next round? | design (fitness) |
| **epibudget** | which variants maximise **epistasis information** under budget B? | **design (structure)** |
| [MoCHI](https://github.com/lehner-lab/MoCHI) | given measurements, **infer** energies & couplings | analysis |

`epibudget` is the **experimental-design front-end** to inference tools like MoCHI: choose what to
measure, then analyse the measurements. It is not a fitness optimiser and not an epistasis-inference
package. See [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md) for the full novelty landscape.

## Quick start

```bash
pip install -e ".[dev]"

# Rank the B=96 most epistasis-informative variants for a target
epibudget allocate --fasta target.fasta --positions 39,40,41,54 --budget 96 --model esm2_t33_650M

# Reproduce the GB1 validation (info-optimal vs fitness-greedy vs random)
epibudget validate --dataset gb1_wu2016 --budgets 48,96,192
```

## The claim we test (and will report either way)

> At equal budget *B*, variants selected by `epibudget` recover the ground-truth epistasis map of GB1
> (Wu et al. 2016, 149,361 measured genotypes from a theoretical 20⁴ space) **better** than the same budget spent on the highest
> predicted-fitness variants, and better than random.

Metric: correlation between the epistasis coefficients inferred from the *B* selected measurements and
the ground-truth coefficients from the full landscape, at *B* ∈ {48, 96, 192}, against fitness-greedy
and random baselines. If the effect is weak or absent, the repo says so — it ships as a rigorous audit
of information-optimal DMS design, not a silent win. See [`docs/VALIDATION.md`](docs/VALIDATION.md).

## Result

<!-- artifact-claims:start -->
**Conjoint-score signal.** On the viable GB1 terms available in the local public-data artifact,
ESM-2 650M conjoint ε has pairwise Spearman **0.302** and third-order Spearman **0.249**
([provisional artifact](artifacts/step1_signal_650m.json)). This supports an epistatic signal in the
scores; it does not validate the masking-variance uncertainty prior.

**Frozen headline (variance-inclusive, full 20-letter alphabet, 650M, *B* ∈ {48, 96, 192}, 20 seeds, run
on a Colab T4) — the prior-free sort wins, so the uncertainty prior is dropped.** The prior-free ablation
`structural-only` (`τ² ≡ const`, ranking purely by loop coverage `n(v)`) has the **higher** full-set
pairwise recovery at every budget — Spearman **0.484 / 0.460 / 0.504**, Pearson **0.514 / 0.526 / 0.573** —
than information-optimal, whose pairwise recovery is Spearman **0.408 / 0.418 / 0.443**, Pearson
**0.458 / 0.479 / 0.504** ([provisional artifact](artifacts/headline_650m.json)). A post-hoc paired analysis
over the terms both methods predict but neither pins (*B* = 48, n = 1,511 matched pairwise terms) puts
structural ahead on precision too — Spearman **0.452** vs **0.537** for info-optimal and structural, a
descriptive Δ −0.085, 95% CI [−0.125, −0.047] that excludes zero (and excludes zero at all three budgets;
[provisional artifact](artifacts/robustness_650m.json)). **There is therefore no evidence that the ESM
masking-variance uncertainty prior improves allocation** — the recovery is carried by the structural `n(v)`
loop-coverage sort — so per [`docs/VALIDATION.md`](docs/VALIDATION.md) the uncertainty prior is dropped from
the claims. The registered decision rule (info vs fitness vs random) is nonetheless **formally supported at
all three budgets**: information-optimal beats fitness-greedy **−0.259 / −0.247 / −0.134** and random
**0.279 / 0.280 / 0.287** on pairwise Spearman *and* Pearson with non-overlapping bootstrap 95% CIs. A
cross-fit scale-sensitivity probe agrees (structural > info > fitness at every order and budget).

**Masking-variance calibration.** At 650M, Spearman(σ², |error|) is **−0.113, 95% CI
[−0.220, −0.002]** and Pearson is **−0.100, 95% CI [−0.198, 0.003]**, n=300
([provisional artifact](artifacts/calibration_650m.json)). This is weak negative rank association, not
evidence of positive uncertainty calibration; Pearson remains compatible with zero, so it is not a
general claim of anti-calibration. At 35M, Spearman is **+0.042, 95% CI [−0.078, +0.157]** and Pearson
is **+0.049, 95% CI [−0.083, +0.180]**, n=300
([provisional artifact](artifacts/calibration_35m.json)). Cross-fitted and order-stratified analyses are
pending.
<!-- artifact-claims:end -->

The defensible current position: conjoint ESM-2 scores contain epistatic signal, and the frozen 650M
headline is formally supported under the registered rule (info-optimal beats fitness-greedy and random) —
but the prior-free structural allocation outperforms information-optimal on both full-set recovery and
matched precision, so the masking-variance uncertainty prior is dropped from the claims.
Masking-perturbation variance has not demonstrated positive calibration.

A downstream-impact benchmark — does a structure-aware budget yield a better training set for ranking
held-out double and triple mutants? — is implemented and specified
([`docs/specs/downstream.md`](docs/specs/downstream.md)); no confirmatory result has been produced yet.

## How it works (3 steps)

1. **Conjoint ESM-2 scoring.** Each candidate variant is scored by mutating *all* its positions onto
   the background and reading the conditional log-likelihood — so genuine, context-dependent epistasis
   appears (additive per-site scoring would make every interaction term identically zero).
2. **Epistasis factor graph.** Nodes = candidate mutations, edges = pairs, hyper-edges = triplets. Each
   interaction ε gets an uncertainty seeded from ESM-2 masking-perturbation dispersion.
3. **Information-optimal allocation.** Under the v1 independent-noise model the variance-reduction
   objective is modular, so allocation is an exact sort of the *B* variants by ESM-uncertainty ×
   loops-braced (the correlated-prior, strictly-submodular form is future work), optionally blended
   with fitness via `--lambda`.

Full math and pseudocode: [`docs/SPEC.md`](docs/SPEC.md). Background on epistasis, the Walsh-Hadamard
formalism, and why this is well-posed: [`docs/RESEARCH_EPISTASIS.md`](docs/RESEARCH_EPISTASIS.md).

## Constraints

Python 3.12+ · CPU-first, GPU-capable (`--device auto|cuda`) · public data only (GB1, UniProt) · no
claim that the full 650M variance-inclusive run is practically CPU-tractable.

## Citation & prior art

This tool stands on: Wu et al. 2016 (GB1 landscape), Poelwijk et al. 2016/2019 (Walsh-Hadamard
epistasis formalism), Faure & Lehner 2024 (MoCHI), and recent work on epistasis in protein language
models (Amir et al. 2024). Full references in [`docs/RESEARCH_EPISTASIS.md`](docs/RESEARCH_EPISTASIS.md)
and [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md).

## License

MIT.
