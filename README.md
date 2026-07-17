# epibudget

**Experimental-design methods for budgeted protein-epistasis mapping with zero-shot ESM-2 signals;
evaluated on GB1.**

`epibudget` ranks candidate protein variants under a fixed budget of *B* wells and compares allocation
heuristics for mapping pairwise and third-order epistasis. It is zero-shot (ESM-2), CPU-first and
GPU-capable, and is evaluated on the measured, viable subset of the public GB1 four-site landscape.

---

## The idea in one picture

Standard practice ranks variants by **predicted fitness** and tests the top *B*. Mapping epistasis asks a
different question: which measurements expose interaction structure across singles, doubles and triples?

The design principle is borrowed from **geodetic triangulation surveys**: when you chain measurements
across a network, the weakest link is never the least-precise instrument — it is the *poorly-braced
triangle*, the loop that fails to close. Surveyors don't add better instruments; they measure the
**redundant loops** that localise and cancel error. Transposed to a protein, this motivates prioritising
variants that *close epistatic loops* and comparing that structural heuristic with fitness-greedy, random
and ESM-weighted alternatives.

`epibudget` builds a factor graph over candidate mutations, seeds each interaction with an
ESM-2-derived dispersion, and ranks variants by a fixed modular weight: ESM dispersion times the number
of epistatic loops braced. A slider (`--lambda`) trades this heuristic against predicted fitness.

## Where it sits (and where it doesn't)

| Tool | Question it answers | Stage |
|------|---------------------|-------|
| ALDE / BO-EVO | which variants maximise **fitness** next round? | design (fitness) |
| **epibudget** | how do budgeted allocation heuristics expose **epistasis structure**? | **design (structure)** |
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

## The hypothesis under evaluation

> At equal budget *B*, variants selected by `epibudget` recover the ground-truth epistasis map of GB1
> (Wu et al. 2016, 149,361 measured genotypes from a theoretical 20⁴ space) **better** than the same budget spent on the highest
> predicted-fitness variants, and better than random.

Metric: order-stratified correlation and squared-error change between the prior and the epistasis map
inferred after revealing *B* measurements, at *B* ∈ {48, 96, 192}. See
[`docs/VALIDATION.md`](docs/VALIDATION.md).

## Result

<!-- artifact-claims:start -->
**Conjoint-score signal.** On the viable GB1 terms available in the local public-data artifact,
ESM-2 650M conjoint ε has pairwise Spearman **0.302** and third-order Spearman **0.249**
([provisional artifact](artifacts/signal_650m.json)). This supports an epistatic signal in the
scores; it does not validate the masking-variance uncertainty prior.

**Masking-variance calibration.** At 650M, Spearman(σ², |error|) is **−0.113, 95% CI
[−0.220, −0.002]** and Pearson is **−0.100, 95% CI [−0.198, 0.003]**, n=300
([provisional artifact](artifacts/calibration_650m.json)). This is weak negative rank association, not
evidence of positive uncertainty calibration; Pearson remains compatible with zero, so it is not a
general claim of anti-calibration. At 35M, Spearman is **+0.042, 95% CI [−0.078, +0.157]** and Pearson
is **+0.049, 95% CI [−0.083, +0.180]**, n=300
([provisional artifact](artifacts/calibration_35m.json)). Cross-fitted and order-stratified analyses are
pending.

**Comparative allocation status.** No comparative recovery claim is current. The existing 650M recovery
artifacts require re-reporting because the structural control has exact within-order ties and the
method-specific calibration slope confounds low-coverage comparisons. Pairwise and third-order results
must remain separate, and pooled recovery is diagnostic only.
<!-- artifact-claims:end -->

The defensible current position is narrower: conjoint ESM-2 scores contain epistatic signal, while
masking-perturbation variance has not demonstrated positive calibration or acquisition value.

A downstream-impact benchmark asks whether a structure-aware budget yields a better *training set* for
ranking held-out double and triple mutants ([`docs/specs/downstream.md`](docs/specs/downstream.md)). On
GB1, a decision-eligible run finds structure-aware selection beats fitness-greedy and random across all
20 salted partitions (S_macro-AUC Δ +0.342 vs fitness, +0.175 vs random); the masking-variance prior
adds nothing. An exploratory, non-decision-eligible replication on the TrpB four-site landscape
(Johnston 2024; enzyme catalysis) reproduces the direction — structure-aware beats random and
fitness-greedy 20/20 (Δ +0.135, +0.286), with fitness-greedy worse than random on both. Both results are
provisional and not yet registered artifacts; detail and caveats in
[`docs/experiments/trpb-downstream-generalization-20260716.md`](docs/experiments/trpb-downstream-generalization-20260716.md).

## How it works (3 steps)

1. **Conjoint ESM-2 scoring.** Each candidate variant is scored by mutating *all* its positions onto
   the background and reading the conditional log-likelihood — so genuine, context-dependent epistasis
   appears (additive per-site scoring would make every interaction term identically zero).
2. **Epistasis factor graph.** Nodes = candidate mutations, edges = pairs, hyper-edges = triplets. Each
   interaction ε gets an uncertainty seeded from ESM-2 masking-perturbation dispersion.
3. **Modular allocation heuristic.** Under the v1 independent-noise model the variance-reduction
   objective is an exact sort by ESM dispersion × loops-braced, optionally blended with fitness via
   `--lambda`. It is not a calibrated posterior-optimal design.

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
