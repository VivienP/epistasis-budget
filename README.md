# epibudget

**Spend your wet-lab budget on the variants that teach you the most about epistasis — not the ones with the highest predicted fitness.**

`epibudget` allocates a fixed experimental budget of *B* wells across candidate protein variants
(singles, doubles, triples) to **maximally reduce uncertainty about the epistatic structure** of the
fitness landscape. It is zero-shot (ESM-2), runs on a CPU, and is validated end-to-end on the complete
GB1 landscape.

> Status: initialised — spec and validation protocol frozen, implementation in progress.
> See [`docs/SPEC.md`](docs/SPEC.md) and [`docs/ROADMAP.md`](docs/ROADMAP.md).

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
ESM-2-derived uncertainty, and greedily selects the *B* variants that maximise total expected
reduction in epistasis uncertainty. A single slider (`--lambda`) trades this exploration against
plain fitness exploitation.

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
> (Wu et al. 2016, complete 20⁴ landscape) **better** than the same budget spent on the highest
> predicted-fitness variants, and better than random.

Metric: correlation between the epistasis coefficients inferred from the *B* selected measurements and
the ground-truth coefficients from the full landscape, at *B* ∈ {48, 96, 192}, against fitness-greedy
and random baselines. If the effect is weak or absent, the repo says so — it ships as a rigorous audit
of information-optimal DMS design, not a silent win. See [`docs/VALIDATION.md`](docs/VALIDATION.md).

## How it works (3 steps)

1. **Conjoint ESM-2 scoring.** Each candidate variant is scored by mutating *all* its positions onto
   the background and reading the conditional log-likelihood — so genuine, context-dependent epistasis
   appears (additive per-site scoring would make every interaction term identically zero).
2. **Epistasis factor graph.** Nodes = candidate mutations, edges = pairs, hyper-edges = triplets. Each
   interaction ε gets an uncertainty seeded from ESM-2 masking-perturbation dispersion.
3. **Information-optimal allocation.** Greedy submodular selection of the *B* variants that maximise
   total expected reduction in epistasis uncertainty (a BALD-style, variance-reduction acquisition),
   optionally blended with fitness via `--lambda`.

Full math and pseudocode: [`docs/SPEC.md`](docs/SPEC.md). Background on epistasis, the Walsh-Hadamard
formalism, and why this is well-posed: [`docs/RESEARCH_EPISTASIS.md`](docs/RESEARCH_EPISTASIS.md).

## Constraints

Python 3.12+ · CPU only · public data only (ProteinGym, GB1, UniProt) · `$0` compute.

## Citation & prior art

This tool stands on: Wu et al. 2016 (GB1 landscape), Poelwijk et al. 2016/2019 (Walsh-Hadamard
epistasis formalism), Faure & Lehner 2024 (MoCHI), and recent work on epistasis in protein language
models (Amir et al. 2024). Full references in [`docs/RESEARCH_EPISTASIS.md`](docs/RESEARCH_EPISTASIS.md)
and [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md).

## License

MIT.
