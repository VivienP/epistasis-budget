# epibudget

> Rank a shortlist of *B* protein variants for a DMS campaign designed to reveal epistatic
> structure, rather than maximize predicted fitness.

`epibudget` is a Python CLI for budgeted protein experimental design. It combines conjoint ESM-2
scores with an epistasis factor graph, then compares the resulting allocation with fitness-greedy,
structure-only, random, and practice-oriented baselines on public GB1 and TrpB landscapes.

## The idea in one picture

![epibudget workflow from protein target to ranked experimental shortlist](figures/epibudget_illustration.png)

The design borrows from geodetic triangulation: a measurement network becomes informative when it
closes poorly constrained loops. In a protein landscape, selected variants brace interaction loops
across singles, doubles, and triples.

## Where it sits (and where it doesn't)

| Tool | Question | Stage |
|---|---|---|
| ALDE / BO-EVO | Which variants maximize fitness next? | fitness design |
| **epibudget** | Which variants expose epistatic structure under budget? | structure design |
| [MoCHI](https://github.com/lehner-lab/MoCHI) | Which energies and couplings explain measured data? | inference |

`epibudget` selects measurements. It is neither a fitness optimizer nor an epistasis-inference
package. See [Prior art](docs/PRIOR_ART.md) for the full comparison.

## Quick start

Install from source with Python 3.12 or later:

```bash
git clone https://github.com/VivienP/epistasis-budget.git
cd epistasis-budget
python -m pip install .
```

Rank variants for a target FASTA and write the shortlist to `allocation.json`:

```bash
epibudget allocate --fasta path/to/target.fasta --positions 39,40,41,54 \
  --budget 96 --model esm2_t12_35M --n-perturbations 2 --out allocation.json
```

Run a smoke-scale GB1 validation after fetching the public dataset:

```bash
python scripts/fetch_gb1.py
epibudget validate --dataset gb1_wu2016 --model esm2_t12_35M --alphabet ACDGV \
  --budgets 48 --seeds 3 --n-perturbations 2 --device cpu
```

This smoke command is not the registered benchmark. Use the frozen settings in
[the validation protocol](docs/VALIDATION.md) to reproduce scientific results.

## The claim we test

> At equal budget *B*, does the ESM-weighted loop-bracing allocation recover the pairwise epistasis
> map of GB1 better than fitness-greedy and random allocation?

The benchmark reports Spearman and Pearson recovery separately for pairwise and third-order terms at
*B* in {48, 96, 192}. Measured fitness enters only after selection.

## Result

**Inconclusive.** The current corrective GB1 analysis found mixed evidence, remains provisional, and
is not eligible for a public comparative claim. The separate downstream GB1 and exploratory TrpB
reports are also provisional and do not change this map-recovery verdict. See
[the validation protocol](docs/VALIDATION.md) for the decision rules and evidence boundaries.

## How it works

1. **Score conjointly.** Apply every mutation in a variant before reading ESM-2 conditional
   log-likelihoods, preserving context-dependent interaction signal.
2. **Build the factor graph.** Represent candidate mutations as nodes and pairwise or third-order
   interactions as edges and hyperedges.
3. **Allocate the budget.** Use `--method structural` to rank by loops braced alone, or the default
   `--method info` to weight those loops by masking-perturbation dispersion. `--lambda` blends the
   info weight with predicted fitness.

See [the specification](docs/SPEC.md) for the model and pseudocode.

## Constraints

- Python 3.12 or later; CPU by default, CUDA opt-in with `--device cuda` or `--device auto`.
- Public protein landscapes only; GB1 epistasis analyses use complete, positive-fitness loops.
- The full ESM-2 650M variance-inclusive workflow is not presented as CPU-practical.
- Masking-perturbation variance has not demonstrated positive uncertainty calibration.
- `allocate` exposes both the ESM-weighted and structure-only allocation modes.

See [Constraints & limitations](docs/LIMITATIONS.md).

## Reproducing the benchmarks

The [validation protocol](docs/VALIDATION.md) defines the frozen settings and decision rules. GPU run
instructions live in [the 650M runbook](docs/headline_650m_colab.md), with notebooks indexed in
[notebooks/README.md](notebooks/README.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, offline quality gate, and pull-request
requirements.

## Citation & prior art

The scientific background and references are in [Research: epistasis](docs/RESEARCH_EPISTASIS.md).
The positioning against adjacent methods is in [Prior art](docs/PRIOR_ART.md).

## License

MIT.
