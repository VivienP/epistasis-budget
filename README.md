# epibudget

> Rank *B* protein variants that expose mutation interactions, rather than only variants a model
> predicts as fit.

`epibudget` is a Python CLI for budgeted protein experimental design. Its default strategy selects
variants by the number of interaction contrasts they appear in. Conjoint ESM-2 scores provide optional
fitness and masking-dispersion signals, and the benchmark keeps those contributions separate.

## The idea in one picture

![epibudget workflow from protein target to ranked experimental shortlist](figures/epibudget_illustration.png)

The design borrows from geodetic triangulation: a measurement network becomes more informative as it
covers more of the contrasts you want to estimate. In a protein landscape, selected variants cover
interaction contrasts across singles, doubles, and triples. The implemented objective rewards that
coverage additively; it contains no term for *completing* a contrast, so it does not by itself make
any interaction identifiable.

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
  --budget 96 --method structural --model esm2_t12_35M \
  --n-perturbations 0 --out allocation.json
```

Run a smoke-scale GB1 validation after fetching the public dataset:

```bash
python scripts/fetch_gb1.py
epibudget validate --dataset gb1_wu2016 --model esm2_t12_35M --alphabet ACDGV \
  --budgets 48 --seeds 3 --n-perturbations 2 --device cpu
```

This smoke command is not the registered benchmark. Use the frozen settings in
[the validation protocol](docs/VALIDATION.md) to reproduce scientific results.

## The claims we test

> At equal budget *B*, does a coverage-driven plate train a fixed learner to rank held-out double and
> triple mutants better than fitness-greedy and random allocation? Does ESM masking dispersion improve
> on mutation-order coverage alone?

The benchmark decides GB1 and TrpB separately. Measured fitness enters only after selection. The
epistasis-contrast comparison that this project originally headlined is now reported only as a
labelled diagnostic; see [the remediation record](docs/AUDIT_REMEDIATION_20260728.md) and
[protocol amendment 2](docs/specs/prospective-amendment-2.md).

## Result

An [independent mathematical audit](docs/AUDIT_REMEDIATION_20260728.md) withdrew the former
map-recovery claim: the epistasis-contrast correlation is reproducible but dominated by lower-order
measurements shared algebraically with the truth. It remains a diagnostic, not evidence of
epistasis-map reconstruction.

The historical v1 artifacts evaluate particular coverage-biased plates with a fixed
main-effects-plus-pairwise ridge:

| log2-budget AUC of S_macro | GB1 | TrpB |
|---|---:|---:|
| `structural` − `fitness` | +0.342 (20/20 partitions) | +0.286 (20/20) |
| `info` − `structural` at B=192 | +0.007 (15/20) | −0.025 (0/20) |

These are historical observations, not promoted estimates of the acquisition method. The later v2
reruns sampled only tie seed 0 and are not tracked public artifacts; neither v1 nor v2 estimates
performance across tie seeds. Masking-dispersion weighting did not pass its incremental v1 gate in
either tested landscape; this is not evidence that protein language models or uncertainty-aware
design are generally ineffective.

`structural` means interaction-loop coverage, not protein 3D structure. On these four-site landscapes
it reduces to singles, then doubles, then triples, with a seeded tie-break. A contrast over an
unassayed residue pair is exactly unidentifiable by this learner, so the project reports variant
ranking rather than map reconstruction. Corrected-recovery numbers remain withheld until validated
artifacts exist. See the [remediation record](docs/AUDIT_REMEDIATION_20260728.md) and
[limitations](docs/LIMITATIONS.md).

Read the [project write-up](https://vivienperrelle.com/journal/designing-protein-experiments-for-epistasis)
for a longer account of the work.

## How it works

1. **Score conjointly.** Apply every mutation in a variant before reading ESM-2 conditional
   log-likelihoods, preserving context-dependent interaction signal.
2. **Build the factor graph.** Represent candidate mutations as nodes and pairwise or third-order
   interactions as edges and hyperedges.
3. **Allocate the budget.** Use `--method structural` for the evaluated coverage strategy.
   `--method info` weights that coverage by ESM masking dispersion (it requires
   `--n-perturbations > 0`); `--lambda` blends either graph score with predicted fitness. The
   objective is a prior trace reduction under a diagonal masking-dispersion prior — it contains no
   loop-closure term, so it does not reward completing a contrast.

See [the specification](docs/SPEC.md) for the model and pseudocode.

## Constraints

- Python 3.12 or later; CPU by default, CUDA opt-in with `--device cuda` or `--device auto`.
- Public protein landscapes only; GB1 epistasis analyses use complete, positive-fitness loops.
- The full ESM-2 650M variance-inclusive workflow is not presented as CPU-practical.
- Evidence is limited to two four-site landscapes and one fixed downstream learner.
- Masking-perturbation dispersion did not pass its incremental gate in the tested artifacts.
- On a four-site landscape the coverage score is three-valued, so `structural` needs a declared
  tie seed to be reproducible.
- `allocate` retains both modes so the ESM contribution remains testable and reproducible.

See [Constraints & limitations](docs/LIMITATIONS.md).

## Citation & prior art

The scientific background and references are in [Research: epistasis](docs/RESEARCH_EPISTASIS.md).
See also [Prior art](docs/PRIOR_ART.md), the [validation protocol](docs/VALIDATION.md), and
[contributor setup](CONTRIBUTING.md).

## License

MIT.
