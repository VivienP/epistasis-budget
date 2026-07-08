# Prior art & the defensible wedge

Before committing three weeks, we mapped the literature around every candidate framing. This document
records what exists, what is therefore *not* novel, and the precise contribution `epibudget` claims.
Being explicit about this is itself a credibility signal: the failure mode of a portfolio project is
re-building something that shipped 12 months ago.

## The one-paragraph positioning

The literature is crowded on two fronts — (a) **inferring** epistasis from data, and (b) selecting
variants to **maximise fitness**. It is empty on the intersection we target: **selecting variants,
under a fixed budget and before any measurement, to maximise information about the epistatic structure
itself.** `epibudget` is an experimental-design front-end (what to measure) that feeds inference tools
like MoCHI (what the measurements mean), using an ESM-2 zero-shot uncertainty prior and a
loop-closure / BALD-style acquisition.

## Landscape

| Work | What it does | Relation to epibudget |
|------|--------------|-----------------------|
| **MoCHI** (Faure & Lehner, Genome Biology 2024) | Fits interpretable models; quantifies energies, couplings, epistasis, allostery **from** DMS data | **Complementary / downstream.** MoCHI analyses measurements; epibudget chooses which to take. epibudget's output is a natural MoCHI input. |
| **ALDE** (Nat. Commun. 2025), **BO-EVO** (Brief. Bioinform. 2023) | Active learning / Bayesian optimisation to reach **high-fitness** variants | Different objective (fitness, not epistasis information). epibudget's `--lambda` slider can reproduce fitness-greedy as a special case, used as the baseline to beat. |
| **MULTI-evolve** (Arc Institute) | Heuristic: pick 15–20 beneficial singles, test **all** their pairwise combinations | The current practical design. A fixed heuristic, not information-optimal; epibudget generalises and (claim) improves on it at equal budget. |
| **Amir et al. 2024** (arXiv:2405.06645, `InteractionRecovery`) | Walsh–Hadamard **recovery** of higher-order interactions **from ESM-2** predictions | **Closest prior art.** Shares the ESM-2 + WH machinery. Does *recovery/interpretation*, not budgeted experimental design, and does not use uncertainty to select experiments. We reuse their kind of decomposition but for a different job. |
| **PLMs Capture Epistasis Zero-Shot** (bioRxiv 2025.09.14.676130) | Characterises what epistasis PLMs capture zero-shot | Justifies our ESM-2 uncertainty prior; descriptive, no design method. |
| **BALD** (Houlsby et al. 2011), **D-optimal design**, **Statistical Guide to DMS design** (Genetics 2016) | General optimal-experimental-design theory | The method we import. Not previously applied to *epistasis-structure* selection in sequence space as a package. |
| **Benchmarking UQ for Protein Engineering** (PLOS CB 2025) and few-shot fitness works | Uncertainty for **fitness** prediction | Adjacent; targets fitness UQ, not epistasis-term UQ for design. |

## What is therefore NOT our contribution (say it plainly)

- The Walsh–Hadamard decomposition of ESM-2 predictions — **taken** (Amir et al. 2024, open-source).
- That PLMs capture epistasis zero-shot — **taken** (bioRxiv 2025.09.14.676130).
- Inferring energies/couplings from DMS data — **taken** (MoCHI).
- Active learning to maximise fitness — **taken** (ALDE, BO-EVO).

## The wedge (what we do claim)

1. **Objective reframing.** Allocate a fixed budget to maximise **reduction of uncertainty about the
   epistasis coefficients** (a factor-graph / loop-closure information objective), rather than to
   maximise predicted fitness.
2. **Zero-shot uncertainty prior.** Seed interaction uncertainties from ESM-2 conjoint-score dispersion
   under masking perturbations — no wet-lab data required at round 0.
3. **An open-source, practitioner-facing design tool** that outputs a ranked shortlist of B variants
   with expected information gain, and slots in front of inference tools like MoCHI.
4. **A validated, honest benchmark** on the complete GB1 landscape: does information-optimal selection
   recover the epistasis map better than fitness-greedy and random at equal B? Reported either way.

## Sibling ideas we rejected after the same novelty check (for the record)

These scored well in the brainstorm but were **pre-empted** — kept here so we don't circle back:

- **contamcheck** (ProteinGym × UniRef contamination) — pre-empted by *Beware of Data Leakage from
  Protein LLM Pretraining* (bioRxiv 2024) and LiveProteinBench (2025).
- **reyfit** (a-priori landscape-ruggedness → strategy) — pre-empted by *Learning-Based Estimation of
  Fitness Landscape Ruggedness for Directed Evolution* (bioRxiv 2024).
- **esm-nshot** (few-shot recalibration) — pre-empted by *Enhancing efficiency of PLMs with minimal
  wet-lab data through few-shot learning* (Nat. Commun. 2024) and in-context-learning few-shot works.
- **dms-noise-ceil** (replicate noise ceiling) — pre-empted by Livesey & Marsh DMS benchmarking (2023).
- **anticonsensus** (conservation ≠ fitness) — the core finding is established (VEPs predict beneficial
  variants poorly; Livesey & Marsh 2020).

## Residual risk

`epibudget`'s niche is adjacent to a very active BO/active-learning field. The differentiation rests on
the *objective* (epistasis-structure information, not fitness) and the *zero-shot prior*. If a reviewer
says "this is just BALD applied to epistasis terms" — yes, deliberately, and the contribution is the
concrete, validated, ESM-2-seeded instantiation that no one has packaged. The honest benchmark against
fitness-greedy is what turns the claim from plausible to demonstrated.
