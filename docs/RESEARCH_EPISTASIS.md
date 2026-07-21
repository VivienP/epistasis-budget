# Deep research: epistasis, higher-order interactions, and the information geometry of measuring them

This document is the scientific foundation for `epibudget`. It explains what epistasis is, the two
mathematical formalisms that define it, why higher-order epistasis is the hard and interesting part,
what the GB1 landscape gives us, how protein language models represent epistasis, and — crucially —
why *choosing which variants to measure* is an information-theoretic problem. Every design decision in
[`SPEC.md`](SPEC.md) traces back to a claim here.

> Citations are to real, verifiable sources. Where an author list could not be verified, the reference
> is given by title + venue + year rather than invented names.

---

## 1. What epistasis is

Epistasis is the phenomenon whereby the effect of a mutation depends on the genetic background in
which it occurs — i.e. the phenotype of a combinatorial mutant is **not** the additive sum of the
effects of its constituent mutations. Formally, for two mutations *i* and *j* with fitness effects
Δ(i) and Δ(j) measured against a reference, the pairwise epistasis term is

```
ε(i,j) = Δ(ij) − Δ(i) − Δ(j)
```

ε = 0 means additivity; ε ≠ 0 means the mutations interact.

Epistasis is not a nuisance term — it is the central obstacle in protein engineering. It is estimated
that roughly **half** of the beneficial mutations that fix during laboratory evolution cannot be
explained from their individual effects on the starting protein, precisely because of epistasis
(reviewed in *Addressing epistasis in the design of protein function*, PNAS 2024). Complete knowledge
of every single-mutant effect does **not** determine the outcome of double mutants, let alone
higher-order combinations.

### Three qualitative classes (Weinreich et al.)

- **Magnitude epistasis** — interactions change the *size* of an effect but not its sign. All fitness
  steps keep the same direction; paths stay monotone.
- **Sign epistasis** — a mutation is beneficial on one background and deleterious on another.
- **Reciprocal sign epistasis** — each of two mutations is individually deleterious but their
  combination is beneficial (or vice versa). This is what creates **fitness valleys** and makes some
  adaptive paths inaccessible to stepwise selection (Weinreich et al. 2006, *Science*; Poelwijk et al.
  2007, *Nature*). Higher-order epistasis is *required* to convert reciprocal-sign into magnitude
  epistasis and vice versa — it reshapes which trajectories exist.

The practical consequence for design: the variants that matter most for an engineering campaign
(multi-mutants that jump valleys) are exactly the ones whose behaviour is *least* predictable from
lower-order data. That unpredictability is a measurable quantity — and reducing it is what `epibudget`
optimises.

---

## 2. Higher-order epistasis (HOE)

Pairwise epistasis is itself background-dependent: ε(i,j) can change depending on whether a third
mutation *k* is present. That third-order dependence is **higher-order epistasis**:

```
ε(i,j,k) = Δ(ijk) − Δ(ij) − Δ(ik) − Δ(jk) + Δ(i) + Δ(j) + Δ(k)
```

This alternating-sign, inclusion–exclusion form generalises to any order and is the discrete analogue
of a mixed partial derivative of the fitness function over the hypercube of mutations.

Key empirical facts (from *Should evolutionary geneticists worry about higher-order epistasis?*,
Curr. Opin. Genet. Dev. 2013; *The influence of higher-order epistasis on biological fitness landscape
topography*, 2018; *Higher-order epistasis and phenotypic prediction*, PNAS 2022):

- The **magnitude of epistatic terms declines with order on average** — first-order (additive) usually
  dominates, then pairwise, then third-order, etc. This is why sparse models work at all.
- But there are **notable exceptions** — specific high-order terms that are large and consequential.
  These exceptions "deserve experimental scrutiny." **Finding those exceptions cheaply is precisely
  `epibudget`'s target**: they are the loops where the model's uncertainty is high and the payoff of
  measurement is greatest.

---

## 3. The two formalisms — and why they matter for us

There are two inequivalent definitions of epistasis, and conflating them is a classic error. They are
unified by a single object: the **weighted Walsh–Hadamard transform (WHT)** of the fitness landscape
(Poelwijk, Krishna & Ranganathan 2016, *The Context-Dependence of Mutations: A Linkage of Formalisms*,
PLOS Comp. Biol.; Poelwijk et al. 2019, *Nature Communications*).

- **Biochemical / local epistasis.** Effects are measured relative to a single **wild-type reference**.
  ε(i,j) uses the WT background. In WHT terms, fixing a reference sequence *sub-samples* specific terms
  of the Hadamard matrix — "the reference sequence picks out the terms that concern the wild-type
  background." This is the operationally relevant quantity for a protein-engineering campaign that
  starts from one WT.
- **Background-averaged / ensemble epistasis.** Effects are averaged over the entire space of
  backgrounds. This is more robust to local idiosyncrasies and is the formalism used by inference tools
  like MoCHI, but it requires (or infers) many backgrounds.

> **Design decision #1 (see SPEC §Scoring).** `epibudget` targets **biochemical / WT-referenced**
> epistasis, because (a) it matches the practitioner's actual starting point (one WT), and (b) it is
> exactly what ESM-2 *conditional* scoring on the WT background produces. Background-averaged epistasis
> is documented here as a future extension, not the v1 target.

### The Walsh–Hadamard transform in one paragraph

Encode each of *n* mutated sites as a bit (0 = WT, 1 = mutant). The fitness vector over the 2ⁿ
hypercube can be written as a linear combination of Walsh functions; the coefficients are the epistatic
interactions by order (order-0 = mean, order-1 = additive main effects, order-2 = pairwise, …). The
transform is orthogonal, so the coefficients cleanly partition the landscape's variance across orders.
For amino acids there are 20 states per position, not 2; the **multiallelic extension** of the WHT
(*An extension of the Walsh–Hadamard transform to calculate and model epistasis in genetic landscapes
of arbitrary shape and complexity*, PLOS Comp. Biol. 2024) generalises this to arbitrary alphabets and
gives formulae to extract individual coefficients without building the full 20ⁿ matrix — important for
tractability.

---

## 4. The GB1 landscape — our ground truth

GB1 (the B1 domain of streptococcal protein G, an IgG-binding domain) is the canonical benchmark for
higher-order epistasis because two complementary complete datasets exist:

- **Olson, Wu & Sun 2014** (*A comprehensive biophysical description of pairwise epistasis throughout an
  entire protein domain*, Current Biology): all single and **double** mutants across 55 positions
  (~536,000 variants) — the reference map of pairwise epistasis.
- **Wu, Olson, Fowler & Sun 2016** (*Adaptation in protein fitness landscapes is facilitated by indirect
  paths*, eLife): a dense measured subset of the theoretical combinatorial landscape at four epistatically-coupled sites in the
  β1–β2 loop — **V39, D40, G41, V54**. The theoretical space contains 20⁴ = 160,000 variants; the
  local public-data artifact contains 149,361 measured rows across all four mutation orders.

The Wu 2016 four-site block makes `epibudget` testable on every positive-fitness interaction whose full
inclusion–exclusion loop is measured. Terms with a dead or absent member are excluded rather than
imputed. Within this conditional domain, we can (a) compute true epistasis coefficients and (b)
*simulate* a budgeted experiment by revealing the measured fitness of selected variants. GB1 is also
where ESM-2's landscape is reported to be dominated by pairwise and third-order interactions (§5),
which is exactly the regime `epibudget` reasons about.

> **Design decision #2 (see VALIDATION.md).** GB1/Wu-2016 is the validation substrate. It has only four
> *positions* (so only C(4,3)=4 position-triplets), but each is instantiated across thousands of
> amino-acid combinations — plenty of statistical power for the ε↔information claim. We therefore frame
> the deliverable as *validating the selection principle*, not as a whole-protein positional heatmap
> (which no public higher-order dataset supports). Honesty about this scope is a feature, not a caveat.

---

## 5. Epistasis in protein language models

Recent work establishes that protein language models represent epistasis **zero-shot**, without any
fitness supervision:

- *On Recovering Higher-order Interactions from Protein Language Models* (Amir et al., arXiv:2405.06645,
  2024) computes Walsh–Hadamard coefficients of ESM-2's predicted landscape via sparse Fourier recovery,
  extracting additive, pairwise and higher-order terms at sub-linear sample complexity. Open-source
  (`InteractionRecovery`). **This is the closest prior art and defines what is *not* novel** — see
  [`PRIOR_ART.md`](PRIOR_ART.md).
- *Protein Language Models Capture Structural and Functional Epistasis in a Zero-Shot Setting* (bioRxiv
  2025.09.14.676130) reports that GB1's ESM-2 landscape is dominated by pairwise and third-order
  interactions, that intermediate model sizes (~650M) are best, and that a nonlinear transform of raw
  scores shifts the signal from structural contacts toward functional couplings.

**The conjoint-scoring subtlety (the single most important implementation fact).** ESM-2 assigns a
score to any sequence. The epistasis it "believes in" exists *only* in the context-dependence of its
conditional token distributions. If a double mutant is scored as the sum of two independent single-site
masked-marginal scores on the WT background, the pairwise term is **identically zero by construction**.
To surface real model epistasis you must score the mutant **conjointly**: place all mutations of the
variant on the background and read the joint conditional log-likelihood (a pseudo-log-likelihood over
the mutated positions in the mutated context). This is the same "conditional / joint scoring"
distinction that separates zero-shot from rescue-mutation scoring in the literature (cf. masked-marginal
vs conditional scoring, Meier et al. 2021). It is invariant #1, enforced in [`SPEC.md`](SPEC.md) §3.

> **Design decision #3.** ε terms are computed from **conjoint** ESM-2 conditional scores on the WT
> background. A unit test (`test_epsilon_not_identically_zero`) fails loudly if additive scoring ever
> sneaks back in.

---

## 6. Measuring epistasis is an experimental-design problem

Here is the pivot from *analysis* to *design*, and the reason `epibudget` exists.

Given a budget of *B* wells, the standard move (ALDE, BO-EVO, and "test all pairwise of the top
beneficials", e.g. Arc Institute's MULTI-evolve) selects variants to **maximise expected fitness**. But
if the goal is to *learn the epistatic structure* — to build a model that generalises to untested
multi-mutants — the right objective is different: choose the variants whose measurement most **reduces
uncertainty about the epistasis coefficients**.

This is classical optimal experimental design, imported into sequence space:

- **Bayesian Active Learning by Disagreement (BALD)** (Houlsby et al. 2011) and **D-optimal design**
  select experiments that maximise expected information gain / minimise posterior variance of the model
  parameters — here, the epistasis terms.
- The **geodetic-triangulation intuition** (the project's founding collision): in a survey network the
  error is localised and cancelled not by better instruments but by measuring **redundant loops that
  must close**. The epistatic analogue of a closing loop is a triple {i,j,k} together with its faces
  {ij, ik, jk} and vertices {i,j,k}: measuring the members of a loop pins down ε(i,j,k). The most
  informative *B* variants are those that best "brace" the epistasis factor graph.

Under the v1 independent-noise Gaussian model this expected uncertainty reduction is **modular**, not
strictly submodular: `info_gain(M, v) = τ²_v · n(v)` does not depend on what has already been selected,
so allocation is a single exact sort rather than an iterative greedy loop — no combinatorial search,
tractable on CPU, and exactly optimal for that stated modular objective rather than (1 − 1/e)-near
optimal. The loop-closure / diminishing-returns intuition above is therefore **not realized in v1**; it
would require correlated priors across variants, which are out of v1 scope (SPEC.md §5, §11). This is
the engine in [`SPEC.md#acquisition`](SPEC.md).

> **Why this is well-posed and not circular.** The *uncertainties* that drive selection come from ESM-2
> (a masking-perturbation / ensemble dispersion of the conjoint scores — a zero-shot proxy for "how
> unsure is the model about this interaction"). The *ground truth* that scores the selection comes from
> the held-out GB1 measurements. Selection never sees the labels it is later graded on.

---

## 7. Design implications — how this research constrains `epibudget`

| Research fact | Consequence in the tool |
|---------------|-------------------------|
| Additive scoring ⇒ ε ≡ 0 (§5) | conjoint conditional scoring is mandatory; guarded by a unit test |
| Biochemical vs ensemble epistasis (§3) | v1 targets WT-referenced (biochemical) epistasis; matches ESM conditional scoring |
| HOE declines with order but has large exceptions (§2) | value = *uncertainty* of a term, not its expected magnitude — chase the exceptions |
| WHT partitions variance by order (§3) | WHT is used only on complete synthetic grids; real-GB1 truth uses WT-referenced complete loops |
| GB1/Wu-2016 provides dense higher-order measurements (§4) | it is the validation substrate; scope is conditional on complete positive-fitness loops |
| Uncertainty reduction is modular under v1 independent noise (§6) | allocation is a single exact sort, CPU-tractable; diminishing returns not realized |
| MoCHI infers, doesn't design (§6, PRIOR_ART) | `epibudget` is the design front-end that feeds inference tools |

---

## References

1. Weinreich, Delaney, DePristo, Hartl. *Darwinian evolution can follow only very few mutational paths
   to fitter proteins.* Science, 2006.
2. Poelwijk, Kiviet, Weinreich, Tans. *Empirical fitness landscapes reveal accessible evolutionary
   paths.* Nature, 2007.
3. Weinreich, Lan, Wylie, Heckendorn. *Should evolutionary geneticists worry about higher-order
   epistasis?* Curr. Opin. Genet. Dev., 2013.
4. Olson, Wu, Sun. *A comprehensive biophysical description of pairwise epistasis throughout an entire
   protein domain.* Current Biology, 2014.
5. Poelwijk, Krishna, Ranganathan. *The Context-Dependence of Mutations: A Linkage of Formalisms.*
   PLOS Computational Biology, 2016.
6. Wu, Olson, Fowler, Sun. *Adaptation in protein fitness landscapes is facilitated by indirect paths.*
   eLife, 2016. (The four-site GB1 combinatorial landscape.)
7. Sailer, Harms. *Detecting High-Order Epistasis in Nonlinear Genotype-Phenotype Maps.* Genetics, 2017.
8. Poelwijk, Socolich, Ranganathan. *Learning the pattern of epistasis linking genotype and phenotype in
   a protein.* Nature Communications, 2019.
9. *Higher-order epistasis and phenotypic prediction.* PNAS, 2022.
10. *An extension of the Walsh–Hadamard transform to calculate and model epistasis in genetic landscapes
    of arbitrary shape and complexity.* PLOS Computational Biology, 2024.
11. Faure, Lehner. *MoCHI: neural networks to fit interpretable models and quantify energies, energetic
    couplings, epistasis and allostery from deep mutational scanning data.* Genome Biology, 2024.
12. Amir et al. *On Recovering Higher-order Interactions from Protein Language Models.* arXiv:2405.06645,
    2024.
13. *Protein Language Models Capture Structural and Functional Epistasis in a Zero-Shot Setting.*
    bioRxiv 2025.09.14.676130, 2025.
14. Houlsby, Huszár, Ghahramani, Lengyel. *Bayesian Active Learning for Classification and Preference
    Learning* (BALD). arXiv:1112.5745, 2011.
15. *Addressing epistasis in the design of protein function.* PNAS, 2024.
16. Meier et al. *Language models enable zero-shot prediction of the effects of mutations on protein
    function* (ESM-1v; masked-marginal vs conditional scoring). NeurIPS, 2021.
