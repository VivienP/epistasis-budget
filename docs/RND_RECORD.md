# R&D record

This document is the record of every approach the project tried: what each one was meant to test, what it
produced, and whether it was kept, narrowed, superseded or abandoned. It spans scoring, acquisition,
inference, validation, benchmarking, data and infrastructure.

Abandoned approaches and negative results are recorded here deliberately. A dead end killed for a stated
reason is a result, and several entries below record the retraction of a claim this project once published.
Omitting them would make the surviving claims look better than the evidence supports.

The live, decision-eligible result is the downstream-impact benchmark: at equal budget, a structure-aware
plate is a better *training set* for ranking held-out double and triple mutants than a fitness-greedy,
practice-heuristic or random plate. Pure loop-bracing (`structural`) carries it; the ESM masking-dispersion
prior (`info`) adds nothing over it. Its numbers and protocol live in
`experiments/trpb-downstream-generalization-20260716.md` and `specs/downstream.md` and are not restated here.

Verdicts used below: **kept** (live and load-bearing), **narrowed** (retained under a reduced claim),
**superseded** (replaced by a better instrument), **abandoned** (killed), **inconclusive** (frozen or run
without reaching a verdict).

---

## Summary

| Approach | Theme | Verdict |
| --- | --- | --- |
| Conjoint (in-mutant-context) scoring instead of additive per-site scoring | scoring | kept |
| Masked-row de-duplication and cross-variant batching | scoring | kept |
| MC-dropout as the zero-shot uncertainty proxy | scoring | abandoned |
| Submodular loop-closure acquisition on a linear-Gaussian factor graph | acquisition | narrowed |
| Acquisition targets interaction uncertainty rather than predicted magnitude | acquisition | narrowed |
| The λ exploitation slider between info-gain and predicted fitness | acquisition | narrowed |
| Tie degeneracy of the structural sort, and the tie-break sensitivity replay | acquisition | kept |
| ESM-2 masking-perturbation dispersion as the calibrated uncertainty prior | acquisition | abandoned |
| Isotropic, label-free, coefficient-aware D-optimal acquisition | acquisition | abandoned |
| Order-restricted "weighted-pairs" D-optimal acquisition | acquisition | abandoned |
| WT-anchor correction (WT-centred log fitness so ΔG(∅) = 0) | inference | kept |
| Per-method calibration slope as a sign-setting confound; shared cross-fit slope | inference | kept |
| Independent-noise error propagation for σ²(ε) over a loop | inference | narrowed |
| Correlated-error prior over ΔG as an inference repair (gate 3) | inference | narrowed |
| Compressed-sensing (Fourier LASSO / ridge) coefficient-recovery baseline | inference | narrowed |
| Sparse-Bayesian coefficient model with sequential D-optimal acquisition | inference | abandoned |
| Background-averaged (ensemble) ε and the inference-tool handoff | inference | abandoned |
| Pre-allocation signal gate on conjoint ESM-2 epistasis signal | validation | kept |
| Label-leakage barrier: a single reveal point for measured fitness | validation | kept |
| Breadth vs precision decomposition of the recovery metric | validation | kept |
| Permutation null (label shuffle) control on the downstream benchmark | validation | kept |
| Determinism control: bit-identical re-runs and input-order invariance | validation | kept |
| Post-hoc robustness suite A1/A2/A3 | validation | narrowed |
| Walsh-Hadamard spectrum as a ground-truth identity check on real GB1 | validation | narrowed |
| Corrected-CV (Nadeau-Bengio) interval as the primary downstream gate | validation | narrowed |
| Corrective zero-GPU replay over fixed selections (gate 2) | validation | inconclusive |
| First uncertainty-prior calibration pass (n = 150, prose-only) | validation | superseded |
| Shared informed-union evaluation subset as the common grading set | validation | superseded |
| Pre-registered stop rule and closure-check fallback for a failed gate | validation | superseded |
| Pooled cross-order ε correlation as a reportable decision statistic | validation | abandoned |
| Pre-amendment confirmatory downstream campaign, stopped in flight | validation | abandoned |
| Downstream-impact benchmark on GB1 | benchmark | kept |
| Reordering the programme: downstream impact before second-landscape generalization | benchmark | kept |
| Structural-only ablation (τ² ≡ 1, rank by loops braced) | benchmark | narrowed |
| TrpB downstream generalization at `n_perturbations = 0` | benchmark | narrowed |
| MULTI-evolve-style practice-heuristic baseline | benchmark | narrowed |
| Reduced-alphabet (ADEF) 35M fast run as an evidence base | benchmark | narrowed |
| ESM-circular downstream diagnostic and its scale mismatch | benchmark | narrowed |
| Supplementary 650M deterministic-only recovery run | benchmark | superseded |
| Comparative epistasis-map recovery headline (frozen 650M) | benchmark | abandoned |
| Exploratory TrpB map-recovery smoke | benchmark | abandoned |
| Confirmatory second-landscape TrpB map-recovery benchmark | benchmark | inconclusive |
| GB1 dataset-completeness claim corrected to a measured subset | data | narrowed |
| PSD95-PDZ3 as the planned generalization landscape | data | superseded |
| Immutable provenance, checksummed artifacts, machine-checked claim registry | infrastructure | kept |
| GPU acceleration without a GPU-specific code path | infrastructure | kept |
| networkx and scikit-learn as runtime dependencies | infrastructure | abandoned |
| Reproducible headline-figure demo notebook | infrastructure | abandoned |

---

# Scoring

## Conjoint (in-mutant-context) scoring instead of additive per-site WT-background scoring

- **Verdict:** kept

**Question.** Can multi-mutant ΔG be read off summed single-site masked-marginal scores on the wild-type
background — the cheap path — or must every mutation be present in the context when each mutated residue is
read?

**What was built.** `ConjointScorer` in `src/epibudget/scoring.py`. `_score_one` builds the fully mutated
sequence first (`apply_mutations`), then for each mutated position masks *that* position while all other
mutations remain present, and sums `log P(mut_aa) − log P(wt_aa)` at the masked site (`_delta_g_pass`,
lines 303-334). A BOS off-by-one guard (`_assert_token_alignment`, lines 336-346) asserts the token at index
`p+1` really is the intended mutant residue. The forbidden additive form is kept as a *test-only* reference,
`additive_delta_g` (lines 60-66), documented "Never call this from the scoring path" and referenced nowhere
in `src/` outside its own definition. The batched throughput path `score_batch` is held to the per-variant
path as its parity oracle.

**What was measured.** Three layers.

1. Algebraic, offline: ε is inclusion-exclusion over the WT-referenced loop
   (`src/epibudget/epistasis.py:50-64`), so an additive ΔG map cancels term-for-term at order 2 and order 3.
   Pinned from both sides — additive map ⇒ ε ≈ 0 (`tests/test_scoring.py:41-49`,
   `tests/test_epistasis.py:69-72`) and a map with a +0.7 injected interaction ⇒ ε = 0.7
   (`tests/test_scoring.py:52-65`) — so a regression that silently collapses scoring to additive fails even
   with no model present.
2. End-to-end on real data: `test_epsilon_not_identically_zero` (`tests/test_scoring.py:88-114`) scores GB1
   singles and doubles over the four assay sites with ESM-2 35M and asserts `Var[ε] > 0`.
3. At the headline model, `scripts/gb1_epistasis_signal.py` (using `ConjointScorer(..., n_perturbations=0)`,
   line 136) reports pooled `Var[ε_pred]` plus order-stratified Spearman of predicted vs measured ε.

**Outcome.** The additive failure mode is structural, not empirical: ε ≡ 0 by construction for every
interaction term, so the cheap path destroys the measured object rather than approximating it. Conjoint
scoring produces genuine non-additivity. The full `tests/test_scoring.py` suite including the slow ESM-2
paths runs 9 passed in 496 s, so the real-slice `Var[ε] > 0` assertion holds as shipped. At 650M the
committed artifact `artifacts/signal_650m.json` records `var_eps_pred_pooled = 0.7771307634544032`
(n = 257 pairwise, 97 third-order, seed 0), with `spearman_pairwise = 0.302` and `spearman_third = 0.249`;
that artifact carries `evidence_classification: "traceable_not_rerun"`. `Var[ε_pred]` rises with model size
across the three sizes tabulated in `SIGNAL_GATE.md:45-49` (0.361 at 35M → 0.530 at 150M → 0.777 at 650M).

**Why this verdict.** The failure is provable rather than probabilistic, so it was killed by construction and
locked behind permanent tests instead of being re-litigated per run. It survived the later retrenchment
intact: `README.md:99-100` states the defensible position as "conjoint ESM-2 scores contain epistatic signal,
while masking-perturbation variance has not demonstrated positive calibration or acquisition value", and
`README.md:93-97` withdraws the comparative recovery claim entirely. Conjoint scoring is the piece left
standing when the claims above it were narrowed.

Two scoping notes, taken conservatively. The CLI's `[PASS/FAIL] invariant #1` line
(`src/epibudget/cli.py:141-144`) is a printed diagnostic computed from `predicted_epistasis_signal`
(`src/epibudget/validate.py:445-448`, thresholded at a float64 roundoff tolerance) — it does not raise or
exit non-zero; the *blocking* enforcement is the test suite. And the two gates are separable: "conjoint
scoring is non-additive" (structural, holds at every model size) is a stronger and independent claim from
"conjoint ε correlates with measured ε" (empirical, clears ≈ 0.2 per order only at 650M, on two seeds, with
`README.md:79-82` labelling the supporting artifact provisional).

**Evidence.**
- `src/epibudget/scoring.py:1-18` (module docstring, invariant statement), `:60-66` (`additive_delta_g`,
  test-only), `:262-278` (`_score_one`), `:303-334` (`_delta_g_pass`), `:336-346` (BOS alignment guard).
- `src/epibudget/epistasis.py:1-6`, `:50-64` (ε definitions that cancel under additivity).
- `tests/test_scoring.py:41-49`, `:52-65`, `:88-114`; `tests/test_epistasis.py:50-72`.
- `src/epibudget/validate.py:137-144`, `:445-448`; `src/epibudget/cli.py:141-144`.
- `SPEC.md:89-108` (§3.1 and the "Forbidden shortcut" block); `RESEARCH_EPISTASIS.md:155-167` (§5 "the
  conjoint-scoring subtlety", design decision #3), `:211`; `SIGNAL_GATE.md:9-10`, `:45-55`.
- `artifacts/signal_650m.json`; `scripts/gb1_epistasis_signal.py:136`; `README.md:79-82`, `:93-100`, `:121-123`.
- Commits `ade5263` (conjoint scorer with masking dispersion), `c7b7228` (batched and de-duplicated masked
  forwards, GPU device).

**Open discrepancy, not part of this verdict.** `SPEC.md:118-120` still describes `delta_g` as the *mean*
over the K perturbation passes, whereas the code computes `delta_g` from a single unperturbed deterministic
pass (`extra_mask=()`, `src/epibudget/scoring.py:276`) and uses the perturbation passes only for
`var_delta_g`. The module docstring (`scoring.py:8-12`) matches the code; the spec pseudo-code does not.

## Scoring throughput unblock: masked-row de-duplication and cross-variant batching

- **Verdict:** kept

**Question.** The 650M variance-inclusive headline needs `var_delta_g` (16 masking perturbations) on every
one of 29,678 candidates — roughly 1.39M short forward passes. Was that cost intrinsic to the scoring
method, or an artefact of scoring one variant at a time? And could it be cut without moving a single
published number?

**What was built.**
- `src/epibudget/scoring_plan.py` — a pure planner with no `torch`/`transformers` import. It enumerates every
  masked-marginal forward as plain data (`Row = (masked_seq, read_pos, mut_aa, wt_aa, pass_id, site_index)`),
  de-duplicates by `(masked_seq, read_pos)` (`dedup`), and reassembles per-variant `delta_g` / `var_delta_g`
  (`finalize`). `variant_key` is the single source of truth for per-variant RNG seeding, shared with the
  reference path, so perturbation draws are identical by construction. Being torch-free, the whole
  planning/de-dup/finalisation layer is testable offline with no ESM-2 download.
- `ConjointScorer.score_batch` (`src/epibudget/scoring.py:179`) consumes the plan and forwards the unique
  rows in `batch_size` chunks. The per-variant `ConjointScorer.score` is deliberately retained, untouched, as
  the parity oracle.
- Throughput knobs and resumability: `--device cpu|cuda|auto`, `--threads`, `--batch-size`
  (`src/epibudget/cli.py:188-189`, `:258-261`), and `src/epibudget/scored_cache.py`, a resumable
  scored-variant cache bound to immutable run metadata.
- `scripts/bench_scoring.py` — the measuring instrument, which also computes the reference-vs-batched score
  gap so a "speed-up" that silently changed numbers would be caught.

The physical insight: masking site `p` erases the residue there, so one forward of the mutant sequence with
`p` masked yields the conditional distribution for *all* 19 substitutions at `p`. Nineteen rows collapse to
one.

**What was measured.** (a) Unique-forward counts over the full 20-letter four-site GB1 pool; (b) wall-clock
and variants/s of reference vs batched paths; (c) `max_abs_delta_g_gap` and `max_abs_var_delta_g_gap` between
the two paths.

**Outcome.**
- De-duplication, full 20-letter deterministic pass: **29,678 → 4,564 unique forwards**, pinned by a passing
  offline test (`tests/test_scoring_plan.py::test_dedup_full_pool_matches_4564`) and decomposing exactly as
  4 singles + 228 order-2 + 4,332 order-3 rows. 29,678 / 4,564 = 6.50x fewer forward *calls*; the 86,716
  planned masked rows (76·1 + 2,166·2 + 27,436·3) / 4,564 = 19.0x fewer masked *rows*.
- Parity: `max_abs_delta_g_gap = 0.0` **and** `max_abs_var_delta_g_gap = 0.0` at both model sizes
  (`artifacts/bench_650m.json`, `artifacts/bench_35m.json`).
- Measured end-to-end speed-up, 650M (alphabet `AC`, 64 variants, 4 perturbations, batch 32, 12 threads,
  CPU): 502.574 s → 374.65 s, 0.127 → 0.171 variants/s, **1.341x**. 35M (alphabet `ACD`, 137 variants):
  84.203 s → 53.719 s, **1.567x**.
- The unblock did its job: the frozen headline subsequently ran to completion over all 29,678 candidates at
  `n_perturbations=16` on GPU (`artifacts/headline_650m.json`: `device = cuda`, full 20-letter alphabet).

**Scope caveat worth recording.** In the *benchmark* configurations the de-duplication contributes almost
nothing — `dedup_ratio` is 1.111 (650M) and 1.144 (35M), because the restricted `AC`/`ACD` alphabets share
few masked rows. The measured 1.341x/1.567x is therefore predominantly batching and thread tuning; the 6.5x
de-dup gain applies to the full-alphabet pass and is evidenced by a row-counting test, not by a wall-clock
measurement. The two numbers are not the same claim, and the repository does not conflate them.

**Why this verdict.** Kept, because it is throughput-only by construction and demonstrated to be so: the
reference path survives untouched purely to serve as an oracle, the benchmark records a zero score gap at
both model sizes, and `tests/test_scoring.py::test_optimized_batch_matches_reference` (slow-marked) asserts
agreement plus — the assertion that actually matters — that the info-optimal and fitness-greedy *selections*
built from either path are identical at budgets 8 and 16. That is what licenses an optimisation to sit
underneath a scientific claim.

The more interesting part of the record is a retraction. An earlier version of `LIMITATIONS.md` claimed 0.66
forward-rows/s at batch 1 versus 1.84 at batch 32 (about 2.8x), and extrapolated the 1.39M-forward variance
pass to "~8-9 days of CPU". Those figures came from an undocumented comparison whose artifacts were never
committed. Commit `e6c1bb0` deleted both, replacing them with the narrowly-scoped benchmark that *is* backed
by an artifact, plus the explicit sentence that the earlier undocumented batch-1 comparison "is excluded from
public claims", and "neither a complete CPU duration nor a Colab duration is published without a matching run
artifact". The larger, more flattering speed-up number was dropped because only the recorded configuration is
claimable.

A second, narrower conclusion also survived: most of the cost was implementation and removable, but the
residual variance-pass cost is not removable by de-duplication, since background masks are essentially
non-shareable across candidates. Rather than quietly rescoping the headline to fit a CPU-only budget, the
CPU-only execution constraint was dropped and the frozen run moved to a GPU. `headline_650m_colab.md` is
presented as a reproducible recipe, explicitly "not a runtime promise", with a throughput-measurement cell
that extrapolates an ETA before committing to the full run.

**Evidence.**
- `src/epibudget/scoring_plan.py` (`dedup`, `finalize`, `variant_key`); `src/epibudget/scoring.py:179`
  (`score_batch`), `:168` (`score` as oracle); `src/epibudget/cli.py:188-189`, `:258-261`;
  `src/epibudget/scored_cache.py`; `scripts/bench_scoring.py`.
- `tests/test_scoring_plan.py:65` and `:28-32`; `tests/test_scoring.py:149-184`, `_PARITY_ATOL = 1e-6` at
  `:31`, selection-identity assertions at `:182-184`.
- `artifacts/bench_650m.json`, `artifacts/bench_35m.json`, `artifacts/headline_650m.json`.
- `LIMITATIONS.md` §1; `headline_650m_colab.md` (the ~1.39M-forward figure, line 5).
- Commits `711e649`, `c7b7228`, `a69bbed`, `50cb70d`, `8487cd9`, `4abec4b`, `5bff991`, `5109299`, `e6c1bb0`.

## MC-dropout as the zero-shot uncertainty proxy

- **Verdict:** abandoned

**Question.** The acquisition objective needs a per-variant uncertainty term (`var_delta_g`) without any
measured labels. Can that be obtained the standard way — Monte-Carlo dropout, i.e. `K` stochastic forward
passes through ESM-2 with dropout left active at inference, taking the variance of the conjoint ΔG across
passes?

**What was built.** Nothing. MC-dropout was named as one of two candidate mechanisms in the original design
(`SPEC.md` §3.2, "either MC-dropout at inference (dropout enabled) or randomised masking order"), but no
implementation ever entered the tree — no code path enables dropout or calls `.train()` on the model. The
implemented substitute is background-context masking dispersion in `src/epibudget/scoring.py`
(`ConjointScorer`, `n_perturbations`/`mask_fraction`): each of `K` passes masks a random subset (default 15%)
of the positions *not* mutated by the variant, and `var_delta_g` is the variance of the conjoint score across
those passes.

**What was measured.** Inspection of the released ESM-2 checkpoint configurations rather than an experiment —
the failure is structural, so no run was warranted. Verified for all three checkpoints the project uses
(`facebook/esm2_t12_35M_UR50D`, `esm2_t30_150M_UR50D`, `esm2_t33_650M_UR50D`): every one ships
`hidden_dropout_prob = 0.0` and `attention_probs_dropout_prob = 0.0`, with `classifier_dropout = null`.
(`token_dropout = true` is present but is ESM's deterministic masked-token rescaling, not a stochastic
regulariser — it does not supply MC-dropout variance.)

**Outcome.** Dead on arrival. With every dropout probability at zero the `K` passes are bit-identical, so
`var_delta_g` would be identically zero for every variant and the uncertainty term in the acquisition
objective would carry no signal at all. The substitute is what shipped, and it is guarded rather than
assumed: `tests/test_scoring.py::test_var_delta_g_is_positive_and_deterministic` asserts that
background-context masking yields strictly positive dispersion *and* that it is reproducible for a fixed seed
— the two properties MC-dropout would have failed and passed respectively.

**Why this verdict.** Abandoned, not falsified: the mechanism is unavailable in the released weights, so
there was never an empirical question to answer. It is recorded because killing it is what makes the
masking-perturbation design legible — without this note the choice of a non-standard uncertainty proxy looks
arbitrary, when in fact the standard one is inapplicable to this model family.

**Evidence.**
- `SPEC.md` §3.2 (lines 110-121) — current text: masking perturbations, with the explicit note that
  MC-dropout is not used because ESM-2 dropout probability is 0.
- `src/epibudget/scoring.py` module docstring, lines 8-12 — same rationale at the point of use.
- `tests/test_scoring.py:117-133` — `test_var_delta_g_is_positive_and_deterministic`.
- `427c898` (initial scaffold) — SPEC §3.2 as first written still offered MC-dropout as an option.
- `ade5263` — the decision point; `git show ade5263:src/epibudget/scoring.py` already carries the "ESM-2 ships
  with dropout probability 0" rationale, and the commit message states "var_delta_g uses background-context
  masking since ESM-2 dropout is 0".
- `ebd2876` — SPEC §3.2 rewritten to match the code, removing the stale MC-dropout option.
- Checkpoint configs verified in the local model cache
  (`models--facebook--esm2_t{12_35M,30_150M,33_650M}_UR50D/.../config.json`).

**Gap.** The "dropout is 0" premise is asserted in prose in two places but is not itself asserted by any
committed test — nothing would fail if a future checkpoint shipped non-zero dropout. The shipped substitute
is covered; the reason it exists is not.

---

# Acquisition

## Submodular loop-closure acquisition on a linear-Gaussian epistasis factor graph

- **Verdict:** narrowed

**Question.** The founding analogy is geodetic: don't buy a better instrument, measure the redundant loops
that must close. Formalised, that asks whether expected reduction in epistasis uncertainty is a *strictly
submodular* set function — genuine diminishing returns as loops get braced — which would license greedy batch
selection under the classic (1 − 1/e) approximation bound and make a lazy-greedy priority queue a free
speed-up.

**What was built.** `src/epibudget/graph.py` — `EpistasisFactorGraph`: each candidate variant's ΔG carries an
independent Gaussian prior `N(ΔG_hat, τ²)` with `τ² = var_delta_g` (ESM-2 masking-perturbation dispersion);
each interaction coefficient ε(S) is the fixed ±1 inclusion-exclusion combination over its loop, so
`σ²(ε(S)) = Σ_{T∈loop} τ²_T`; measuring a variant reveals ΔG exactly and collapses that τ² to 0. It exposes
`posterior_variance`, `total_uncertainty`, `info_gain`. `src/epibudget/acquisition.py` provides `allocate()`
with the λ exploitation slider (λ=0 dispersion × loop-coverage, λ=1 exactly `fitness_greedy`). The
lazy-greedy priority queue was specified but **never implemented** (no occurrence of `lazy` anywhere under
`src/`).

**What was measured.** Analytic derivation of `info_gain(M, v)` under the shipped v1 model (independent
priors, zero-noise exact measurement), pinned by toy-graph tests in `tests/test_graph.py`: a non-strict
submodularity contract test and an exact-modularity regression tripwire. Separately, the loop-incidence count
`n(v)` was enumerated over the four-site, 20-letter candidate universe.

**Outcome.** The submodularity ambition was not realised — and the correction went in *both* directions.

- `info_gain(M, v) = τ²_v · n(v)`, where `n(v)` is the number of interaction loops containing `v`. This is
  independent of the already-measured set `M`, so the objective is **modular**: the degenerate special case
  of submodular where the diminishing-returns inequality holds with *equality*. Greedy is therefore not
  merely (1 − 1/e)-near-optimal but *exactly* optimal — and collapses to a single stable sort by a fixed
  weight, with no iterative greedy loop. The (1 − 1/e) bound and lazy-greedy are vacuous in v1. The
  loop-closure / diminishing-returns intuition, the project's founding image, is not realised.
- Worse for the method's selling point, `n(v)` dominates `τ²` and is **constant within each order**.
  Recomputed independently against `epibudget.epistasis.interaction_loop`: order 1 → 76 variants, all
  `n(v) = 1140`; order 2 → 2,166 variants, all `n(v) = 39`; order 3 → 27,436 variants, all `n(v) = 1`. The
  ranking front-loads singles then doubles; at small budgets it may measure **no triples at all**, and the
  ESM dispersion only breaks ties *within* an order.
- Downstream consequence: the `structural-only` ablation (`τ² ≡ 1`) has three distinct weight values in
  total, so its within-order ranking is an exact tie resolved by `enumerate_candidates`' site-major input
  order. That tie-break rule alone predicts the frozen artifacts' integers with no ESM input (GB1 B=48 pooled
  `n_informed` 17,700 predicted vs 17,700 observed; B=96/192 17,782; `n_pinned` 20/116; TrpB B=24 14,301,
  B=48 17,582).

**Why this verdict.** Narrowed, not abandoned: the mechanism ships and is the acquisition rule actually used,
but the theoretical claim was cut back to its honest form everywhere it appears rather than defended. Strict
submodularity would require correlated priors across variants, explicitly placed out of v1 scope. The wrong
claim originated in the initial scaffold (`427c898`: "Submodular greedy acquisition", "Submodular ⇒ (1 − 1/e)
near-optimal, cheap", "Optional lazy-greedy … identical result by submodularity") and was corrected at
implementation time — `graph.py` carried the honest "Submodularity claim (honest wording)" paragraph in its
very first commit. The same narrowing later forced the public framing down a second step, from
"information-optimal allocation (the thesis)" to "the v1 ESM-dispersion × loop-coverage heuristic, not
calibrated posterior-optimal design".

**Evidence.**
- `src/epibudget/graph.py` module docstring ("Submodularity claim (honest wording)" — modular, equality not
  strict, explicitly not a general A-optimal-design theorem); `info_gain` docstring "(≥ 0, modular)".
- `src/epibudget/acquisition.py` module docstring ("Modular budget allocation…"), `allocate` docstring
  ("`info_gain` is modular (graph.py), so this is a single stable sort — no iterative greedy loop").
- `SPEC.md:181-187` (§5 "Submodularity (honest form)"), `:207-230` (§6, including "the (1 − 1/e) submodular
  bound and the lazy-greedy priority queue are only relevant for a future correlated-prior model"),
  `:245-250` (`structural-only` ablation).
- `LIMITATIONS.md:70-80` §3 — "`info_gain` is modular, not strictly submodular" and "The modular weight
  structurally front-loads low-order variants".
- `VALIDATION.md:98`, `:380-385`; `experiments/trpb-smoke-20260713.md:215-250` (the 1140/39/1 table and the
  artifact-matching tie-break prediction).
- Commits `427c898` (scaffold, pre-correction), `ad83451` (graph, honest wording from birth), `4b983f5` (spec
  aligned: "Submodular ⇒ (1 − 1/e) near-optimal" replaced by "modular … greedy is *exactly* optimal"),
  `862b0ff` (acquisition), `168a144` (README: "greedily selects … maximise total expected reduction" → "ranks
  by a fixed weight"), `943833d` (further public narrowing to "dispersion × loop-coverage proxy").
- Tests `tests/test_graph.py:115-131` — `test_info_gain_is_submodular_non_strict_contract` and
  `test_info_gain_is_exactly_modular_under_independent_noise` (the tripwire, expected to fail if correlated
  priors are ever introduced). `pytest tests/test_graph.py tests/test_acquisition.py` → 20 passed.
- `n(v)` counts reproduced independently by enumerating all order-2/3 interactions over four sites × 19
  substitutions through `epibudget.epistasis.interaction_loop`: distinct `n(v)` = {1140}, {39}, {1}.

## Acquisition targets interaction uncertainty rather than predicted magnitude

- **Verdict:** narrowed

**Question.** Should a fixed measurement budget be spent on the interactions the model predicts to be
*large*, or on the interactions the model is *unsure* about? The premise is an empirical regularity in the
epistasis literature: epistatic magnitude declines with interaction order on average, so ranking by predicted
magnitude concentrates the budget on terms that are already well predicted, while the scientifically valuable
terms are the high-order exceptions — precisely where the model's uncertainty is high
(`RESEARCH_EPISTASIS.md:69-73`).

**What was built.** The objective is variance reduction over the interaction map, not a magnitude score.
`EpistasisFactorGraph` (`src/epibudget/graph.py`) gives every candidate variant an independent Gaussian prior
with τ² = `var_delta_g` and defines each interaction's prior variance as the plain sum of τ² over its
inclusion-exclusion loop. `total_uncertainty` is Σσ² over all interactions; `info_gain(M, v)` is its
reduction from measuring `v` (`graph.py:78-88`). Under the v1 independent-noise model this collapses to a
fixed weight `info_gain(∅, v) = τ_v² · n(v)` (`graph.py:62-66`), and `allocate`
(`src/epibudget/acquisition.py:28-73`) is a single stable sort on that weight at λ=0. There is no
`abs(epsilon_hat)` anywhere in the selection path — the only magnitude-based ranker in the repository is the
explicit control, `fitness_greedy` (`acquisition.py:76-79`, == `allocate(λ=1)`, top-B by predicted ΔG).

The ablation isolating the *uncertainty* half is a separate graph: `structural_graph`
(`src/epibudget/validate.py:362-371`) and the `unit_map` graphs in
`src/epibudget/downstream.py:2729,2734,2634` pin τ² ≡ 1, so the same sort reduces to `n(v)` — pure loop
bracing, ESM dispersion removed. Its purpose is stated in the docstring: "If info-optimal (which uses the
real τ²) does not beat selection by this graph, the ESM uncertainty prior contributes nothing to the
allocation" (`validate.py:365-367`; also `SPEC.md` §7 item 5).

**What was measured.** Two independent probes: the downstream-impact benchmark (`specs/downstream.md`), whose
two relevant pre-registered contrasts are `structural − fitness` (uncertainty-reduction objective vs
predicted magnitude) and `info − structural` (the τ² ablation), gated at ≥16/20 salted partitions positive
(`downstream.py:812-813`); and the uncertainty-prior calibration (`scripts/calibrate_uncertainty.py`), which
asks whether `var_delta_g` tracks absolute prediction error at all.

**Outcome.**
- *The "not magnitude" half is strongly vindicated.* GB1, R=20 × K=5 × 20 seeds, 650M,
  `n_perturbations = 16`, B ∈ {48, 96, 192}: `structural − fitness` is **20/20 partitions positive, mean
  S_macro-AUC +0.3423**, `structural_downstream_supported = true`. Per-method S_macro at B = 48/96/192:
  structural 0.423/0.572/0.587 vs fitness 0.123/0.194/0.272 — **fitness-greedy is worse than random**
  (0.260/0.359/0.474). TrpB reproduces the direction (`structural − fitness` 20/20, +0.286; fitness again
  below random), exploratory only.
- *The "uncertainty" half is not supported.* `info − structural` at B=192: **15/20 positive, below the 16/20
  gate, mean delta +0.0074** → `esm_uncertainty_supported = false`. The run is `decision_eligible = true`, so
  this is a real null, not a power failure.
- *Calibration is consistent with that null.* Spearman(τ², |error|) = **−0.113**, 95% CI [−0.220, −0.002];
  Pearson **−0.100**, CI [−0.198, 0.003] (`artifacts/calibration_650m.json`). No positive association; at 35M
  both intervals include zero.
- *Structural side-effect, documented.* Because `n(single) ≈ 1140`, `n(pair) ≈ 39`, `n(triple) = 1`, the
  modular weight is dominated by `n(v)`: the allocation measures ~all singles then doubles and at small
  budgets **may measure no triples at all**. τ² only breaks ties *within* an order (`LIMITATIONS.md` §3).
  Guarded behaviourally by `tests/test_validate.py:542-548`.

**Why this verdict.** Narrowed, not kept. Two separable propositions were bundled into one principle and they
came apart under measurement. The *rejection of predicted magnitude* survives decisively and is now the
strongest empirical result in the repository — the magnitude ranker is not merely beaten, it is beaten by
random on both landscapes. But the *positive* content originally attached to "uncertainty" — per-variant ESM
masking dispersion — contributes nothing: the τ² ablation is inside the noise band and the prior shows no
positive error calibration. What survives is still an uncertainty-reduction objective, but a purely
*structural* one: rank by how many unresolved interaction loops a variant braces, with model uncertainty
removed. The winning method (`structural`) is constructed only inside the benchmark harnesses; `allocate`
itself still always weights the graph by `var_delta_g`, so the shipped default is the unsupported variant and
the validated variant is the ablation — a gap the README states plainly (`README.md:127-136`,
`LIMITATIONS.md` §7).

**Evidence.**
- Objective and modularity: `src/epibudget/graph.py:1-19`, `:62-66`, `:78-88`.
- Acquisition sort and the magnitude control: `src/epibudget/acquisition.py:28-79`; `SPEC.md` §6 (lines
  205-229) and §7 item 5 (lines 245-252).
- τ² ablation: `src/epibudget/validate.py:362-371`; `src/epibudget/downstream.py:2576-2582`, `:2632-2636`,
  `:2728-2736`.
- Decision gates read from the GB1 downstream artifact `report/20260715T111312Z/downstream.json`:
  `decision.structural_gate` (20/20, +0.3422903), `decision.esm_gate` (15/20 vs threshold 16, +0.0073908,
  `supported: false`, `decision_eligible: true`).
- Calibration: `artifacts/calibration_650m.json` (`spearman_sigma2_abserror = -0.11282`,
  `spearman_ci95 = [-0.22029, -0.00205]`).
- Narrative and caveats: `experiments/trpb-downstream-generalization-20260716.md`; `LIMITATIONS.md` §3, §5,
  §7; `RESEARCH_EPISTASIS.md:69-73`.
- Tests (all pass): `tests/test_acquisition.py:43-75`, `tests/test_graph.py:105-131`,
  `tests/test_validate.py:542-548` → 20 passed.
- Commits `ad83451`, `862b0ff`, `56f7f99`, `1dda421`, `ebd2876`.

**Gap.** The design rule in its explicit form — value is the *uncertainty* of ε, not the magnitude of ε,
because higher-order magnitude declines with order and the payoff sits in the uncertain exceptions — has no
committed home. `RESEARCH_EPISTASIS.md` states the literature premise and `SPEC.md` states the resulting
objective, but neither records the rejected alternative (ranking by |ε̂|) as a deliberate, argued choice.

## The λ exploitation slider between info-gain and predicted fitness

- **Verdict:** narrowed

**Question.** Selection can rank candidates by expected information gain about the epistasis map, or by
predicted fitness. Is there a useful intermediate design — does blending the two (0 < λ < 1) buy anything
that neither endpoint gives?

**What was built.** `allocate()` in `src/epibudget/acquisition.py:28-73` implements the full convex blend
`score(v) = (1−λ)·minmax(info_gain(v)) + λ·minmax(delta_g(v))`, with `_minmax` (lines 19-25) mapping a
degenerate all-equal input to all-zeros. Both endpoints are special-cased to bypass the normalisation, which
is 0/0 when either score is constant across the pool: λ=1 sorts by raw `delta_g`, λ=0 by raw info-gain.
`expected_info_gain` in the returned `Allocation` is always the raw info-gain of each selected variant, never
the blended score (line 69). Shipped as `--lambda` on the CLI (`src/epibudget/cli.py:183`, default 0.0) and
as a validated config field bounded to [0,1] (`src/epibudget/types.py:96-98`). Specified in `SPEC.md` §6
(lines 205-230). Introduced by `862b0ff`.

**What was measured.** The interior was never evaluated on any scientific metric. Every benchmark harness
calls `allocate` at λ = 0.0 only: `validate.py:457,459`; `downstream.py:2579-2580`; `robustness.py:559,561`;
`gate2.py:449`; `gate3.py:303`. The fitness arm of every benchmark is built by `fitness_greedy(pool, budget)`
(`downstream.py:2526`, `validate.py:458`), not by `allocate(λ=1.0)` — so even the λ=1 endpoint is exercised
scientifically only through its equivalent code path. `git log -- src/epibudget/acquisition.py` shows three
commits and no λ-sweep; no script, notebook, doc or artifact passes an interior λ.

What the interior *does* have is unit coverage of its mechanics, not of its value:
`tests/test_acquisition.py:71-75` (λ=0.5 returns exactly `budget` distinct variants) and
`tests/test_validate.py:507-518` (mean predicted ΔG of the selection is monotone non-decreasing over
λ ∈ {0, 0.25, 0.5, 0.75, 1} on a synthetic pool, and strictly higher at λ=1 than λ=0). That proves the slider
slides. It says nothing about whether any interior point produces a better experimental design.

**Outcome.** No interior point has ever been benchmarked, and the endpoint evidence gives it no case to
answer. On the downstream training-set-quality benchmark the fitness endpoint is worse than *random* on both
landscapes — GB1 S_macro at B = 48/96/192: random 0.260/0.359/0.474 vs fitness 0.123/0.194/0.272; TrpB random
0.197/0.271/0.354 vs fitness 0.081/0.128/0.149
(`experiments/trpb-downstream-generalization-20260716.md:28,44`). Mixing in a component that is below chance
has no evidenced upside.

A second, sharper limit: the design that actually won the downstream benchmark is not reachable from the
slider at any λ. The winner is `structural` — the τ² ≡ 1 ablation, where the graph is weighted by a constant
instead of by `var_delta_g`, reducing the info weight to loops-braced alone. That graph is constructed only
inside the benchmark harnesses (`validate.py:451`, `downstream.py`, `gate2.py`); the CLI `allocate` command
always builds its graph with `var_delta_g` (`cli.py:215-216`). So `--lambda` interpolates between the
*ESM-dispersion-weighted* info heuristic and fitness — a line that does not pass through the validated
design. This is stated in `README.md:126-131`.

**Why this verdict.** Narrowed rather than abandoned: the slider stays in the codebase because the λ=1
endpoint is a genuine, tested reduction — `allocate(λ=1)` reproduces `fitness_greedy` as an *ordered list*,
not merely as a set (`tests/test_acquisition.py:43-46`), which keeps the baseline honest by construction and
makes the label-inversion no-leakage canary (`tests/test_acquisition.py:80-86`) meaningful on the same code
path. But nothing in the evidence base rests on the interior, no claim cites it, and the two facts above
(fitness below random; the validated design off the line) mean an interior sweep would be searching a segment
with a known-bad endpoint for a design that is known not to lie on it. It is retained as a shipped knob and a
specification-level generalisation, explicitly not as a validated method.

**Evidence.**
- `src/epibudget/acquisition.py:19-25` (`_minmax`), `:28-73` (`allocate`, blend at `:56-63`), `:76-79`
  (`fitness_greedy`); `src/epibudget/cli.py:183`, `:215-218`; `src/epibudget/types.py:96-98`.
- λ=0.0 call sites: `src/epibudget/validate.py:457,459`; `src/epibudget/downstream.py:2579-2580`;
  `src/epibudget/robustness.py:559,561`; `src/epibudget/gate2.py:449`; `src/epibudget/gate3.py:303`.
- Fitness arm built via `fitness_greedy`, not λ=1: `src/epibudget/downstream.py:2526`,
  `src/epibudget/validate.py:458`.
- `tests/test_acquisition.py:43-46`, `:49-55`, `:58-62`, `:71-75`, `:80-86`;
  `tests/test_validate.py:507-518`.
- `SPEC.md:205-230`, `:238-239`; `README.md:126-131`;
  `experiments/trpb-downstream-generalization-20260716.md:26,28,40,44,63-67`.
- Commit `862b0ff`; `git log --oneline -- src/epibudget/acquisition.py` (no sweep commit).

## Tie degeneracy of the structural sort, and the tie-break sensitivity replay

- **Verdict:** kept

**Question.** The `structural-only` ablation ranks candidates by loop coverage `n(v)` alone. Does that weight
actually order candidates, or is its whole within-order ranking an exact tie silently resolved by candidate
enumeration order? And if it is a tie, does the ablation's verdict — "the ESM masking-dispersion prior earns
no credit over pure loop-bracing" — survive other, equally valid resolutions of that same tie?

**What was built.**
- `src/epibudget/validate.py:362` `structural_graph`: an `EpistasisFactorGraph` with τ² ≡ 1.0 for every
  variant, so its greedy weight collapses to `n(v)` (`src/epibudget/graph.py:63`, `weight[v] = τ_v² · n(v)`).
- The tie is resolved implicitly at `src/epibudget/acquisition.py:55` — `allocate(λ=0)` is a single Python
  `sorted` (stable), so equal weights keep the input order emitted by `data.enumerate_candidates`
  (`src/epibudget/data.py:46`), which is order-major, then site-major, then residue-alphabetical.
- `src/epibudget/gate2.py` (1,812 lines, commit `621cfce`) replays the frozen 650M cache under 100 seeded
  within-stratum permutations of each exact score tie (`_score_strata` / `_permuted_strata_order`, lines
  315-336; `structural_seeded`, tie-break id `exact-score-strata-canonical-pcg64-v1`). The original single
  prefix is retained beside it as `structural_legacy_prefix` (`input-order-stable-v1`) for diagnosis only.
- An ad-hoc replay over the frozen cache substituting alternative deterministic tie-breaks of the identical
  tie class. Its results are written up in the experiment note; no replay script was committed.

**What was measured.** (a) The number of distinct `n(v)` values per mutation order over the full GB1
universe. (b) `n_informed` / `n_pinned` predicted from the tie-break rule alone, with no ESM input, against
the frozen artifacts. (c) Pooled Spearman ε-recovery at B=48 under each alternative tie-break, over the same
frozen 650M cache. (d) In gate 2, the registered pairwise statistic: `info − structural_seeded` post-hoc
Pearson and Spearman over 100 permutations, at B ∈ {48, 96, 192}, under two calibration regimes.

**Outcome.**
- `n(v)` takes exactly three values across all 29,678 candidates: 1140 (76 singles) / 39 (2,166 doubles) / 1
  (27,436 triples) — reproduced independently. The weight carries *zero* within-order information, so at any
  B ≤ 76 `structural-only` is literally "take the first B singles in enumeration order".
- Reproduced with no ESM score read at all — enumeration order alone predicts the frozen GB1 artifact's
  coverage exactly: B=48 → `n_informed` 17,700 and `n_pinned` 0; B=96 → 17,782 / 20; B=192 → 17,782 / 116.
  All six match `report/20260711T091947Z/metrics.json`. The first 48 picks are 19 at site 38, 19 at site 39,
  10 at site 40 and **none at site 53**.
- Tie-break sensitivity at B=48, pooled Spearman, reproduced from `report/scored_650m.jsonl`: as-run
  enumeration order **0.2470** (= the frozen artifact), reversed enumeration order **0.0355**, balanced
  12-per-site **0.1736**, against info-optimal's **0.1997** (also the frozen artifact value). **Under
  reversed enumeration order the ablation's verdict flips and info-optimal wins.** Twenty random draws from
  the tie class land near ρ ≈ 0.15 (two independently constructed families: mean 0.147, sd 0.054, 3/20 beat
  info-optimal; and mean 0.152, sd 0.044, 2/20 beat info-optimal). The as-run prefix sits in the extreme
  upper tail of that distribution — an unreplicated lucky draw.
- At B=96 and B=192 all 76 singles are forced and only the doubles are tied; there, structural beats
  info-optimal under **20/20** tie-breaks (B=96 draws span 0.1720-0.2440 vs info 0.1644; B=192 span
  0.2178-0.2756 vs info 0.1752). The verdict is genuinely robust at those budgets.
- Gate 2 supplies the registered *pairwise* version over 100 seeded ties
  (`report/20260714T104137Z/gate2.json`, `aggregates.tau2`): at B=48 all four cells straddle zero (e.g.
  operational Spearman q025 −0.0531, q50 +0.0294, q975 +0.1085) — info and structural are not separable; at
  B=96 all four cells are negative (structural robustly ahead); at B=192 three negative and one straddling.
  Overall `tau2_status = inconclusive`; the registered decision is `inconclusive_zero_gpu`, with
  `public_claim_eligible = false`.

**Why this verdict.** Kept — this is a first-class negative finding about the project's own control, and it
changed what the project is allowed to claim. It showed that a headline reported as holding at *every* budget
is tie-break-dependent at the smallest one, and it did so by a cheap, falsifiable prediction (derive the
artifact's five-digit coverage counts from the sort rule alone, with the model switched off) that matched
exactly. The response was not to delete the ablation but to bound it: the legacy prefix is demoted to a
diagnostic, the tie is randomised 100 ways under a registered statistic, and no direction is promoted to a
public claim. The rule generalized — the same degeneracy argument later disqualified `info` in the TrpB
downstream run, where `n_perturbations = 0` makes `var_delta_g` exactly zero for all 29,678 variants and its
selection likewise reduces to an arbitrary tie-break; those numbers are explicitly not claimed.

One honest limit on the replay itself: the three deterministic tie-breaks reproduce to four decimals, but the
published 20-random-draw summary (mean 0.1449, sd 0.045, min 0.0567, max 0.2168, "4 of 20 beat info", "as-run
above the maximum") is not reproducible as stated — the RNG construction for those draws was never recorded.
Two independent reconstructions give the same qualitative picture and comparable means, but in one of them 1
of 20 draws does exceed the as-run value. The load-bearing conclusions (reversed order flips the B=48
verdict; the as-run prefix is an upper-tail draw; B=96/192 are robust) survive both reconstructions; the
"above the maximum of all 20" phrasing does not, and should be stated as "in the extreme upper tail".

**Evidence.**
- `src/epibudget/validate.py:362-372`; `src/epibudget/graph.py:63`; `src/epibudget/acquisition.py:51-55`;
  `src/epibudget/data.py:46-77`.
- `src/epibudget/gate2.py:315-336`, `:440-514`, `:69-72`; commit `621cfce`.
- `tests/test_gate2.py:345` `test_corrected_selections_are_input_order_invariant_and_legacy_is_not` — the
  finding is codified as a test: reversing the input pool changes the legacy structural selection and nothing
  else.
- `report/20260711T091947Z/metrics.json`; `report/20260714T104137Z/gate2.json` (sha256
  `cb24af4f0ffd025260b430fa069075653608d048d8efd2a42c05b47384149fe5`).
- `experiments/trpb-smoke-20260713.md` §6.2; `VALIDATION.md` §"Corrective Gate 2" and defect 2 in the TrpB
  consequence list; `LIMITATIONS.md` §6b bullet 2; `README.md` §"Comparative allocation status".
- `src/epibudget/downstream.py:812` — `("structural", "fitness", "auc")` is still the downstream benchmark's
  primary contrast, so the same structural graph remains the live control;
  `experiments/trpb-downstream-generalization-20260716.md:44-49` applies the identical degeneracy argument to
  disqualify `info` at `n_perturbations = 0`.
- Reproductions run against `report/scored_650m.jsonl` and `data/proteingym/gb1_wu2016.csv` using
  `validate.map_recovery` / `infer_epistasis`; no committed script performs this replay.

## ESM-2 masking-perturbation dispersion (`var_delta_g`) as the calibrated uncertainty prior

- **Verdict:** abandoned

**Question.** The acquisition objective's only ingredient beyond structure is `var_delta_g` — the dispersion
of the conjoint ΔG score across K random background-masking passes — used as the per-variant prior variance
τ². It is a usable zero-shot uncertainty signal only if it is *larger where the model is more wrong*. Does it
track the model's actual per-variant prediction error, and does weighting by it select a better measurement
budget than loop-coverage alone?

**What was built.**
- `src/epibudget/scoring.py:280` `_var_delta_g` — K stochastic background-masking passes over the conjoint
  ΔG, returning `np.var(passes)`; `n_perturbations=0` short-circuits to 0.0.
- `src/epibudget/graph.py:66` — the linear-Gaussian factor graph sets `weight[v] = τ_v² · n(v)` with
  `τ² = var_delta_g`; `acquisition.py` sorts by that weight.
- `src/epibudget/validate.py:362` `structural_graph` — the prior-free ablation, `τ² ≡ 1`, so
  `info_gain(∅, v) = n(v)`. Pre-registered *before* any headline existed (commit `c934bb2`; implemented
  `6900206`).
- `src/epibudget/calibrate.py` + `scripts/calibrate_uncertainty.py` (commit `b84d0ee`) — pure, ESM-free
  calibration math: put ΔĜ on the measured scale with the through-origin slope `b` (reusing
  `validate._calibrate_slope`), form `|b·ΔĜ − ΔG_measured|`, correlate σ² against it with a 1000-resample
  percentile bootstrap CI. Tests in `tests/test_calibrate.py`.

**What was measured.** Three independent lines. (1) *Mechanistic calibration:* Spearman and Pearson of σ² vs
|calibrated error| with bootstrap 95% CIs, n=300 covered GB1 variants (order composition 1/28/271), seed 0,
`n_perturbations=16`, full 20-letter alphabet, at both ESM-2 35M and 650M. (2) *Map-recovery ablation:*
pairwise/third-order recovery of `info` vs the prior-free `structural-only` graph at B ∈ {48, 96, 192}.
(3) *Downstream training-set quality:* the `info − structural` contrast on S_macro-AUC over 20 salted
partitions (R=20 × K=5 × 20 seeds), gate = 16/20 positive partitions.

**Outcome.** Null, at every line, and not rescued by scale.

| model | Spearman(σ², \|err\|) | 95% CI | Pearson | 95% CI | slope b |
|---|---|---|---|---|---|
| 35M  | +0.042 | [−0.078, +0.157] | +0.049 | [−0.083, +0.180] | 1.131 |
| 650M | −0.113 | [−0.220, −0.002] | −0.100 | [−0.198, +0.003] | 0.791 |

Both 35M intervals include zero; at 650M the Spearman interval excludes zero but the Pearson one does not —
so a weak *negative rank* association is reported and the stronger "anti-calibration" claim is explicitly
refused. Scaling 35M → 650M makes the prior marginally worse, not better.

Downstream (GB1, decision-eligible): `info − structural` positive in **15/20** partitions, below the 16/20
sign gate, mean S_macro-AUC **+0.007** → not supported. On the TrpB replication the `info` numbers are not
interpretable at all: at `n_perturbations = 0`, `var_delta_g` is exactly 0 for all 29,678 variants, so
`info`'s weights are degenerate and its selection collapses to an arbitrary tie-break — the reason an n≠16
run is pinned non-decision-eligible.

Map-recovery ablation: the frozen 650M run had structural-only ahead of `info` at every budget on both
correlations, which is what originally triggered the pre-registered drop clause. That *interpretation has
since been withdrawn* — the structural score is exactly tied within mutation order and the run used a single
deterministic enumeration-order tie-break, so it was an artifact. The corrective seeded re-analysis is
`inconclusive_zero_gpu`; the dispersion-versus-seeded-structural-ties contribution is "inconclusive overall".
Conservatively, that line is now evidence for neither side — but it never at any point produced evidence
*for* the prior.

**Why this verdict.** The prior fails the mechanistic test it was introduced to pass: dispersion does not
track where the model is wrong, so it cannot legitimately seed τ². The pre-registered ablation clause ("if
info-optimal is not > structural-only, the ESM uncertainty prior is dropped from the claims",
`VALIDATION.md:129-134`) fired, and the one decision-eligible task (downstream training-set quality)
independently found it adds nothing over pure loop-bracing. The project thesis was narrowed accordingly to
"conjoint ESM-2 scores carry epistatic signal" without "ESM uncertainty guides allocation"
(`README.md:99-100`).

Two things make this a properly-killed dead end rather than a quiet retreat. The ablation was frozen *before*
the headline existed, so the kill criterion was not chosen after seeing the result. And the weaker first-pass
figures (a 150-point sample, no confidence intervals) were superseded by the powered 300-point run with
CIs, and then **banned from ever reappearing**: the two superseded Spearman values and their sample size
are listed in `forbidden_literals` in `artifacts/claim_map.json`, and so are deliberately not reproduced
in this document.

Residual caveats a reader should carry: both calibration artifacts are
`evidence_classification: traceable_not_rerun`; the 650M one is `status: supplementary` and the 35M one
`status: smoke_test`; both were generated from a dirty tree (`code_state: "dirty"`, base commit `1a1f30a`).
The analysis is conditional on measurable positive fitness, since only positive-fitness rows are
log-transformable. Cross-fitted and order-stratified calibration analyses remain pending.

**Evidence.**
- `artifacts/calibration_650m.json` (`spearman_sigma2_abserror` −0.11281947577195303, `spearman_ci95`
  [−0.22029495928435333, −0.0020540231354003033], `pearson_ci95`
  [−0.1977108668742965, +0.003127218239102498], `calibration_slope_b` 0.7912697622444083);
  `artifacts/calibration_35m.json` (+0.04245069389659885, ci95
  [−0.07833717141239625, +0.15682759846645067]); `artifacts/manifest.json`.
- `src/epibudget/calibrate.py`; `scripts/calibrate_uncertainty.py`; `src/epibudget/scoring.py:280-301`;
  `src/epibudget/graph.py:60-67`; `src/epibudget/validate.py:362-372`; `tests/test_calibrate.py`.
- `VALIDATION.md:129-134` (mandatory ablation + drop clause), `:217-231` (historical ablation result and its
  withdrawal), `:241-248` (corrective gate, `inconclusive_zero_gpu`); `LIMITATIONS.md` §5; `SPEC.md:247`;
  `README.md:84-100`.
- `experiments/trpb-downstream-generalization-20260716.md:24-25` and `:46-50`; `specs/downstream.md`.
- Commits `c934bb2`, `6900206`, `5a56314`, `b84d0ee`, `5bff991`, `e75535d`.

## Isotropic, label-free, coefficient-aware D-optimal acquisition

- **Verdict:** abandoned

**Question.** The information-optimal selection is singles-heavy, and a single mutant carries no information
about a pairwise Fourier coefficient. Is the binding constraint therefore the *acquisition* rather than the
inference — would a coefficient-aware optimal experimental design, chosen purely from the design geometry
with no access to labels, recover the epistasis map better at matched budget?

**What was built.** `_doptimal_order` in `src/epibudget/coeff_recovery.py:645-691` — a greedy Bayesian
D-optimal (maximum posterior-variance) design over the closed-form multiallelic Fourier kernel, under an
isotropic N(0, I) prior on the coefficients with observation variance `_DOPTIMAL_OBS_VAR = 1e-2` (line 642).
It reads only the design (`_kernel_cross` / `_order_symmetric_kernel`) and never a measured label — the label
barrier is stated in the docstring at line 656. It is prefix-consistent (the budget-B design is the first B
of a longer greedy run) via an O(N·B²) GP-style rank-1 update (lines 677-691). It is wired into
`run_coeff_recovery` as an extra selection arm alongside `info` and `random` (lines 734, 748), so all arms
share estimators, budgets and metric. Module-level `public_claim_eligible: bool = False`
(`coeff_recovery.py:504`) — this is a diagnostic, never a selection path.

**What was measured.** Residualized pairwise Spearman of recovered ε against truth, controlling for the true
main-effect skeleton, under `doptimal` vs `info` vs `random` (5 seeds) selection at matched budgets
B ∈ {48, 96, 192}, with the same Fourier-LASSO and Fourier-ridge estimators and the same metric as the
compressed-sensing baseline. Protocol in `specs/step6-coefficient-recovery.md`. Selection is label-free;
measured labels enter only after selection is fixed, via `reveal_measured_fitness`
(`coeff_recovery.py:714-723`).

**Outcome.** Clean negative — the coefficient-aware design does not help, and at two of three budgets it
actively hurts. Fourier-LASSO pairwise residualized recovery −0.166 / +0.063 / −0.023 at B = 48/96/192,
versus roughly 0.10-0.21 for the ESM-weighted `info` selection and ≈0 for random. Order-3 recovery remained
≈0 throughout. Diagnosed cause: the isotropic prior spans all orders, so the design spends most of its budget
trying to resolve the order-3 subspace — 27,436 triple-mutant coefficients out of a 29,678-variant universe
(corroborated in `src/epibudget/gate2.py:59`, `tests/test_data.py:38`, `VALIDATION.md:143`) — which is
unrecoverable at any of these budgets. The `info`/`random` decision rule itself was left untouched; the
D-optimal arms are diagnostic-only (`coeff_recovery.py:620-637`).

**Why this verdict.** The result kills the "easy acquisition win" hypothesis: generic experimental-design
geometry recovers no pairwise signal above random, so the budget — not the acquisition — is the binding
constraint, and whatever weak pairwise signal exists is carried by the ESM-weighted loop-bracing structure
rather than by the design math. Crucially, the one identified escape hatch (the isotropic prior leaking
budget to order 3) was not left standing as an excuse: it was made testable by parameterising the target
subspace and run with `orders=(1, 2)`. That follow-up removed the harm but did not create a win (pairwise
residualized recovery +0.038 / +0.024 / +0.008, i.e. ≈ random, far below `info`), which triangulated the same
conclusion from a second direction and exhausted the acquisition lever. The approach is therefore abandoned
rather than merely narrowed.

Two caveats keep the negative honest: the numbers are cache-only zero-GPU diagnostics with no CLI entry point
(no reference to `coeff_recovery` outside its own module in `src/` or `pyproject.toml`), so they are not
reproducible from the public tree; and the order-restricted follow-up is a *reduced-model* D-optimal design,
assuming order-3 negligible rather than actively fighting order-3 → pairwise aliasing, so the rigorous
comparator (full-order model with a pairwise-only target functional, c-/Ds-optimality) is expected to confirm
rather than overturn this.

**Evidence.**
- Commit `6ff14ce7b101868dcad827a432d7b0daff118aea` — "feat(coeff-recovery): Step 6 compressed-sensing +
  D-optimal + weighted-pairs recovery"; +1065 lines. `git log -S"_doptimal_order"` returns this commit only:
  the function was never modified after introduction.
- `src/epibudget/coeff_recovery.py:640-691`, `:504`, `:734,748`, `:714-723`.
- `tests/test_coeff_recovery.py:140-150` and `:184-197`. Both pass offline → 2 passed.
- `specs/step6-coefficient-recovery.md` — protocol, leakage barrier and decision rule; its `both_weak` branch
  explicitly names the acquisition/data bottleneck.
- `VALIDATION.md:18-31` — the committed qualitative record: "neither an isotropic nor a pairwise-targeted
  label-free D-optimal acquisition beats the ESM-weighted selection", together with the explicit admission
  that the numbers are not committed and that `coeff_recovery` exposes no CLI command.
- `PRIOR_ART.md:24`; `RESEARCH_EPISTASIS.md:183` — D-optimal design positioned as imported prior art.

**Gap.** The numeric results (−0.166 / +0.063 / −0.023 isotropic; +0.038 / +0.024 / +0.008 order-restricted;
the ~0.10-0.21 `info` reference line) are re-expressed here from an uncommitted working document. The
committed tree carries the qualitative conclusion but not the numbers.

## Order-restricted "weighted-pairs" D-optimal acquisition

- **Verdict:** abandoned

**Question.** The preceding label-free D-optimal acquisition placed an isotropic N(0, I) prior over all
interaction orders, so on GB1 it spent most of a small budget resolving the vast third-order subspace it
cannot recover at those budgets. Does restricting the design's *target subspace* to singles + pairs —
concentrating the budget where the map is plausibly recoverable — make pure experimental-design geometry
competitive with the ESM-weighted selection?

**What was built.** `_order_symmetric_kernel` in `src/epibudget/coeff_recovery.py:241` — a closed-form
Fourier kernel restricted to an arbitrary set of interaction orders. Grouping modes by their non-constant
support `T` factors the full kernel as `prod_{s in T}(agree_s − 1/q) * (1/q)^(n−|T|)`, so the order-*r*
contribution is `e_r * (1/q)^(n−r)` with `e_r` the *r*-th elementary symmetric polynomial of the per-site
non-constant agreements. It is evaluated by the elementary-symmetric recurrence over `n_sites+1`
accumulators of shape (N, M), i.e. O(n_sites·N·M) peak memory, without ever materialising the ~29.7k-mode
design. It drives `_doptimal_order(..., orders=(1, 2))` (`coeff_recovery.py:645`), a greedy
max-posterior-variance design with a rank-1 GP update; the selection reads only the kernel, never a measured
label. `run_coeff_recovery` adds it as a fourth selection arm, `doptimal_pairs`, alongside `info`, `random`
(5 seeds) and the isotropic `doptimal` (`coeff_recovery.py:735-749`).

**What was measured.** Residualized pairwise and third-order Spearman recovery of the reconstructed epistasis
map versus truth — the rank correlation after controlling for the true main-effect skeleton — at
B ∈ {48, 96, 192}, for both the Fourier-LASSO and Fourier-ridge estimators, with the pairwise-targeted design
compared against the isotropic D-optimal design, `random`, and the ESM-weighted `info` selection. Protocol
frozen in `specs/step6-coefficient-recovery.md`; the run is cache-only, pure-numpy, and flagged
`public_claim_eligible = False` (`coeff_recovery.py:504`).

**Outcome.** Subspace targeting repaired the isotropic design's pathology but did not make it competitive.
Fourier-LASSO pairwise residualized recovery came out at roughly +0.04 / +0.02 / +0.01 at B = 48/96/192
(ridge about −0.07 / +0.04 / +0.02) — consistently at or above the isotropic design's roughly
−0.17 / +0.06 / −0.02, which was often actively negative, so restricting the target subspace does remove the
*harm*. But the result sits at ≈ random (~0) and far below the ESM-weighted `info` selection (~0.10-0.21) at
every budget and both estimators. Third-order recovery remained ≈ 0 throughout. The recorded decision
categories were unchanged: pairwise `esm_pipeline_ahead`, third-order `both_weak`. Those categories are
computed only from the Fourier-LASSO/`info` arm (`_decide`, `coeff_recovery.py:602-637`) — the D-optimal arms
are reported cells that did not shift the standing decision, rather than inputs to it.

**Why this verdict.** Three acquisition strategies triangulate one conclusion: ESM-weighted loop-bracing
~0.10-0.21; isotropic D-optimal ~−0.17..+0.06; pairwise-targeted D-optimal ~0.00..+0.04. Only the
ESM-weighted selection recovers pairwise signal above random. Pure label-free experimental-design geometry —
the compressed-sensing / optimal-experimental-design lever — does not beat random at B ≤ 192, so the signal
is carried by the ESM prior rather than the acquisition mathematics or the choice of coefficient basis, and
the budget is the binding constraint. The acquisition lever was therefore declared exhausted and effort
redirected. One caveat was registered against the design itself and is worth preserving: this is a
*reduced-model* D-optimal design — the orders-1..2 kernel serves as both the model and the target — so it
assumes third-order effects are negligible rather than actively fighting order-3-to-pairwise aliasing. The
rigorous comparator (a full-order model with a pairwise-only target functional, i.e. c- or Ds-optimality) was
never run; it was expected to confirm rather than overturn the finding, which makes the negative result
well-supported but not formally airtight.

**Evidence.**
- `src/epibudget/coeff_recovery.py:241` (`_order_symmetric_kernel`, derivation in the docstring), `:645`,
  `:735`, `:602`.
- `tests/test_coeff_recovery.py:153` (order-additivity and agreement with the unrestricted kernel), `:168`
  (the orders-1,2 sub-kernel pinned against the explicit order-1,2 character Gram matrix at n=4, where an
  order-3 block exists and must be excluded), `:184` (the targeted design is distinct, prefix-consistent, and
  selects a genuinely different set than the isotropic one). Verified passing → 4 passed.
- Commit `6ff14ce`, whose message records the finding that the ESM prior, not the acquisition geometry,
  carries the weak pairwise signal and that order-3 is unrecoverable at B ≤ 192.
- `VALIDATION.md:18-31` (committed at `ebd2876`) — the standing amendment, plus the statement that these are
  internal diagnostics whose numbers are not committed and that `coeff_recovery` exposes no CLI command.
- `specs/step6-coefficient-recovery.md` — defines `esm_pipeline_ahead`, `both_weak`, and
  `public_claim_eligible = false`.

**Gap.** The qualitative conclusion has a committed home in `VALIDATION.md`; the per-budget numbers quoted
above, and the reduced-model / aliasing caveat, do not.

---

# Inference

## WT-anchor correction: WT-centred log fitness so ΔG(∅) = 0 on a non-unit-fitness reference

- **Verdict:** kept

**Question.** The ε operator is a WT-referenced inclusion-exclusion sum whose loop enumerator
`interaction_loop` iterates `range(1, n + 1)` — it excludes the empty set, so ΔG(∅) = 0 is a *structural*
assumption, never an explicit term. Does that assumption survive on a landscape whose reference genotype is
not normalised to fitness 1.0, or was the whole ε machinery silently depending on an accident of the first
dataset's normalisation?

**What was built.** `wt_centered_log_fitness` in `src/epibudget/epistasis.py:30-48`: it requires the
wild-type key to be present, rejects any non-finite value, requires a strictly positive reference, drops
non-positive non-reference variants, and returns ΔG(v) = log f(v) − log f(reference) with the reference
pinned to an exact `0.0`. The convention is routed through every consumer of measured fitness rather than
applied at one call site — validation truth and revealed calibration labels (`src/epibudget/validate.py:396`,
`:434`), robustness truth and cross-fit slopes (`src/epibudget/robustness.py:155`, `:543`), and the
cached-replay gate (`src/epibudget/gate2.py:520`). The pre-fix code built
`landscape_dg = {v: log(f) for v, f in landscape.items() if f > 0.0}` with no normalisation at all (visible
in the `e00b2c3` diff of `validate.py`).

A second, related change landed with it: the CLI's invariant-#1 gate was re-pointed at *predicted* ESM
epistasis variance (`var_predicted_epsilon` / `predicted_epistasis_signal` / `predicted_epistasis_tolerance`,
`src/epibudget/cli.py:140-147`), with measured truth variance `var_epsilon` demoted to a separately-printed
descriptive number. Before that, the serialized `var_epsilon` was measured-truth variance and did not certify
the non-additivity invariant it appeared to report.

**What was measured.** Every TrpB ε term, `var_epsilon`, and η² (the between-order share of pooled ε
variance) were recomputed from the raw CSV under both the old uncentred and the new centred transform, and
GB1 was checked for bit-exactness. Recomputed independently from `data/proteingym/trpb_johnston2024.csv` and
`data/proteingym/gb1_wu2016.csv` via `load_trpb` / `load_gb1` → `ground_truth_epistasis`.

**Outcome.** A real, silent, landscape-dependent correctness bug.

- GB1's WT has f = 1.0 exactly, so ln f(∅) = 0 and centring is a **bit-exact no-op** — the centred and legacy
  ΔG maps compare equal as dicts, and so do their truth-term lists. No GB1 number moved.
- TrpB's Tm9D8\* parent has f = 0.408073925, so ln f(∅) = −0.896307 ≠ 0. Consequence: every pairwise ε was
  shifted by exactly **+0.896307** and every third-order ε by exactly **−0.896307** (constant to the digit
  across all 1,784 pairwise and 15,925 third-order terms), **manufacturing** between-order separation.
- `var_epsilon` (population variance, `np.var`, ddof = 0): **4.844110 as-run vs 3.930211 re-anchored** — the
  reported figure was **+23.3% inflated**.
- η²: GB1 **0.079**; TrpB as-run **0.257** (3.2× GB1); TrpB re-anchored **0.085**, landing on GB1's value.
  The inflated between-order share was entirely the anchor artifact.
- The arithmetic tell was already visible in the artifact: `info` at B = 48 reported pooled ρ = 0.2891 *above
  both of its own sub-orders* (pairwise 0.0434, third 0.1321). A pooled correlation can only exceed both its
  parts through between-group separation.
- The correction is not rank-neutral for recovery. A constant within-order offset leaves truth-only Pearson
  and Spearman invariant, but the through-origin calibration slope is refitted after centring —
  `b_centered = b_raw − g₀·Σx/Σx²`. Fully measured loops shift with the truth; partially measured loops
  retain prior members scaled by the changed slope, so their ε̂ moves non-uniformly. A regression test
  exhibits this on a hand-built landscape with f(∅) ≠ 1: the calibration slope is 1.5 centred vs 1.0
  uncentred, and pairwise Spearman is 1.0 centred vs 0.5 uncentred.

**Why this verdict.** The fix is live in every measured-fitness path, regression-pinned, and cheap; it
removes a correctness dependency on an accident of the first dataset's normalisation. GB1 is bit-exact, so
retaining it costs nothing already banked. The honest residual caveat is recorded rather than glossed:
re-anchoring fixes the ΔG(∅) = 0 *convention*, it does not make TrpB's max-normalised, sign-crossing `ln f` a
valid free energy — 3.930 is the anchor-consistent `var_epsilon`, not an "honest" one.

What was *not* done is as much a part of the verdict as what was. The affected historical TrpB outputs were
declared uninterpretable and left standing with a correction notice, not retroactively repaired: all
historical TrpB recovery coefficients, correlations and truth-map variance are invalid, while selection
identities, attempted/revealed counts, coverage, hit-rate and run configuration are explicitly preserved as
still-valid descriptive outputs because they do not depend on centring. The correction itself is retained,
routed and tested; only the downstream historical numbers were retired.

The broader lesson the episode carries: the bug was latent on GB1 *precisely because* GB1's wild-type fitness
is 1, so it survived the entire first landscape undetected and only a second landscape exposed it. That is a
concrete argument for generalization runs as correctness tests, not merely as breadth claims.

**Evidence.**
- `src/epibudget/epistasis.py:30-48` (`wt_centered_log_fitness`); `:69-77` (`interaction_loop`, the
  `range(1, n + 1)` empty-set exclusion, with the ΔG ≡ 0 assumption stated in its docstring); `:51-66`
  (`epsilon_pairwise` / `epsilon_third`, "ΔG(∅) = 0 by convention").
- Routing: `src/epibudget/validate.py:396`, `:434`; `src/epibudget/robustness.py:155`, `:543`;
  `src/epibudget/gate2.py:520`.
- `experiments/trpb-smoke-20260713.md` §6.1 (lines 182-212) — the full diagnosis, offsets, variance and η²
  numbers, and the slope-refit argument; lines 3-9 — the correction notice fixing what is and is not
  invalidated; lines 115-117 — the pooled-above-sub-orders tell; line 164 — the `Var[ε_true]` comparison row.
- `LIMITATIONS.md:148-160` (§6b) — the committed statement of the defect and its corrective status.
- `SIGNAL_GATE.md:28-30` — the invariant now stated as ΔG(v) = ln(f(v)/f(reference)); `SPEC.md:161-163` —
  ground-truth ε defined over `wt_centered_log_fitness` of measured fitnesses.
- Tests (all passing; `pytest tests/test_epistasis.py tests/test_validate.py` → 60 passed):
  `tests/test_epistasis.py:98-110` (exact `0x0.0p+0` reference and multiplicative scale invariance),
  `:113-133` (missing / non-positive / non-finite reference rejection), `:136-140` (non-positive
  non-reference variants dropped), `:143-160` (inclusion-exclusion recovered at arbitrary order);
  `tests/test_validate.py:148-208` (`test_f0_not_one_changes_within_order_spearman_when_not_centered` — the
  slope 1.5-vs-1.0 and Spearman 1.0-vs-0.5 regression pin), `:212-241` and `:243-252` (bit-exactness for a
  unit reference, synthetic and on the real GB1 CSV).
- Commits: `e00b2c3` "refactor(core): audit hardening of core modules + offline tests" — introduces
  `wt_centered_log_fitness`, replaces the unnormalised `log(f)` maps in `validate.py` / `robustness.py`, and
  adds the predicted-epistasis invariant fields. `943833d` — restates the `SIGNAL_GATE.md` invariant in ratio
  form and lands the §6.1 analysis note. `621cfce` — adds `gate2.py`, a consumer of the corrected convention.
  Note that `621cfce` does **not** touch `epistasis.py` at all (`git log -S`, `git show 621cfce --stat`); the
  fix commit is `e00b2c3`.
- Independent recomputation reproduced every number to the digit: f(parent) = 0.408073925,
  ln f = −0.8963069…, per-order offsets ±0.896307 constant, `var_epsilon` 4.844110 → 3.930211 (+23.3%), η²
  0.2574 → 0.0848, term counts 1,784 + 15,925 = 17,709; GB1 f(WT) = 1.0, centred map equal to legacy, var
  2.6162, η² 0.0793 over 17,782 terms.

**Gap.** The observation that the previously serialized truth-map variance had never actually exercised the
non-additivity invariant, and that the CLI gate was deliberately re-pointed at predicted ESM epistasis while
truth variance stayed descriptive, has no committed home — the code carries the behaviour, but not the
rationale.

## Per-method through-origin calibration slope as a sign-setting confound, and the shared cross-fit mitigation

- **Verdict:** kept

**Question.** The epistasis inferrer prices every unmeasured loop member with an ESM prior rescaled by a
through-origin slope `b` fit on *that method's own revealed set*. Does a recovery correlation therefore
measure the quality of a method's selection, or the sign and magnitude of a per-method nuisance parameter?

**What was built.**
- Diagnosis of the estimator itself: `esm_prior_mu` in `src/epibudget/validate.py:147-161` sets
  `mu[v] = revealed[v]` if measured, else `b * esm[v]`, with `b` from `_calibrate_slope`
  (`src/epibudget/validate.py:109-125`), fit only on the revealed set.
- Mitigation probe (A2 of a three-part post-hoc robustness module): `src/epibudget/robustness.py`, added in
  `00ee7ad` with a CLI entry point in `bad1577`, spec `specs/robustness.md`. `crossfit_slopes` (lines
  145-166) fits one method-independent through-origin slope per fold, out-of-fold, over the full measurable
  candidate set; folds come from `variant_fold` (line 140), a deterministic, label-free
  `variant_key(sorted(variant)) % 5`. `infer_epistasis_crossfit` (lines 169-191) re-runs inference with each
  unmeasured member priced by its own fold's slope.
- The two-regime reporting was later generalised into the corrective gate-2 path:
  `src/epibudget/gate2.py:65` declares
  `_REGIMES = ("operational_method_specific", "shared_crossfit_5fold")`, and `_slope_integrity_reasons`
  (lines 1393-1437) refuses a report whose shared slope is not method-independent or whose caveat does not
  identify it as non-operational.

**What was measured.** Two things.

1. An algebraic identity, readable directly off the code: `interaction_loop`
   (`src/epibudget/epistasis.py:69-77`) excludes the empty set and `ΔG(∅) ≡ 0`, so ε is a *homogeneous* ±1
   inclusion-exclusion sum over loop members. If no loop member is measured, every member carries
   `b * esm[v]`, hence `ε̂ = b · ε̂_ESM` exactly, and a rank correlation becomes `sign(b) · ρ_prior` — a
   method-independent constant.
2. Empirically: GB1 pairwise/third-order recovery Spearman under each method's own operational slope versus
   the shared five-fold cross-fitted slope, at B ∈ {48, 96, 192}, over the frozen 650M artifacts (29,678
   candidates, 20 seeds).

**Outcome.** Both confirmed.

The magnitude collapse predicted by the identity is visible in the frozen headline.
`artifacts/signal_650m.json` gives the raw ESM ε̂ pairwise Spearman as **+0.30195**. In
`artifacts/headline_650m.json` at B = 48 pairwise the three lowest-coverage arms report −0.2590 (fitness,
coverage 6.31%), −0.2709 (practice, 2.63%) and **+0.2791 (random, 0.52% coverage — the lowest coverage and
the value closest to the bare prior)**. All three have |ρ| ≈ 0.26-0.28; the signs differ, the magnitudes do
not.

Under the shared cross-fit slope (`artifacts/robustness_650m.json`, `scale_sensitivity`), fitness-greedy's
GB1 pairwise recovery **flips sign**:

| B (pairwise) | own slope | shared cross-fit slope |
|---|---|---|
| 48 | −0.25905 | +0.27065 |
| 96 | −0.24715 | +0.27276 |
| 192 | −0.13421 | +0.34894 |

The *ranking* survives: `ranking_agrees = true` for all six order × budget cells,
`structural > info > fitness` throughout. The *effect size* does not: `info − fitness` pairwise at B = 48
goes from +0.6672 (headline; 0.40816 − (−0.25905)) to +0.1604 (cross-fit; 0.43109 − 0.27065). Third order
shows the same flip (fitness −0.0872 → +0.0938 at B = 48; −0.1056 → +0.0900 at B = 96).

Scope limit, verified in code: `_scale_sensitivity` iterates only `("info", "fitness", "structural")`
(`src/epibudget/robustness.py:512`), so A2 was never run for `random` or `practice`; and no TrpB scored cache
was exported, so it was never run on the second landscape.

**Why this verdict.** Kept, on both halves, but with the mitigation bounded from the start rather than after
the fact.

The confound is a confirmed defect of the estimator, not a hypothesis, so it is retained as a permanent
interpretation constraint (`LIMITATIONS.md` §6b) and structurally enforced: gate 2 must report both the
operational method-specific slope and the shared cross-fit attribution regime for every selection.

The shared slope is kept as *attribution evidence only*, never promoted to an operational selection method,
because it consumes full-landscape labels — more label information than any real run has. That bound was
present in the first commit of the module, not bolted on later: `git show 00ee7ad:src/epibudget/robustness.py`
line 63 already carries the caveat string that "never quote as a headline figure and never adopt
crossfit_ranking as the reported method order", it is serialized into every `ScaleSensitivity` record, and
`tests/test_robustness.py::test_report_has_serialized_caveats_and_no_pooled_order` fails if it goes missing.

The concrete consequence is a retraction, not a rhetorical caveat. The README previously claimed
information-optimal "beats fitness-greedy **−0.259 / −0.247 / −0.134**"; commit `943833d` deleted that block
and replaced it with text naming "the method-specific calibration slope confounds low-coverage comparisons"
as one of the two reasons no comparative recovery claim is current. The negative fitness-greedy numbers were
calibration artifacts, not measured anti-recovery.

**Evidence.**
- `src/epibudget/validate.py:109-125` (`_calibrate_slope`), `:147-161` (`esm_prior_mu`).
- `src/epibudget/epistasis.py:51-77` (homogeneous ±1 ε form; loop excludes ∅).
- `src/epibudget/robustness.py:66-70` (caveat), `:140-191` (folds, cross-fit slopes, cross-fit inference),
  `:497-532` (`_scale_sensitivity`, three methods only).
- `src/epibudget/gate2.py:65`, `:73-76`, `:1393-1437` (two-regime reporting + integrity refusal).
- `artifacts/signal_650m.json` → `spearman_pairwise = 0.3019452425119898`; `artifacts/headline_650m.json` →
  B=48 pairwise: fitness −0.25905 @ cov 0.0631, practice −0.27093 @ 0.0263, random +0.27910 @ 0.0052;
  `artifacts/robustness_650m.json` → `result.scale_sensitivity` (all six cells, `ranking_agrees = true`),
  `result.pair_differences` info−structural pairwise Spearman at B=48: delta −0.07633,
  CI [−0.11768, −0.03844], `excludes_zero = true`.
- `experiments/trpb-smoke-20260713.md` §6.3 (lines 270-313), §7 "Scientific transfer — UNINTERPRETABLE";
  `LIMITATIONS.md` §6b (line 172 ff.).
- Commits `00ee7ad`, `bad1577`, `82c898a`, `943833d`.
- `python -m pytest tests/test_robustness.py -q` → 16 passed.

## Independent-noise error propagation for σ²(ε) over a loop

- **Verdict:** narrowed

**Question.** Can the per-interaction prior variance σ²(ε(S)) be seeded by simply summing each loop member's
ESM masking-perturbation dispersion `var_delta_g`, i.e. by assuming the ΔG score errors of distinct candidate
variants are uncorrelated?

**What was built.** `predicted_epistasis` in `src/epibudget/epistasis.py:108-141` propagates variance through
the WT-referenced inclusion-exclusion sum. Because the inclusion-exclusion coefficients are all ±1,
coefficient² = 1 and the propagated variance degenerates to a plain sum over the loop:
`sigma2 = sum(var[member] for member in loop)` (`epistasis.py:137`), implementing
σ²(ε(S)) = Σ_{∅≠T⊆S} var_delta_g(T). A missing lower-order loop member raises `KeyError` rather than
defaulting to 0, which would understate σ² and bias acquisition (`epistasis.py:130-136`). The same model is
what makes the acquisition objective modular: `info_gain(M, v) = var_delta_g(v)·n(v)`, independent of the
measured set (`SPEC.md:172-187`), so `allocate` is a single sort rather than an iterative greedy loop.

**What was measured.** Three probes, none of which argues the assumption from theory.

1. The uncertainty-prior calibration asks whether `var_delta_g` correlates with the absolute calibrated ESM
   error per variant — the seed's own validity, upstream of the independence assumption
   (`artifacts/calibration_650m.json`, n=300, ESM-2 650M, 16 masking perturbations, calibration slope
   b = 0.791).
2. The gate-3 probe attacks the independence assumption directly: it replaces the diagonal prior with a
   correlated-error covariance Σ_e = τ_a² G Gᵀ + σ_r² I (additive random effects over shared sub-mutations),
   conditions on the measured set, and sweeps λ = σ_r²/τ_a² from ∞ (the exact pin baseline) down to 0⁺,
   scoring squared-error gain and Δspearman at B ∈ {48, 96, 192} (`specs/gate3-correlated-inference.md`,
   `src/epibudget/gate3.py`).
3. The downstream benchmark's `info − structural` ablation isolates the σ² channel: `info` is `structural`
   reweighted by `var_delta_g`, so the contrast measures exactly what the uncertainty prior contributes
   (`specs/downstream.md:459,468`).

**Outcome.** The assumption survives in code but is demoted to a labelled first approximation.

1. Calibration: Spearman(σ², |error|) = **−0.113**, 95% CI [−0.220, −0.002]; Pearson = **−0.100**, 95% CI
   [−0.198, 0.003] at 650M — weak negative rank association, not positive calibration; at 35M both intervals
   include zero (`README.md:84-91`, `artifacts/calibration_650m.json`, `artifacts/calibration_35m.json`).
2. Gate 3: the correlated-error structure is real and consequential — pinning true ΔG onto some loop members
   breaks the error cancellation that ε's ±1 difference otherwise provides, roughly doubling squared error
   (`sse_gain ≈ −0.9 … −1.0`), and a correlated-error inferrer repairs the squared-error calibration but not
   the underlying main-effect-sharing confound (`specs/gate3-correlated-inference.md:5-11`,
   `VALIDATION.md:18-31`, commit `621cfce`).
3. Downstream, on the decision-eligible GB1 run, structure-aware selection wins and the masking-variance
   prior adds nothing (`README.md:106-108`).

The gate-3 mechanism is reproduced in an offline synthetic test: on a dominantly additive-error landscape,
`pin_sse_gain < 0` and some finite λ on the frontier beats it (`tests/test_gate3.py:181-198`).

**Why this verdict.** Narrowed, not abandoned or kept. The propagation rule is still the shipped seed and is
unit-tested exactly as a loop sum (`tests/test_epistasis.py:193-219`), so the mechanism is retained. What was
withdrawn is the epistemic status attached to it. The direction of the bias induced by real correlated score
noise is **not** derivable from the ±1 inclusion-exclusion structure, so it is explicitly not claimed to be
conservative — it is stated as an assumption and checked empirically rather than argued
(`LIMITATIONS.md:85-89`; docstring `epistasis.py:118-123`). Both empirical checks came back against it: the
seed shows no positive calibration, and the independence premise is demonstrably false for
nested/overlapping variants scored from related contexts. The knock-on consequence is registered too: the
geodetic loop-closure / diminishing-returns intuition is not realised, because under independent noise
`info_gain` is modular rather than strictly submodular — strict submodularity would require correlated
priors, out of scope for v1 (`LIMITATIONS.md:70-74`, `SPEC.md:181-187`). The live decision-eligible result
therefore rests on the pure loop-bracing (`structural`) channel, not on this variance channel.

**Evidence.**
- `src/epibudget/epistasis.py:108-141` — `predicted_epistasis`, the σ² loop sum and the explicit ASSUMPTION
  block (`Cov[ΔG(T), ΔG(T′)] = 0`, "first approximation, not claimed conservative").
- `LIMITATIONS.md:85-89` (§3 Modeling); `:70-74` (modular-not-submodular consequence).
- `SPEC.md:158-160` and `:172-187` (§5) — the Gaussian model, σ²(ε(S)) = Σ τ²_T, and the "submodularity
  (honest form)" statement.
- `specs/gate3-correlated-inference.md` (whole file, esp. lines 5-11, 13-34, 47-55).
- `src/epibudget/gate3.py:1-10`; `_LAMBDA_GRID` at `gate3.py:43-56`.
- `VALIDATION.md:18-31` — standing amendment: the correlated-error inferrer repairs squared-error calibration
  but not the confound.
- `artifacts/calibration_650m.json` — `spearman_sigma2_abserror = -0.1128`,
  `pearson_sigma2_abserror = -0.0998`, CIs, n = 300, `calibration_slope_b = 0.7913`; mirrored in
  `README.md:84-91`.
- `tests/test_epistasis.py:193-219` (σ² is the loop sum, 3 terms pairwise, 7 terms third-order), `:222-226`
  (missing lower-order member raises); `tests/test_gate3.py:119-131`, `:181-198`.
- `specs/downstream.md:459,468`; `README.md:106-108`.
- Commits `ecc89c8`, `315a30a`, `621cfce`.

**Caveat on evidence depth.** The gate-3 λ-frontier numbers themselves are not committed to the public tree —
`gate3` exposes no CLI command and its run outputs are not among the tracked artifacts, a limitation the
repository states about itself (`VALIDATION.md:29-31`). The qualitative gate-3 conclusion used above is
committed; its quantitative frontier is reproducible only from the module and its synthetic tests.

## Correlated-error prior over ΔG as an inference repair (gate 3)

- **Verdict:** narrowed

**Question.** An earlier diagnostic (gate 2) found that revealing information-selected measurements improved
the *rank* of recovered epistasis but nearly *doubled* the squared error (`sse_gain ≈ −0.9 … −1.0`). The
diagnosed mechanism: ε is an inclusion-exclusion *difference*, so positively-correlated nested ESM error
largely cancels inside the loop; hard-pinning the true ΔG on some loop members destroys that cancellation.
The question was whether a correlated-error prior over ΔG closes the squared-error gap without destroying the
rank gain — i.e. whether the *inference* model needed repair or replacement, with selection held frozen.

**What was built.** `src/epibudget/gate3.py` (574 lines) — a cache-only, zero-GPU probe that swaps *only* the
inferrer and never selects. It places a Gaussian prior `z ~ N(μ₀, Σ_e)` in ΔG space with `μ₀(v) = b·esm(v)`
(`b` the leakage-safe shared cross-fit slope reused from gate 2, `gate3.py:400`), and models the ESM prior
error as an additive random effect over shared sub-mutations plus an independent residual:
`Σ_e = τ_a²·G Gᵀ + σ_r²·I` with `G[v, effect] = 1[effect ⊆ v]` and an effect basis of `single` or
`single+pair` (`_sub_effects`/`_effect_index`/`_incidence`, `gate3.py:128-159`). Conditioning on the measured
set is done in the effect-space ridge form `â = (GₘᵀGₘ + λI)⁻¹ Gₘᵀ eₘ` with `λ = σ_r²/τ_a²` (`_ridge_blup`,
`gate3.py:162-168`); its equivalence to the n-dimensional Gaussian conditioning is asserted numerically by
`tests/test_gate3.py::test_ridge_blup_equals_gaussian_conditioning:85-100`. A 12-point λ frontier runs from
`inf` down to `1e-3` (`_LAMBDA_GRID`, `gate3.py:43-56`); `λ = inf` short-circuits to the gate-2 pin
construction and is pinned as a sanity invariant by
`tests/test_gate3.py::test_lambda_inf_frontier_point_equals_pin:119-131`. λ is fitted by generalized
cross-validation on the *measured* errors only (`_gcv_lambda`, `gate3.py:171-194`).

The probe also introduced the diagnostic that ended up mattering most: a **residualized,
main-effect-controlled recovery metric**. `_skeleton` (`gate3.py:230-242`) computes
`k(S) = Σ_{T ∈ loop(S), T measured} c_T·true(T)` — the component the pinned ε̂ and the truth share by
construction — and `_residualize`/`_partial_pearson`/`_partial_spearman` (`gate3.py:245-270`) score rank
recovery after removing it. Its behaviour is pinned by two synthetic tests: a purely shared control is
stripped (`test_partial_corr_strips_a_purely_shared_control:143-155`) while genuine signal beyond the control
survives (`test_partial_corr_preserves_signal_beyond_control:158-167`).

**What was measured.** Per budget `B ∈ {48, 96, 192}`, per effect basis, per interaction order (pairwise and
third): `sse_prior`, `sse_gain` for the pin and the correlated posterior, Δspearman/Δpearson against the
ESM-prior baseline, the full λ-frontier `(λ, sse_gain, Δspearman)`, and bootstrap-over-terms 95% CIs — plus
the residualized (skeleton-controlled) Spearman and its bootstrap CI (`OrderCell`, `gate3.py:59-87`;
`_order_cell`, `gate3.py:310-384`). Selection was the frozen, label-free gate-2 information allocation
(`_info_selection`, `gate3.py:299-303`). The decision was restricted to operational budgets `B ≥ 96` because
λ is unidentifiable on the saturated single-only design (`_OPERATIONAL_BUDGET`, `gate3.py:463`;
`lambda_identified`, `gate3.py:412`).

**Outcome.** Registered decision `calibration_repair_rank_confounded`. Both halves of the result are real and
they point in opposite directions:

- The correlated prior **does** repair the squared-error calibration — which establishes that the
  independent-variant inference was genuinely mis-specified, not merely noisy. The mechanism is reproduced
  end-to-end on a synthetic landscape with dominantly additive ESM error: pinning drives `pin_sse_gain < 0`,
  and some finite λ on the frontier beats it
  (`tests/test_gate3.py::test_correlated_prior_beats_pin_on_additive_error_landscape:181-198`).
- The residualized diagnostic showed the apparent rank gain was **mostly a main-effect-sharing confound**.
  Genuine pairwise interaction recovery is weak — residualized Spearman **0.20 at B = 96 and 0.29 at
  B = 192** — and **order 3 is not recovered at all (0.0 at both budgets)**. These figures are committed
  twice: as prose in `specs/step6-coefficient-recovery.md:5-6` and as the hard-coded reference constant
  `_ESM_REFERENCE = {"pairwise": {96: 0.20, 192: 0.29}, "third": {96: 0.0, 192: 0.0}}` in
  `src/epibudget/coeff_recovery.py:74-77`.

The report is marked `public_claim_eligible = False` by construction (`gate3.py:565`).

**Why this verdict.** Narrowed, not kept and not abandoned: the repair worked on the loss it targeted and
failed the honest version of its own test. The shipped decision rule (`_decide`, `gate3.py:466-504`)
requires, at `B ≥ 96`, *both* `sse_gain ≥ 0` *and* a residualized rank gain whose bootstrap CI lower bound
exceeds zero, precisely so that "a repair that merely reverts to the prior or rides the confound fails"
(`gate3.py:469-471`). The correlated prior cleared the first condition and not the second, so neither
`repair_confirmed` nor a clean replace verdict could be claimed. The consequence recorded in the committed
protocol is a narrowing of the project thesis, not a fix: "pairwise map recovery is weak, and its raw rank
gain is largely a main-effect-sharing confound; a correlated-error inferrer repairs the squared-error
calibration but not the confound" (`VALIDATION.md:18-31`). This is what redirected effort away from repairing
the inference model and toward changing the estimand entirely — first to an external compressed-sensing
baseline on the same frozen selections (`specs/step6-coefficient-recovery.md`,
`src/epibudget/coeff_recovery.py`), then to the downstream-impact benchmark.

Two honest caveats, both verifiable in the tree:

- **The probe is not reproducible from the public tree.** `gate3` exposes no CLI command (the CLI defines
  `allocate`, `validate`, `robustness`, `gate2`, `downstream`, `score` — no `gate3`;
  `src/epibudget/cli.py`), and no gate3 report artifact was ever persisted (`git ls-files artifacts` lists 13
  files, none of them a gate3 report). The repository states this limitation itself: "the numbers live in an
  uncommitted working roadmap, and `gate3` / `coeff_recovery` expose no CLI command, so they are not
  reproducible from the public tree and no claim above is decision-eligible" (`VALIDATION.md:29-31`).
- **The committed spec's decision rule is stale relative to the shipped code.**
  `specs/gate3-correlated-inference.md:47-53` still describes the original three-way rule
  (`repair_current_core` / `replace_phase2` / `inconclusive_zero_gpu`) and does not mention the residualized
  metric at all, even though the spec and the module landed in the same commit. The criterion that actually
  produced the verdict — the main-effect-controlled gain — exists only in `gate3.py` and in the
  `VALIDATION.md` amendment. The finer-grained claim that the residualized gain over the ESM prior is
  statistically significant only at B = 192 could not be confirmed against any committed document or
  artifact; treat it as unverified.

**Evidence.**
- `src/epibudget/gate3.py` — module docstring `:1-10`; λ grid `:43-56`; `OrderCell` `:59-87`; `_ridge_blup`
  `:162-168`; `_gcv_lambda` `:171-194`; `_skeleton` `:230-242`; residualization `:245-270`; `_order_cell`
  `:310-384`; `evaluate_budget` `:387-460`; `_decide` `:466-504`; `public_claim_eligible=False` `:565`.
- `tests/test_gate3.py` — `:85-100`, `:119-131`, `:143-167`, `:181-198`. All 9 tests pass offline →
  9 passed.
- `specs/gate3-correlated-inference.md` — question and the gate-2 `sse_gain ≈ −0.9…−1.0` figure `:5-11`;
  model and Σ_e `:13-33`; leakage barrier `:35-40`; decision rule (stale) `:47-55`.
- `specs/step6-coefficient-recovery.md:5-6`; `src/epibudget/coeff_recovery.py:71-77`; `VALIDATION.md:18-31`.
- Commit `621cfce` "feat(gate): Step 5 metric hardening + Gate-3 correlated-inference diagnostic" — adds
  `gate3.py` (574), `test_gate3.py` (198), the spec (55); its message registers the decision verbatim
  ("registered: calibration_repair_rank_confounded"). Later touched only by `4318c4f`, a lint-driven style
  change.

**Gap.** The per-budget, per-cell λ-frontier and residualized results summarized above are not reproduced in
any committed document; the committed record states only the aggregate verdict.

## Compressed-sensing (multiallelic Walsh-Hadamard / Fourier LASSO and ridge) coefficient-recovery baseline

- **Verdict:** narrowed

**Question.** The established method for budgeted epistasis recovery is sparse L1 recovery of the
Walsh-Hadamard (Fourier) spectrum, with GB1 as the canonical testbed. Unlike the indicator basis `1[S ⊆ v]`,
whose rows are sparse and which therefore cannot extrapolate, the multiallelic Fourier basis is dense per
variant, so sparsity lets it predict *unmeasured* ε. Does that standard structure-only estimator, fit on the
same frozen selections and scored with the same metric, do better than the ESM inclusion-exclusion pipeline?
(`specs/step6-coefficient-recovery.md:3-14`)

**What was built.** `src/epibudget/coeff_recovery.py` (791 lines, pure numpy, no scikit-learn):

- a dense multiallelic Fourier design over the four GB1 sites — per-site alphabet with WT first, per-site
  orthonormal contrast basis reused from `epistasis._orthonormal_contrast_basis(20)`, characters
  `χ_m(v) = ∏_s B_s[m_s, idx_s(v)]`, orders 1..3 kept, p = 29,678 columns (76 + 2,166 + 27,436) against
  B ≤ 192 rows (`_build_fourier_config`, `_design_matrix`);
- `fourier_lasso` — coordinate-descent LASSO with soft-thresholding, unpenalized intercept, warm-started
  descending λ-path with active-set-restricted sweeps, λ by K-fold CV on measured rows only (`_N_LAMBDA = 20`,
  `_CD_MAX_SWEEPS = 200`, `coeff_recovery.py:61-69`);
- `fourier_ridge` — an L2 companion over a fixed log-spaced λ grid, same CV;
- a deliberate design choice on how ε is read back: rather than inverting the basis, the fit reconstructs
  `ΔĜ(v) = X_U[v]·β̂ + c` for every loop member and applies the *same* `_epsilon` operator the pipeline uses.
  This sidesteps the mismatch between the Fourier coefficients, which are background-averaged, and ε, which
  is WT-referenced by construction (`specs/step6-coefficient-recovery.md:35-38`).

Tests in `tests/test_coeff_recovery.py` (211 lines, 10 tests, offline) pin LASSO correctness against
soft-thresholded OLS on an orthonormal design, exact sparse-support recovery, a Parseval invariant (on a
*complete* small landscape the full-design least-squares fit reconstructs ΔG exactly and its squared
coefficients per order match `wht_spectrum`), and the compressed-sensing property itself — fit on 60 of 81
genotypes, reconstruct, correlate against the 21 held out.

**What was measured.** For each budget B ∈ {48, 96, 192} and each selection (`info` = the frozen label-free
ESM allocation; `random` = uniform over seeds): raw Pearson/Spearman and SSE of ε̂ against truth, plus the
residualized Spearman controlling for the true-main-effect skeleton with a bootstrap-over-terms CI, per order
∈ {pairwise, third}, per estimator. The ESM pipeline's residualized recovery is the reference line, entered
as a fixed committed constant — pairwise 0.20 at B=96 and 0.29 at B=192, third order 0.0 — not re-derived
inside the comparison (`coeff_recovery.py:71-78`; `specs/step6-coefficient-recovery.md:5-6`). The registered
decision rule has four outcomes: `compressed_sensing_competitive`, `esm_pipeline_ahead`, `both_weak`,
`inconclusive` (`specs/step6-coefficient-recovery.md:53-63`; `_classify_cell` / `_decide`,
`coeff_recovery.py:586-637`).

**Outcome.** Pairwise residualized Spearman: Fourier-LASSO 0.099 (B=96) / 0.175 (B=192); Fourier-ridge 0.132
/ 0.118 — all four below the ESM pipeline's 0.20 / 0.29. Third order ≈ 0 for every estimator at every budget.
Recovery under random selection ≈ 0 throughout. Registered decision: pairwise `esm_pipeline_ahead`, third
`both_weak`.

Three readings follow, and the third did not survive:

1. The ESM prior genuinely beats a standard structure-only compressed-sensing baseline for pairwise ordering
   — the one comparative result that went the project's way.
2. Recovery is nonetheless weak in absolute terms (~0.1-0.3 residualized), and order-3 is unrecoverable at
   these budgets by *any* estimator here — expected, given 27,436 third-order coefficients against B ≤ 192
   measurements, and pre-registered as an admissible outcome.
3. `random ≈ 0` while `info ≈ 0.1-0.2` was initially read as the acquisition carrying the signal. Follow-up
   acquisition work refuted this: a label-free isotropic D-optimal design and a pairwise-targeted
   (order-restricted kernel) D-optimal design both failed to beat the ESM-weighted selection, so the ESM
   prior rather than the acquisition geometry carries the weak pairwise signal and the budget is the binding
   constraint (commit `6ff14ce` message; `VALIDATION.md:22-27`).

Engineering finding worth keeping: the constant mode of the per-site contrast basis is `±1/√q`, the sign
being whatever QR returns. Off-support sites contribute that constant factor once per site, so using `+1/√q`
instead of the actual `B_s[0,0]` sign-flips the reconstruction on odd `(n−r)`, i.e. orders 1 and 3 — the fit
and the predictor would disagree. The design multiplies by the actual `B_s[0,0]`, the hazard is documented in
place, and consistency between design, `_character`, `_reconstruct` and `_kernel_cross` is what the
extrapolation test enforces (`coeff_recovery.py:181-192`; `tests/test_coeff_recovery.py:106-137`).

**Why this verdict.** Narrowed rather than kept outright. What was kept: the module, its spec and its tests
are all live at HEAD, and the qualitative conclusion — that the ESM prior beats a structure-only
compressed-sensing baseline for pairwise ordering while order-3 stays unrecoverable — is recorded in a
committed document (`VALIDATION.md:22-27`). What was narrowed: the estimator was rejected as a replacement
inferrer and confined to a diagnostic lane that never feeds selection — `public_claim_eligible = False` is
the dataclass default, fitting reads labels only through `data.reveal_measured_fitness` after the label-free
selection is already fixed, and no CLI command exposes it (no reference to `coeff_recovery` anywhere else
under `src/`). The comparative claim itself was then explicitly demoted: `VALIDATION.md:27-31` states that no
decision-eligible comparative claim is current, and that because the numbers live outside the committed tree
and the module has no CLI entry point, the result is not reproducible from the public repository.

Two methodological caveats the repository states about itself, which bound how much the "ESM ahead" reading
can carry. The ESM reference is a hardcoded constant, so the comparison is against a remembered number rather
than a co-run pipeline. And `_classify_cell` reduces the spec's CI-separation rule to a direct point
comparison, precisely because that reference is a constant and not a resampled distribution
(`coeff_recovery.py:586-592`). The margin between 0.099-0.175 and 0.20-0.29 is therefore a point-estimate
gap, not a CI-separated one.

**Evidence.**
- `src/epibudget/coeff_recovery.py` — docstring and diagnostic-lane declaration at lines 1-11; LASSO/CV
  constants 61-69; ESM reference constant 71-78; Fourier config 84-120; design matrix and QR sign convention
  176-192; `_reconstruct` 203-219; `public_claim_eligible: bool = False` at line 504; `_classify_cell` /
  `_decide` 586-637.
- `specs/step6-coefficient-recovery.md` — pre-registered question, model, leakage barrier, outputs and
  four-way decision rule (63 lines, whole file).
- `tests/test_coeff_recovery.py` — Parseval/round-trip test at lines 89-103; extrapolation test at 106-137.
  Run offline at HEAD: 10 passed.
- `VALIDATION.md:19-31` — the committed standing amendment: qualitative conclusion, refutation of the
  acquisition reading, and the explicit "no decision-eligible comparative claim is current".
- Commit `6ff14ce7b101868dcad827a432d7b0daff118aea`, *feat(coeff-recovery): Step 6 compressed-sensing +
  D-optimal + weighted-pairs recovery* — adds all three files, +1065 lines. Commit `ebd2876` later corrected
  the in-code citation for the reference constant (it pointed at `VALIDATION.md`, which does not carry the
  values; it now points at the spec, which does) — docstrings and comments only, no behaviour change.

**Gap.** The per-estimator numeric results (0.099 / 0.175 / 0.132 / 0.118) are not recorded in a committed
document; they exist only in an uncommitted working roadmap, a gap `VALIDATION.md:29-31` acknowledges in the
committed tree.

## Sparse-Bayesian coefficient model (horseshoe / spike-and-slab) with sequential D-optimal acquisition

- **Verdict:** abandoned

**Question.** Should the v1 independent-variant acquisition/inference core be *replaced* by a
coefficient-native sparse-Bayesian model — a horseshoe or spike-and-slab prior over the epistasis
coefficients with a calibrated posterior — driving multi-round sequential acquisition under the D-optimal
increment `Δ_D = ½·log(1 + hᵀΣh/σ²)`? The v1 objective is modular rather than submodular
(`info_gain(M, v) = var_delta_g(v)·n(v)` does not depend on the measured set), so the intended
diminishing-returns / loop-closure behaviour is not realised; correlated priors across variants would be
needed to realise it (`LIMITATIONS.md:70-73`).

**What was built.** Nothing of the sparse-Bayesian model itself. `horseshoe` and `spike-and-slab` appear
nowhere in the repository at any point in its history — `git log --all -S"horseshoe"` and
`git log --all -S"spike-and-slab"` both return empty, and `git grep -i` over `HEAD` and the working tree
exits 1. No module, no test, no report artifact.

What *was* built is the cheaper single-shot precursor that priced the idea: `_doptimal_order` in
`src/epibudget/coeff_recovery.py:645-691` — a greedy Bayesian D-optimal (max-posterior-variance) design over
the multiallelic Fourier kernel under an **isotropic N(0, I)** prior (dense, not sparse), selecting a whole
batch at round 0 with a rank-1 GP-style update, using only the design kernel and never a measured label. Its
`orders=(1, 2)` variant restricts the target subspace to singles+pairwise via the closed-form
order-restricted kernel (`_order_symmetric_kernel`). Both are wired into `run_coeff_recovery` as the
`doptimal` and `doptimal_pairs` selections (`coeff_recovery.py:734-749`) and pinned by offline tests for
prefix-consistency, label-freedom and subspace distinctness (`tests/test_coeff_recovery.py:140-196`).
Introduced in commit `6ff14ce`.

**What was measured.** The full model was never run. The decision was priced against a three-way acquisition
comparison on the frozen GB1 selections at `B ∈ {48, 96, 192}`, cache-only and zero-GPU: ESM-weighted
loop-bracing (`info`) versus isotropic label-free D-optimal versus pairwise-targeted label-free D-optimal
versus random, graded by residualized pairwise Spearman recovery against a Fourier-LASSO / kernel-ridge
compressed-sensing baseline (protocol: `specs/step6-coefficient-recovery.md`).

**Outcome.** All three label-free design-geometry strategies failed to beat the ESM-weighted selection.
Restricting the D-optimal target subspace to orders 1-2 removed the isotropic design's active harm but left
it at roughly random. Order-3 stayed unrecoverable at every budget tested. The committed conclusion: "neither
an isotropic (6B) nor a pairwise-targeted (6C) label-free D-optimal acquisition beats the ESM-weighted
selection — pure experimental-design geometry does not recover the map above random, so the ESM prior (not
the acquisition) carries the signal and the budget is the binding constraint" (`VALIDATION.md:24-27`).

The per-budget recovery coefficients behind that sentence are *not* independently checkable from the public
tree: no `coeff_recovery` report artifact exists under `report/`, and `coeff_recovery` exposes no CLI command
— a limitation the repository states about itself (`VALIDATION.md:29-31`). The numbers are therefore treated
here as unverified; only the direction and the decision are substantiated.

**Why this verdict.** Killed on evidence, not on effort or on a technical failure. The sparse-Bayesian
model's whole value proposition is better acquisition geometry, and three independent acquisition strategies
had already triangulated that acquisition is not the binding constraint at these budgets. Building a fourth,
more expensive design prior had no plausible path to changing the conclusion, so the effort was redirected to
the one un-exercised value claim — shifting the estimand from map recovery to the downstream-impact benchmark
(ranking held-out double/triple mutants). Multi-round sequential design survives as the acknowledged
long-term direction while remaining explicitly out of v1 scope (`SPEC.md:344`; `LIMITATIONS.md:210-211`). The
gate that would have chosen between repairing the v1 model and replacing it closed inconclusive and
explicitly declined to make that selection (`VALIDATION.md:247-248`).

**Evidence.**
- `src/epibudget/coeff_recovery.py:640-691` (`_DOPTIMAL_OBS_VAR`, `_doptimal_order`, isotropic and
  orders-(1,2) branches); `:734-749` (both selections registered in `run_coeff_recovery`).
- `tests/test_coeff_recovery.py:140-150`, `:184-196`.
- Commit `6ff14ce`.
- `specs/step6-coefficient-recovery.md` (63 lines; states the decision rule and that
  `public_claim_eligible = false`).
- `VALIDATION.md:18-31` (standing amendment), `:247-248` (the corrective gate declines the architecture
  selection).
- `LIMITATIONS.md:70-73` (modular, not submodular; correlated priors out of scope), `:210-211` (no
  multi-round sequential design); `SPEC.md:344`.
- Absence check: `git log --all -S"horseshoe"`, `git log --all -S"spike-and-slab"` → no commits;
  `git grep -i "horseshoe\|spike-and-slab" HEAD` → exit 1.

**Gap.** The design sketch of the sparse-Bayesian model and the explicit deferral rationale are recorded only
in an uncommitted working document; both are re-expressed above.

## Background-averaged (ensemble) ε and the inference-tool handoff

- **Verdict:** abandoned

**Question.** The tool positions itself as the experimental-design front-end to downstream
epistasis-inference packages (`README.md:33-38`). Inference tools in that ecosystem consume
*background-averaged* (ensemble) epistasis, whereas the selection pipeline produces *WT-referenced*
(biochemical) ε. Should the background-averaged basis be built so that a budgeted selection can be handed off
to an inference fit — and does a fit on the `info` selection beat the same fit on fitness-greedy?

**What was built.** Nothing for this purpose. The ε operator is WT-referenced by construction:
`src/epibudget/epistasis.py:1-3` states the coefficients are "the wild-type sub-sampling of the multiallelic"
transform, and `wt_centered_log_fitness` anchors every ΔG to the WT reference
(`src/epibudget/epistasis.py:31-47`). No module, CLI surface, or test references an ensemble basis or an
inference-tool export — grep for `background_averaged|ensemble_eps|to_mochi|mochi` over `src/` and `tests/`
returns nothing.

The one place a background-averaged basis *was* built is `src/epibudget/coeff_recovery.py`, whose
compressed-sensing baseline fits a character/Fourier design `X[v, m] = chi_m(v)`
(`coeff_recovery.py:176-196`). That basis is background-averaged, but it is used strictly as an internal
fitting device: the fit is projected back to ΔĜ and re-scored through the same WT-referenced ε operator
(`coeff_recovery.py:541-542`, importing `_epsilon` at line 33). The design spec is explicit that this
"sidesteps the WT-referenced-vs-background-averaged distinction (the Fourier β are background-averaged; the
recovered ε̂ are WT-referenced by construction)" (`specs/step6-coefficient-recovery.md:35-38`). So the bridge
component exists in code but was deliberately never surfaced as an output and never wired to an inference
tool.

**What was measured.** Nothing. No fit on a budgeted selection was ever compared across acquisition
strategies for an inference handoff.

**Outcome.** Not attempted. The scientific rationale for the choice is recorded rather than tested:
WT-referenced ε was selected because (a) it matches the practitioner's actual starting point — one wild type
— and (b) it is exactly what conditional ESM-2 scoring on the WT background produces
(`RESEARCH_EPISTASIS.md:93-96`). Background-averaged ε is more robust to local idiosyncrasies but "requires
(or infers) many backgrounds" (`RESEARCH_EPISTASIS.md:89-91`), which the zero-shot single-background scoring
path does not supply.

The repository disclaims the positioning gap in its own limitations register rather than leaving the README
claim unqualified: "Background-averaged (ensemble) ε — the basis inference tools like MoCHI consume — is out
of scope for v1. The 'feeds MoCHI' story is not yet real." (`LIMITATIONS.md:82-83`), restated in the summary
as "No background-averaged ε, no MoCHI handoff, no multi-round sequential design — all deliberately out of
scope for v1." (`LIMITATIONS.md:210-211`). The front-end framing in `README.md:36` and the complementarity
claim in `PRIOR_ART.md:19` are therefore design intent, not a demonstrated handoff.

**Why this verdict.** Scoped out at project inception, with an explicit conditional promotion path that was
never triggered. The original scaffold declared it out of scope with the condition "promote to the v1.1
ambition layer *only if* the MoCHI integration is pursued" (`git show 427c898 -- docs/SPEC.md`, out-of-scope
section). The integration was never pursued, so the bridge was never promoted; the current wording keeps the
same conditional — "It is the bridge to inference tools like MoCHI, and is a future extension only if that
integration is pursued" (`SPEC.md:338-342`). The empirical effort went instead to the downstream-impact
benchmark, which answers a question about training-set quality that does not require the ensemble basis at
all. This is a deliberate non-start kept off the critical path, not a tried-and-failed direction — but it is
worth recording because the public positioning advertises a capability the code does not have, and the
limitations register is the only thing preventing that from being an overclaim.

**Evidence.**
- `LIMITATIONS.md:82-83`, `:210-211` — the explicit disclaimer, both added in commit `315a30a`.
- `SPEC.md:338-342` — out-of-scope for v1 with the conditional promotion path;
  `git show 427c898 -- docs/SPEC.md` — the original scaffold's identical conditional scoping.
- `RESEARCH_EPISTASIS.md:89-96` — the two inequivalent epistasis definitions and the recorded design decision
  to target the WT-referenced one.
- `src/epibudget/epistasis.py:1-3,31-47`; `src/epibudget/coeff_recovery.py:33,176-196,541-542` and
  `specs/step6-coefficient-recovery.md:35-38` (commit `6ff14ce`).
- `README.md:33-38`, `PRIOR_ART.md:19` — the front-end positioning that remains undemonstrated.
- Negative check: no match for `background_averaged|ensemble_eps|to_mochi|mochi` in `src/` or `tests/`.

---

# Validation

## Pre-allocation de-risk signal gate: does conjoint ESM-2 scoring carry measurable GB1 epistasis signal?

- **Verdict:** kept

**Question.** Before committing effort to a factor graph, an acquisition rule or any budget-allocation method
built on ESM-2, two things had to be true of the underlying signal. (1) Conjoint scoring — mutating *all*
positions onto the background and reading each mutated residue's conditional log-likelihood *in* the mutated
context — must be genuinely non-additive, i.e. `Var[ε_pred] > 0`. A per-site additive shortcut makes every
interaction term identically zero by construction, so the whole program would be built on an artifact. (2)
The predicted interaction term must actually track the measured one, at Spearman ≳ 0.2, judged **per
interaction order** rather than pooled.

**What was built.**
- `scripts/fetch_gb1.py` — fetches the Wu et al. 2016 four-site GB1 landscape
  (`SaProtHub/Dataset-GB1-fitness`) and records provenance: source URL, download date, sha256, row count, WT
  sequence and mutation-order composition into `data/proteingym/provenance.json` (data itself is never
  committed).
- `src/epibudget/data.py` — typed genotype loader asserting the WT residues V/D/G/V at GB1 sites
  V39/D40/G41/V54 (0-indexed 38/39/40/53).
- `src/epibudget/scoring.py` — `ConjointScorer`. A positional guard, `_assert_token_alignment`
  (`scoring.py:336-345`), checks that the token at index `p+1` is the intended mutant residue, catching the
  ESM BOS-prepend off-by-one that would silently score the wrong position.
- `src/epibudget/epistasis.py` — WT-referenced inclusion-exclusion operators `epsilon_pairwise` (line 51) and
  `epsilon_third` (line 56).
- `scripts/gb1_epistasis_signal.py` — the standalone gate spike (originally
  `scripts/spike_gb1_epistasis.py`).

**What was measured.** Spearman correlation between ESM-predicted and measured WT-referenced ε, plus
`Var[ε_pred]`, swept across ESM-2 35M / 150M / 650M. Measured ΔG(v) = ln(fitness(v)/fitness(reference)), with
fitness(reference) = 1 for GB1. Sampling: 50 amino-acid combos per position-pair, 40 per position-triple,
seed 0 (`--k-pair 50 --k-tri 40 --seed 0` defaults, `gb1_epistasis_signal.py:103-117`), yielding n = 257
pairwise / 97 third-order instances. Scoring deterministic at `n_perturbations=0`. Two design choices matter
for interpretation: single-mutant terms are subtracted with coefficient −1 on both the predicted and measured
side, so the correlation isolates *interaction* agreement rather than riding on ESM's well-known
single-mutant fitness signal; and dead variants (fitness 0, no log-fitness) are **dropped, never imputed**,
so any interaction with a dead or missing constituent is excluded.

**Outcome.** PASS at 650M. Signal rises monotonically with model capacity:

| ESM-2 | pairwise ρ | third-order ρ | Var[ε_pred] |
|-------|-----------:|--------------:|------------:|
| 35M   | 0.085 | 0.108 | 0.361 |
| 150M  | 0.167 | 0.131 | 0.530 |
| 650M  | 0.302 | 0.249 | 0.777 |

Exact values from `artifacts/signal_650m.json`: `spearman_pairwise` 0.3019452425119898, `spearman_third`
0.24890858405217753, `var_eps_pred_pooled` 0.7771307634544032. Gate #1 holds at every size. Gate #2 holds
only at 650M — both orders clear ≈ 0.2 independently. A second seed at 650M reproduces it
(`report/spike_gb1_650M_seed1.json`: pairwise 0.3047, third 0.2312, Var 0.7922) on a slightly different
sample (n = 262 / 102, since the sampler re-draws per seed). Pooled Spearman (0.114 / 0.120 / 0.316) is
recorded for context only — pooling a 3-term and a 7-term ε into one correlation distorts the estimate and
overstates significance through shared sub-terms.

The 35M model fails the bar on both orders (0.085 / 0.108) and was consequently fixed as the fast smoke-test
path only, never the headline; empirical claims use 650M.

**Why this verdict.** Both pre-stated gate conditions cleared independently, per order, at the headline
model, so the downstream graph and acquisition work rested on a measured effect rather than an assumption.
The result was never reversed: when the comparative epistasis-map recovery claims were retired (`943833d`),
this one survived, and it remains the only ESM-score-level empirical claim in the README's artifact-checked
block. Four honest qualifications keep this from being a stronger verdict than "kept":

- The claim's *domain* was narrowed after the fact. Commit `e6c1bb0` corrected an overreach describing GB1 as
  "the complete four-site landscape"; it is a 149,361-row measured subset of a 160,000-genotype theoretical
  space, with 29,477 dead variants and 10,639 genotypes absent. The README now scopes the claim to "the
  viable GB1 terms available in the local public-data artifact".
- The artifact is not decision-grade. `artifacts/manifest.json` records `artifacts/signal_650m.json` as
  `status: "supplementary"`, `evidence_classification: "traceable_not_rerun"`, `code_state: "dirty"` —
  traceable to a generation command and base commit, but not re-run under a clean tree. Two seeds are
  explicitly indicative, not a frozen protocol.
- The gate is deliberately narrow in what it licenses. It establishes that conjoint scores carry epistatic
  signal; it does **not** validate the masking-variance uncertainty prior, which the README states outright
  and which later work found to add nothing.
- Instance non-independence: sampled interactions share sub-ΔG terms, so an exact p-value from the reported n
  would be optimistic. Point estimates stand; significance is treated as indicative. Dead-variant exclusion
  also removes the strongest live→dead negative-epistasis cases, which if anything deflates the estimate.

**Evidence.**
- `SIGNAL_GATE.md` — gate definition (lines 7-13), data provenance (15-24), method (26-38), result table
  (45-51), verdict (53-62), caveats (64-74).
- `artifacts/signal_650m.json` — headline numbers, `evidence_classification: traceable_not_rerun`.
- `report/spike_gb1.json` (35M), `spike_gb1_150M.json`, `spike_gb1_650M.json`, `spike_gb1_650M_seed1.json` —
  the per-size and per-seed runs.
- `artifacts/manifest.json` — sha256, `generation_command`
  (`python scripts/gb1_epistasis_signal.py --model facebook/esm2_t33_650M_UR50D --seed 0 --out report/spike_gb1_650M.json`),
  `base_commit_sha 1a1f30aabd11bb50af6208bef983f2d017352b97`, `status: supplementary`.
- `artifacts/claim_map.json` — claim ids `signal.pairwise_spearman` / `signal.third_spearman`, bound to
  `README.md` anchors via json-pointers into `artifacts/signal_650m.json`; the registry is itself sha256'd
  into the manifest by `scripts/build_public_artifacts.py:333-337`.
- `README.md:78-82` (the artifact-checked claim), line 99 (the narrowed defensible position).
- `src/epibudget/scoring.py:336-345` (BOS alignment guard), `src/epibudget/epistasis.py:51,56` (ε operators),
  `scripts/gb1_epistasis_signal.py:103-136` (sampling params, deterministic scorer).
- Invariant pinned by tests: `tests/test_scoring.py::test_additive_scoring_yields_zero_epistasis` (line 41,
  additive shortcut ⇒ ε ≡ 0), `::test_nonadditive_landscape_yields_nonzero_epistasis` (offline mirror),
  `::test_epsilon_not_identically_zero` (line 89, `@pytest.mark.slow`, real GB1 slice at 35M, asserts
  `Var[ε] > 0`). The two offline tests pass on the current tree.
- Commits `486853b` (GB1 loader + fetch + provenance), `ade5263` (conjoint scorer), `ecc89c8` (ε operators
  and WHT spectrum), `74833cb` (the gate spike and gate note, originally `docs/STEP1_GATE.md` and
  `scripts/spike_gb1_epistasis.py`), `e6c1bb0` (dataset-completeness correction), `a25efbe` (rename to
  `SIGNAL_GATE.md` / `scripts/gb1_epistasis_signal.py`), `943833d` (retirement of the comparative claims,
  which this result survived).

## Label-leakage barrier: a single reveal point for measured fitness

- **Verdict:** kept

**Question.** The benchmark grades a selection method on how well the resulting measurements recover (or
predict) fitness. If any selector could see the measured labels it is later graded on, the headline number
would be inflated in a way that still looks plausible. How do you make that failure mode structurally
impossible rather than merely unlikely?

**What was built.** A one-way boundary, present from the first commit rather than retrofitted:

- `src/epibudget/data.py:204-212` — `reveal_measured_fitness(landscape, selected)` is the sole function that
  reads the fitness column, and it is only callable *with a selection already in hand*. Its docstring states
  the invariant explicitly.
- Selection-side modules carry no path to a label. `src/epibudget/acquisition.py:15-16` imports only `graph`
  and `types`; `src/epibudget/graph.py:27-28` imports only `epistasis` and `types`. Neither imports `data`,
  and neither takes a landscape argument. `graph.py:35-36` records that the uncertainty model consumes no
  fitness value.
- The benchmark drivers keep the ordering. In `src/epibudget/downstream.py:2511-2597` all five methods
  (`fitness`, `practice`, `random`, `info`, `structural`) receive only `pool`, a sequence of `ScoredVariant`
  carrying ESM `delta_g` / `var_delta_g`; the evaluator closure at `downstream.py:2647-2649` calls
  `reveal_measured_fitness(dict(land), list(selected))` *after* the method has returned.
  `src/epibudget/validate.py:396` and `validate.py:10` do the same for the original recovery harness.
- The same estimator is applied to every method, so only the selected set differs — otherwise a leak could
  hide in an inference asymmetry (`VALIDATION.md`, threats table).

**What was measured.** Four independent guards, one behavioural and three structural:

- Behavioural canary — `tests/test_acquisition.py:78-86`: invert the ESM `delta_g` ranking and the greedy
  top-B must become *disjoint* from the original. Selection tracks the predicted score and nothing else; a
  hidden label channel would keep some overlap.
- Signature guards — `tests/test_acquisition.py:89-93` (`allocate`, `fitness_greedy`) and
  `tests/test_validate.py:524-528` (`random_selection`, `practice_heuristic`) assert no selector accepts a
  `landscape` / `fitness` / `measured` / `labels` parameter.
- `tests/test_robustness.py:358-366` pins `_deterministic_selections`'s parameter set to exactly
  `{scored, budget, max_order}`, so selection recomputation cannot acquire a label input later.
- Source-level guard — `tests/test_downstream.py:950-964` inspects the source of the primary supervised
  predictor (`FeatureSpace.design_matrix`, `active_columns`, `fit_ridge`, `select_alpha`,
  `select_alpha_main_only`, `_build_fold_context`) and fails if `delta_g`, `var_delta_g`, `infer_epistasis`
  or `esm_prior_mu` appear — the mirror-image guard keeping ESM out of the *grading* side.

**Outcome.** The barrier held across every lane added afterwards, and the guards pass on the current tree
(the five tests above: 5 passed). The later diagnostic lanes respect the ordering by construction:

- The correlated-inference probe's ridge hyper-parameter is chosen by generalized cross-validation on the
  *measured errors only* (`src/epibudget/gate3.py:171-172`, `_gcv_lambda`), on labels obtained via
  `reveal_measured_fitness` at `gate3.py:522-528`. Evaluation terms are unmeasured pair/third-order ε and
  never enter selection, slope fitting, or hyper-parameter fitting
  (`specs/gate3-correlated-inference.md:35-40`).
- The compressed-sensing recovery baseline fits and cross-validates only on revealed labels, after an
  ESM-only (`info`) or label-free (`random`) selection is fixed (`src/epibudget/coeff_recovery.py:711-722`;
  `specs/step6-coefficient-recovery.md:40-44`).

One adjacent claim was narrowed, and the narrowing is documented rather than quietly dropped: an earlier text
asserted that changing ESM values could only move three diagnostic fields. That is false — `delta_g` /
`var_delta_g` are legitimate acquisition inputs, so perturbing them legitimately changes which variants are
selected and therefore the downstream metrics. It was replaced by two narrower invariants (clean-predictor
isolation at a fixed plate; diagnostic isolation) with the explicit note that ESM-dependent selection "is
intended behavior, not a leak" (`specs/downstream.md:404-420`). The label barrier itself was untouched by
that correction.

**Why this verdict.** Kept because it is cheap, absolute, and guards the one bug class that would be
invisible in the result: a leak produces a wrong-but-plausible headline rather than a crash or an obvious
anomaly, which is the worst possible failure for a project whose entire output is a verification claim.
Enforcing it by module topology plus signature assertions — rather than by review discipline — means a future
contributor cannot reintroduce it without deleting a test. Treated as blocking in review throughout.

**Evidence.** `src/epibudget/data.py:204-212`; `src/epibudget/acquisition.py:15-16`;
`src/epibudget/graph.py:27-36`; `src/epibudget/downstream.py:2511-2597, 2647-2649, 2719-2721`;
`src/epibudget/validate.py:10, 396, 428`; `src/epibudget/gate3.py:171-172, 522-528`;
`src/epibudget/coeff_recovery.py:711-722`; `tests/test_acquisition.py:1-6, 78-93`;
`tests/test_validate.py:521-528`; `tests/test_robustness.py:358-366`; `tests/test_downstream.py:948-964`;
`VALIDATION.md:55-62` ("Simulation of a budgeted experiment") and `:400` (threats table, rows "Selection
leaks labels" and "Inference step does the work, not selection"); `specs/downstream.md:509-520`;
`specs/step6-coefficient-recovery.md:40-44`; `specs/gate3-correlated-inference.md:35-40`. Origin commit
`427c898` already contains `reveal_measured_fitness` with the barrier docstring verbatim; the
acquisition-side signature guard arrives with `862b0ff`.

**Gap.** Two pieces of the reasoning have no committed home. First, the standing detection procedure: before
accepting any change to the selection path, enumerate every symbol that can reach a fitness value and confirm
none is reachable from `acquisition.py`, `graph.py`, or a selector signature. Second, the rationale for
treating this as the single highest-severity defect class in the codebase (a leak yields a plausible wrong
number, not a visible failure), which motivates the redundancy of four guards where one would nominally do.
Relatedly, `src/epibudget/coeff_recovery.py:714` points a reader at a project-contract document that is not
tracked in the repository; the barrier's public statement should point at `VALIDATION.md` instead.

## Breadth vs precision decomposition of the recovery metric

- **Verdict:** kept

**Question.** A selection method can score high on full-set epistasis-map recovery for two very different
reasons: it *measured* many ε terms outright (breadth), or it *predicted* the unmeasured ones well
(precision). Which one drives the headline recovery correlation — and is the metric therefore partly
tautological?

**What was built.** An additive per-method/per-order reporting split inside the recovery report, fixed before
the 650M headline existed. `src/epibudget/validate.py`: `_informed` (line 230, at least one loop member
measured) and `_pinned` (line 235, the *entire* interaction loop measured, so ε is recovered exactly);
`_order_metric` (lines 244-278) emits `n_informed` / `coverage_fraction` / `n_pinned` (breadth) and
`pearson_predicted` / `spearman_predicted` computed over terms that are informed but **not** pinned
(precision). Fields declared on `OrderMetric` (lines 52-72). The split is derived from the selection set only
— `_informed`/`_pinned` take `measured` and term-loop membership, never a fitness label, and there is no
`|ε| > threshold` answer-key restriction. Pre-registered in `VALIDATION.md` §Metrics ("Breadth vs precision
(pre-registered, additive)", HEAD lines 82-94) one commit before the implementation. The split later became
the substrate for the A1 matched-precision comparison in `src/epibudget/robustness.py` (`common_precision`,
lines 314-325, over `predicted(A) ∩ predicted(B)`, "so a breadth advantage cannot masquerade as a precision
advantage").

**What was measured.** Every method × budget × order cell of every recovery artifact carries the split
alongside the primary correlation. It was never a replacement for the primary statistic and never entered the
frozen decision rule.

**Outcome.** The split made the tautology visible and bounded it.

- 650M full-alphabet headline, B=48 pairwise (`artifacts/headline_650m.json`): **`n_pinned` = 0 for all five
  methods**, while coverage already reaches 0.873 (info) and 0.955 (structural). So the B=48 numbers are pure
  prediction, not read-off.
- Breadth grows with budget: pairwise `n_pinned` = 17 (info) / 20 (structural) at B=96, and 75 (info) / 116
  (structural) at B=192, at 100% coverage for structural.
- Third order stays at `n_pinned` = 0 for info and structural at every budget — no triple loop is ever fully
  closed.
- The reduced-alphabet smoke run is the contrasting exhaustion regime: `artifacts/smoke_recovery_35m.json`
  shows structural pinning **57 of 58** pairwise terms at B=96 (info 47, fitness 26), i.e. recovery there is
  almost entirely "measure the pairs".
- Payoff on the matched-precision follow-up (`artifacts/robustness_650m.json`, `common_precision`, info vs
  structural, pairwise): structural is ahead of info on the *same* predicted terms at all three budgets —
  Spearman 0.5368 vs 0.4518 (n=1511), 0.4538 vs 0.4206 (n=1762), 0.5029 vs 0.4574 (n=1627), with all three
  difference CIs excluding zero. This is a descriptive difference, explicitly labelled "NOT a hypothesis
  test".

Stated conclusion, committed in `LIMITATIONS.md` §4: pairwise recovery ≈ "measure the pairs", third-order
recovery ≈ "measure the triples" — order-matched and close to trivial; a non-tautological advantage must
appear in precision.

**Why this verdict.** Kept, and load-bearing twice over. First, it is the analysis that made the recovery
estimand's weakness legible in a quantified way rather than as a hand-wave, and it did so on a schedule that
blocks post-hoc tuning (reporting frozen before the numbers). Second, its matched-precision extension is what
withdrew credit from the ESM masking-uncertainty prior — structural-only beats info-optimal on identical
predicted terms, so the win could not be attributed to the uncertainty prior. Its diagnosis is cited as the
explicit motivation for moving the project's live claim off recovery entirely: `VALIDATION.md`
§"Post-registration downstream-impact protocol" opens its rationale with "The frozen recovery statistic is
partly tautological (`docs/LIMITATIONS.md` §4)".

**Evidence.**
- Pre-registration: commit `c934bb2` "docs(validation): pre-register the structural-only ablation and
  breadth/precision reporting".
- Implementation: commit `6900206` "feat(validate): add structural-only ablation, breadth/precision split,
  run provenance" — replaces "the vacuous shared informed-union".
- Code: `src/epibudget/validate.py` lines 52-72 (`OrderMetric`), 230-237 (`_informed` / `_pinned`), 244-278
  (`_order_metric`), 537-546 (random-baseline seed averaging of the same fields);
  `src/epibudget/robustness.py` lines 6-7, 304-325.
- Tests: `tests/test_validate.py::test_order_metric_reports_breadth_and_precision` (line 531);
  `::test_breadth_and_precision_split_counts_pinned_vs_predicted` (line 552); line 451 pins `n_pinned` under
  a determinism check.
- Docs: `VALIDATION.md` §Metrics lines 82-94, §"Common identities" lines 177-185 (the non-neutral-intersection
  caveat), §"Post-registration downstream-impact protocol" line 268; `LIMITATIONS.md` §4 lines 99-120.
- Artifacts: `artifacts/headline_650m.json`, `artifacts/smoke_recovery_35m.json`,
  `artifacts/robustness_650m.json`.

## Permutation null (label shuffle) control on the downstream benchmark

- **Verdict:** kept

**Question.** Is the `structural − random` downstream advantage genuine epistatic-learning signal, or an
artifact of the benchmark machinery — something any structured plate would produce even if the
variant→fitness relationship were destroyed?

**What was built.** No new module. The control exploits an existing seam: `downstream_report` takes the
ESM-scored candidates and the fitness landscape as two separate arguments
(`src/epibudget/downstream.py:2694-2714`, `scored: Sequence[ScoredVariant]`,
`landscape: Mapping[Variant, float]`), so a seeded random bijection variant→fitness can be substituted for
the real landscape while every selection — which reads only ESM scores — is bit-identical. Two committed
invariants make that substitution sound rather than assumed: fold assignment provably cannot see labels
(`tests/test_downstream.py:1277`, a signature guard asserting `assign_outer_folds` takes no
`landscape`/`fitness`/`labels`/`measured` parameter), and the primary predictor is AST-guarded against
reading any ESM field (`tests/test_downstream.py:951`). The paired determinism check is committed as
`PYTHONHASHSEED` subprocess tests: `tests/test_downstream.py:1252` and `tests/test_robustness.py:331`.

**What was measured.** `structural − random` on the primary statistic `S_macro`-AUC
(`S_macro = ½(ρ_doubles + ρ_triples)`, target-blind / attempted-budget), under shuffled labels versus the
real labels, on both landscapes, at 10 salted partitions rather than the registered 20. Plus a bit-identity
re-run comparison.

**Outcome.** The effect collapses, as a true null requires. Reported: GB1 falls from +0.175 to **+0.0046**
(~40x smaller), TrpB from +0.135 to **+0.0135** (~10x smaller), with every method's `S_macro ≈ 0`. A small
residual survives permutation and is named rather than rounded away — a coverage/diversity floor, ~3% of the
GB1 effect and ~10% of TrpB's (10/10 partitions still positive on TrpB) — leaving ~97% (GB1) and ~90% (TrpB)
of the advantage attributed to real signal. Re-runs are bit-identical.

The *baselines* reproduce exactly from the run artifacts. Recomputing the target-blind / attempted-budget
`s_macro_auc` partition aggregates gives GB1 `structural − random` mean **+0.1749**, 20/20 partitions
positive, and TrpB **+0.1349**, 20/20 positive (`report/20260715T111312Z/downstream.json`,
`report/20260716T154715Z/downstream.json`). Both determinism tests pass on demand.

Two honesty notes:

- **The permuted-run numbers are not independently confirmable.** `+0.0046` and `+0.0135` appear only in the
  committed prose and in the commit message of `1dda421`. No permuted `downstream.json` is retained under
  `report/`, and no committed CLI flag or script reproduces the control — `epibudget downstream`
  (`src/epibudget/cli.py:761-782`) exposes no label-permutation option. The control was run through the
  library seam ad hoc. The mechanism is sound and cheap to re-run, but the run itself is attested, not
  reproducible from the repository as it stands.
- **The residual's stated mechanism is a diagnosis, not a measurement, and its wording overreaches.** The
  note explains the floor as a structure-aware plate "spanning the mutation orders more evenly."
  Mutation-order composition of the selected plates (computed from the raw records; label-independent, so
  identical under permutation) shows something sharper: at B=48 `structural` selects 48.0 singles / 0.0
  doubles / 0.0 triples while `random` selects 0.1 / 3.5 / 44.4. Neither is "even" — structural concentrates
  on the low orders the pairwise ridge is parameterized on, random is dominated by triples. Only at B=192
  (structural 76.0 / 116.0 / 0.0 vs random 0.7 / 14.2 / 177.1) does structural span more orders. The residual
  is better described as a design-conditioning floor than as evenness.

**Why this verdict.** Kept because it is a real false-positive test with the power to have killed the live
result, and it was reported with its residual quantified rather than declared zero — the only place the
downstream effect is decomposed into signal versus artifact instead of claimed whole. The reduced scale is
disclosed in the committed note with its cause (the machine hit a RAM ceiling at R=20) and the argument that
a null's collapse and a determinism check do not depend on partition count. That argument is reasonable:
`partitions` only controls how many salted repeats are averaged, and the registered decision gate already
refuses eligibility below 20 (`src/epibudget/cli.py:778-787`), so the control is explicitly positioned as a
diagnostic and never as a decision-eligible run.

**Evidence.**
- `experiments/trpb-downstream-generalization-20260716.md:70-87` ("Robustness controls").
- Commit `1dda421f4c836fa4f287d174ca431dd2e4f7512b` — "docs: add permutation-null + determinism controls to
  the downstream note"; +19 lines, doc only; message carries the full numbers. Ancestor of HEAD (`ebd2876`),
  which touched only lines 63-68 of that file and left the controls section unchanged.
- `LIMITATIONS.md:200-205` — the null is carried into the limitations surface ("with a permutation null
  leaving ~90% of the effect as real signal").
- Baselines: `experiments/trpb-downstream-generalization-20260716.md:23` (GB1 +0.175, 20/20) and `:39` (TrpB
  +0.135, 20/20), recomputed from `partition_aggregates` in `report/20260715T111312Z/downstream.json`
  (`dataset = gb1_wu2016`) and `report/20260716T154715Z/downstream.json` (`dataset = trpb_johnston2024`,
  `partitions = 20`, `protocol_profile_mismatches = ["n_perturbations"]`,
  `declared_protocol_profile_conforming = false`). Both report directories are git-ignored, so these are
  local artifacts a reader cannot fetch.
- Seam and guards: `src/epibudget/downstream.py:2694-2723`; `tests/test_downstream.py:1277` (fold assignment
  label-independent), `:951` (predictor never reads ESM), `:1252` (`PYTHONHASHSEED` reproducibility);
  `tests/test_robustness.py:331`. Both determinism tests pass → 2 passed.

## Determinism control: bit-identical downstream re-runs and input-order invariance

- **Verdict:** kept

**Question.** Is the downstream-impact benchmark a pure function of its declared configuration, or does the
reported effect move with incidental things the science does not depend on — the order in which candidate
variants happen to arrive, the order in which raw per-fold records happen to be emitted, or the per-process
hash salt? A metric that drifts under any of those is not measuring what it claims to measure.

**What was built.** Three layers, all in the engine rather than in the analysis scripts:

- Canonicalization at the engine boundary. `downstream_report` sorts the incoming `scored` sequence by
  identity before anything reads it:
  `scored_sorted = sorted(scored, key=lambda sv: canonical_id(sv.variant))`
  (`src/epibudget/downstream.py:2724`). Every downstream structure — the candidate universe, the variance
  map, the ESM map, the factor graphs — is derived from `scored_sorted`, so no selection, graph construction,
  or seeded sample can see the caller's arrival order.
- Identity-derived, label-free fold assignment. `canonical_id` is a canonical JSON serialization of the
  sorted mutation list (`downstream.py:73-80`); outer and inner folds sort by
  `(sha256(salt:canonical_id), canonical_id)` and assign `rank % n_folds` (`downstream.py:129-135`), which is
  reorder-stable by construction and never reads a fitness value.
- Canonical ordering of the forensic raw-record arrays before serialization (`downstream.py:2816-2817`),
  keyed on the registered record key with the full canonical record payload as tie-breaker.

The same discipline was applied to the robustness module after a concrete cross-process bug: a paired
bootstrap was indexing a Python `set` of term tuples, whose iteration order is salted by `PYTHONHASHSEED`, so
`robustness.json` was not byte-identical across processes. The fix is
`common = sorted(_predicted_terms(sa, truth_terms) & _predicted_terms(sb, truth_terms))`
(`src/epibudget/robustness.py:325`), which also makes the paired bootstrap build `rows_a[i]` and `rows_b[i]`
from the same `common[i]` term rather than from two independently ordered lists.

**What was measured.** Byte-level identity of the serialized report under perturbations that must not matter:

- `tests/test_downstream.py:921-945` — run `downstream_report` on a pool, on its reverse, and on a seeded
  shuffle of it; assert `model_dump_json()` is identical in all three.
- `tests/test_downstream.py:1252-1274` — write a small reproduction script to a temp dir and run it twice in
  a subprocess under `PYTHONHASHSEED=0` and `PYTHONHASHSEED=1`; assert identical stdout. This exercises the
  frozen salts plus canonical sorting across process boundaries, where same-process set ordering would
  otherwise mask a defect.
- `tests/test_downstream.py:134-140` and `:163-169` — outer folds are unchanged under reversal of the input;
  inner folds are unchanged per identity under reversal.
- `tests/test_downstream.py:2525-2538` — with raw records injected in original vs arbitrary permutation, the
  report payload (volatile fields excluded) is identical, and `method_budget`, `partition_aggregates` and
  `corrected_cv_companions` match baseline.
- `tests/test_robustness.py:331-353` — the same `PYTHONHASHSEED` 0-vs-1 subprocess check for
  `robustness_report`.

Separately, the actual downstream configuration was re-run and compared.

**Outcome.** All the committed checks hold. Running the determinism-relevant subset: `python -m pytest
tests/test_downstream.py -k "byte_identical_under_an_arbitrary_permutation or reproducible_across_processes
or reorder_stable or inner_folds"` → 5 passed; `python -m pytest tests/test_robustness.py -k
"reproducible_across_processes or paired_alignment or permut"` → 1 passed. The report is byte-identical
(timestamps and explicitly-permitted runtime fields excepted) under an arbitrary permutation of the input
`scored` sequence and across processes with different hash seeds.

The re-run of the real configuration is recorded as bit-identical in the frozen experiment note
(`experiments/trpb-downstream-generalization-20260716.md:83-84`). That control, like the permutation null run
alongside it, was executed at 10 partitions rather than the full 20 because the machine hit a RAM ceiling at
R=20; the note argues the determinism conclusion is scale-independent (lines 86-87), which is reasonable —
byte-identity does not depend on partition count — but the re-run itself is asserted in prose, with no
committed artifact diff a reader can re-check. The automated tests are the reproducible part of this control;
the full-scale re-run is not.

**Why this verdict.** Kept, because the control did real work rather than confirming an assumption. The
`sorted(common)` finding is the proof: the seeded paired bootstrap in `robustness.py` genuinely produced
different output across processes until set iteration was canonicalized. Without the subprocess
`PYTHONHASHSEED` test that defect would have been invisible in a normal same-process test run. The
determinism requirement is treated as a fail condition on the engine, not as a nice property: if a number
moves when only the hash seed or the arrival order changes, it is a bug in the engine and not a finding about
proteins. The controls are cheap, offline, and permanently committed, so they keep holding as the code
changes.

**Evidence.**
- `src/epibudget/downstream.py:73-80` (`canonical_id`), `:129-135` (fold ordering), `:2717-2724`
  (engine-boundary canonicalization + its docstring), `:2816-2817` (canonical forensic record order).
- `src/epibudget/robustness.py:325` (`sorted(...)` on the common-term intersection).
- `tests/test_downstream.py:921-945`, `:1252-1274`, `:134-140`, `:163-169`, `:2525-2538`;
  `tests/test_robustness.py:331-353`.
- `specs/downstream.md:509-520` ("No leakage / determinism (enforced at the engine boundary)"), `:329-331`,
  `:576-578`; `specs/robustness.md:242-248` (the `PYTHONHASHSEED`-salted set-iteration finding), `:264-268`.
- `experiments/trpb-downstream-generalization-20260716.md:83-87`.
- Commit `ec9b5f7` — introduced both the permutation byte-identity test and the `PYTHONHASHSEED` subprocess
  test; commit `1dda421` — recorded the re-run result in the frozen experiment note.

## Post-registration robustness suite A1/A2/A3

- **Verdict:** narrowed

**Question.** Three things the frozen map-recovery statistic could not settle. (1) Is a higher recovery
correlation *breadth* or *precision*? A method can win simply by informing more terms. (2) Is a recovery gap
an artifact of per-method slope fitting? `infer_epistasis` fits one through-origin ESM-to-fitness slope on
each method's *own* revealed set, so methods that reveal differently get different slopes. (3) Is the frozen
rule's non-overlapping-CI criterion a valid difference test? It compares two separate intervals, never the
difference itself. (`specs/robustness.md` §Why, lines 13-27.)

**What was built.** `src/epibudget/robustness.py` (added in `00ee7ad`) plus the `epibudget robustness` CLI
command (`bad1577`, now `src/epibudget/cli.py:343-390`); spec in `specs/robustness.md`. Three analyses, all
post-hoc on an already-scored candidate cache — the module imports no torch, no model, no network.

- **A1** `common_precision()` (`robustness.py:314`) correlates both methods on
  `sorted(predicted(A) ∩ predicted(B))`, where "predicted" means *informed but not pinned*
  (`_predicted_terms`, `robustness.py:304-308`) — the terms where a method had to predict ε rather than read
  it off.
- **A2** `crossfit_slopes()` (`robustness.py:145`) fits one through-origin slope per fold on the measurable
  candidates *outside* that fold, using a deterministic variant-identity partition (`variant_fold`), five
  folds, method-independent. `infer_epistasis_crossfit` then prices each unmeasured loop member by *that
  member's* fold slope — per-member, not one slope per loop (`robustness.py:169-190`).
- **A3** `paired_difference()` bootstraps `corr(A) − corr(B)` on index-aligned terms with one shared
  resample; `hierarchical_random_difference()` (`robustness.py:412`) handles the random arm by resampling
  seeds in the outer loop and drawing a *fresh* term-resample inside each drawn seed, so seed variance nests
  outside term variance. `_N_BOOTSTRAP = 1000` resamples (`src/epibudget/validate.py:47`).

Two engineering decisions worth naming. Every caveat is a **serialized string field**, not a code comment:
`_DIFF_INTERPRETATION` and `_CROSSFIT_CAVEAT` are Pydantic field defaults (`robustness.py:63-88`), so the
JSON artifact carries its own limits as data — enforced by
`test_report_has_serialized_caveats_and_no_pooled_order`. And the CLI re-enumerates the candidate universe
and calls `validate_cache_against_universe` (`src/epibudget/scored_cache.py:173-207`), which **raises, never
warns**, on a missing sidecar or any cache whose identity set is not exactly the requested universe —
including a same-count swap a bare count check would miss. A truncated cache is rejected rather than silently
analysed over a smaller universe.

**What was measured.** GB1 (`gb1_wu2016`), ESM-2 650M, 29,678 candidates, budgets 48/96/192, 20 random seeds,
5 folds, pairwise and third order never pooled. Metrics: per-pair Spearman/Pearson on matched terms with
descriptive difference CIs and the mean informed fraction of each arm (so unequal coverage depth on the
matched set stays visible); global-vs-cross-fitted method ranking agreement; and paired/hierarchical
difference intervals for info vs fitness, info vs structural, info vs random (36 pair-difference records).
Artifact `artifacts/robustness_650m.json`, provenance `source_run_id: 20260711T102440Z`,
`evidence_classification: traceable_not_rerun`.

**Outcome.**

- **A1.** Structural ahead of info on matched-term precision at all three budgets. At B = 48 on 1,511 matched
  pairwise terms: info Spearman 0.4518 vs structural 0.5368, Δ −0.0850, 95% CI [−0.1251, −0.0465], excludes
  zero. Third order at B = 48 on 15,379 terms: Δ −0.0655 [−0.0803, −0.0497]. The info-vs-structural pairwise
  CI excludes zero at B = 96 (Δ −0.0331) and B = 192 (Δ −0.0456) too. By contrast info vs fitness at B = 48
  pairwise is Δ +0.128 [−0.207, +0.426] — does **not** exclude zero, on only 101 common terms.
- **A2.** The cross-fit ranking agreed with the frozen ranking in all six cells (3 budgets × 2 orders):
  `structural > info > fitness` under both the global and the cross-fitted slope, `ranking_agrees: true`
  everywhere. But it simultaneously **destroyed the effect sizes it agreed with**: fitness-greedy's pairwise
  correlation at B = 48 flips from −0.259 (global) to +0.271 (cross-fit), so the structural-vs-fitness gap
  collapses from 0.743 to 0.233. The sign flips in 5 of 6 cells (the exception is third order at B = 192,
  already +0.047 globally). This is the cleanest evidence in the suite that the headline gap over
  fitness-greedy was substantially a per-method slope-fitting artifact.
- **A2 was never extended.** It covers only `("info", "fitness", "structural")` (`robustness.py:512`) — never
  the random arm, never practice, never TrpB. Only `artifacts/robustness_650m.json` exists.
- Tests: `tests/test_robustness.py`, 16 tests, all pass offline and synthetic (no ESM, no network).

**Why this verdict.** Narrowed rather than kept, on three independent grounds, all self-declared in committed
prose rather than discovered later.

1. **Not a pre-registration.** `VALIDATION.md:169-175` states outright that this section cannot claim the
   bias protection of a blind pre-registration, because the qualitative shape of the results (the uncertainty
   prior looking unhelpful; structural-only beating random and fitness-greedy) was already visible when the
   three devices were chosen. It fixes the method before its own numbers exist — which still blocks tuning to
   a number — and nothing more.
2. **Two biases are carried, not absorbed.** The A1 intersection is flagged as a *non-neutral* subsample: a
   term is informed more easily the larger its loop (7 members at third order vs 3 at pairwise), and
   info-optimal's hub bias and fitness-greedy's high-ΔG bias concentrate on the same popular positions — so
   "both methods inform it" correlates with loop size and structural popularity, not only with skill
   (`VALIDATION.md:181-185`). A2's single shared slope removes a per-method calibration confound only by
   *assuming* the ESM-to-fitness relationship is homogeneous across whatever subpopulation each method leaves
   unmeasured (`VALIDATION.md:192-196`).
3. **Barred from the headline by construction.** The serialized caveat forbids adopting `crossfit_ranking` as
   the reported method order and forbids quoting A2 as an operational recovery figure; the difference CIs are
   labelled descriptive, never hypothesis tests. The suite is a companion diagnostic that constrains
   interpretation, not a producer of claims.

**Narrowed further by a later retraction.** A1's B = 48 pairwise numbers had been the strongest quantitative
support for "structural beats info on precision, not just breadth", and were published with six verified
claim entries (`precision.info_sp`, `precision.structural_sp`, `precision.delta`, `precision.delta_ci_lo`,
`precision.delta_ci_hi`, `precision.n_common`) pinning README text to `/result/common_precision/1`. All six
were deleted in `943833d`; no `precision` claim remains in `artifacts/claim_map.json` and no such figure
remains in `README.md`. The reason is not that A1 was computed wrong — it is that the *comparand* was invalid:
the structural score is exactly tied within mutation order, so the frozen run's structural arm was one
deterministic enumeration-order tie-break, an unreplicated draw with no variance over its own tie-break. The
"structural wins at every budget" interpretation is withdrawn, and the corrective seeded re-analysis is
itself inconclusive and public-claim-ineligible, so it does not rescue the converse claim either
(`LIMITATIONS.md:130-135`; `VALIDATION.md:224-227`). The A1 machinery survives; the number it produced does
not.

**Minor traceability weakness.** `CommonPrecision` and `PairDifference` serialize no `budget` field. Budget
is implied only by list position under the `for budget in budgets` loop (`robustness.py:585-641`), which is
why the retracted claim entries had to pin `/result/common_precision/1` by bare index. Nothing else in the
artifact recovers which budget a record belongs to.

**Evidence.**
- `src/epibudget/robustness.py` — module docstring lines 1-18; caveat constants 63-74; `crossfit_slopes`
  145-166; `infer_epistasis_crossfit` 169-190; `paired_difference` 251-269; `_predicted_terms` 304-308;
  `common_precision` 314-379; `hierarchical_random_difference` 412-465; A2 method tuple line 512; budget loop
  585-641.
- `src/epibudget/validate.py:47` (`_N_BOOTSTRAP = 1000`); `src/epibudget/scored_cache.py:173-207` (the
  reject-don't-warn cache guard); `src/epibudget/cli.py:343-390`.
- `specs/robustness.md` — §Why 13-27, §Non-goals 29-33, §A1/A2/A3 80-140.
- `VALIDATION.md:164-205`, `:214-227`; `LIMITATIONS.md:130-135`.
- `artifacts/robustness_650m.json` — `provenance.source_run_id` `20260711T102440Z`; `result.note`;
  `result.common_precision[1]`, `[3]`, `[0]`; `result.scale_sensitivity[0..5]`.
- `tests/test_robustness.py` — 16 tests, all pass offline.
- Commits `00ee7ad` (module + spec + tests, 1,252 lines), `bad1577` (CLI command), `82c898a`
  (`VALIDATION.md` status), `943833d` (retraction: six `precision.*` claim entries removed).

**Corrections to prior accounts.** The paired bootstrap uses 1,000 resamples, not 2,000
(`validate.py:47`). Six claim entries were retracted, not five
(`git show 943833d -- artifacts/claim_map.json`). The cross-fit sign flip for fitness-greedy holds in 5 of 6
cells, not all 6.

## Walsh-Hadamard spectrum as a ground-truth identity check on real GB1

- **Verdict:** narrowed

**Question.** The project computes ground-truth epistasis two ways in principle: WT-referenced
inclusion-exclusion loops, and the variance-by-order decomposition of the multiallelic Walsh-Hadamard
transform. Could the second be run on the real GB1 landscape as an independent identity check on the first —
a second formalism agreeing with the loop-based ε would be strong evidence that neither is mis-implemented?

**What was built.** `wht_spectrum(dg, sites)` in `src/epibudget/epistasis.py:233`, a separable multiallelic
transform: `_orthonormal_contrast_basis` (`:161`) builds a q×q orthonormal basis per site whose row 0 is the
mean mode, `_landscape_tensor` (`:189`) materialises the dense ΔG tensor and infers each site's alphabet from
observed mutations, and `_wht_forward` (`:181`) contracts one basis per axis. Returned values are squared
coefficients aggregated by interaction order, order 0 excluded. Exported from the package
(`src/epibudget/__init__.py:19,42`).

**What was measured.** Two separate things. (1) Correctness on complete synthetic grids —
`tests/test_epistasis.py:276-403`: orthonormal round-trip and Parseval on a random tensor; Parseval against
the population variance of a complete 3-site landscape; an additive landscape giving exactly zero at orders 2
and 3 *and* every WT-referenced ε of order ≥ 2 exactly zero (the two formalisms agreeing); an injected pure
pairwise mode showing at order 2 in the spectrum and as ε(0,1) = 4c in the loop formalism with no leakage to
order 3; and the q=3 multiallelic aggregation path. Ten tests, all passing. (2) Applicability to real GB1 —
coverage of the measured four-site grid, recorded in `artifacts/dataset_gb1.json`.

**Outcome.** Not usable as a real-data check. The transform requires a complete dense tensor: every residue
combination over the chosen sites must be present, else `_landscape_tensor` raises "incomplete landscape: not
every residue combination is present" (`src/epibudget/epistasis.py:228`). The GB1 artifact gives
`theoretical_genotypes` 160,000, `measured_rows` 149,361, `dead_rows` 29,477 (fitness 0, `ln` undefined),
`missing_rows` 10,639, so only `live_rows` 119,884 are log-transformable — roughly a quarter of the grid is
unusable and the dense tensor cannot be built at all. At order 4 alone, 25,129 of 130,321 genotypes are dead
and 9,147 absent. `wht_spectrum` is therefore validated on synthetic complete grids only; on real GB1 it is
context/reporting, not an identity check. No script or notebook in the repository calls it on real data — its
only live consumers are the synthetic tests and one cross-check.

**Why this verdict.** Narrowed rather than abandoned, for two reasons.

First, the alternative — imputing the dead and absent cells to complete the tensor — was rejected in favour
of a hard failure. Dead and missing constituents are dropped, never imputed (invariant #3,
`SIGNAL_GATE.md:29-31`), the same rule `ground_truth_epistasis` follows by skipping any interaction with an
incomplete loop (`src/epibudget/epistasis.py:144-158`); an interaction whose loop touches a dead variant is
simply unrecoverable ground truth. Raising is what stops the spectrum from being quietly misapplied to a
holed landscape and reported as if it were a clean decomposition. The bias this leaves is stated and signed:
dropping the dead removes the strongest negative-epistasis cases, biasing the tested domain toward
all-viable interactions, which if anything *deflates* the measured signal (`LIMITATIONS.md:58-62`).

Second, the function kept a real job in a narrower scope. It serves as the independent reference for the
compressed-sensing coefficient-recovery work: `test_fourier_design_round_trips_and_matches_wht_spectrum`
(`tests/test_coeff_recovery.py:89-103`) checks that the Fourier design matrix reconstructs ΔG exactly and
that its squared coefficients match `wht_spectrum` under Parseval, on a complete 27-genotype synthetic
landscape. Two independently written implementations of the same basis agreeing is the identity check that
survived — just on synthetic grids rather than on GB1.

The narrowing is recorded in the committed docs rather than left implicit: the design-implications table
reads "WHT is used only on complete synthetic grids; real-GB1 truth uses WT-referenced complete loops"
(`RESEARCH_EPISTASIS.md:214`), the validation protocol lists the spectrum "for context" (`VALIDATION.md:52`),
and the spec describes it as something `ground_truth_epistasis` can "for completeness" also compute
(`SPEC.md:163`).

One caveat on the strength of the evidence: the applicability question is settled *structurally* — the
transform's precondition versus the measured coverage artifact — not by an executed run that failed on real
GB1. No such run is recorded.

**Evidence.**
- `src/epibudget/epistasis.py:161-253` (`_orthonormal_contrast_basis`, `_landscape_tensor` with the raise at
  `:228`, `_wht_forward`, `wht_spectrum`); `:144-158` (`ground_truth_epistasis`, drop-never-impute).
- `tests/test_epistasis.py:276-403` — ten synthetic-grid tests including
  `test_wht_spectrum_raises_on_incomplete_landscape` (`:352`), which punches one hole in a complete grid and
  asserts `ValueError`. Verified passing → 10 passed.
- `tests/test_coeff_recovery.py:89-103` — Parseval cross-check against the Fourier design matrix. Verified
  passing.
- `artifacts/dataset_gb1.json` — `theoretical_genotypes` 160000, `measured_rows` 149361, `live_rows` 119884,
  `dead_rows` 29477, `missing_rows` 10639, plus the per-order breakdown.
- `LIMITATIONS.md:58-66`; `SIGNAL_GATE.md:29-31`; `RESEARCH_EPISTASIS.md:214`; `VALIDATION.md:52`;
  `SPEC.md:163`.
- Commits `ecc89c8` (introduces `wht_spectrum` with its synthetic tests), `315a30a` (registers the data
  limitation), `6ff14ce` (coefficient recovery, where the Parseval cross-check lands).
- No caller outside tests: `grep -rn "wht_spectrum" scripts/ notebooks/` returns nothing.

## Corrected-CV (Nadeau-Bengio) interval as the primary downstream decision gate

- **Verdict:** narrowed

**Question.** The downstream-impact benchmark resamples one landscape: R=20 salted partitions × K=5 outer
folds, all drawn from a single variant universe. Ordinary CV t-intervals are anti-conservative under that
overlap. The plan was to gate the decision on a Nadeau-Bengio corrected-resampled t interval, whose variance
inflation `var * (1/n + n_test/n_train)` is designed for exactly this reuse. Does that interval actually
license the decision here?

**What was built.** `src/epibudget/downstream.py` carries the interval as `_corrected_cv_formula` (lines
556-628), emitting a `SensitivityInterval` (539-553) with `convention`, `status`, `n_test`, `n_train`,
`ratio`, `n_valid_effects`, `delta_mean`, `sample_variance`, `df`, `se`, `t_critical`, `ci95`.
`_corrected_cv_companion` (1758-1821) wraps it in a `CorrectedCVCompanion` (631-648) that reports **two
separately labelled ratio conventions instead of one authoritative ratio**:
`pool_ratio = n_eval / selectable_pool_size` and `effective_label_ratio = n_eval / effective_train_size`.
Each carries its own sizes and interval, or `status="unavailable"` when its denominator is undefined. Sizes
are averaged only over the budgets the paired statistic itself spans (`mode="at_max"` → `max(budgets)` only;
`mode="auc"` → all budgets), so the reported ratio always describes training sizes the contrast actually
used.

The replacement primary gate is `robustness_gate` (1698-1755), a purely partition-level, 7-point rule
computed from `PartitionAggregate` means with constants `EXPECTED_PARTITIONS = 20`, `SIGN_THRESHOLD = 16`,
`MIN_STRUCTURAL_EFFECT_SIZE = 0.0` (lines 61-67).

**What was measured.** Whether `n_test/n_train` is identifiable in this design. It is not: the "test" set is
an outer held-out fold's *measured* members, while the "train" pool is a **budget-limited selected subset**
of a much larger candidate universe, not an independent draw from the population the test set comes from.
Two defensible denominators exist (the selectable pool identity count vs the post-missingness label count)
with no principled way to choose, so the interval's width is convention-dependent rather than a property of
the data.

**Outcome.** The interval is retained but non-decisional.
`DecisionSummary.structural_downstream_supported` and `esm_uncertainty_supported` are assigned solely from
`struct_gate.supported` / `esm_gate.supported` (`downstream.py:2415-2416`); no corrected-CV field enters any
decision path. `RobustnessGate.supported = sign_pass and global_mean_positive and median_positive and
effect_size_pass`, with `decision_eligible` requiring complete 20/20 partition coverage (1727-1733) — a
missing or wholly degenerate partition fails closed rather than shrinking the denominator. Every companion
object ships a fixed `assumption_warning` (635-641) stating that folds are not i.i.d., that the ratio is not
naturally identified, and that neither interval is a frequentist CI over future wet-lab campaigns or
proteins.

The live GB1/TrpB result note reports only robustness-gate quantities (`structural − fitness` 20/20
partitions positive, S_macro-AUC mean +0.342; `info − structural` 15/20, +0.007, below the 16/20 gate); no
corrected-CV interval is quoted anywhere in it.

**Why this verdict.** Narrowed rather than abandoned. The claim the benchmark can actually support is
stability of the `structural − fitness` effect across the 20 frozen partitions of one landscape, not
frequentist generalization to unseen proteins — four sites and one assay cannot license the latter, and the
ratio the corrected formula needs is not identified in a selection-then-training protocol. The interval still
has descriptive value (it shows how wide the paired effect is under either convention), so it was kept,
labelled, and stripped of authority instead of deleted. The minimum effect size was deliberately frozen at
`0.0` because no prior study establishes a practically meaningful `S_macro`-AUC delta on this benchmark, and
picking a non-zero bar after seeing a favourable exploratory direction would be the exact after-the-fact
threshold-setting the amendment exists to prevent.

**Evidence.**
- Original (superseded) rule: `VALIDATION.md:288-302` — "The decision gate is a Nadeau-Bengio
  corrected-resampled t over R=20 frozen salted partitions × K=5 folds … plus a ≥16/20 partition-mean
  sign-consistency safeguard", with support defined as the AUC contrast excluding zero positive.
- Narrowing: `VALIDATION.md:304-330` (protocol amendment 1) — "reframes the corrected-CV interval as a
  labelled sensitivity companion rather than the primary gate (replaced by an explicit 7-point
  partition-level robustness gate)".
- Spec: `specs/downstream.md:82-131` — "Primary robustness gate (frozen; replaces the corrected-CV interval
  as the primary gate)" and "Corrected-CV interval (demoted to a labelled sensitivity-only companion)",
  including the two-convention rationale; also `:319-320`, `:453`, `:463-471`.
- Code: `src/epibudget/downstream.py:61-67`, `:539-648`, `:1698-1755`, `:1758-1821`, `:2415-2416`.
- Tests: `tests/test_downstream.py:473-501` (formula checked by hand against Nadeau-Bengio; `unavailable`
  when the ratio is undefined; degenerate folds dropped from `n_valid_effects`), `:586-651` (each
  convention's sizes come only from the contrast's own budgets), `:884` (companions reconstruct exactly from
  raw records) → 11 passed.
- Commits: `ec9b5f7` introduced the demoted implementation, the spec section and the tests together
  (`git log -S"Nadeau"` and `-S"sensitivity_only"` on `src/epibudget/downstream.py` both return only
  `ec9b5f7`); `24fdd77` carries the superseded pre-amendment rule text in `VALIDATION.md`.
- Result note with no corrected-CV number: `experiments/trpb-downstream-generalization-20260716.md:16-29`.

## Corrective zero-GPU replay over fixed selections: seeded structural tie-breaks plus shared vs method-specific slopes (gate 2)

- **Verdict:** inconclusive

**Question.** Two mechanical defects were found in the earlier GB1 map-recovery comparison, and both could
manufacture the reported effect on their own:

1. the `structural-only` control has no within-order signal at all — the loop-coverage count `n(v)` is
   constant per order (1140 singles / 39 doubles / 1 triple), so with a flat uncertainty prior the greedy
   weight takes exactly three distinct values and the within-order ranking is an exact tie silently resolved
   by candidate-enumeration input order. The control was therefore one unreplicated draw with zero variance
   over its own tie-break;
2. the calibration slope was fitted per method, so a method with no measured loop member reduces to
   `ε̂ = b · ε̂_ESM` and its recovery is `sign(b) · ρ_prior` — the sign of a nuisance parameter rather than a
   property of its selection.

With both controlled, does *any* comparative claim survive on the already-computed 650M GB1 universe —
specifically, does the ESM masking-dispersion term contribute anything over structural loop-bracing? Three
outcomes were registered in advance: `repair_current_core`, `replace_phase2_current_model`,
`inconclusive_zero_gpu`.

**What was built.** `src/epibudget/gate2.py` (1,812 lines) and its `epibudget gate2` CLI command
(`src/epibudget/cli.py:527`), with `tests/test_gate2.py` (1,318 lines). It replays the completed 650M GB1
scored cache at zero new GPU cost — no model inference, no new labels. Mechanism: `_score_strata` groups
candidates into exact-score strata and `_permuted_strata_order` (`gate2.py:329-337`) draws 100 independent
seeded permutations *within* each tie stratum, replacing the single enumeration-order prefix; every selection
is then evaluated under two slope regimes, `operational_method_specific` (what the pipeline actually does)
and `shared_crossfit_5fold` (a method-independent five-fold cross-fit used strictly as attribution evidence,
never as a selection method). Pairwise and third-order terms are evaluated separately and never pooled for a
decision. The decision rule itself is a pure function, `decide_gate2` (`gate2.py:1046-1076`), and all five of
its branches are pinned by a parametrized test (`tests/test_gate2.py:800-826`); it fails closed to
`inconclusive_zero_gpu` when eligibility is not satisfied.

**What was measured.** One run at B ∈ {48, 96, 192}, 20 random seeds, 100 structural tie seeds, 5 cross-fit
folds, 2,000 paired term bootstraps, serializing 372 selection records, 377 slope records and 1,488
order-stratified evaluation cells with per-array SHA-256 hashes. Cache identity is verified rather than
trusted (expected vs observed identity, validator status `passed`). Registered statistics: per-budget change
in pairwise Pearson and Spearman *and* relative squared-error gain after revealing the selected measurements
(all three 95% intervals must agree in sign for a budget to count as positive or negative —
`_evidence_status`, `gate2.py:759-766`); the dispersion contribution as info's value against the
100-permutation structural distribution in four cells per budget (2 regimes × 2 statistics); and strict
calibration sign reversal between the two slope regimes at ≥2 budgets.

**Outcome.** Registered decision `inconclusive_zero_gpu`, architecture-decision eligibility satisfied.

- *Inference is mixed, at every budget.* Revealing the measurements improves rank but worsens squared error:
  ΔSpearman +0.150 / +0.163 / +0.193 and ΔPearson +0.184 / +0.200 / +0.229 at B = 48 / 96 / 192, all with
  intervals excluding zero positive, while relative squared-error gain is −0.915 / −1.024 / −0.939 with
  intervals entirely below zero. Every budget therefore scores `inconclusive`.
- *Dispersion contributes nothing detectable over seeded structural ties.* B = 48 inconclusive in all four
  cells; B = 96 **negative in all four** (info below the structural median, e.g. operational Pearson −0.041
  [−0.059, −0.027]); B = 192 negative in three cells, with shared-slope Spearman crossing zero (−0.008
  [−0.017, +0.001]). Overall status inconclusive.
- *No registered contrast reverses sign* across the two slope regimes at the required two budgets — but the
  effect sizes differ by a large factor, which is the substantive finding of the slope separation: the
  info-vs-fitness Pearson difference is 0.729 under the operational method-specific slope and only 0.203
  under the shared cross-fit slope at B = 48 (0.742 → 0.217 at B = 96; 0.648 → 0.217 at B = 192). Roughly two
  thirds of the apparent info-over-fitness advantage was carried by the per-method slope, not by the
  selection.

The run is permanently marked `status = provisional` and `public_claim_eligible = false` — it executed from a
dirty worktree (`code_state = "dirty"`), a property recorded in the artifact itself rather than asserted
afterwards.

**Why this verdict.** The registered outcome is literally inconclusive, and nothing about it was salvaged
into a claim in either direction. It does not rescue the earlier comparative recovery headline, and —
importantly — it also does not support the converse claim that ESM dispersion actively hurts allocation, nor
does it choose between repairing and replacing the acquisition model. The two mechanical defects turned out
to be large enough to dissolve the original comparison without being large enough to establish its opposite.
That is exactly why it was run as a cache-only replay first: it cost no GPU time and it prevented an
expensive confirmatory campaign from being run against a control whose selection was effectively "the first
B singles in enumeration order". The code is retained as an auditable diagnostic, not as a source of results;
the confirmatory second-landscape map-recovery run stayed deferred on the strength of this outcome. The
separately-registered downstream-impact benchmark was explicitly *not* blocked by it and did subsequently
run. The implementation is kept; the scientific question is not resolved, and the conservative reading is
taken here since the artifact's own registered decision field reads `inconclusive_zero_gpu`.

**Evidence.**
- `src/epibudget/gate2.py` — `PROTOCOL_VERSION = "gate2-v1"` and `RUN_TYPE = "post_hoc_corrective_gate2"` at
  lines 41-42; `_permuted_strata_order` at 329-337; `_evidence_status` at 759-766; `_inference_evidence` at
  769-852; `decide_gate2` at 1046-1076.
- `tests/test_gate2.py:800-826` — parametrized coverage of all decision branches plus the fail-closed
  ineligible path. `python -m pytest tests/test_gate2.py -q` passes (30 tests).
- `src/epibudget/cli.py:527` — the `gate2` command; provenance capture at 439-490.
- Commit `621cfced1fad767c8e449b343393014174f22681` — the only commit touching `gate2.py`; adds 1,812 + 1,318
  lines.
- `VALIDATION.md:229-256` §"Corrective Gate 2 (provisional, non-public)"; `:224-227` (historical comparative
  conclusions no longer current); `LIMITATIONS.md:148-177` §6b (both defects and their corrective status);
  `SPEC.md:286-304` (the command documented as GB1-only and CPU-only).
- Local artifact `report/20260714T104137Z/gate2.json`, SHA-256 recomputed and matching the value recorded in
  `VALIDATION.md`: `cb24af4f0ffd025260b430fa069075653608d048d8efd2a42c05b47384149fe5`. Its provenance block
  records `execution_commit fc6436a90911080e83b439f9e776d3086c393dcf`, `code_state "dirty"`,
  `code_diff_sha256 2e876ac1…`, `scored_cache_validator_status "passed"`, and a 363-second CPU runtime. Note
  that `report/` is not tracked, so a public reader can follow the recorded hash and the exact command but
  cannot open the artifact.

**Gap.** The per-budget, per-cell breakdown of the dispersion-contribution and calibration results summarized
above is not reproduced in any committed document — the committed record states only the aggregate verdicts.

## First uncertainty-prior calibration pass (n = 150, prose-only)

- **Verdict:** superseded

**Question.** Does the ESM-2 masking-perturbation dispersion (`var_delta_g`, σ²) track the model's real
per-variant prediction error? If it does not, the uncertainty prior cannot be a useful acquisition signal,
and the mechanism behind the observed allocation ablation is explained.

**What was built.** The first pass was not built at all in the reproducible sense: the correlation was
computed ad hoc and written straight into `LIMITATIONS.md` §5 with no committed script, no CI, and no
artifact. The replacement is a real measurement path: `src/epibudget/calibrate.py` (calibration slope `b`,
absolute error `|b·ΔĜ − ΔG_measured|`, Spearman and Pearson, `bootstrap_ci` with 1000 resamples seeded at
`seed+1`/`seed+2`) driven by `scripts/calibrate_uncertainty.py` (defaults `--n 300`, `--n-perturbations 16`,
`--max-order 3`, `--seed 0`), with `tests/test_calibrate.py`. Provenance was added on top:
`artifacts/calibration_35m.json` and `artifacts/calibration_650m.json` (checksummed,
JSON-pointer-addressed), `artifacts/claim_map.json`, and the validator `src/epibudget/artifacts.py` invoked
by `scripts/validate_artifacts.py`.

**What was measured.** Spearman(σ², |calibrated error|) at ESM-2 35M and 650M on GB1 variants. The first pass
reported point estimates only at n = 150. The replacement reports Spearman and Pearson with bootstrap 95% CIs
at n = 300, with the sampled composition recorded in the artifact (order 1/2/3 = 1/28/271, seed 0,
`data_sha256` `2f115d4e…`, `evidence_classification: traceable_not_rerun`).

**Outcome.** First pass: two small positive Spearman values, at 35M and at 650M, on a 150-point sample,
published as "a confirmed null, at both model sizes" (`5a56314`). Those figures are now hard-banned
strings and are deliberately not restated here. Scripted replacement: **35M Spearman +0.042, 95% CI [−0.078, +0.157]; 650M
Spearman −0.113, 95% CI [−0.220, −0.002]**, both n = 300 (`artifacts/calibration_*.json`,
`/result/spearman_sigma2_abserror` = 0.04245… and −0.11282…). The sign at 650M reversed from small-positive
to small-negative, and the CI there marginally excludes zero — so the committed reading was narrowed rather
than merely re-precised: `README.md` and `LIMITATIONS.md` §5 now state a weak negative rank association,
explicitly **not** a general anti-calibration claim, since Pearson (−0.100, CI [−0.198, 0.003]) remains
compatible with zero. The load-bearing conclusion that survives is only the negative one: σ² does not show
positive association with error, so it is not usable as an acquisition signal.

**Why this verdict.** The qualitative direction held, but the numbers themselves were unbacked by any
runnable code and underpowered, which is exactly the failure mode the project treats as unpublishable. Rather
than deleting them quietly, the three superseded literals were promoted to
hard-banned strings: `artifacts/claim_map.json` `forbidden_literals`, enforced over `README.md` and every
`docs/**/*.md` by `src/epibudget/artifacts.py:203-215`, wired into the pre-commit hook and CI by `bee30d7`,
and pinned by `tests/test_artifacts.py:96-101`. A stale number cannot silently reappear in public prose.
`python scripts/validate_artifacts.py` passes at HEAD (exit 0).

**Evidence.**
- `5a56314` — `LIMITATIONS.md` §5, introduces the two uncalibrated Spearman values and their sample size;
  the commit message repeats them.
- `b84d0ee` — adds `scripts/calibrate_uncertainty.py`, `src/epibudget/calibrate.py`,
  `tests/test_calibrate.py`; `--n` default is 300 and the bootstrap is present from birth.
- `188b5ae` — adds `artifacts/claim_map.json` (with `forbidden_literals`),
  `artifacts/calibration_{35m,650m}.json`, `src/epibudget/artifacts.py`, `scripts/validate_artifacts.py`.
- `bee30d7` — `scripts/hooks/pre-commit:48` and `.github/workflows/ci.yml:41` run the validator.
- `5bff991` — removes the n = 150 literals from `LIMITATIONS.md`, replacing them with the n = 300 values and
  CIs.
- Current state: `README.md:84-91`, `LIMITATIONS.md:122-129` (HEAD), `src/epibudget/artifacts.py:211`,
  `tests/test_artifacts.py:55` and `:96-101`.

## Shared informed-union evaluation subset as the common grading set

- **Verdict:** superseded

**Question.** Map-recovery correlations over the full truth-term set are dominated by terms no method
touched, where every method reports the same prior-driven ε̂. Can a *single subset of terms, fixed identically
across all methods*, be used as the common grading set so that per-method correlations stay mutually
comparable?

**What was built.** The first GB1 recovery harness (`src/epibudget/validate.py`, added in `478b7ac`) carried
an `informed_union: frozenset[Term]` parameter threaded through `run_validation` → `map_recovery` →
`_order_metric`, plus three report fields on `OrderMetric`: `n_informed_union`, `pearson_informed`,
`spearman_informed`. The subset was built from the selections only (no ground-truth peeking) as the **union**
over all four deterministic methods *and* all random seeds:

```python
informed_union = frozenset(
    term for term in term_set
    if any(_informed(term, m) for m in (*det_measured.values(), *random_measured))
)
```

Note that it is a union (`any`), not an intersection.

**What was measured.** Per method and per order (pairwise / third / pooled), Pearson and Spearman of inferred
vs ground-truth ε restricted to `informed_union` — the same term identities for every method — reported
alongside each method's own `coverage_fraction`. It was pre-registered in `VALIDATION.md` as a companion
diagnostic, explicitly additive and never part of the decision rule.

**Outcome.** Retired before it produced a committed number. `6900206` removed the parameter and all three
fields, its message stating the replacement was made "replacing the vacuous shared informed-union". No
committed artifact carries `n_informed_union`: the first artifacts landed in `e75535d`, after the removal —
so the vacuity verdict is a design-time judgment recorded in a commit message, not a measured result in the
repository. The mechanism is nonetheless legible from the code: because the union takes `any` over every
method plus every random seed, it is dominated by the *broadest* selection, so a narrow-coverage method is
graded largely on terms it never informed. For those terms `ε̂ = b · ε̂_ESM` exactly (`LIMITATIONS.md` §6b),
i.e. the shared ESM prior scaled by a per-method slope — so every method's union correlation collapses toward
the same prior correlation and the subset carries almost no discriminating signal.

The replacement, pre-registered in `c934bb2` before the headline existed, splits the same concern into two
explicit quantities per method and per order:

- **breadth** — `n_pinned`, terms whose *entire* interaction loop is measured, hence recovered exactly;
- **precision** — `pearson_predicted` / `spearman_predicted` over terms the method informs but does *not*
  pin, where it must genuinely predict ε.

A third revision (`00ee7ad`) addressed the residual comparability problem the union had been trying to solve:
method-specific precision sets contain different term identities, so `common_precision` in
`src/epibudget/robustness.py` restricts any *direct* A-vs-B precision comparison to
`sorted(predicted(A) ∩ predicted(B))` — a genuine intersection this time, and pairwise rather than global. On
the frozen 650M run (`artifacts/robustness_650m.json`, `result.common_precision[0]`, info vs fitness,
pairwise): `n_common = 101`, `spearman_a = 0.1095`, `spearman_b = -0.0182`, `delta = 0.1278` with 95% CI
`[-0.2072, 0.4257]`, `excludes_zero = false`, labelled "descriptive difference on matched terms; NOT a
hypothesis test".

**Why this verdict.** Superseded, not abandoned: the goal (comparable grading across methods) was kept, the
device was replaced. A global union fixed across methods buys nominal comparability at the cost of
discrimination, because comparability is achieved by grading every method on the broadest method's terms. The
breadth/precision split makes the underlying tautology explicit and testable — "a non-tautological advantage
must show up in precision, not only in breadth" — and the later pairwise intersection restores comparability
only where it is actually earned, with a committed caveat that the intersection is not a neutral subsample (a
term is informed more easily the larger its loop, and both info-optimal's hub bias and fitness-greedy's
high-ΔG bias concentrate on the same popular positions). The chain's terminus is worth recording: `943833d`
removed the common-support precision deltas from `artifacts/claim_map.json` along with the other comparative
recovery claims, so neither the original union metric nor its successor's deltas stand as current public
claims.

**Evidence.**
- `478b7ac` — introduced `informed_union`, `n_informed_union`, `pearson_informed`, `spearman_informed` in
  `src/epibudget/validate.py` (union construction via `any(_informed(...))`).
- `6900206` — removed all four symbols; message: "replacing the vacuous shared informed-union".
- `c934bb2` — pre-registers the structural-only ablation and breadth/precision reporting in `VALIDATION.md`.
- `00ee7ad` — `common_precision` at `src/epibudget/robustness.py:314-325`
  (`_predicted_terms(sa, …) & _predicted_terms(sb, …)`).
- `943833d` — retired the common-support precision deltas from `artifacts/claim_map.json`
  (`precision.info_sp`, `precision.structural_sp`, `precision.delta`, `precision.delta_ci_lo`,
  `precision.delta_ci_hi`, `precision.n_common`).
- Current state: `src/epibudget/validate.py:55-72` (`OrderMetric` docstring and fields), `:255-277`
  (`_order_metric` breadth/precision computation); `_pinned` helper.
- Tests: `tests/test_validate.py:532-538`, `:552-561`.
- Committed rationale: `LIMITATIONS.md` §4 ("Map-recovery is partly tautological", "Method-specific precision
  correlations are not directly comparable"); `VALIDATION.md` breadth-vs-precision section and the "Common
  identities" caveat.
- Data: `artifacts/robustness_650m.json`, `result.common_precision[0]`.

**Gap.** The mechanism by which the union collapses toward the shared prior correlation is a reconstruction
from the code; it is not stated in a committed document.

## Pre-registered stop rule and closure-check fallback for a failed signal gate

- **Verdict:** superseded

**Question.** Before any machinery was built on ESM-2 conjoint scores, what happens if the scores turn out to
carry no epistasis signal at all? The pre-commitment was meant to make the de-risk gate a real go/no-go
rather than a formality: name the failure action in advance, so a negative gate cannot be quietly absorbed
and the factor graph and acquisition built anyway on absent signal.

**What was built.** Two halves, and only one of them is public.

The gate itself was built and committed: `scripts/spike_gb1_epistasis.py` (later
`scripts/gb1_epistasis_signal.py`), the conjoint scorer in `src/epibudget/scoring.py`, and the
non-additivity guard `tests/test_scoring.py::test_epsilon_not_identically_zero` (line 89). Its criteria and
result are recorded in `SIGNAL_GATE.md`.

The failure branch was written only as a planning-document line: on a failed gate, stop, and either repivot
the project to a differently-framed "closure-check" objective or reframe the objective outright — explicitly
not proceed to the graph or the acquisition. No module, script, spec or test named `closure-check` was ever
written. The string appears in no committed blob at HEAD and in no commit in the repository's history.

**What was measured.** Nothing — this is a decision rule fixed before the gate ran, not an experiment. The
gate it guarded was measured: Spearman between ESM-predicted and measured WT-referenced inclusion-exclusion ε
on GB1, judged per interaction order (pairwise, third) rather than pooled, at three ESM-2 sizes, seed 0,
n = 257 pairwise instances and 97 third-order.

**Outcome.** The fallback was never exercised, because the gate passed. `SIGNAL_GATE.md` records **PASS** at
650M: pairwise ρ **0.302**, third-order ρ **0.249**, `Var[ε_pred]` **0.777**; 150M gives 0.167 / 0.131 /
0.530 and 35M gives 0.085 / 0.108 / 0.361, so the fast model would not have cleared the ≈0.2 bar on either
order. A second seed at 650M reproduces the effect (pairwise 0.305, third 0.231), so the pass is not a
sampling fluke.

The `closure-check` fallback left no trace: `git grep -i closure-check HEAD` returns nothing, and
`git log --all -S"closure-check"` returns no commit. A public reader can see the gate's criteria and its
pass, but not what would have happened had it failed.

**Why this verdict.** Superseded rather than abandoned: the *practice* the stop rule encoded — name the
failure branch before the numbers exist — survived and was carried into committed documents at every later
gate, in a stronger form with explicitly named outcomes.

- `VALIDATION.md:108-122` freezes one statistic before any result exists and states the negative branch in
  the document itself: on failure "the report headline is the observed relationship (partial, null, or
  negative), stated plainly, with the same figures."
- `VALIDATION.md:129-135` goes further and pre-commits a kill: "If info-optimal ≈ structural-only, the ESM
  uncertainty prior does nothing to the allocation and must be dropped from the claims."
- `specs/gate3-correlated-inference.md:47-56` and `specs/step6-coefficient-recovery.md:53-62` each enumerate
  named terminal verdicts including the failure ones (`replace_phase2`; `both_weak`), and both mark the probe
  `public_claim_eligible = false`.

And unlike the original fallback, the successor rule was actually triggered. The downstream benchmark's
`info − structural` contrast failed its own pre-registered sign gate — 15/20 partitions against a 16/20
requirement, mean +0.007, recorded as "not supported"
(`experiments/trpb-downstream-generalization-20260716.md:24`). The structure-only selection won; the ESM
masking-dispersion prior added nothing, exactly the branch `VALIDATION.md:133` had pre-committed to
reporting.

So the specific artifact (a one-line stop rule naming an unbuilt `closure-check` pivot) is dead and
undocumented publicly; the discipline it represented was replaced by committed, machine-checkable, and
demonstrably enforced decision rules.

**Evidence.**
- `SIGNAL_GATE.md` — gate criteria (lines 7-13), size table (lines 45-51), `Verdict: PASS` (line 53),
  two-seed caveat (lines 66-69).
- `tests/test_scoring.py:89` — `test_epsilon_not_identically_zero`, the enforced half of gate criterion #1.
- Commit `74833cb` "feat(validation): add epistasis de-risk spike and Step 1 gate note" — added
  `docs/STEP1_GATE.md` (+72) and `scripts/spike_gb1_epistasis.py` (+197); renamed to `SIGNAL_GATE.md` in
  `a25efbe`.
- `git grep -i closure-check HEAD` → no output; `git log --all -S"closure-check"` → no output.
- `VALIDATION.md:108-135`; `specs/gate3-correlated-inference.md:47-56`;
  `specs/step6-coefficient-recovery.md:53-62`; `experiments/trpb-downstream-generalization-20260716.md:24`.

**Gap.** The stop rule and the `closure-check` pivot exist only in an uncommitted planning document; they are
re-expressed above.

## Pooled cross-order ε correlation as a reportable gate/decision statistic

- **Verdict:** abandoned

**Question.** Pairwise ε (a 3-term inclusion-exclusion contrast) and third-order ε (a 7-term one) are both
"epistasis coefficients". Can they be concatenated into a single Spearman/Pearson correlation that serves as
*the* number — the signal gate's pass/fail statistic and the allocation benchmark's decision statistic?

**What was built.** `map_recovery` in `src/epibudget/validate.py:281` emits three `OrderMetric` rows —
`pairwise`, `third`, and `pooled` (the concatenation, `validate.py:301,307`). The pre-allocation gate script
computes the same three (`spearman_pooled_context_only`). Around this, the reporting and decision surfaces
were progressively rewritten to be order-stratified: the CLI table prints the third row under the literal
label `pooled (diagnostic only; cross-order)` (`src/epibudget/cli.py:150-151`); the post-hoc robustness
report emits per-order rows only and no pooled row at all (`src/epibudget/robustness.py:122`); the downstream
benchmark's primary statistic is the order-stratified mean `S_macro = ½(ρ_doubles + ρ_triples)` with pooled
demoted to a companion (`src/epibudget/downstream.py:484-492`).

**What was measured.** Two independent checks. (1) *A priori, on GB1 at the signal gate*: per-order Spearman
by model size — 35M 0.085/0.108, 150M 0.167/0.131, 650M 0.302/0.249 (pairwise/third) — against the pooled
value 0.114 / 0.120 / 0.316 (`SIGNAL_GATE.md:44-51`). (2) *A posteriori, on TrpB*: the same five allocation
methods scored per order and pooled at B ∈ {24, 48} (`experiments/trpb-smoke-20260713.md` §3), plus a
recomputation of the ε anchor from the source CSV (§6.1).

**Outcome.** Pooling was shown to be able to manufacture signal that exists in neither constituent order. On
TrpB at B = 48, method `info` scores pooled ρ = 0.2891 while its own sub-orders are pairwise ρ = 0.0434 and
third ρ = 0.1321 — a pooled rank correlation can exceed *both* of its parts only through between-group
separation, i.e. the two orders occupying different regions of both axes
(`experiments/trpb-smoke-20260713.md:76,95,106,115-117`). The separation there was an artefact: the
historical TrpB path took ΔG = ln f(v) without normalising by f(reference) = 0.408074, and the ε operators
exclude the empty set, so every pairwise ε carried a constant offset of +0.896307 and every third-order ε
−0.896307 (§6.1). The arithmetic was confirmed independently: ln(0.408074) = −0.8963067; the
inclusion-exclusion coefficients sum to −1 at order 2 and +1 at order 3, giving exactly the ∓ln f(ref)
offsets reported. The between-order share of pooled ε variance follows: η² = 0.257 as-run vs 0.085 after
re-anchoring, the latter landing on GB1's 0.079 (§6.1). A constant *within-order* shift leaves per-order
ranks invariant, which is why the per-order numbers survive the defect and the pooled one does not. The
note's own verdict is that pooled recovery is "invalid" for this run (§7).

**Why this verdict.** The mechanism was called before any of these numbers existed: pooling a 3-term and a
7-term ε into one correlation distorts the estimate and overstates significance because the two populations
share sub-ΔG terms, so instances are not independent. That reasoning is in the gate definition from its first
commit (`74833cb`) and still stands verbatim in `SIGNAL_GATE.md:11-13`. The frozen allocation decision rule
was correspondingly narrowed (`1617c9d`) from an unqualified
`map_recovery(info) − map_recovery(fitness)` to a named statistic — pairwise Spearman **and** pairwise
Pearson — with the explicit note that "pooling orders can be distorted by between-order separation"
(`VALIDATION.md` "Decision rule (frozen)"). The TrpB run was therefore a confirmation of a predicted failure
mode, not its discovery, and it closed the question: the demonstration that a pooled value can sit above both
its parts leaves no reading under which pooled is safe to decide on.

The abandonment is of the *decision and gate role*, not of the computation. The pooled row is still emitted
for backward compatibility and is everywhere labelled non-decisional (`VALIDATION.md:74,80`;
`README.md:96`), and the constraint is enforced by tests rather than by prose: `tests/test_robustness.py:248-259`
asserts no `pooled` order appears in the serialized robustness report, and `tests/test_cli.py:337` asserts
the CLI prints the "pooled (diagnostic only" label. Both pass. Zero pooled claims exist in the citable-claim
registry `artifacts/claim_map.json`, at HEAD and at `943833d^`.

One detail did not survive verification: `map_recovery` was **not** originally a single pooled correlation at
the code level. Its first commit (`478b7ac`) already returned all three rows. What was pooled-shaped was the
*documentation* — `VALIDATION.md` described the metric as "correlation between inferred and true ε over all
pairwise + third-order terms" and listed pooled as a co-equal reported number, wording replaced at `943833d`.

**Evidence.**
- `SIGNAL_GATE.md:11-13` (gate #2, judged per order, not on the pooled number), `:44-51` (per-order table +
  "Pooled Spearman, reported for context only: 0.114 / 0.120 / 0.316").
- `experiments/trpb-smoke-20260713.md:76` (pairwise 0.0434), `:95` (third 0.1321), `:100` ("Pooled (17,709
  terms) — invalid, see §6.1"), `:106` (pooled 0.2891), `:115-117` (the arithmetic tell), `:186-200`
  (constant ±0.896307 offsets; η² 0.079 / 0.257 / 0.085), `:333-334` ("Pooled is invalid").
- `VALIDATION.md:74`, `:80`, `:112`, `:245`, `:289`.
- `src/epibudget/validate.py:281-310`; `src/epibudget/cli.py:150-151`; `src/epibudget/robustness.py:122`;
  `src/epibudget/downstream.py:484-492`; `specs/robustness.md:75`; `specs/downstream.md:429`.
- Commits `74833cb` (gate defined per-order a priori; the spike script's field is literally
  `spearman_pooled_context_only`), `1617c9d` (decision rule narrowed to pairwise), `478b7ac` (harness already
  order-stratified), `00ee7ad` (robustness "per order (never pooled)"), `943833d` (prose demotion + the TrpB
  note).
- Tests `tests/test_robustness.py:248-259`, `tests/test_cli.py:337` — both verified passing.

## Pre-amendment confirmatory downstream campaign, stopped in flight by self-audit

- **Verdict:** abandoned

**Question.** Does a structure-aware budget produce a training set that supports better held-out ranking than
a fitness-greedy budget? And, before reading any confirmatory number from the run that would answer it: does
the implementation actually match the frozen protocol?

**What was built.** A downstream-impact benchmark (`src/epibudget/downstream.py`) with its spec
(`specs/downstream.md`), and a confirmatory-scale execution — `R=20` partitions × `K=5` outer folds × 20
random seeds — started under that original implementation. Neither the pre-amendment module nor the original
spec text is in the committed history: `specs/downstream.md` and `downstream.py` both first appear at
`ec9b5f7`, already carrying `Status: amended` and `AMENDMENT_VERSION = "protocol-amendment-1"`.

**What was measured.** Two things. (a) The intended science: the `structural − fitness` contrast on the
`S_macro` learning-curve AUC over `B ∈ {48, 96, 192}`. (b) The actual measurement that decided the campaign's
fate: a pre-read review of the implementation against the frozen spec, conducted while the run was still in
flight.

**Outcome.** The in-flight run produced a *favourable* exploratory smoke direction on `structural − fitness`,
and was then stopped without writing any artifact. The review had surfaced six deviations, several of them
individually sufficient to make a positive headline unfalsifiable:

1. the sign-consistency gate silently lowered its 16/20 threshold to the count of *surviving* partitions
   instead of requiring all 20;
2. no raw per-fold record trail — random-seed metrics were averaged before serialization, so the corrected-CV
   statistics and sign counts could not be independently recomputed;
3. the inner-fold count (3) and the alpha grids existed in code but were never named in the frozen spec text;
4. the main-only and no-triples-transfer regimes reused the full model's alpha instead of running their own
   inner CV;
5. the three mandatory ESM diagnostics were absent from the implementation;
6. the report's content depended on the input order of `scored`.

Amendment 1 was then frozen before any confirmatory number was read, with an explicit disposition clause
recording that the favourable direction informed no frozen value. `MIN_STRUCTURAL_EFFECT_SIZE` was
deliberately fixed at `0.0` — the weakest non-trivial bar — on the stated ground that choosing a magnitude
threshold *after* seeing a favourable direction is precisely the after-the-fact threshold-picking the
amendment exists to prevent.

The claim that no artifact survived is consistent with the repository: `report/` contains no `20260712*`
directory, and the only two `downstream.json` files on disk (`report/20260715T111312Z/`, GB1;
`report/20260716T154715Z/`, TrpB) both stamp `amendment_version = "protocol-amendment-1"`, i.e. both are
post-amendment reruns.

The fixes are verifiable. `robustness_gate` now fails closed: `decision_eligible=False`, `supported=None`,
`status="insufficient_valid_partitions"` whenever coverage is incomplete, never a reduced denominator. The
GB1 confirmatory rerun serializes 4,800 deterministic and 24,000 random raw fold records — the trail that
deviation (2) lacked — and its gates read `structural − fitness`: 20/20 partition means positive, global mean
delta 0.3423, median 0.3427, `supported=true`; `info − structural` at `B=192`: 15/20 positive,
`sign_pass=false`, `supported=false` despite a positive global mean (0.0074). That ESM gate failing at
exactly one partition below the pre-frozen 16/20 threshold is the clearest demonstration of why the
self-lowering denominator had to go.

**Why this verdict.** The campaign itself is abandoned, not narrowed: its numbers were destroyed rather than
banked, no artifact was written, and the disposition clause designates every pre-amendment execution
non-decision-use. The next run is labelled a confirmatory *rerun*, not an untouched first test, and stamps
`protocol_version`/`amendment_version` in every artifact so the distinction cannot be lost downstream. What
survived is the amendment, not the campaign: raw-record schema, fail-closed missing-partition policy,
regime-separated hyperparameter tuning, the three ESM diagnostics, cache/provenance hardening,
canonical-order enforcement, and the demotion of the Nadeau-Bengio corrected-CV interval from primary gate to
labelled sensitivity companion (its train/test ratio is not naturally identified in a selection-then-training
design, where the "train" pool is a budget-limited *selected* subset rather than an independent sample from
the test population).

One limit on corroboration, stated plainly: because the pre-amendment code and the original spec text were
never committed, the six deviations and the favourable smoke direction are attested only by the project's own
committed narrative — they cannot be reconstructed from a diff. What *is* independently verifiable is that
each named defect has a corresponding fix present in the code, tests, and artifacts.

**Evidence.**
- `specs/downstream.md` lines 3-13 (Status: amended, the seven-item deviation list), 18-25 (disposition of
  prior executions), 69-80 (degenerate-metric / fail-closed partition policy), 82-106 (7-point robustness
  gate and the `MIN_STRUCTURAL_EFFECT_SIZE = 0.0` rationale), 112-131 (corrected-CV demoted to companion).
- `VALIDATION.md` lines 304-330 ("Protocol amendment 1 — downstream-impact benchmark").
- `src/epibudget/downstream.py` lines 54-67 (`AMENDMENT_VERSION`, `N_INNER_FOLDS = 3`, `GRID_MAIN`,
  `GRID_PAIR`, `EXPECTED_PARTITIONS = 20`, `SIGN_THRESHOLD = 16`, `MIN_STRUCTURAL_EFFECT_SIZE = 0.0`); lines
  1698-1755 (`robustness_gate`, fail-closed); `_esm_circular` / `_esm_offset_supervised` /
  `select_alpha_esm_offset` around lines 399, 1267, 1313.
- `tests/test_downstream.py:678` `test_robustness_gate_4_positive_16_missing_is_not_eligible`; `:736`
  `test_robustness_gate_exact_zero_partition_mean_is_not_positive`; `:743`
  `test_robustness_gate_never_reduces_denominator_below_expected_partitions`; `:921`
  `test_report_is_byte_identical_under_an_arbitrary_permutation_of_scored`; `:2106`
  `test_no_incomplete_r20_variant_ever_becomes_decision_eligible`. All pass → 8 passed.
- Commits `ec9b5f7` (spec + module + tests, already post-amendment), `381d1be` (downstream CLI command),
  `79137f0` (`--n-perturbations` pinned in the confirmatory profile).
- Artifacts: `report/20260715T111312Z/downstream.json` (GB1, `amendment_version` stamped,
  `cli_protocol_profile_conforming=true`, 4,800 + 24,000 raw records, `structural_gate.supported=true`,
  `esm_gate.supported=false`); `report/20260716T154715Z/downstream.json` (TrpB,
  `cli_protocol_profile_conforming=false`). No `report/20260712*` directory exists.

---

# Benchmark

## Downstream-impact benchmark on GB1 — does a structure-aware budget yield a better training set for ranking held-out multi-mutants?

- **Verdict:** kept

**Question.** The project's original headline statistic — how much of an epistasis map a budget recovers — is
partly tautological: a method scores well partly by *measuring* more of the map, and the inference step keeps
the ESM prior for every unmeasured term, so a high score can restate the prior rather than test it. The
non-tautological reframing: at equal initial budget *B*, does a method's selected plate constitute a better
**training set** for a fixed supervised learner asked to rank **held-out** double and triple mutants? This
grades the map's usefulness for a downstream decision instead of its coverage.

**What was built.** `src/epibudget/downstream.py` (2,847 lines) plus an `epibudget downstream` CLI command
(`src/epibudget/cli.py:761`). Deterministic order-stratified SHA-256 outer folds over the order-2/3 universe;
singles are never held out (`n_candidates` 29,678 vs `n_eval_universe` 29,602 — the 76-variant difference is
exactly the 4 sites × 19 non-WT singles). For each fold every method selects *B* from `universe \ held-out`,
zero-shot, and is scored on the identical measured members of the held-out set. The primary learner is a
pure-numpy generalized-dual ridge (`fit_ridge`, `downstream.py:223`) on one global fixed dictionary shared by
all methods, so methods differ only in training data and never in model capacity: `FeatureSpace`
(`downstream.py:141`) builds 76 reference-coded main effects + 2,166 reference-coded pairwise indicators =
2,242 columns. No third-order columns, no ESM feature. α is chosen by inner CV on the outer training set
only. Two estimands (target-blind primary, target-aware companion) and two missingness regimes run side by
side. A frozen `CONFIRMATORY_PROFILE` (`downstream.py:846`) pins protocol version, partitions, folds,
budgets, alphabet, max order, `n_perturbations`, seeds, estimands, regimes and methods, and is checked at the
CLI boundary and again inside the decision summary. Every summary is a pure function of immutable raw
per-fold records serialized first.

**What was measured.** `S_macro = ½(ρ_doubles + ρ_triples)` of predicted vs raw held-out fitness, with
NDCG@B, hit-rate@B, regret@B, an epistasis-uplift and a no-triples→held-out-triples transfer test as
secondaries. Structural claim gated on the learning-curve AUC contrast `structural − fitness` over
B ∈ {48, 96, 192}; ESM-prior claim gated on `info − structural` at B = 192 only. Gate is a 7-point
partition-level robustness rule (`robustness_gate`, `downstream.py:1698`): complete 20/20 partition
coverage, ≥16/20 partition means strictly positive, positive global mean, positive median, effect size above
threshold — failing closed to `insufficient_valid_partitions` rather than shrinking the denominator. Run:
R=20 salted partitions × K=5 outer folds × 20 seeds, ESM-2 650M, `n_perturbations = 16`, target-blind /
attempted-budget.

**Outcome.** `structural_downstream_supported = true`, `decision_eligible = true`,
`protocol_profile_mismatches = []`, raw-record coverage exact (4,800 deterministic + 24,000 random cells;
zero missing, duplicate or unexpected). Recomputed from the artifact's partition aggregates:

- `structural − fitness` (S_macro-AUC): 20/20 partitions positive, mean **+0.3423**, median +0.3427.
- `structural − random` (S_macro-AUC): 20/20 positive, mean **+0.1749**.
- `info − structural` (S_macro at B=192): **15/20 — below the 16/20 sign gate**, mean **+0.0074** →
  `esm_uncertainty_supported = false`. The ESM masking-dispersion prior adds nothing over pure loop-bracing.
- `practice − structural`: 0/20 positive, mean −0.3922.

Per-method S_macro at B = 48 / 96 / 192: info 0.476 / 0.551 / 0.594; structural 0.423 / 0.572 / 0.587; random
0.260 / 0.359 / 0.474; fitness 0.123 / 0.194 / 0.272; practice 0.058 / 0.141 / 0.244. Both fitness-greedy and
the MULTI-evolve-style "practice" heuristic are **worse than random** as training sets — the strongest
negative result in the run.

**Why this verdict.** This is the one estimand that escapes the coverage tautology, and the gate it was
pre-committed to passes cleanly on the artifact. The design closes the second, less visible tautology too:
the primary predictor consumes no ESM-derived feature and never calls the prior-inclusive inference path,
enforced by source-inspection guards over `design_matrix`, `active_columns`, `fit_ridge`, `select_alpha`,
`select_alpha_main_only` and `_build_fold_context` (`tests/test_downstream.py:951`, `:1055`); the full
114-test module passes. It also independently reproduces the masking-variance null found elsewhere in the
project.

Scope is genuinely narrow and four caveats are load-bearing:

1. **Not a blind pre-registration, and the docs say so.** `VALIDATION.md:260-266` states the protocol was
   frozen *after* structural-only's apparent advantage was already known and committed, so it "cannot claim
   that bias protection"; it only blocks tuning to a computed number. A confirmatory-scale pre-amendment run
   produced a favourable direction and was stopped without writing any artifact — disclosed at
   `VALIDATION.md:317-321`.
2. **The effect-size arm of the gate is vacuous.** `MIN_STRUCTURAL_EFFECT_SIZE = 0.0`
   (`downstream.py:64`), so "effect size passes" only means the mean is positive — the code comments this as
   the honest choice since no non-zero threshold was defensible pre-result.
3. **The corrected-CV companion is convention-dependent and one convention crosses zero.** For
   `structural − fitness`, the `pool_ratio` convention gives 95% [0.314, 0.371] but the conservative
   `effective_label_ratio` convention gives [−0.065, 0.750]. The spec pre-emptively demoted this object to a
   labelled sensitivity companion (never the gate) in protocol amendment 1, which predates the run, so the
   demotion is not post-hoc. The zero-crossing under the conservative convention is not stated in any
   committed prose.
4. **Provenance of the winning run is not fully pinned.** The artifact records `code_state = "dirty"` against
   `execution_commit fc6436a`, with 25 changed scientific files including `src/epibudget/downstream.py`
   itself. A `code_diff_sha256` is recorded but the diff is not committed, so no single SHA reproduces the
   exact executed code.

Two controls are reported: a permutation null (shuffling fitness across variants collapses
`structural − random` from +0.175 to +0.0046 on GB1, ~40× smaller, leaving ~3% as a coverage/diversity
floor) and a bit-identical determinism re-run. The null is a genuine false-positive test and materially
strengthens the result — but note it has **no committed implementation**: a repository-wide search finds no
label-shuffle flag, script or retained run artifact, and both controls ran at 10 partitions rather than 20
due to a RAM ceiling. The number is therefore documented, not independently reproducible.

The artifact remains `status = provisional` under the git-ignored `report/` and is not registered in
`artifacts/`. Claimable scope stays retrospective, single-landscape (GB1; TrpB corroborates at
`n_perturbations = 0` but is non-decision-eligible), single-assay, single primary learner, one-step, with no
sequential selector update.

**Evidence.**
- Artifact: `report/20260715T111312Z/downstream.json` (git-ignored) — `decision` block: `structural_gate`
  sign_positive 20/20, `global_mean_delta` 0.3422903257144479, statistic `s_macro_auc`, `supported: true`;
  `esm_gate` sign_positive 15/20, `global_mean_delta` 0.007390820815059291, statistic `s_macro_at_192`,
  `supported: false`; `raw_record_coverage` 4800/4800 + 24000/24000, all anomaly counts 0.
  `provenance.scored_cache_identity_observed` → `facebook/esm2_t33_650M_UR50D`, `n_perturbations: 16`,
  `candidate_count: 29678`.
- Code: `src/epibudget/downstream.py` — module docstring (leakage argument), `PROTOCOL_VERSION`
  `"epibudget-downstream-v1"` (:56), `MIN_STRUCTURAL_EFFECT_SIZE = 0.0` (:64), `FeatureSpace` (:141),
  `fit_ridge` (:223), `CONFIRMATORY_PROFILE` (:846), `robustness_gate` (:1698);
  `src/epibudget/validate.py:362` `structural_graph` (τ² ≡ 1 ablation = pure loop-bracing);
  `src/epibudget/cli.py:761`.
- Tests: `tests/test_downstream.py:951` and `:1055` (leakage/circularity guards), 114 tests, all pass.
- Docs: `specs/downstream.md` (frozen protocol; §"Corrected-CV interval (demoted to a labelled
  sensitivity-only companion)" at :112, §"Protocol amendment 1 addendum" at :134); `VALIDATION.md:258-330`;
  `LIMITATIONS.md:185-207`; `README.md:102-118`;
  `experiments/trpb-downstream-generalization-20260716.md`.
- Commits: `ec9b5f7` (add downstream-impact benchmark), `381d1be` (expose downstream CLI), `e00b2c3` (audit
  hardening + offline tests), `8bff940` (dataset-generic via `--dataset`), `79137f0` (`--n-perturbations`
  pinned in the confirmatory profile), `56f7f99` (record the downstream result), `1dda421` (permutation-null
  + determinism controls).

## Reordering the research programme: downstream impact before second-landscape generalization

- **Verdict:** kept

**Question.** The headline comparative claim was map recovery: at equal budget B, which selection method best
reconstructs the pairwise/third-order epistasis map. That statistic was recognised as partly tautological — a
method scores high largely because it *measured* the terms it is graded on (breadth), not because it
predicted the unmeasured ones (precision) — and the uncertainty prior looked null. Two candidate next
investments existed: (A) reproduce the recovery comparison on a second protein, or (B) change the estimand on
GB1 to a downstream-impact test. Which should run first? The argument for B-before-A: generalizing a coverage
artefact reproduces the artefact, so the mechanism on GB1 has to be settled before it is worth porting
anywhere.

**What was built.** Not a module — a sequencing decision, plus the machinery it prioritised:

- `specs/downstream.md` (591 lines) and `src/epibudget/downstream.py` (2,847 lines), with
  `tests/test_downstream.py` (2,664 lines), added in `ec9b5f7`; `epibudget downstream` exposed in
  `src/epibudget/cli.py` (`381d1be`).
- The protocol frozen in a committed document *before* any downstream number existed: `VALIDATION.md:258`
  "Post-registration downstream-impact protocol", committed `24fdd77`, with all three narrative outcomes
  pre-enumerated at `VALIDATION.md:300-302` — structural beats fitness, info does not beat structural,
  nothing beats fitness. Tightened by "Protocol amendment 1" (`VALIDATION.md:304`).
- `8bff940` then routed the benchmark's landscape/sites/WT/alphabet through `data.resolve_dataset`
  (`--dataset`), so the *second landscape entered through the downstream benchmark* rather than through the
  recovery comparison. Its own message states the purpose: enabling the confirmatory downstream benchmark on
  TrpB as a generalization check.

**What was measured.** The ordering itself is a decision, not a measurement. What is checkable is its
consequence: which comparative claim ended up decision-eligible, and what the one out-of-order
second-landscape attempt returned. The downstream estimand is: at equal initial budget B, is a method's
selected plate a better training set for a fixed pairwise-ridge learner ranking *held-out* double/triple
mutants — `S_macro = ½(ρ_doubles + ρ_triples)`, over R=20 salted partitions × K=5 folds, gated on a ≥16/20
partition sign count.

**Outcome.** The ordering held, and both halves of its rationale were borne out.

- GB1 downstream ran first and is the only decision-eligible comparative result:
  `report/20260715T111312Z/downstream.json` gives `structural_downstream_supported = true`,
  `structural − fitness` S_macro-AUC `global_mean_delta = 0.3423` with `sign_positive = 20`/20;
  `esm_uncertainty_supported = false` (`info − structural` at B=192: `sign_positive = 15` < threshold 16,
  mean +0.0074). Structure alone won; the ESM masking-dispersion prior added nothing.
- The second landscape followed, as a downstream replication only:
  `report/20260716T154715Z/downstream.json` (TrpB, Johnston 2024), `structural − fitness` +0.2864, 20/20
  partitions, but `status = nonconforming_protocol_profile`, `decision_eligible = false` — observed
  `n_perturbations = 0` against the frozen 16. Direction replicates; nothing is established by it.
- The pre-registered second-landscape *map-recovery* run is still deferred: `VALIDATION.md:332` "Second
  landscape — TrpB (pre-registered, run DEFERRED)".
- The one early, out-of-order attempt at second-landscape recovery is documented as a technical success and a
  scientific non-result: `experiments/trpb-smoke-20260713.md` — "exploratory · non-confirmatory · not
  decision-eligible · recovery invalidated", off-protocol grid (B ∈ {24,48}, 5 seeds against the frozen
  {48,96,192}/≥20) and a broken WT anchor that makes its recovery coefficients uninterpretable.

**Why this verdict.** Kept, on evidence from both directions. The one second-landscape recovery run that did
happen early returned nothing usable — precisely the "generalizing a coverage artefact is wasted effort"
prediction. Meanwhile the recovery-comparison claims were retired wholesale: `943833d` removed 24 retracted
comparative claims from `artifacts/claim_map.json` (−216 lines) and narrowed the README thesis. Had
generalization run first, it would have been generalizing claims that were about to be withdrawn. The
downstream benchmark, prioritised instead, is what survived — it is the only comparative result the
repository currently stands behind (`README.md:102-117`), and it is the one that answers the "you just
measured the scaffold" objection, because the learner reads only revealed fitness labels and never the
held-out variant's ESM score or the prior-inclusive `infer_epistasis` output.

The reordering was also self-protecting in a second way: the fallback framing to adopt if the uncertainty
prior came back null — structure-aware allocation wins at equal budget while ESM zero-shot uncertainty adds
nothing on top of loop structure — was written down before the results, so adopting it was not post-hoc
rationalisation.

**Evidence.**
- Ordering as executed: `git log --format="%h %ad %s" --date=iso` — `ec9b5f7` downstream benchmark;
  `24fdd77` protocol frozen; TrpB recovery smoke run `20260713T135240Z`; `943833d` thesis narrowed;
  `8bff940` `--dataset`; `79137f0` `--n-perturbations`; `56f7f99` result recorded; `1dda421` controls.
- `VALIDATION.md` lines 252-262 (factual correction: downstream ran, TrpB recovery still blocked), 258-302
  (frozen protocol), 332-359 (deferred second-landscape recovery), 361+ (the invalidated smoke).
- `LIMITATIONS.md` §4 lines 97-104 — the tautology argument that triggered the reorder; §5 lines 122-135 —
  the withdrawn structural-only comparison.
- `experiments/trpb-downstream-generalization-20260716.md` — full result, per-method S_macro tables,
  permutation-null and determinism controls; `experiments/trpb-smoke-20260713.md` lines 1-20 — the
  invalidated out-of-order recovery attempt.
- Artifacts read directly (git-ignored `report/`, so not reader-followable):
  `report/20260715T111312Z/downstream.json` `decision.structural_gate` / `decision.esm_gate`;
  `report/20260716T154715Z/downstream.json` `decision.structural_gate`.
- `src/epibudget/downstream.py`, `tests/test_downstream.py`, `specs/downstream.md`,
  `src/epibudget/cli.py:761`.

**Gap.** The pre-commitment to that fallback framing currently lives only in a working document.

## Structural-only ablation (τ² ≡ const, rank by loops braced n(v)) — the pre-registered prior-free control

- **Verdict:** narrowed

**Question.** What does the ESM masking-dispersion prior actually contribute to the *allocation*, as opposed
to the structural loop-coverage sort it multiplies? The kill rule was frozen in writing before any 650M
number existed: "If info-optimal ≈ structural-only, the ESM uncertainty prior does nothing to the allocation
and must be dropped from the claims; if info-optimal > structural-only, that gap is the contribution"
(`VALIDATION.md:129-134`, committed in `c934bb2`).

**What was built.** `structural_graph` in `src/epibudget/validate.py:362-371` — the same modular greedy sort
with τ² ≡ 1.0, so `info_gain(∅, v) = n(v)`, the number of interaction loops containing `v`. Made a mandatory
baseline in every figure and table alongside info / fitness / random / practice (`VALIDATION.md:126-137`).
Implemented in `6900206`. Unit-pinned by `tests/test_validate.py:543-549`. The same unit-weight graph is
reused as the control in `src/epibudget/gate2.py:296` and as the *primary contrast* of the downstream
benchmark (`src/epibudget/downstream.py:812`, `("structural", "fitness", "auc")  # primary contrast`).

**What was measured.** Stage 1: per-order and pooled Spearman/Pearson map recovery at B ∈ {48, 96, 192},
650M, full 20-letter alphabet, 20 seeds, bootstrap CIs; plus a post-hoc paired precision comparison on the
terms both methods inform but neither fully pins. Stage 2, after the defect was found: (a) predicting
`n_informed` from the tie-break rule alone with zero ESM input; (b) replaying the frozen scored cache under
alternative, equally valid tie-breaks of the same exact score tie; (c) 100 seeded within-stratum permutations
of each exact score stratum in `gate2` (`_DEFAULT_STRUCTURAL_SEEDS = 100`, `src/epibudget/gate2.py:51`).

**Outcome — stage 1 (the prior-free control won).** From `artifacts/headline_650m.json`, pairwise Spearman /
Pearson at B = 48 / 96 / 192:

| method | Spearman | Pearson |
|---|---|---|
| structural | 0.4845 / 0.4602 / 0.5042 | 0.5143 / 0.5262 / 0.5728 |
| info | 0.4082 / 0.4183 / 0.4431 | 0.4579 / 0.4789 / 0.5042 |
| random | 0.2791 / 0.2795 / 0.2875 | 0.3113 / 0.3108 / 0.3170 |

On matched terms (`artifacts/robustness_650m.json`, `common_precision`) at B = 48, pairwise,
n_common = 1511: structural 0.5368 vs info 0.4518, descriptive Δ = −0.0850, 95% CI [−0.1251, −0.0465]; the
pairwise Δ excludes zero at all three budgets (−0.085 / −0.033 / −0.046). The earlier 35M reduced-alphabet
smoke (`artifacts/smoke_recovery_35m.json`, alphabet `ADEF`, 307 candidates) pointed the same way at B = 96:
structural 0.9695 vs info 0.7604. Per the pre-registered rule the masking-dispersion prior was dropped from
the claims (`VALIDATION.md:217-223`).

**Outcome — stage 2 (the control itself was found defective).** In a four-site, 20-letter universe `n(v)` is
constant *within* each mutation order — {1140} for 76 singles, {39} for 2,166 doubles, {1} for 27,436 triples
— so with τ² ≡ 1 the greedy weight takes exactly three distinct values and carries zero within-order
information. `allocate` uses Python's stable `sorted`, so the within-order ordering is an exact tie resolved
by `enumerate_candidates`' site-major, residue-alphabetical input order. At any B ≤ 76, structural-only is
literally "take the first B singles in enumeration order" (`experiments/trpb-smoke-20260713.md:214-243`).
Confirmed by predicting `n_informed` from the tie-break with no ESM input, giving exact five-digit matches
against the frozen artifacts: GB1 B=48 17,700; B=96/192 17,782; `n_pinned` 20/116; TrpB 14,301 / 17,582.
Replaying the frozen GB1 cache under other tie-breaks of the same tie (pooled Spearman, B=48): as-run 0.2470,
reversed enumeration order 0.0355, balanced 12-per-site 0.1736, 20 random draws mean 0.1449 (sd 0.045, min
0.0567, max 0.2168), info-optimal 0.1997. Under the reversed order info-optimal wins, and the as-run value
sits in the extreme upper tail of the random-draw distribution. At B = 96/192, where all 76 singles are
forced and only doubles are tied, structural still beats info under 20/20 tie-breaks
(`experiments/trpb-smoke-20260713.md:245-264`).

The 100-seed corrective replay reaches a registered verdict of **inconclusive** on the τ² contribution
(`VALIDATION.md:229-245`, decision `inconclusive_zero_gpu`). Its per-budget shape sharpens the narrowing:
once the tie is randomised, at B = 48 info-optimal beats the *median* seeded structural draw (roughly 70-90%
of the 100 draws, depending on statistic and calibration regime), i.e. the frozen headline's direction does
not survive at that budget; at B = 96 and B = 192 structural still beats info in 100/100 draws under the
operational slope. Neither direction is promoted to a public claim.

**Why this verdict.** Narrowed rather than abandoned, on three grounds the repository states explicitly.
(1) The comparative reading "structural wins at every budget" is withdrawn, because at B = 48 the result is
an unreplicated, lucky tie-break draw rather than a property of a method — a control whose selection is "the
first B candidates in enumeration order" cannot carry a comparative claim (`LIMITATIONS.md:130-134`,
`:162-167`). (2) The *conclusion the ablation was built to deliver* — that the masking-dispersion prior earns
no credit — survives at B = 96/192 and is corroborated independently downstream, so the ablation did its job;
what failed is the strength of the claim, not its direction. (3) The object was retained, not deleted:
`structural_graph` is still the diagnostic legacy prefix in gate 2 and still the primary contrast of the
downstream benchmark, where `structural − fitness` passes its robustness gate on GB1 (20/20 partitions
positive, S_macro-AUC mean +0.342, `structural_downstream_supported = true`) while `info − structural` fails
(15/20, below the 16/20 sign gate) (`LIMITATIONS.md:186-196`,
`experiments/trpb-downstream-generalization-20260716.md:21-25`). The withdrawal was carried through to the
machine-readable registry, not just to prose: `943833d` deleted 216 lines of `artifacts/claim_map.json`
covering 24 retracted comparative claims, and the current `artifacts/claim_map.json` contains zero
occurrences of "structural".

**Evidence.**
- Pre-registration and kill rule: `VALIDATION.md:126-137`; commit `c934bb2`.
- Implementation: `src/epibudget/validate.py:362-371`; commit `6900206`; test
  `tests/test_validate.py:543-549`.
- Stage-1 numbers: `artifacts/headline_650m.json`, `artifacts/robustness_650m.json` (`common_precision`,
  n_common 1511, Δ −0.0850, CI [−0.1251, −0.0465]), `artifacts/smoke_recovery_35m.json`; narrative at
  `VALIDATION.md:214-227`.
- Defect analysis: `experiments/trpb-smoke-20260713.md:214-268`; `LIMITATIONS.md:162-170`;
  `VALIDATION.md:381-384`.
- Corrective replay: `src/epibudget/gate2.py:51`, `:296`, `:436-514`; `VALIDATION.md:229-245`.
- Withdrawal: commit `943833d`; `artifacts/claim_map.json` (no `structural` claims remain);
  `LIMITATIONS.md:130-134`.
- Retained role: `src/epibudget/downstream.py:812`;
  `experiments/trpb-downstream-generalization-20260716.md:21-25`; `LIMITATIONS.md:186-196`.

**Gap.** The per-budget seeded breakdown above is not in a committed document; the committed record states
only the overall "inconclusive".

## TrpB downstream generalization at `n_perturbations = 0` (exploratory replication)

- **Verdict:** narrowed

**Question.** Is the downstream-impact finding — that a structure-aware plate is a better *training set* for
ranking held-out double/triple mutants than a random or fitness-greedy plate — specific to GB1, or does it
hold on a biochemically independent combinatorial landscape? TrpB (Johnston 2024) is enzyme catalysis against
GB1's IgG-Fc binding, so a shared direction is not explainable by assay chemistry.

**What was built.** Two enabling changes to the benchmark CLI, then a run.
- `8bff940` routed the downstream command's landscape/sites/WT/alphabet through `data.resolve_dataset` behind
  `--dataset` (`src/epibudget/cli.py`), so the benchmark is no longer GB1-hardcoded; default `gb1_wu2016`
  reproduces prior behaviour exactly, and an unknown `--dataset` is rejected before any load.
- `79137f0` added `--n-perturbations` and, critically, *pinned* it into `CONFIRMATORY_PROFILE`
  (`src/epibudget/downstream.py:835-850`) and threaded it through `protocol_profile_conformance`
  (`:876-914`). A cache scored with n ≠ 16 is therefore structurally incapable of being reported as
  decision-eligible.
- The ESM-2 650M score cache was produced off-machine on a T4 at `n_perturbations = 0`
  (`notebooks/colab/trpb_650m_n0.ipynb`, committed as run provenance in `078bedd`), which is what made a
  zero-GPU-budget second landscape affordable: `structural` vs `random` needs only `delta_g`, not the
  masking-variance passes.

**What was measured.** Same protocol scale as the GB1 confirmatory run — R=20 partitions × K=5 outer folds ×
20 seeds, budgets {48, 96, 192}, max_order 3, target-blind / attempted-budget — over a 29,678 variant
candidate universe (29,602 evaluated). Primary statistic `S_macro = ½(ρ_doubles + ρ_triples)`, contrasts
aggregated as learning-curve AUC per partition with a sign gate at 16/20. Secondary "find winners" metrics
(`hit_rate@B`, `ndcg@B`, `regret`) at B=192. The only deviation from the frozen confirmatory recipe is
`n_perturbations = 0`.

**Outcome.** The direction replicates; every number was confirmed directly from the run artifact:

| contrast (S_macro-AUC) | partitions positive | mean |
| --- | --- | --- |
| structural − random | 20/20 | +0.1349 |
| structural − fitness | 20/20 | +0.2864 |
| practice − structural | 0/20 | −0.2449 |

Per-method S_macro at B = 48/96/192: structural 0.337/0.426/0.443; random 0.197/0.271/0.354; fitness
0.081/0.128/0.149; practice 0.098/0.144/0.266. Fitness-greedy is again *worse than random* as a training set,
as on GB1. Secondary at B=192: `hit_rate` structural 0.439 > practice 0.280 > random 0.260 > fitness 0.107;
`ndcg` structural 0.892 > random 0.816 > practice 0.757 > fitness 0.616; `regret` (lower better) practice
0.002 < random 0.051 < structural 0.059 — practice edges structural only for finding the single
highest-fitness variant, not for covering or ranking the landscape.

The run is pinned `status = nonconforming_protocol_profile`, `decision_eligible = false`, `supported = null`,
with `protocol_profile_mismatches = ["n_perturbations"]` as the sole mismatch.

**The `info` arm self-invalidates, and the pin is what caught it.** At `n_perturbations = 0` the masking
dispersion `var_delta_g` is exactly 0 for all 29,678 cache records (verified by scanning the cache). The
factor graph weights each candidate `weight[v] = τ²_v · n(v)` (`src/epibudget/graph.py:66`), so with τ² ≡ 0
every candidate's `info_gain` is identically 0; the `info` selector is a stable sort on that constant key
(`src/epibudget/acquisition.py:51-55`, `lambda_=0.0`), which degenerates to "take the first B in canonical
identity order". Reproduced on a synthetic pool: all gains 0.0, selection equals the input prefix.
`structural` is the same graph with a unit variance map (`downstream.py:2733-2734`), so it still ranks by
loop-bracing count `n(v)` and remains meaningful.

This matters because the degenerate arm *would have passed its gate*. The TrpB `esm_gate`
(`info − structural`) scores 17/20 positive against a 16/20 threshold, mean +0.0128 — nominally a pass. The
properly-scored GB1 run at n=16 gives 15/20, mean +0.0074 — a fail, `esm_uncertainty_supported = false`. So
reading the n=0 `info` arm at face value would have manufactured support for the ESM masking-variance prior
on exactly the statistic the conforming run rejects. The `info` S_macro (0.381/0.427/0.456) and the
`info − structural` contrast are declared artifacts and are not claimed.

**Why this verdict.** Narrowed, not kept: the ambition was a second-landscape *confirmation* that would
establish generality; what was obtained is a direction-replication that corroborates it. The artifact
declares itself non-decision-eligible and the gate outputs are `null` rather than `true`, so the run cannot
carry a publishable claim on its own — that needs the n=16 confirmatory rerun, deferred on GPU cost. The
engineering judgement worth recording is that the conformance pin was added *before* the run and made the
non-conforming profile non-decision-eligible by construction, which is precisely what prevented a
favourable-looking but meaningless `info` result from being harvested.

**Caveats recorded with the result.** Two landscapes only; retrospective; single primary learner (pairwise
ridge); one-step selection with no sequential update. ~871 of 160,000 TrpB fitness values (~0.5%) are imputed
rather than measured and unflagged in the source mirror. Both `downstream.json` artifacts are
`status = provisional`, live under the git-ignored `report/`, and are not registered public artifacts — a
reader can follow the committed note and re-run the recorded command, but cannot fetch the JSON from the
repository.

**Related controls.** A permutation null (seeded label shuffle) collapses TrpB's `structural − random` from
+0.135 to +0.0135 at 10 partitions, leaving ~10% as a coverage/diversity floor; re-running the same
configuration is bit-identical. Both controls ran at 10 partitions because the machine hit a RAM ceiling at
R=20.

**Evidence.**
- `experiments/trpb-downstream-generalization-20260716.md` — TrpB section lines 31-50, secondary metrics
  52-59, reading 61-68, frozen caveats 89-97, exact command 99-105.
- Run artifact `report/20260716T154715Z/downstream.json` (git-ignored): `decision.structural_gate` and
  `decision.esm_gate` both carry `decision_eligible=false`, `supported=null`,
  `status=nonconforming_protocol_profile`; `protocol_profile_mismatches=["n_perturbations"]`;
  `provenance.execution_commit = 79137f0…`, `code_state = clean`; raw-record coverage 4800 deterministic +
  24000 random cells, zero missing/duplicate.
- Score cache `report/scored_trpb_650m_n0.jsonl` + `.meta.json` (git-ignored): 29,678 records, all
  `var_delta_g == 0`; sidecar records `n_perturbations: 0`, `device: cuda`,
  `model_id: facebook/esm2_t33_650M_UR50D`.
- GB1 comparator `report/20260715T111312Z/downstream.json`: conforming, `structural−fitness` 20/20 +0.3423
  supported, `esm_gate` 15/20 +0.0074 not supported.
- Commits `8bff940`, `79137f0`, `56f7f99`, `1dda421`, `ebd2876` (wording tightened to "corroborates … rather
  than establishing it"), `078bedd` (Colab notebooks as provenance).
- Code: `src/epibudget/graph.py:62-88`; `src/epibudget/acquisition.py:44-73`;
  `src/epibudget/downstream.py:2579-2580, 2724-2735`; profile pin at `downstream.py:835-850, 876-914`.
- Test: `tests/test_downstream.py:1387` `test_protocol_profile_conformance_flags_wrong_n_perturbations`
  asserts `mismatches == ["n_perturbations"]` for an n=0 cache — passes offline.
- Supporting docs: `notebooks/README.md` (n_perturbations table, load-bearing difference);
  `specs/trpb-exploratory.md` (dataset provenance, missing/duplicate/invalid rules); `VALIDATION.md:347`
  (871/160,000 imputed).

## MULTI-evolve-style practice-heuristic baseline (top beneficial singles, then all their pairwise combinations)

- **Verdict:** narrowed

**Question.** Directed-evolution practice picks a handful of beneficial single mutants and tests all their
pairwise combinations. If the budgeted-design claim is only ever compared against random and fitness-greedy,
it may be beating a strawman. Does the real wet-lab design heuristic, run at the same budget, do better than
the proposed structure-aware allocation — on epistasis-map recovery, and on downstream training-set quality?

**What was built.** `practice_heuristic(candidates, budget)` in `src/epibudget/validate.py:323-359`: rank
singles by predicted ΔG, expand `k` top singles into all valid cross-site pairs (same-site pairs are not
variants and are skipped), grow `k` until at least `budget` distinct pairs exist, then rank those pairs by
their own predicted ΔG and keep the top `budget`. Only order-2 variants spend budget; it under-fills rather
than raising if the pool is too small. It is strictly zero-shot — it reads `delta_g` only, never a measured
label. Registered as a baseline in four places: the recovery harness (`validate.py:460`), the
downstream-impact benchmark (`src/epibudget/downstream.py:800`, `:2527`), the gate-2 path
(`src/epibudget/gate2.py:444`, `:473-478`), and the supplementary recovery script
(`scripts/headline_650m_supplementary.py:140`).

**What was measured.** Two independent protocols. (1) Map recovery: Spearman/Pearson of inferred vs
ground-truth ε terms at B ∈ {48, 96, 192}, per interaction order, GB1 four-site landscape, ESM-2 650M,
`n_perturbations = 16`, 20 seeds. (2) Downstream impact: `S_macro = ½(ρ_doubles + ρ_triples)` for a fixed
pairwise-ridge learner trained on each method's revealed plate and asked to rank held-out double/triple
mutants, R=20 × K=5 × 20 seeds, GB1 (confirmatory) and TrpB (exploratory, `n_perturbations = 0`).
`practice − structural` (AUC) is one of the four pre-registered contrast pairs
(`specs/downstream.md:459-461`). Secondary "find winners" metrics: `hit_rate@B`, `ndcg@B`, `regret`.

**Outcome.** Poor for epistasis structure, on both protocols.

- Map recovery, GB1 650M headline: pairwise Spearman **−0.271** at B=48 (Pearson −0.272), i.e. worse than
  uninformative; 0.282 at B=96, 0.305 at B=192. `hit_rate` 0.0 at B=48 and B=96, 0.0052 at B=192.
- Downstream S_macro (B = 48 / 96 / 192), GB1: practice **0.058 / 0.141 / 0.244** against random
  0.260 / 0.359 / 0.474 and structural 0.423 / 0.572 / 0.587. TrpB: practice **0.098 / 0.144 / 0.266**
  against random 0.197 / 0.271 / 0.354 and structural 0.337 / 0.426 / 0.443. On both landscapes it is a
  **worse training set than a random plate**. On TrpB, `practice − structural` is **0/20 partitions** —
  structural wins every one.
- Its one genuine win, reported rather than buried: on TrpB at B=192, `regret` (best available minus best
  found in top-B; lower is better) is **0.002** for practice against 0.059 for structural, 0.051 random,
  0.060 fitness. It is the best method at finding the single highest-fitness variant — which is the job it
  was actually designed for. It still loses on coverage and ranking at the same budget (`hit_rate@B` 0.280 vs
  structural 0.439; `ndcg@B` 0.757 vs 0.892).

**Why this verdict.** Narrowed, not killed: the baseline is retained everywhere it answers the "are we
beating a strawman?" question — the recovery harness, the downstream benchmark, gate 2 — but its scope was
deliberately cut in two ways. First, it was never admitted to the frozen decision rule, which stays on info
vs fitness vs random; practice and structural-only are declared reported companions that determine framing,
not the decision (`SPEC.md:242-244`, `VALIDATION.md:128,136-137`). Second, it was excluded from the post-hoc
robustness analyses, which recompute selections for info/fitness/structural/random only. That exclusion was
already true in code — `src/epibudget/robustness.py` imports `random_selection` and `structural_graph` but
not `practice_heuristic`, and `artifacts/robustness_650m.json` contains no `practice` record — while the spec
prose still listed it. The resolution was to correct the prose to match the code rather than extend the code
to match stale prose (commit `8517b96`). The negative-recovery result at B=48 is the substantive finding: an
all-pairs-of-top-singles plate concentrates budget on a few high-ΔG positions, which is close to optimal for
hill-climbing to one winner and close to worst-case for estimating an interaction map.

**Evidence.**
- Implementation: `src/epibudget/validate.py:323-359` (selector), `:454-474` (registration in
  `run_validation`).
- Downstream/gate-2 registration: `src/epibudget/downstream.py:800-801`, `:815`, `:2527`;
  `src/epibudget/gate2.py:39`, `:444`, `:473-478`; `scripts/headline_650m_supplementary.py:140`.
- Recovery numbers: `artifacts/headline_650m.json`, `result.results` entries with `method == "practice"`
  (B=48 pairwise `spearman = -0.2709300398322422`, `pearson = -0.2723860622842808`). Artifact provenance
  `source_run_id 20260711T091947Z`, `evidence_classification: traceable_not_rerun`.
- Downstream numbers: `experiments/trpb-downstream-generalization-20260716.md:27-29` (GB1), `:41,:43-44`
  (TrpB), `:54-59` (hit_rate / ndcg / regret). Present in the committed HEAD version.
- Scope narrowing: commit `8517b96a00571e3dcdd76da6645c6dc979366213` "docs(robustness): drop
  practice_heuristic from the reuse list", diffing `specs/robustness.md` lines 41-49 and 56-60;
  `src/epibudget/robustness.py:44-58` (import list, no `practice_heuristic`);
  `artifacts/robustness_650m.json` (no `practice` substring).
- Companion-not-decision status: `SPEC.md:242-244`, `:260`; `VALIDATION.md:128`, `:136-137`;
  `specs/downstream.md:158`, `:337`, `:459-461`, `:487`.
- Tests: `tests/test_validate.py:280-288` (returns exactly `budget` cross-site order-2 variants), `:315` (all
  five methods present in a run), `:493-494` (label-permutation invariance of `n_informed`), `:524-528`
  (structural no-label-leakage guard: the selector signature may not accept a landscape or labels).
- Prior-art framing of the heuristic: `PRIOR_ART.md:21`.
- Introducing commits: `478b7ac` (recovery harness and baselines), `ec9b5f7` (downstream benchmark).

**Unverified.** The GB1 `practice − structural` sign count is not stated in the committed experiment note
(only the per-method S_macro series is), although the pair is pre-registered in `specs/downstream.md:459-460`.
The underlying `report/*/downstream.json` runs are git-ignored.

## Reduced-alphabet (ADEF) 35M fast run as an evidence base

- **Verdict:** narrowed

**Question.** Can a restricted per-site candidate alphabet (`ADEF`, ~307 candidates) scored with the small
ESM-2 35M model stand in for the full-alphabet 650M benchmark? The run costs minutes on CPU instead of
GPU-hours, so if its method ranking were trustworthy it would have been the cheap evidence base for every
empirical claim.

**What was built.** A `--alphabet` option on the `validate` command (`src/epibudget/cli.py:253-254`, "a
reduced set keeps the fast-model run tractable"), applied through
`enumerate_candidates(..., allowed_aa=alphabet, ...)` at `src/epibudget/cli.py:288`. The alphabet is carried
into every result record as provenance (`candidate_alphabet`, `src/epibudget/cli.py:308,331`). One complete
allocate → reveal → infer → score run was executed and archived as `artifacts/smoke_recovery_35m.json`.

**What was measured.** Five allocation methods (`info`, `structural`, `fitness`, `practice`, `random`) at
B ∈ {48, 96}, 20 seeds, `n_perturbations=16`, `device=cpu`, on `gb1_wu2016` with 307 candidates and 132 truth
terms (`artifacts/manifest.json`, `generation_command`). Per method and per order (pairwise / third /
pooled): Spearman recovery correlation, `coverage_fraction`, `n_pinned` (breadth — terms whose full loop is
measured) and `spearman_predicted` (precision — correlation over informed-but-not-pinned terms).

**Outcome.** Numerically strong and structurally uninterpretable.

- Pairwise Spearman at B=96: `structural` 0.9695, `info` 0.7604, `fitness` 0.6007, `random` 0.2302,
  `practice` 0.2242.
- At B=96 `coverage_fraction` is 1.000 for `info`, `structural` and `practice`; `structural` pins 57 of 58
  pairwise terms, and its precision correlation `spearman_predicted` is `null` — essentially nothing was left
  unpinned to predict.
- B=96 consumes 96/307 ≈ 31% of the pool, so "measure the low-order scaffold broadly" wins by construction
  rather than by better prediction.
- Third-order recovery is erratic and moves the wrong way with budget: `info` 0.2104 (B=48) → −0.0250 (B=96);
  `structural` 0.3178 (B=48) → 0.0200 (B=96).

A second, independent disqualifier: the underlying ε signal at 35M is weak — pairwise ρ 0.085 and third-order
ρ 0.108 predicted-vs-measured, against 0.302 / 0.249 at 650M (`SIGNAL_GATE.md:47`).

**Why this verdict.** Narrowed, not abandoned: the configuration survives as a cheap end-to-end pipeline
exercise, but was permanently barred from carrying any empirical claim. Two reasons, each sufficient. (1) It
is an *exhaustion regime* — with a pool that small, a budget of 48-96 measures a large fraction of it, so the
benchmark cannot separate breadth from precision, which is exactly the distinction the whole comparison turns
on. (2) The 35M model's epistasis signal is too weak to rank methods on. The correct regime, `pool ≫ B`, was
pre-registered instead: full 20-letter alphabet, ~76 singles / ~2,166 doubles / ~27,436 triples, "so a win
cannot be an artefact of pool exhaustion" (commit `c934bb2`). The frozen headline run duly used 29,678
candidates and 17,782 truth terms (`artifacts/headline_650m.json`).

The ordering is the load-bearing part: the pre-registration (`c934bb2`) and the exhaustion caveat
(`315a30a`) both landed **before** the 650M headline artifact existed (`e75535d`). The demotion was a
protocol decision taken on structural grounds, not a rationalisation of an unwelcome number — and this
matters, because the ADEF run is the first place `structural` was seen beating `info`, a result that later
held up in the full-alphabet regime.

One correction against the working record: continuous integration does **not** run the 35M pipeline. The CI
gate is offline by design and runs `pytest -q -m "not slow and not data"` (`.github/workflows/ci.yml`), while
every 35M-instantiating test is marked `@pytest.mark.slow` (`tests/test_scoring.py:88,117,149,187`). The 35M
smoke path is opt-in and locally invoked, and `SIGNAL_GATE.md:58` describing it as a "CI smoke-test"
overstates what the pipeline actually executes.

**Evidence.**
- `artifacts/smoke_recovery_35m.json` — `candidate_alphabet: "ADEF"`, `n_candidates: 307`,
  `n_truth_terms: 132`, `model_id: esm2_t12_35M`, `seeds: 20`, `budgets: [48, 96]`; `structural` B=96
  pairwise `spearman: 0.9694853732812454`, `n_pinned: 57` of `n_terms: 58`, `coverage_fraction: 1.0`,
  `spearman_predicted: null`; `info` B=96 pairwise `spearman: 0.7604355716878403`.
- `artifacts/manifest.json` — the entry for that path carries `"status": "smoke_test"` and
  `generation_command: epibudget validate --model esm2_t12_35M --alphabet ADEF --budgets 48,96 --seeds 20
  --n-perturbations 16 --device cpu`. The demotion is machine-readable, not only prose: `status` is a typed
  literal at `src/epibudget/artifacts.py:32`.
- `LIMITATIONS.md:106-110` (§4, exhaustion regime, 57/58 pinned) and `:47-49` (§1, "remain a smoke test, not
  the headline"); `VALIDATION.md:139-145` ("Headline regime (pre-registered)", full 20-letter alphabet,
  `pool ≫ B`); `SIGNAL_GATE.md:47`, `:58`.
- Commits `c934bb2` (pre-registers the `pool ≫ B` headline regime and the breadth/precision split),
  `315a30a` (registers the exhaustion caveat), `e75535d` (frozen 650M headline artifact), `188b5ae` (adds the
  smoke artifact with provenance).
- `src/epibudget/cli.py:185,253-254,288,308,331` — the `--alphabet` knob and its provenance propagation.

## ESM-circular downstream diagnostic (`_esm_circular`) and its log1p-vs-WT-centred scale mismatch

- **Verdict:** narrowed

**Question.** How much of the downstream ranking signal could a purely ESM-prior-based predictor produce? The
downstream benchmark's whole point is to escape the map-recovery tautology, so it needs a deliberately
circular control that shows what the tautology looks like — the score you get when predictions for held-out
variants are allowed to restate their own ESM prior.

**What was built.** `_esm_circular` in `src/epibudget/downstream.py:1313-1336`, one of three explicitly
non-primary ESM diagnostics. It reuses `epibudget.validate.esm_prior_mu`
(`src/epibudget/validate.py:147-161`) — the exact posterior-mean mechanism `infer_epistasis` conditions on —
rather than reimplementing it, so the control is the real mechanism and not an approximation of it. Because a
fold's evaluation set and its revealed set are disjoint by construction, every predicted value collapses to
`b * esm[v]`, where `b` is the through-origin calibration slope fit on the revealed labels
(`validate._calibrate_slope`, `src/epibudget/validate.py:109-125`).

**What was measured.** `S_macro` (the order-stratified statistic `½(ρ_doubles + ρ_triples)`,
`downstream.py:484-492`) under the prior-collapsed predictor, serialized per fold as `esm_circular_s_macro`
alongside `esm_zero_shot_s_macro` and `esm_offset_s_macro`.

**Outcome.** The diagnostic is defective on a scale contract and this was caught, recorded, and quarantined
rather than patched or shipped silently. `_esm_circular` feeds `log1p(fitness)` labels to `esm_prior_mu`
(`downstream.py:1325`), but `esm_prior_mu`'s `revealed` argument is contractually WT-centred log fitness
(`validate.py:171`), and `_calibrate_slope`'s intercept-free fit is only valid because both scales are
anchored at zero at WT (`validate.py:110-119`). `log1p` puts WT at +0.693, not 0, so the anchor is wrong.

Quantification of the severity, using the repository's own functions against the committed GB1 landscape
(`data/proteingym/gb1_wu2016.csv`) and the 650M scored cache, over the 28,186 variants present in both (WT
fitness = 1.0; 94.7% of variants fall below WT):

- `b` under `log1p` labels (what the code actually does) = **−0.0163**
- `b` under WT-centred log fitness (the contract) = **+1.528**

The slope does not merely change magnitude — **it changes sign**. That matters exactly once, and decisively:
for a held-out variant the prediction is `b * esm[v]`, and Spearman is invariant to the magnitude of `b` but
not its sign. So the defect is inert for every purpose except the one thing the diagnostic reports, whose
direction it inverts.

This is visible in the GB1 run's raw records: across all 28,800 of them, `|esm_circular_s_macro|` equals
`|esm_zero_shot_s_macro|` exactly (0 exceptions), and 26,800 of the 28,800 carry the opposite sign. Two
consequences follow. First, the exact equality independently confirms the disjointness premise in the
function's docstring — a single pinned measured value would break it. Second, under the current metric the
diagnostic carries no information beyond the zero-shot control times the sign of a nuisance parameter, and
that sign is currently the wrong one for most selections.

**Why this verdict.** Narrowed, not kept and not abandoned. The diagnostic itself survives — the question it
asks is a mandated part of the protocol — but its claim scope was cut to a labelled non-primary lane that can
touch no decision surface, and the defect was booked as public debt with a mandatory precondition on any
rerun rather than left as a code comment. Three things hold that narrowing in place mechanically, not just by
convention:

1. the module docstring states the three ESM diagnostics are a separate, explicitly-labelled, non-primary
   path that *may* restate the prior (`downstream.py:1-14`);
2. the spec forbids all three from entering the structural claim, the ESM-uncertainty claim, the robustness
   gate, sign consistency, or the primary `S_macro`/AUC (`specs/downstream.md:401-402`);
3. a test corrupts only those three fields on every raw record and asserts the decision pipeline does not
   move by a single bit while the diagnostic columns do (`tests/test_downstream.py:1064-1115`, Invariant B in
   `specs/downstream.md:417-420`).

The scoped fix was chosen over the broad one deliberately: the primary learner's `log1p(raw fitness)`
response is a frozen spec'd choice (`specs/downstream.md:372`), so replacing the downstream prediction target
globally would corrupt the decision-eligible path in order to repair a diagnostic. The registered remedy is a
downstream-specific calibration scale local to `_esm_circular`.

This is also the second appearance of one mechanism, not an isolated bug. The limitations register already
carries the same failure mode for low-coverage recovery: with no measured loop member, `ε̂ = b · ε̂_ESM`
exactly, so the reported sign is `sign(b)`, a nuisance parameter rather than a property of the selection
(`LIMITATIONS.md:172-177`). A through-origin ESM calibration slope governing the sign of a prior-collapsed
quantity is a recurring hazard in this codebase.

**Evidence.**
- `src/epibudget/downstream.py:1313-1336` — `_esm_circular`; in-code TODO at `:1324`; `log1p` labels at
  `:1325`; module docstring `:1-14`.
- `src/epibudget/validate.py:147-161` (`esm_prior_mu`), `:109-125` (`_calibrate_slope`, WT-anchor rationale),
  `:171` (`infer_epistasis` docstring — `revealed` is WT-centred log-fitness ΔG).
- `LIMITATIONS.md:179-182` (the registered scale-mismatch entry), `:172-177` (the same slope-sign mechanism
  in the recovery path).
- `specs/downstream.md:391-395` (diagnostic 1 defined as the tautology control), `:401-402` (never
  decisional), `:417-420` (Invariant B), `:372` (frozen `log1p` primary response).
- `tests/test_downstream.py:1050-1052` (diagnostic must reuse the real `esm_prior_mu` mechanism), `:1064-1115`
  (diagnostic fields never feed the decision pipeline), `:951-964` (primary predictor never touches
  `esm_prior_mu` or `infer_epistasis`).
- Commits `ec9b5f7` (introduced `_esm_circular` with the downstream benchmark), `e00b2c3` (added the in-code
  TODO), `943833d` (added the public limitations entry).
- The per-record numbers above come from the GB1 downstream run artifact, which lives under an ignored
  `report/` tree and is `status = provisional` — it is referenced from `LIMITATIONS.md:186-195` but is not
  itself a committed artifact a reader can retrieve.

**Gap.** The quantified severity — that the mismatch inverts the sign of `b` on real GB1 data (−0.0163 vs
+1.528), and that the diagnostic consequently collapses to ± the zero-shot control in 28,800 of 28,800
records — is not recorded in a committed document. The committed register states the mismatch exists and
confines it, but does not state that its practical effect is a sign inversion of the reported statistic
rather than a rescaling.

## Supplementary 650M deterministic-only recovery run (`n_perturbations = 0`) as a no-GPU workaround

- **Verdict:** superseded

**Question.** The frozen 650M recovery comparison needs a `var_delta_g` estimate — 16 extra
background-masking forward passes per variant — which is impractical on CPU. Could the full-alphabet,
`pool ≫ B` comparison still be obtained on CPU by dropping the variance pass entirely, and would the
resulting var-independent arms be trustworthy?

**What was built.** `scripts/headline_650m_supplementary.py` (183 lines, added whole in commit `4b717b1`). It
runs the standard validation pipeline with `ConjointScorer(..., n_perturbations=0)` (line 100) over the full
20-letter, four-site GB1 pool, and reports only the methods whose selection does not read `var_delta_g`:
`fitness-greedy`, `random`, `practice`, and the `structural-only` ablation (`structural_graph`, τ²
identically 1, `src/epibudget/validate.py:362-371`). The cost saving is real and mechanical:
`ConjointScorer._var_delta_g` short-circuits to `0.0` when `n_perturbations <= 0`
(`src/epibudget/scoring.py:290-291`), and the deterministic pass de-duplicates the 29,678 enumerated variants
down to 4,564 unique forwards — a claim pinned by a passing offline test,
`tests/test_scoring_plan.py::test_dedup_full_pool_matches_4564`.

Critically, the script refuses to pass itself off as the headline. Its docstring states info-optimal is
**omitted, not dropped to win** (it degenerates when τ² is identically 0), and the emitted payload carries
two self-limiting fields (lines 164-168): `info_optimal: "deferred (needs the var_delta_g pass...)"` and a
`note` recording that "the VALIDATION.md decision rule (info vs fitness vs random) is not evaluated here."

**What was measured.** Map-recovery correlation (Spearman/Pearson against ground-truth epistasis terms,
bootstrap CIs, per order: pairwise / third / pooled) for the four var-independent methods at
B ∈ {48, 96, 192}, 20 random seeds, full `ACDEFGHIKLMNPQRSTVWY` alphabet, 29,678 candidates, 17,782 truth
terms, `device=cpu`.

**Outcome.** From `artifacts/supplementary_recovery_650m.json` — pairwise Spearman:

| method | B=48 | B=96 | B=192 |
|---|---|---|---|
| structural-only | 0.4845 | 0.4602 | 0.5042 |
| random | 0.2791 | 0.2796 | 0.2875 |
| practice | −0.2709 | 0.2821 | 0.3048 |
| fitness-greedy | −0.2591 | −0.2472 | −0.1342 |

So the run did what it was built for: it established a full-set structural-only recovery line well above
random, with fitness-greedy actively *anti*-correlated at every budget.

The cross-run agreement claim needs correcting. Against the later GPU headline
(`artifacts/headline_650m.json`, `device=cuda`, `n_perturbations=16`, same 29,678 candidates and 17,782 truth
terms), the shared var-independent arms are **not** byte-identical, contrary to how this run has been
described. Comparing all 36 shared method × budget × order Spearman values, only two match exactly —
structural pairwise at B=96 and B=192, precisely the cells where `coverage_fraction` is 1.0. Every other cell
differs at roughly the 1e-6 level (e.g. fitness B=48 pairwise: `-0.2590519896866100` supplementary vs
`-0.2590490176916245` headline). The runs agree to about five to six significant figures — genuinely strong
mutual corroboration across two devices — but "identical" overstates it.

**Why this verdict.** Three independent reasons, in ascending severity:

1. *Self-declared as additive.* The run never claimed to substitute for the frozen protocol; it omits
   info-optimal by construction and says so in both code and artifact. Its manifest entry records
   `"status": "supplementary"` with `"evidence_classification": "traceable_not_rerun"`.
2. *Superseded on the merits.* The variance-inclusive headline subsequently ran on GPU over the same pool
   (`artifacts/headline_650m.json`, run `20260711T091947Z`, stamped `configuration.colab_commit: 3ba75eb`),
   supplying the info-optimal arm this run had to defer. The workaround's reason to exist expired.
3. *The line it established turned out to be an artifact.* The structural-only result is not a structural
   finding at all. Because `n(v)` is constant within each mutation order (1140 singles / 39 doubles / 1
   triple), the τ²-identically-1 weight takes only three distinct values, and `allocate`'s stable `sorted`
   resolves the resulting exact tie by `enumerate_candidates`' site-major, residue-alphabetical emission
   order. At any B ≤ 76, "structural-only" literally means "take the first B singles in enumeration order."
   This was confirmed by predicting the artifact's integers from the tie-break rule alone with no model input
   — and those predictions match *this run's own numbers* exactly: pooled `n_informed` 17,700 at B=48, 17,782
   at B=96/192, `n_pinned` 20 and 116. Replaying the same cache under other equally valid tie-breaks of the
   identical tie collapses pooled Spearman from 0.2470 (as-run) to 0.0355 (reversed enumeration order); under
   reversal, info-optimal wins at B=48. Consequently the "structural wins at every budget" interpretation is
   recorded as **withdrawn**.

One further epistemic cost is recorded in the repository and worth carrying forward: because this run was
computed and committed *before* the post-hoc robustness analyses were designed, the qualitative shape of its
result (uncertainty prior looking unhelpful, structural-only beating random and fitness-greedy) was already
visible when those analyses were chosen. The project explicitly forfeits any claim to blind pre-registration
for that section as a result. Getting a cheap answer early cost the ability to claim the answer was unbiased.

Two details could **not** be substantiated and are dropped: the "~40 min CPU" runtime appears in no committed
document, and the byte-identity claim is refuted above.

**Evidence.**
- `scripts/headline_650m_supplementary.py` — docstring lines 1-19 (scope disclaimer, info-optimal omission);
  line 100 (`n_perturbations=0`); lines 132, 139 (structural ablation, `lambda_=0.0`); lines 162-183
  (self-limiting payload fields).
- `artifacts/supplementary_recovery_650m.json` — lines 10-29 (config, `info_optimal: deferred`, `note`,
  29,678 candidates, 17,782 truth terms, `device: cpu`); lines 97-159 (structural B=48, pooled `n_informed`
  17,700); lines 375-377 (`n_pinned` 20); lines 635-637 (`n_pinned` 116).
- `artifacts/headline_650m.json` — the superseding GPU run (`device: cuda`, `n_perturbations: 16`).
- `artifacts/manifest.json` — entry for `artifacts/supplementary_recovery_650m.json`,
  `status: supplementary`, source run `20260710T101945Z`, with the exact generation command.
- `artifacts/claim_map.json` — contains no reference to either the supplementary or the headline artifact; 24
  comparative recovery claims were removed in commit `943833d`.
- `src/epibudget/scoring.py:290-291` — the `n_perturbations <= 0` short-circuit that makes the workaround
  possible; `src/epibudget/validate.py:362-371` — `structural_graph`.
- `tests/test_scoring_plan.py::test_dedup_full_pool_matches_4564` — passes; pins the 29,678 → 4,564
  de-duplication.
- `LIMITATIONS.md:39-45` (the supplementary run's registered scope), `:130-134` ("structural wins at every
  budget" withdrawn), `:162-170` (structural-only has no within-order signal);
  `experiments/trpb-smoke-20260713.md` §6.2, lines 214-264; `VALIDATION.md:169-175` (records that this run's
  qualitative result preceded, and therefore compromised the blindness of, the post-hoc robustness
  analyses).
- Commit `4b717b1` — "feat(validate): add supplementary 650M deterministic-only recovery"; message states
  "Explicitly not the frozen headline: info-optimal is omitted, not dropped to win."

## Comparative epistasis-map recovery headline (frozen 650M) — withdrawn

- **Verdict:** abandoned

**Question.** At equal budget *B*, does an information-optimal allocation (ESM masking dispersion × loops
braced, `lambda_=0`) recover GB1's ground-truth epistasis map better than the same budget spent
fitness-greedily, and better than random? The decision statistic was frozen before any result existed:
pairwise-order Spearman **and** Pearson, `recovery(info) − recovery(fitness) > 0` and
`recovery(info) > recovery(random)`, both with non-overlapping bootstrap 95% CIs, at a majority of
*B* ∈ {48, 96, 192} (`VALIDATION.md` §"Decision rule (frozen)", lines 108-121). The protocol also fixed
mandatory companions — `practice` and a prior-free `structural-only` ablation — with the rule written in
advance that if info ≈ structural, "the ESM uncertainty prior does nothing to the allocation and must be
dropped from the claims" (`VALIDATION.md` lines 123-137).

**What was built.** `src/epibudget/validate.py` — the recovery harness. Zero-shot selection (`allocate` /
`fitness_greedy` / `practice_heuristic` / `random_selection`), then `data.reveal_measured_fitness` as the
single point where labels enter (`validate.py:396`), then `infer_epistasis` (closed-form posterior mean of
the `graph.py` linear-Gaussian model — Tikhonov shrinkage toward the through-origin-calibrated ESM ΔĜ prior
with precision `1/var_delta_g`), then `map_recovery` with bootstrap CIs. The *same* inferrer runs for every
method; only the selected set differs (`validate.py:456-476`). The `structural-only` ablation is the
identical factor graph with τ² ≡ 1, so ranking collapses to `n(v)` alone (`validate.py:451`). Results were
wired into a checksummed public artifact layer with a JSON-pointer claim map (`artifacts/headline_650m.json`,
`artifacts/robustness_650m.json`, `artifacts/claim_map.json`, `scripts/build_public_artifacts.py`).

**What was measured.** Per-order Spearman and Pearson between inferred and true ε, with `bootstrap-over-terms`
CIs for deterministic methods and `bootstrap-over-seeds` for random, plus a pre-registered split of breadth
(`n_pinned`) from precision (correlation over informed-but-not-pinned terms). Frozen configuration, confirmed
in the artifact: `gb1_wu2016`, `esm2_t33_650M`, full 20-letter alphabet, 29,678 candidates, 17,782 truth
terms, `n_perturbations=16`, 20 seeds, `max_order=3`, `device=cuda` (Colab T4), branch tip `3ba75eb`.

**Outcome.** Two results, in tension.

Registered rule: **formally supported at all three budgets.** Pairwise Spearman — info 0.408 / 0.418 / 0.443,
fitness −0.259 / −0.247 / −0.134, random 0.279 / 0.280 / 0.288; Pearson agrees (info 0.458 / 0.479 / 0.504 vs
random 0.311 / 0.311 / 0.317). Info-vs-random and info-vs-fitness CIs are non-overlapping at all three
budgets on both correlations (verified directly from `artifacts/headline_650m.json`).

Mandatory ablation: **the prior-free control beat the method it was meant to isolate.** `structural-only`
pairwise Spearman 0.485 / 0.460 / 0.504 and Pearson 0.514 / 0.526 / 0.573 — higher than info at every budget
(full-set CIs overlap at most budgets; Pearson at *B*=192 is the one non-overlapping cell). The post-hoc
matched-term precision comparison agrees: at *B*=48 over the 1,511 pairwise terms both methods inform and
neither pins, Spearman 0.452 (info) vs 0.537 (structural), Δ = −0.085, 95% CI [−0.125, −0.047], excluding
zero (`artifacts/robustness_650m.json`, `common_precision`). So the ESM masking-perturbation uncertainty
prior earned no credit; the recovery was attributed to the structural `n(v)` loop-coverage sort.

Then the whole comparison was withdrawn. Commit `943833d` deleted **24** comparative claims (216 lines,
verified by counting removed `"id"` keys in the diff) from `artifacts/claim_map.json` and updated the
registry SHA in `artifacts/manifest.json`; the README's entire frozen-headline paragraph was removed and
replaced by "No comparative recovery claim is current." `artifacts/claim_map.json` now holds 17 claims, none
with a `headline.*` id.

**Why this verdict.** Two mechanical defects, both found after the fact, make the effect sizes
uninterpretable in either direction:

1. **The structural control carries no within-order information.** `n(v)` is constant per order (1140
   singles / 39 doubles / 1 triple), so with τ² ≡ 1 the greedy weight takes exactly three distinct values and
   the within-order ranking is an exact tie broken by `enumerate_candidates`' site-major enumeration order.
   `structural-only` was therefore a single unreplicated draw with no variance over its tie-break, and
   "structural wins at every budget" is not a finding about structure (`LIMITATIONS.md` §6b).
2. **A per-method calibration slope can set the sign of a low-coverage method's recovery.** With no measured
   loop member, ε̂ = b · ε̂_ESM exactly, so a near-zero-coverage method reports `sign(b) · ρ_prior` — a
   nuisance parameter, not a property of its selection. Fitness-greedy's pairwise coverage was
   0.063 / 0.097 / 0.277 and practice's 0.026 / 0.052 / 0.102, precisely the regime where this bites
   (`LIMITATIONS.md` §6b; coverage read from the artifact).

A third, older limitation was already on record: map-recovery is **partly tautological** — a method can score
high by having *measured* terms (breadth) rather than predicted them (precision) (`LIMITATIONS.md` §4).

The corrective analysis (gate 2, `src/epibudget/gate2.py`) re-ran the comparison over the completed 650M
cache with 100 seeded structural tie-breaks and a method-independent five-fold cross-fit slope. Its
registered decision was `inconclusive_zero_gpu`: pairwise correlations improve at every budget while relative
squared-error gain is negative at every budget, and no registered calibration contrast reverses sign at the
required two budgets. It is `status=provisional`, `public_claim_eligible=false`. So the converse claim — that
ESM masking dispersion *does* improve allocation — is not supported either.

The decision taken was to retract the interpretation without rewriting history: the artifacts were left
byte-identical (`artifacts/headline_650m.json` has exactly one commit in its history, `e75535d`) and remain
listed in `artifacts/manifest.json` as `primary`, but are labelled "superseded for current claims" and are no
longer cited. The estimand itself was moved to a separate downstream-impact benchmark, which asks whether a
structure-aware budget yields a better *training set* for ranking held-out double/triple mutants — a
different question, decided independently.

**Evidence.**
- `artifacts/headline_650m.json` — `provenance.source_run_id` `20260711T091947Z`, `source_sha256`
  `c11dbe00a1…`, `evidence_classification: traceable_not_rerun`; all recovery numbers and CIs above read
  directly from `result.results`.
- `artifacts/robustness_650m.json` — `result.common_precision`, info-vs-structural pairwise entry
  (`n_common: 1511`, `delta: -0.08502…`, `excludes_zero: true`).
- Commit `943833d` "docs: narrow thesis; retire retracted comparative claims from README and registry" —
  `artifacts/claim_map.json` −216 lines / 24 claim ids, `artifacts/manifest.json` SHA update, README headline
  paragraph removed.
- Commit `e75535d` "feat(artifacts): record frozen 650M headline and precision analysis" — the original
  record; its message already states the ablation outperforms info-optimal.
- `README.md:93-96` "Comparative allocation status. No comparative recovery claim is current."
- `VALIDATION.md:108-137` (frozen decision rule and mandatory baselines), `:207-227` ("Historical outcome —
  frozen 650M headline … superseded for current claims"), `:229-248` (corrective gate 2,
  `inconclusive_zero_gpu`).
- `LIMITATIONS.md` §4 (tautology), §5 (structural comparison withdrawn), §6b (exact within-order tie;
  per-method slope sign).
- `src/epibudget/validate.py:396` (single label entry point), `:449-476` (shared inferrer, τ² ≡ 1 ablation);
  `src/epibudget/acquisition.py:28-69` (`allocate`, modular sort at `lambda_=0`).

## Exploratory TrpB map-recovery smoke (second-landscape transfer of the recovery harness)

- **Verdict:** abandoned

**Question.** Do the allocation and evaluation abstractions run on a second, biochemically independent
four-site landscape — TrpB (Johnston et al. 2024, *PNAS* 121(32) e2400439121; enzyme catalysis) rather than
GB1 (IgG-Fc binding) — without GB1-specific assumptions, and does the GB1 map-recovery result transfer with
them?

**What was built.** A generic four-site loader shared by both landscapes: `_load_landscape` and `load_trpb`
in `src/epibudget/data.py`, with the Tm9D8\* parent registered as `TRPB_SITES = (182, 183, 226, 227)` /
`TRPB_WT_AT_SITES = ("V","F","V","S")` / `TRPB_WT_SEQUENCE` (`data.py:31-37, 141-143`). Genotypes are
recovered by diffing full-length 397-residue sequences against the parent, so mutant-string formatting in the
source mirror is irrelevant. A dataset registry (`DatasetSpec`, `resolve_dataset`, `data.py:176-182`) routes
the whole validate path through `--dataset trpb_johnston2024` (`src/epibudget/cli.py:243-280`).
`scripts/fetch_trpb.py` downloads the CSV, checksums it, loads it through the real loader so the
reference-residue asserts fire at fetch time, and writes a `provenance_trpb.json`.
`src/epibudget/trpb_explore.py` (553 lines) is a raw-row profiler that classifies duplicate / conflicting /
missing-label / invalid-residue rows *before* the dict loader collapses them — `load_trpb` returns a mapping
and therefore silently swallows exactly that structure. Every profiler artifact is stamped
`RUN_TYPE = "exploratory_non_decision_eligible"` and `decision_eligible: bool = False`
(`trpb_explore.py:43, 143-144`). `scripts/explore_trpb.py` is the notebook entry point. The exploratory
status was pre-registered in `specs/trpb-exploratory.md` at commit `4077f55`, before the run.

**What was measured.** The production `epibudget validate` harness on TrpB: `esm2_t33_650M`, full 20-letter
alphabet, `device = cuda`, `scorer_seed = 0`, `n_perturbations = 16`, 29,678 order-1..3 candidates, 17,709
truth terms (1,784 pairwise + 15,925 third), at `B ∈ {24, 48}` with 5 seeds. That grid is deliberately off
the pre-registered protocol of `VALIDATION.md` §"Second landscape — TrpB", which freezes B ∈ {48, 96, 192}
and ≥ 20 seeds. Metric: Spearman ρ between inferred and true ε per order (pairwise / third / pooled), with
bootstrap CIs, plus coverage, `n_informed`, `n_pinned` and hit-rate.

**Outcome.** Technical transfer passed; scientific transfer is uninterpretable.

All 10 method × budget cells completed, finite and well-formed. Dataset identity verified directly: the
on-disk CSV hashes to `e94e2bed0a128f505eeedd8890cad64b3113c4a17562908ad8a121fa2a8e205f`, exactly the
`data_sha256` serialized in the report; the report itself is 19,751 bytes hashing to
`c85a9abc051d…f2bd4ca7`, as recorded. `n_candidates = 29678`, `n_truth_terms = 17709`,
`var_epsilon = 4.844110` all reproduce from the local CSV.

The method ranking cannot be read off it, for four independent reasons:

1. **Broken WT anchor.** `interaction_loop` (`epistasis.py:69-77`) iterates `range(1, n+1)` and excludes the
   empty set, so ε structurally assumes ΔG(∅) = 0. The historical `validate.py` at `42ef311` built
   `landscape_dg = {v: log(f) for v, f in landscape.items() if f > 0.0}` — no reference subtraction. GB1
   satisfies the assumption by luck of normalisation (f(WT) = 1.0 exactly); TrpB's parent is f = 0.408073925,
   so ln f(∅) = −0.896307 ≠ 0. Recomputed from both CSVs: every pairwise ε is shifted **+0.896307** and every
   third-order ε **−0.896307** (GB1: 0.0 in both orders). `var_epsilon` is 4.844110 as-run vs **3.930211**
   re-anchored, i.e. **+23.3% inflated**; η², the between-order share of pooled ε variance, is 0.2574 as-run
   vs 0.0848 re-anchored, against GB1's 0.0793 under either transform. The bug manufactures between-order
   separation, which pooled recovery then rewards. The arithmetic tell is visible in the artifact: `info` at
   B=48 has pooled ρ = 0.2891, **above both of its own sub-orders** (pairwise 0.0434, third 0.1321).
2. **`structural − info` is not a clean ablation.** `structural_graph` sets τ² ≡ 1.0
   (`validate.py:362-371`), so its greedy weight is the loop count n(v), which is constant within each order
   in a four-site universe — the within-order ranking is an exact tie resolved by enumeration order.
3. **The per-method calibration slope sets a low-coverage arm's sign.** With no measured loop member,
   ε̂ = b · ε̂_ESM exactly, so such a method reports sign(b) · ρ_prior with ρ_prior method-independent. At
   B=48 pairwise the three ~0%-coverage arms report near-identical magnitudes with opposite signs: `fitness`
   +0.1305 (8 of 1,784 loops informed, 0.45%), `practice` +0.1424 (48), `random` −0.1264 (12). Read naively
   the artifact says "fitness-greedy beats info-optimal on TrpB pairwise" — false, since fitness-greedy
   informed 8 loops.
4. **No contrast has a dispersion estimate.** `run_validation` executes `info`, `fitness`, `structural` and
   `practice` once per budget; only `random` is replicated over `range(seeds)` (`validate.py:455-462`). The
   serialized `ci_method` confirms it: `bootstrap-over-terms` for the four deterministic methods,
   `bootstrap-over-seeds` for `random` alone. Seed-level paired contrasts are outside what the schema can
   express, not merely absent. `n_pinned = 0` in every cell, so no ε loop is fully measured at these budgets.

Recorded provenance gaps: the report schema carries no repository SHA and no `run_type`/`decision_eligible`
field, making it schema-indistinguishable from a confirmatory run; no ESM scored cache was exported, so no
TrpB post-hoc diagnostic is possible without re-paying full GPU scoring; and `data/proteingym/provenance.json`
documents GB1 only, with no `provenance_trpb.json` accompanying the CSV (the fetch script emits one, but the
run's data directory has none).

**Why this verdict.** The run violated its own pre-registered budget grid and seed count by construction, so
no number in it may be reported as *the* TrpB result regardless of quality — and the anchor bug independently
invalidates every recovery coefficient, recovery correlation and truth-map variance it produced. It was
declared a technical success and a scientific non-result, and was deliberately **not** retroactively
repaired: the artifact stands as run, with the correction recorded alongside it. Only anchor-independent
descriptive outputs survive (candidate enumeration, selection identities, run configuration,
attempted/revealed counts, coverage, hit-rate). It is evidence neither for nor against the GB1 claim. Its
lasting value is diagnostic: it is the instrument that surfaced all three metric defects, two of which (the
structural tie-break and the calibration slope) also contaminate GB1 and forced the retraction of the GB1
map-recovery headline. The anchor defect is fixed in current code — `wt_centered_log_fitness`
(`epistasis.py:30-48`) sets the reference exactly to zero and is called at `validate.py:434`. The
confirmatory TrpB recovery re-run under the frozen protocol remains deferred.

**Evidence.**
- `experiments/trpb-smoke-20260713.md` (387 lines; §1 provenance, §2 cell completeness, §3 result tables, §5
  landscape comparison, §6.1-6.3 defects, §7 verdicts, §9 self-limitations).
- `VALIDATION.md` §"Exploratory TrpB smoke — run 20260713T135240Z (non-decision-eligible)" (lines 361-393)
  and §"Second landscape — TrpB (pre-registered, run DEFERRED)" (line 332); `specs/trpb-exploratory.md`.
- `src/epibudget/data.py:31-37, 141-143, 176-182`; `src/epibudget/cli.py:243-280`;
  `src/epibudget/trpb_explore.py:43, 143-144`; `src/epibudget/epistasis.py:30-48, 69-77`;
  `src/epibudget/validate.py:362-371, 434, 455-462`; `scripts/fetch_trpb.py`; `scripts/explore_trpb.py`.
- Commits `4077f55` (exploratory TrpB transfer profile spec), `f8fe1fe` (dataset profiler), `a0f4fd1` (Colab
  entry point), `10a9e29` (native TrpB validation path), `42ef311` (merge of PR #2, `explore/trpb-port`);
  historical transform visible via `git show 42ef311:src/epibudget/validate.py` line 414.
- Run outputs (`report/20260713T135240Z/metrics.json`) and datasets (`data/proteingym/`) are git-ignored by
  design and are not in the repository; their checksums and full result tables are transcribed into the
  committed experiment note above. The `var_epsilon`, η², per-order ε offsets and truth-term counts quoted
  here were recomputed independently from the two landscape CSVs and match the serialized values exactly.

## Confirmatory second-landscape TrpB map-recovery benchmark (pre-registered, never executed)

- **Verdict:** inconclusive

**Question.** Does the GB1 epistasis-map-recovery comparison (info-optimal vs fitness-greedy vs random)
reproduce on a second, mechanistically independent landscape — TrpB, an enzyme-catalysis readout rather than
GB1's IgG-Fc binding assay — under a protocol identical in shape to the GB1 headline?

**What was built.** The full execution path exists, but no confirmatory run was ever performed against it.

- `src/epibudget/data.py` — `TRPB_SITES = (182, 183, 226, 227)`, `TRPB_WT_AT_SITES = ("V","F","V","S")` (the
  assayed Tm9D8* parent, not literal TmTrpB), `TRPB_WT_SEQUENCE`, `load_trpb`, and a
  `DATASETS`/`resolve_dataset` registry (lines 35-37, 141-196) that rejects unregistered identifiers instead
  of silently falling back to GB1.
- `scripts/fetch_trpb.py` — writes the git-ignored `data/proteingym/trpb_johnston2024.csv` plus
  `provenance_trpb.json` (source URL, download date, sha256, row count, reference sequence, order
  composition), and hard-codes the imputed-value caveat in the provenance record (lines 19-21, 83-104).
- `src/epibudget/cli.py` — `epibudget validate --dataset {gb1_wu2016|trpb_johnston2024}` (line 243), with
  `--data` defaulting to the selected dataset's canonical path and provenance recording the actual dataset
  id, sites and WT checksum.
- Tests: `tests/test_data.py::test_load_trpb_parses_sequences_into_variants`,
  `::test_load_trpb_rejects_off_site_mutation`;
  `tests/test_cli.py::test_validate_trpb_records_trpb_metadata` and
  `::test_validate_trpb_csv_not_loaded_via_load_gb1` (a loader-identity spy asserting the TrpB CSV never
  reaches `load_gb1`).
- The protocol itself is frozen prose: `VALIDATION.md` §"Second landscape — TrpB (pre-registered, run
  DEFERRED)" (lines 332-359).

**What was measured.** Nothing under this protocol — deliberately zero numbers. The protocol was frozen first
and pins: `esm2_t33_650M`, full 20-letter alphabet, `B ∈ {48, 96, 192}`, ≥ 20 seeds, `n_perturbations = 16`,
the same decision rule as the GB1 headline (pairwise-order Spearman **and** Pearson map recovery,
non-overlapping bootstrap 95% CIs), the same mandatory baselines (info / fitness / random, plus practice and
structural-only companions), and a commitment to report the result regardless of direction, including if it
weakens the GB1 story.

Two conditioning caveats were registered *in advance* of any number, and both are unusually consequential:

- ~871 of 160,000 TrpB fitness values (~0.5%) are imputed rather than measured, and the public mirror does
  not flag which — so no TrpB number can be reported without that caveat.
- TrpB encodes inactivity as *negative* fitness with **no exact zeros**, unlike GB1 where dead variants are
  exactly `0.0`. The `f > 0` positivity rule therefore drops the negative half of a noisy continuum
  straddling zero while retaining the near-zero positive half, whose `ln f` reaches −13.76 — assay noise
  amplified into large ΔG outliers that inclusion-exclusion then propagates into every ε term touching them.

**Outcome.** Deferred at freeze time and still deferred. No TrpB entry exists anywhere in `artifacts/`
(registry grep returns nothing), and the only `validate` run in `report/` touching `trpb_johnston2024` is
`report/20260713T135240Z/metrics.json`, whose config reads `dataset = trpb_johnston2024`,
`budgets = [24, 48]`, `seeds = 5` — off the frozen grid on both axes, hence a smoke by construction and not a
confirmatory result. That smoke is separately documented as uninterpretable for recovery: it ran with a
broken ε anchor (TrpB's parent is f = 0.408074, so ln f(∅) = −0.896307 ≠ 0 where the machinery assumes 0),
shifting every pairwise ε by +0.896307 and every third-order ε by −0.896307 and inflating `var_epsilon` by
+23.3%.

A TrpB run *did* execute at full protocol scale — but for a different benchmark: the downstream-impact
benchmark, `report/20260716T154715Z/downstream.json`, whose `exact_command` is
`epibudget downstream --dataset trpb_johnston2024 ... --n-perturbations 0 --budgets 48,96,192 --seeds 20
--partitions 20`. That is a training-set-quality test, not map recovery, and at `n_perturbations = 0` it is
`decision_eligible = false`.

**Why this verdict.** Inconclusive rather than abandoned or kept, because it was never executed and the
repository still names it as the only path to a reportable second-landscape recovery result. The freeze order
was the point: running a second landscape and only then choosing how to report it would be landscape /
multiple-comparison cherry-picking, so the protocol was fixed before the data was touched, and the deferral
is recorded as an explicit mitigation in the threats-to-validity table.

Two later developments make the deferral more than administrative and argue against ever simply un-pausing
it. First, the comparative recovery claim this benchmark was built to *confirm* was itself retracted on GB1 —
the headline comparative Spearman/Pearson claims were removed from the claim registry and the "structural
wins at every budget" reading withdrawn as an artifact of a deterministic tie-break over a score that is
exactly tied within mutation order. Second, map recovery is documented as partly tautological: a method can
score high because it *measured* many terms, not because it *predicted* the unmeasured ones. Confirming a
retracted, partly tautological statistic on a second landscape would buy little, which is why the
second-landscape generalization question was in practice carried by the downstream-impact benchmark instead —
corroborating, explicitly not establishing, TrpB transfer.

**Evidence.**
- `VALIDATION.md:332-359` (frozen protocol), `:361-393` (off-protocol smoke and its three defects), `:404`
  (cherry-picking mitigation row), `:250-256` (correction noting the confirmatory recovery run remains
  blocked while downstream is not).
- `LIMITATIONS.md:196-209` ("No second-landscape *recovery* result; the downstream replication is
  corroborating only"); `:99-104` (recovery is partly tautological); `:130-135` (historical structural-only
  comparison withdrawn).
- `experiments/trpb-smoke-20260713.md:148-178` (GB1 vs TrpB conditioning table: inactivity encoded as exactly
  0.0 in 29,477 GB1 rows vs negative in 35,643 TrpB rows with no zeros; 11,893 vs 11,820 terms dropped;
  `ln f` min −13.76), `:210-212` (re-anchoring fixes the convention, not the label).
- `src/epibudget/data.py:35-37,141-196`; `scripts/fetch_trpb.py:19-21,83-108`; `src/epibudget/cli.py:243`;
  `tests/test_data.py:147-208`; `tests/test_cli.py:460-577`.
- Commits `4237f53` (feat(data): add the TrpB second-landscape loader and fetch script — "No run is performed
  here"), `d23603e` (docs(validation): freeze the deferred TrpB second-landscape protocol), `10a9e29`
  (feat(cli): add native TrpB validation path), `943833d` (docs: narrow thesis; retire retracted comparative
  claims from README and registry).
- Run artifacts inspected directly: `report/20260713T135240Z/metrics.json`
  (`dataset=trpb_johnston2024, budgets=[24,48], seeds=5`); `report/20260716T154715Z/downstream.json`
  (`--n-perturbations 0`); `artifacts/` contains no TrpB record.

---

# Data

## GB1 dataset-completeness claim corrected from a complete 20^4 landscape to a measured subset

- **Verdict:** narrowed

**Question.** The repository described the Wu-2016 GB1 four-site block (V39/D40/G41/V54) as "the complete GB1
landscape". Is that true, and if not, what is the actual measured coverage — and does the gap change what the
benchmark is entitled to claim?

**What was built.** A machine-generated dataset-provenance artifact plus a claim registry that binds prose
numbers to it:

- `scripts/build_public_artifacts.py` — `_dataset_payload()` (lines ~228-262) loads the CSV via `load_gb1`,
  buckets every row by mutation order and by live (`fitness > 0`) vs dead (`fitness == 0`), and emits
  `artifacts/dataset_gb1.json` with a SHA-256 of the source file.
- `src/epibudget/artifacts.py` — `validate_public_artifacts()` re-renders each registered claim from its
  artifact JSON via a JSON pointer plus a declared transform, then fails unless the rendered string appears
  verbatim inside the anchor text and the anchor appears in the named document (lines ~186-201). It also
  fails on a list of `forbidden_literals` (retired historical numbers).
- `scripts/fetch_gb1.py` — `EXPECTED_ROWS = 149_361` (line 46), warning on mismatch (lines 77-78), so a
  silently changed upstream mirror is visible at fetch time.

**What was measured.** Row counts of the local public-data artifact against the theoretical
20^4 = 160,000-genotype space, stratified by mutation order and by viability.

**Outcome.** Not complete. `artifacts/dataset_gb1.json`:

- 149,361 measured rows of 160,000 theoretical (10,639 absent); 119,884 live, 29,477 dead.
- Absences concentrate at high order: order 4 is missing 9,147 of 130,321 and order 3 is missing 1,417 of
  27,436; order 2 is missing 75 of 2,166; orders 0-1 are complete.
- The order-1..3 candidate universe the selector actually draws from is therefore 28,186 present of 29,678
  enumerable — 94.97% (`experiments/trpb-smoke-20260713.md:156`, and the arithmetic reproduces from the
  artifact's `by_order` block).
- Usable ground truth is narrower still: real-valued epistasis truth is conditional on positive,
  log-transformable fitness with every loop member present, so the 29,477 dead rows are dropped rather than
  imputed and any interaction whose loop touches one is unrecoverable (`VALIDATION.md:36-44`,
  `LIMITATIONS.md:52-62`).

**Why this verdict.** Narrowed rather than abandoned: the dataset and the benchmark on it both survive, but
the scope statement attached to them was wrong and was rewritten. Commit `e6c1bb0` propagated the correction
through 18 files in one pass — README headline and claim sentence, `VALIDATION.md` (added an explicit
"Factual correction … the decision rule is unchanged" note), `LIMITATIONS.md`, `PRIOR_ART.md` ("a validated,
honest benchmark on the complete GB1 landscape" became "a null-tolerant benchmark on the measurable,
complete-loop subset"), `RESEARCH_EPISTASIS.md`, `SPEC.md`, the gate note, and public docstrings in
`scripts/fetch_gb1.py` and `src/epibudget/data.py` (`load_gb1`'s docstring went from "the complete GB1
four-site landscape" to "the measured GB1 four-site rows"); a test was even renamed
`test_gb1_loads_complete_landscape` → `test_gb1_loads_measured_landscape`.

The correction was then made hard to undo. The figure lives in the registry as claim
`dataset.measured_rows` (`artifacts/claim_map.json`), pointer `/measured_rows` on
`artifacts/dataset_gb1.json`, rendered `149,361`, anchored to the README phrase "149,361 measured genotypes".
`validate_public_artifacts` recomputes it from the artifact, so prose drifting away from the data is a
validation failure, not a silent regression. `pytest tests/test_artifacts.py -q` passes (8 tests) on the
current tree.

The scientific consequence is the point: the deliverable is principle validation on a dense measured subset,
not a whole-landscape result. It also gave the later TrpB work a clean contrast — TrpB is combinatorially
complete over the same 29,678-variant order-1..3 universe (100% vs GB1's 94.97%), though ~871 of its 160,000
labels are imputed rather than measured and unflagged in the mirror (`VALIDATION.md:342-347`,
`scripts/fetch_trpb.py:18-20`).

**Evidence.**
- Commit `e6c1bb08fd0223d18ee5446e6bf68a6b390f04ce` — "docs: correct dataset-completeness and
  empirical-claim overreach", 18 files, +186/−129.
- `artifacts/dataset_gb1.json` (counts and `by_order`, lines 6-43); `artifacts/claim_map.json` — claim
  `dataset.measured_rows`.
- `src/epibudget/artifacts.py:186-215` — claim re-rendering, anchor check, forbidden-literal check.
- `scripts/build_public_artifacts.py:228-262`; `scripts/fetch_gb1.py:10,46,77-78`.
- `VALIDATION.md:34-44`; `LIMITATIONS.md:52-62`; `README.md:69`;
  `experiments/trpb-smoke-20260713.md:148-156`.

## PSD95-PDZ3 as the planned generalization landscape, and the eligibility rule for adding one

- **Verdict:** superseded

**Question.** Which independent combinatorial landscape should carry the "works beyond GB1" generalization
check, and on what criterion does a candidate qualify at all?

**What was built.** A written eligibility rule plus a vetted candidate list, kept in working planning
documents and never implemented. The rule: a candidate protein qualifies only if it has *measured
multi-mutant* data — dense doubles at minimum, triples preferred — because without higher-order measurements
there is no ground-truth ε to recover and the exercise is vacuous; single-mutant-only DMS assays are
disqualified; coverage must be confirmed against the actual data files rather than from memory. First named
target: PSD95-PDZ3 (combinatorial DMS, GEO GSE184042). Alternates: KRAS, a tRNA landscape, and the Olson-2014
GB1 pairwise set. Reporting rule attached to it: report per protein, never pooled; if coverage is partial,
narrow the claim rather than impute. None of PSD95-PDZ3, KRAS, the tRNA landscape or GSE184042 appears
anywhere in the tracked repository — `git grep -i -E 'PSD95|PDZ3|GSE184042|KRAS|tRNA' HEAD` returns nothing
(rc=1), as does the same grep over the working tree. No fetch script, no loader, no dataset id was ever
written for it.

What was built instead is TrpB (Johnston et al. 2024, PNAS 121(32) e2400439121): `scripts/fetch_trpb.py` and
`epibudget.data.load_trpb`, added in commit `4237f53` together with a refactor extracting a generic four-site
loader that both GB1 and TrpB delegate to (`src/epibudget/data.py:98`, `:137`, `:142`).

**What was measured.** Nothing for PSD95-PDZ3 — the target was replaced before any run. The substitution
itself was decided on two stated structural grounds rather than on a measurement: (1) TrpB is combinatorially
complete over the same four-site, 20-letter universe as GB1, so the order-1..3 candidate pool is identical by
construction — 76 singles + 2,166 doubles + 27,436 triples = 29,678 variants for both landscapes, which means
the frozen budget grid `B ∈ {48, 96, 192}` transfers without re-justification
(`specs/trpb-exploratory.md:77`, `:105`); (2) its readout is biochemically independent — enzyme catalysis
versus GB1's IgG-Fc binding — so agreement across the two is a genuine generalization rather than a re-test
of the same assay type (`VALIDATION.md:341-344`).

**Outcome.** Superseded as a target. TrpB then carried both of the generalization workstreams that actually
ran: the exploratory smoke that exposed three defects making the historical recovery comparison unreadable
(`experiments/trpb-smoke-20260713.md`, §6 — a broken WT anchor, a `structural-only` weight that is constant
within each order and therefore an arbitrary tie-break, and a per-method calibration slope that sets the sign
of a low-coverage method's recovery), and the downstream-impact replication
(`experiments/trpb-downstream-generalization-20260716.md`: `structural − random` 20/20 partitions positive,
mean S_macro-AUC +0.135; `structural − fitness` 20/20, +0.286; exploratory only, `n_perturbations = 0`, so
`decision_eligible = false`).

Of the attached rules, the never-pool reporting discipline survived *and* reached a committed home:
`VALIDATION.md:245`, `specs/robustness.md:75-76`, `specs/trpb-exploratory.md:133` ("a per-protein split is
itself an honest finding"), `experiments/trpb-smoke-20260713.md:387` ("GB1 and TrpB are never pooled"). The
*eligibility criterion* did not: no committed document states the general rule that a candidate landscape
needs measured multi-mutant data to qualify. The committed record justifies TrpB specifically; it does not
record the gate that any future candidate must pass.

**Why this verdict.** The gate outlived the target. The specific protein changed because TrpB's structural
match to GB1 made it the cheaper and cleaner transfer test — an identical candidate universe means the second
landscape tests the *method*, not the ability to re-tune a pool size — while still satisfying the
independence requirement that motivated a second landscape in the first place. The eligibility rule that
selected PSD95-PDZ3 is the durable artifact and TrpB passes it too (combinatorially complete through order
4). Superseded, not abandoned: nothing about the criterion was found wrong.

One caution the record supports: the second landscape was run before the GB1 mechanism was settled, and the
TrpB smoke duly returned a technical success and a scientific non-result — the defects it surfaced were
metric-level and would have appeared on any landscape. That is an argument for the sequencing discipline
(settle the mechanism, then generalize), not against TrpB as the choice.

**Unverified claim, dropped.** One source record asserted that ParD3 is queued as the next generalization
check. No occurrence of "ParD3" exists anywhere in the repository — tracked, modified, untracked or ignored
(`rg -i --no-ignore --hidden 'ParD3'` returns nothing; `README.md` line 99 is about comparative allocation
status, not a third landscape). Treat as unsupported.

**Evidence.**
- Absence in the committed repository: `git grep -i -E 'PSD95|PDZ3|GSE184042|KRAS|tRNA' HEAD` → rc=1 (no
  output); same over the working tree.
- Replacement implementation: commit `4237f53` "feat(data): add the TrpB second-landscape loader and fetch
  script" (`scripts/fetch_trpb.py`, `src/epibudget/data.py`, `tests/test_data.py`; +214/−16).
- Protocol frozen before any number existed: commit `d23603e` "docs(validation): freeze the deferred TrpB
  second-landscape protocol" → `VALIDATION.md` §"Second landscape — TrpB (pre-registered, run DEFERRED)", now
  at `VALIDATION.md:332-360`, plus the added threat-table row "Second-landscape cherry-picking".
- Identical candidate universe justifying the swap: `specs/trpb-exploratory.md:77`, `:105`;
  `experiments/trpb-smoke-20260713.md:148`; count 29,678 corroborated in `artifacts/headline_650m.json:24`
  and `specs/downstream.md:528`.
- Landscape comparison table (assay, reference, coverage, order-1..3 presence 29,678/29,678 for TrpB vs
  28,186/29,678 for GB1): `experiments/trpb-smoke-20260713.md:148-165`.
- Never-pool rule in committed docs: `VALIDATION.md:245`; `specs/robustness.md:75-76`;
  `specs/trpb-exploratory.md:133`; `experiments/trpb-smoke-20260713.md:387`.
- Downstream replication numbers: `experiments/trpb-downstream-generalization-20260716.md:31-70`; summarized
  in `README.md:110-117`.

**Gap.** The eligibility rule, the candidate list (PSD95-PDZ3 / KRAS / tRNA / Olson-2014) and the sequencing
argument are re-expressed above from uncommitted working documents.

---

# Infrastructure

## Immutable provenance, checksummed artifacts and a machine-checked public claim registry

- **Verdict:** kept

**Question.** Can every number printed in a public document be mechanically bound to a checksummed result
file, so that a claim cannot drift from its evidence and retracting a published figure becomes an enforced
registry operation rather than a prose edit that reviewers must notice?

**What was built.**

- `artifacts/manifest.json` — one entry per public result file, each carrying `sha256`, `source_run_id`, the
  exact `generation_command`, `base_commit_sha`, `code_state` (`clean|dirty`), `code_diff_sha256`,
  `model_id`, `data_sha256`, `configuration`, `status`, and a six-valued `evidence_classification`
  (`reproduced | traceable_not_rerun | estimated | session_only_uncommitted | unsupported | stale`). Schema
  in `src/epibudget/artifacts.py:19-50`.
- `artifacts/claim_map.json` — each public numeric claim bound to an artifact, a JSON pointer, an allowlisted
  deterministic transform (`identity | round | ratio | grouped`), the exact rendered string, and the README
  anchor text it must appear inside. Plus a `forbidden_literals` blocklist.
- `src/epibudget/provenance.py` — code-diff hashing over a path-ordered tracked+untracked working-tree delta
  (generated `artifacts/` and `report/` excluded so a manifest never hashes itself), and create-only atomic
  JSON publication via `os.link` rather than `os.rename`.
- `src/epibudget/scored_cache.py` — `CacheMetadata` (frozen) pins a resumable ESM score cache to model,
  WT/candidate hashes, candidate count, alphabet, max order, scorer seed and perturbation count via an
  immutable sidecar; `CacheIdentity` re-derives all eight fields independently rather than trusting the
  sidecar, and both expected and observed sides are serialized into run provenance.
- `scripts/validate_artifacts.py`, wired into `scripts/hooks/pre-commit:48` and
  `.github/workflows/ci.yml:41`.

**What was measured.** On every commit and every CI build: manifest schema validity, SHA-256 of every listed
artifact against the file on disk, JSON-pointer resolution plus transform rendering of every claim against
its artifact, presence of each rendered string inside its declared README anchor, and absence of every
`forbidden_literals` entry from `README.md` and all of `docs/**/*.md`. On every analysis run: expected vs
observed cache identity.

**Outcome.**

- Clean at HEAD: `python scripts/validate_artifacts.py` exits 0, silent.
  `pytest tests/test_artifacts.py tests/test_provenance.py tests/test_scored_cache.py` — 36 passed.
- It was load-bearing at the moment it mattered. Commit `943833d` executed the comparative-claim retraction
  as a registry operation: `artifacts/claim_map.json` went from 41 claims to 17 (**exactly 24 removed, none
  added**) — all eighteen `headline.{structural,info,fitness,random}_{sp,pe}_b{48,96,192}` entries plus the
  six `precision.*` common-support entries — and `artifacts/manifest.json` updated the `claim_map.json` SHA
  (`ee847fe8…` → `89ae2b1e…`) in the same commit, so registry and prose could not diverge.
  `claim_map.json` −216 lines.
- The lock is bidirectional and both directions are verifiable in the validator source: deleting a README
  sentence without deleting its claim fails the anchor check (`src/epibudget/artifacts.py:198-201`); editing
  `claim_map.json` without re-hashing it in the manifest fails the checksum check (`:175-179`), because the
  claim map is itself a listed artifact.
- `artifacts/headline_650m.json` and `artifacts/robustness_650m.json` remain **listed** in the manifest as
  historical evidence but are cited by **no** current claim — the 17 surviving claims resolve only to
  `calibration_650m.json` (7), `calibration_35m.json` (7), `signal_650m.json` (2) and `dataset_gb1.json` (1).
- `forbidden_literals` holds three strings — the superseded uncalibrated masking-dispersion
  Spearman values (35M / 650M) and their sample size, published in `5a56314` and now permanently blocked from
  any public Markdown file. Present since the mechanism's first commit `188b5ae`.
- Honest self-labelling rather than flattery: the manifest declares `provisional: true` and
  `requires_remanifest_after_commit: true`; all 11 entries carry `code_state: "dirty"` with a
  `code_diff_sha256`; and of the 11, only `dataset_gb1.json` and `claim_map.json` claim `reproduced` —
  **every empirical result is `traceable_not_rerun`**, i.e. copied unchanged from audited local run files
  rather than re-executed at the manifest commit.
- Cache identity held under real use: the GB1 downstream run serializes `scored_cache_identity_expected` and
  `..._observed` and they match field for field (650M, seed 0, 16 perturbations, 29,678 candidates, full
  20-letter alphabet, max_order 3). The serialization contract is committed (`specs/downstream.md`, "Cache
  integrity"), but the run output carrying the matching values is an untracked local report.

**Why this verdict.** Kept, because it converted the project's negative results from stated to enforced. A
stale or retracted number in the README is a build failure, not a reviewer's catch. Three residual gaps are
recorded rather than papered over, and all three are confirmable from the source:

1. `validate.Report` carries no commit field, so a run's SHA is not recoverable from its artifact. The GB1
   headline works around this by stamping `configuration.colab_commit` (`3ba75ebbe70…`, `LIMITATIONS.md` §6);
   `robustness_650m.json` has no such field, and the TrpB smoke report has none either —
   `experiments/trpb-smoke-20260713.md:38` states plainly that its expected SHA `42ef311…` "cannot be
   confirmed against the JSON", and that reproducibility at HEAD is established from Git, not from the
   artifact.
2. The check is one-directional. `validate_public_artifacts` iterates over registered claims, never over
   numbers found in the README, so a *newly introduced* unregistered number is caught only if it happens to
   match a `forbidden_literals` entry.
3. One working document is excluded from the forbidden-literal scan (`src/epibudget/artifacts.py:205`).

**Evidence.**
- `src/epibudget/artifacts.py` (schemas :19-80; checksum gate :175-179; claim rendering :186-197; anchor gate
  :198-201; forbidden-literal scan :203-216).
- `src/epibudget/provenance.py`; `src/epibudget/scored_cache.py:40-79`.
- `artifacts/manifest.json`; `artifacts/claim_map.json`; `scripts/validate_artifacts.py`;
  `scripts/hooks/pre-commit:48`; `.github/workflows/ci.yml:41`.
- `tests/test_artifacts.py` (rejects changed checksum, documented-number mismatch, forbidden historical
  number, non-allowlisted transform, path escaping the repository).
- Commits `188b5ae` (mechanism introduced, with `forbidden_literals` from day one), `bee30d7` (pre-commit +
  CI wiring), `8487cd9` and `4abec4b` (cache identity + atomic provenance), `943833d` (the retraction),
  `5a56314` (the now-forbidden values as originally published).
- `LIMITATIONS.md` §5-§6; `experiments/trpb-smoke-20260713.md:38-44`; `specs/downstream.md` "Cache integrity"
  / "Provenance" (:524-545).

**Gap.** The run output demonstrating expected-vs-observed cache-identity agreement is an untracked local
report; only the serialization contract is committed.

## GPU acceleration without a GPU-specific code path

- **Verdict:** kept

**Question.** The full 20-letter, variance-inclusive 650M pass (29,678 candidates × 16 masking perturbations)
is not practically CPU-tractable, but a GPU-only implementation would break the offline test suite and the
reproducible default path. Can GPU execution be added as a pure configuration knob — and if GPU output is not
bit-identical to the CPU reference, what evidence is allowed to validate it?

**What was built.** A single scoring implementation with a device parameter, no CUDA-specific branch:

- `src/epibudget/scoring.py:47-57` — `resolve_device()`: `"auto"` resolves to `"cuda"` when
  `torch.cuda.is_available()` else `"cpu"`; `"cpu"` (the default) and explicit `"cuda"` pass through.
  Resolution happens once at model load (`scoring.py:116`), after which tensor placement is the only
  device-dependent code (`scoring.py:121,148,163`).
- `ConjointScorer.__init__(..., device: str = "cpu", ...)` (`src/epibudget/scoring.py:80-91`; interface
  mirrored in `SPEC.md:129-136`). `SPEC.md:345` lists "Any GPU-specific path" as a permanent v1 non-goal;
  `SPEC.md:347` states v1 is a "CPU-first, GPU-capable CLI + library".
- CLI exposure as an option only: `src/epibudget/cli.py:188` and `:258`, `--device cpu|cuda|auto`, defaulting
  to `cpu`.
- Device recorded as provenance, not just as a runtime flag: it is a field of the frozen cache identity
  (`src/epibudget/scored_cache.py:54,126`) and `_ensure_metadata` (`scored_cache.py:133-147`) raises on any
  field mismatch — so resuming a cache under a different device is rejected rather than silently mixing
  devices within one scored set.
- `scripts/bench_scoring.py` — measures reference vs optimized throughput on the *assigned* hardware and
  records `max_abs_delta_g_gap` / `max_abs_var_delta_g_gap` alongside it
  (`bench_scoring.py:66,103,115-119`).
- `headline_650m_colab.md` — the GPU execution recipe; Cell 4 (`:60-69`) measures throughput on the
  actually-assigned GPU and extrapolates an ETA over the 29,678-candidate pool instead of quoting a
  hardware-name-based guess.

**What was measured.**

- *Implementation parity (CPU only).* `tests/test_scoring.py:149-184`
  (`test_optimized_batch_matches_reference`, `@pytest.mark.slow`) runs the de-duplicated batched path against
  the per-variant `score` oracle on a real GB1 slice, single-threaded so CPU BLAS is batch-invariant. It
  asserts a tight tolerance on `delta_g` and `var_delta_g`, then — the assertion that actually carries the
  claim — that the info-optimal selection and the fitness-greedy selection built from either path are
  identical sets at budgets 8 and 16 (`:182-184`).
- *Throughput.* `artifacts/bench_650m.json`: 650M, alphabet `AC`, 64 variants, `n_perturbations=4`, batch 32,
  12 CPU threads.
- *Device resolution.* `tests/test_scoring.py:79-85` asserts pass-through and that `auto` tracks
  `torch.cuda.is_available()` — so the suite passes with or without a GPU present.

**Outcome.**

- Kept as an optional accelerator. The frozen headline executed on a Colab T4:
  `artifacts/headline_650m.json` records `result.device = "cuda"`, `n_candidates = 29678`,
  `n_perturbations = 16`, `seeds = 20`, `model_id = esm2_t33_650M`, alphabet `ACDEFGHIKLMNPQRSTVWY`.
- Throughput artifact (CPU, 650M): 0.127 reference variants/s vs 0.171 optimized variants/s, speed-up 1.341x,
  with `max_abs_delta_g_gap = 0.0` and `max_abs_var_delta_g_gap = 0.0`.
- No complete CPU duration for the frozen run is published, and CPU tractability of that configuration is
  explicitly not claimed (`LIMITATIONS.md:15-17,34-37`; `headline_650m_colab.md:7-11`).
- Cross-device floating-point identity is stated as not assumed (`headline_650m_colab.md:13-14`): the parity
  test covers optimized-vs-reference CPU scoring only.

**Why this verdict.** Separating the capability (faster scoring) from the mechanism (CUDA) is what makes this
keepable. Because the device is a parameter and not a branch, the default path stays CPU, the offline test
suite runs unchanged on machines with no GPU, and there is no second implementation whose numerical behaviour
could drift from the reference. The non-obvious part is the validation rule: the CPU parity test cannot be
the acceptance criterion for a GPU run, since a different BLAS makes GPU output non-bit-identical to the
parity oracle by construction. The evidence admitted for GPU execution is therefore throughput plus identity
of the downstream *selection sets* — the decision-level output the tool actually emits — rather than bitwise
agreement of intermediate scores. Recording the resolved device inside the cache identity is the enforcement
that keeps this honest: a cache cannot be half-CPU and half-GPU without raising.

**Evidence.**
- Commit `c7b7228` — "perf(scoring): batch and de-duplicate masked forwards; add GPU device"
  (`src/epibudget/scoring.py`, `tests/test_scoring.py`; +244/−20).
- `src/epibudget/scoring.py:47-57, 72-91, 116, 121, 148, 163`; `src/epibudget/cli.py:188, 258, 334`;
  `src/epibudget/scored_cache.py:40-57, 115-130, 133-147`.
- `tests/test_scoring.py:79-85, 149-184`; `scripts/bench_scoring.py:66, 103, 115-119`.
- `artifacts/headline_650m.json` (`result.device = "cuda"`, 29,678 candidates, `n_perturbations = 16`);
  `artifacts/bench_650m.json` (0.127 → 0.171 variants/s, 1.341x, gaps 0.0).
- `LIMITATIONS.md:15-17, 34-37`; `SPEC.md:123-125, 129-136, 345, 347`;
  `headline_650m_colab.md:7-14, 60-69, 122, 127-130`.

**Gap.** The substitute-validation rule — throughput plus selection-set identity in place of the CPU parity
test, and why — has no committed home; the committed docs state the weaker fact that cross-device identity is
not assumed, without naming what is accepted instead.

## networkx and scikit-learn as runtime dependencies

- **Verdict:** abandoned

**Question.** The project scaffold declared `networkx>=3.3` and `scikit-learn>=1.5` as runtime dependencies
on the assumption that the epistasis factor graph would need a graph library and that coefficient inference
would need a general-purpose ML toolkit. Does either assumption hold once the modules are actually written?

**What was built.** Neither dependency was ever imported. `src/epibudget/graph.py` (`EpistasisFactorGraph`)
implements the linear-Gaussian factor graph over interaction terms using only the standard library — its
complete import list is `collections.Counter`, `collections.abc.Mapping/Sequence`, plus two internal modules
(`epibudget.epistasis`, `epibudget.types`). The structure it needs is a per-variant incidence count over
interaction loops, not a general graph object, so no graph library earns its place. Later,
`src/epibudget/coeff_recovery.py` hand-rolls the compressed-sensing estimator in numpy: a warm-started
coordinate-descent LASSO along a descending lambda path with active-set-restricted sweeps and a KKT
stationarity check (`_soft_threshold`, `_cd_lasso_path`, `_fourier_lasso_fit`), with lambda chosen by K-fold
CV — covering the one place scikit-learn would plausibly have been used.

**What was measured.** An import audit across the source tree, and a re-audit at the current head:
`grep -rn "networkx\|sklearn\|scikit"` over all `.py`, `.toml`, `.cfg`, `.txt` files and the notebooks
returns exactly one hit repository-wide — the explanatory docstring line `src/epibudget/coeff_recovery.py:7`
("Pure numpy, no scikit-learn"). A history-wide search (`git log --all -S`) for `import networkx`,
`import sklearn` and `from sklearn` returns zero commits on any branch: the two packages were declared at
scaffold time and never once imported.

**Outcome.** Both were removed from `[project.dependencies]`, along with their now-pointless
`ignore_missing_imports` mypy overrides (`networkx.*`, `sklearn.*`). The three survivors in that edit were
re-annotated with the reason they are actually needed — `scipy` for correlations and the bootstrap in the
validation harness, `pandas` for GB1 CSV loading, `torch` for CPU execution. Net change to the manifest: five
lines out, three in. Runtime behaviour is unaffected, since nothing imported the removed packages.

**Why this verdict.** The removal is a truthfulness fix, not a performance one. The project's public framing
is a CPU-only, minimal-footprint tool; a heavyweight package sitting unused in the manifest is a false
statement about what the runtime requires, and it inflates install cost for every user who never benefits.
The decision was reinforced rather than reversed when `coeff_recovery.py` landed: faced with the one genuine
opportunity to pull scikit-learn back in, the estimator was written in numpy instead and the choice recorded
in the module docstring. The committed spec for that module likewise describes it as pure-numpy
(`specs/step6-coefficient-recovery.md:12-13`). No dependency-manifest audit is enforced by a test, so the
property is currently maintained by review rather than by CI.

**Evidence.**
- `38c1e270dd684d890d2d30201642cc396304fb04` — "build(deps): drop unused networkx and scikit-learn"; touches
  `pyproject.toml` only, 3 insertions / 5 deletions.
- `427c898` — initial scaffold; `git show 427c898:pyproject.toml` shows both packages in the original
  dependency list.
- `src/epibudget/graph.py:22-28` — the module's full import block (stdlib + internal only).
- `src/epibudget/coeff_recovery.py:7` — "Pure numpy, no scikit-learn."; `:270-330` `_soft_threshold` and
  `_cd_lasso_path`; `:390-431` `_fourier_lasso_fit` with K-fold CV lambda selection.
- `6ff14ce` — the commit that added `coeff_recovery.py`; `specs/step6-coefficient-recovery.md:12-13`.
- `pyproject.toml:24-36` (current dependency list) and `:66-69` (surviving mypy overrides).

## Reproducible headline-figure demo notebook

- **Verdict:** abandoned

**Question.** Can an outside reader install the package, run the frozen validation command, and regenerate
the headline figure end-to-end from the saved report — i.e. is the main scientific claim reproducible by a
stranger rather than only by its author?

**What was built.** Nothing. `notebooks/gb1_demo.ipynb` has no commit in the repository's history
(`git log --all -- notebooks/gb1_demo.ipynb` returns nothing) and does not exist on disk. Only its
scaffolding shipped: `pyproject.toml:33` declares `matplotlib>=3.9` ("figures for the notebook / report") as
a runtime dependency and `pyproject.toml:44` declares `jupyter>=1.0` ("reproducible demo notebook") as a dev
dependency, but no file in the tree imports matplotlib —
`git grep -l "matplotlib\|pyplot\|savefig" HEAD` matches `pyproject.toml` alone. No plotting code and no
rendered figure exist anywhere; `README.md` embeds no image.

What was built instead, in commit `078bedd`, is a different kind of provenance: two Colab scoring notebooks
with their cell outputs retained — `notebooks/colab/gb1_650m_n16.ipynb` (GB1, A100, `n_perturbations=16`) and
`notebooks/colab/trpb_650m_n0.ipynb` (TrpB, T4, `n_perturbations=0`). They record the GPU runs that produced
the score caches rather than rendering a figure, and they were deliberately kept as two separate notebooks
because merging runs on different hardware and different `n_perturbations` would describe a run that never
happened (`078bedd` commit message; `notebooks/README.md:3-21`).

**What was measured.** Not applicable — the deliverable was infrastructure, never executed.

**Outcome.** The notebook was overtaken by the science it was meant to display. Its specified content was
epistasis-map recovery versus budget *B* for info-optimal / fitness-greedy / random with bootstrap CIs —
precisely the comparative claim that was subsequently withdrawn. `README.md:92-95` now states "No comparative
recovery claim is current", and the standing amendment in `VALIDATION.md:18-30` records that pairwise map
recovery is weak, that its raw rank gain is largely a main-effect-sharing confound, and that no
decision-eligible comparative claim stands.

Cleanup was partial. `notebooks/README.md:23-29` now says plainly that the demo notebook is not committed and
redirects the reader to the live downstream-impact result, but two references to the non-existent file
survive in `VALIDATION.md:157` and `:161`. Those lines sit inside an explicitly frozen pre-registration
document ("This document freezes the protocol before any result exists", `VALIDATION.md:3-5`), so they read
as a record of the planned reproducibility path rather than as a current instruction — but a reader following
them still lands on a file that does not exist. A claimed reference at `SPEC.md:279` is not supported —
`git grep -n "gb1_demo" HEAD` matches only the two `VALIDATION.md` lines.

**Why this verdict.** Rendering a publication-style figure for a retracted comparison would have manufactured
credibility for a claim the project had already narrowed, so building it became the wrong move rather than
merely a deferred one. The reproducibility obligation was met by a different mechanism: a checksummed public
artifact layer (`artifacts/`, twelve JSON payloads plus `artifacts/manifest.json` recording SHA-256, source
run id, base commit SHA, dirty-tree state, a deterministic code-diff digest, and the reconstructed generation
command per artifact), verified by `scripts/validate_artifacts.py`, plus the Colab notebooks as evidence of
the GPU runs. That layer carries numbers and provenance instead of a picture, which is the stricter of the
two.

A second, quieter failure mode drove the same decision. The superseded `notebooks/README.md` told readers the
pipeline was documented in a working-configuration file that is excluded from version control — a
reproducibility pointer no outside reader could follow. Commit `078bedd` removed that dangling pointer.

The planned deliverable was an exit criterion of the project's headline-result milestone, alongside the
ablation verdict on the uncertainty prior; that same milestone had already pre-committed the fallback of
moving the signal to a downstream-impact benchmark if the uncertainty prior turned out null, which is exactly
what happened. So the notebook was not dropped ad hoc — the branch that discards it was written down before
the result was known.

**Evidence.**
- Absence: `git log --all -- notebooks/gb1_demo.ipynb` (empty); file not present in the working tree.
- Unused scaffolding: `pyproject.toml:33`, `:44`; `git grep -l "matplotlib\|pyplot\|savefig" HEAD` →
  `pyproject.toml` only.
- Replacement provenance: commit `078bedd` "docs(notebooks): add Colab GPU scoring notebooks as run
  provenance" (+806 lines across `notebooks/README.md`, `notebooks/colab/gb1_650m_n16.ipynb`,
  `notebooks/colab/trpb_650m_n0.ipynb`).
- Superseded text: `git show 078bedd^:notebooks/README.md` (describes `gb1_demo.ipynb` as the renderer of the
  recovery-vs-budget figure and points at an uncommitted configuration file). Current text:
  `notebooks/README.md:23-29`.
- Surviving dangling references: `VALIDATION.md:157`, `:161`.
- Withdrawal of the depicted claim: `README.md:92-95`; `VALIDATION.md:18-30`.
- Artifact layer that took over the burden: `artifacts/manifest.json`, `artifacts/README.md`,
  `scripts/validate_artifacts.py`.
- Live result the reader is redirected to: `README.md:99-117`, `specs/downstream.md`,
  `experiments/trpb-downstream-generalization-20260716.md`.

**Gap.** The pre-committed fallback that made discarding the notebook a planned branch rather than an ad-hoc
drop is recorded only in an uncommitted milestone document.

---

# Reasoning not yet in a committed document

Eighteen entries above carry reasoning that exists only in this record, in code comments, or in working
documents excluded from version control. Listing them keeps the gap visible instead of letting it disappear.
Each needs a committed home — a spec, a limitation entry, or an experiment note — before it can be cited by a
public reader.

| Entry | Theme | What has no committed home |
| --- | --- | --- |
| Structural-only ablation | benchmark | The per-budget seeded tie-break breakdown; the committed record states only the overall "inconclusive". |
| WT-anchor correction | inference | Why the previously serialized truth-map variance never exercised the non-additivity invariant, and why the CLI gate was re-pointed at predicted ESM epistasis while truth variance stayed descriptive. |
| Corrective zero-GPU replay (gate 2) | validation | The per-budget, per-cell dispersion-contribution and calibration breakdown; only aggregate verdicts are committed. |
| Correlated-error inference repair (gate 3) | inference | The per-budget λ-frontier and residualized-recovery cells; also, the committed spec's decision rule is stale relative to the shipped residualized criterion. |
| Compressed-sensing coefficient recovery | inference | The per-estimator numbers (0.099 / 0.175 / 0.132 / 0.118); the committed tree carries only the qualitative conclusion. |
| Isotropic D-optimal acquisition | acquisition | The per-budget recovery numbers for both the isotropic and order-restricted designs, and the `info` reference line. |
| Order-restricted D-optimal acquisition | acquisition | The per-budget numbers, and the reduced-model / order-3-aliasing caveat on the design itself. |
| Provenance and claim registry | infrastructure | The run output demonstrating expected-vs-observed cache-identity agreement; only the serialization contract is committed. |
| Sparse-Bayesian coefficient model | inference | The design sketch and the explicit deferral rationale. |
| Shared informed-union grading set | validation | The mechanism by which a union subset collapses toward the shared prior correlation — a reconstruction from code, stated nowhere. |
| Headline-figure demo notebook | infrastructure | The pre-committed fallback that made discarding the notebook a planned branch rather than an ad-hoc drop. |
| PSD95-PDZ3 generalization landscape | data | The eligibility rule for adding a landscape (measured multi-mutant data required), the candidate list, and the settle-then-generalize sequencing argument. |
| Stop rule and closure-check fallback | validation | The original stop rule and the `closure-check` pivot it named. |
| ESM-circular downstream diagnostic | benchmark | The quantified severity: the scale mismatch inverts the sign of `b` (−0.0163 vs +1.528) and collapses the diagnostic to ± the zero-shot control in 28,800 of 28,800 records. |
| GPU acceleration | infrastructure | The substitute-validation rule (throughput plus selection-set identity in place of the CPU parity test) and its justification. |
| Programme reordering | benchmark | The pre-commitment to the fallback framing adopted when the uncertainty prior came back null. |
| Label-leakage barrier | validation | The standing detection procedure for selection-path changes, and the argument for treating label leakage as the highest-severity defect class. |
| Acquisition targets uncertainty, not magnitude | acquisition | The rejected alternative (ranking by \|ε̂\|) as a deliberate, argued choice rather than an unstated default. |

Two further gaps are adjacent rather than entry-level, and are recorded in the entries themselves: the
permutation-null control has no committed implementation or retained artifact (only prose and a commit
message carry its numbers), and the zero-crossing of the downstream corrected-CV companion under the
conservative `effective_label_ratio` convention appears in no committed prose.
