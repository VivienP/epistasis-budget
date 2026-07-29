# Audit remediation — 2026-07-28

An independent mathematical audit found two critical and four high-severity defects in the
map-recovery evaluation, tie handling, calibration comparison, and TrpB label accounting. This
record distinguishes implementation corrections from empirical results. It does not promote a new
result.

## Current status

The original map-recovery interpretation is withdrawn. The tracked downstream v1 results remain
historical observations on GB1 and TrpB, not estimates over the acquisition method's tie
distribution. The later v2 reruns are retrospective, single-tie-seed diagnostics whose recorded
provenance describes the tree at process completion rather than the code loaded at process start.
They are not public artifacts. No downstream v3 or corrected-recovery artifact exists.

| Evidence class | Current standing |
|---|---|
| Downstream v1 artifacts | historical, retained for traceability |
| Downstream v2 reruns | local diagnostic only; one seeded plate per landscape |
| Downstream v3 | implementation only; no empirical artifact |
| Corrected recovery | implementation only; no empirical artifact |

Every corrective analysis of GB1 or TrpB is retrospective. It cannot be made prospective by
renaming it, rerunning it, or moving it into the README.

## C-1 — shared-skeleton confounding

The former map-recovery correlation compared a predicted contrast with the measured contrast after
copying purchased lower-order labels into both quantities. If `M` is the measured part of a loop,
`U` its unmeasured part, and `c_T` the inclusion-exclusion sign, then both sides contain

```text
k(S) = sum_{T in M} c_T * fitness(T).
```

Consequently, correlation could increase because the same measured skeleton was present on both
sides, even when prediction of the unmeasured terms did not improve. The former correlation is now
named `raw_*_with_skeleton` and retained only as a diagnostic.

The corrected recovery schema reports the skeleton association, skeleton-controlled partial
association, term census, and relative SSE gain separately. It also includes the model-free
`singles_zero_prior` baseline. None of these fields is currently backed by a tracked corrected-run
artifact, so this document records no corrected recovery number.

## C-2 — correlation was not an error-reduction gate

A correlation can rise while squared prediction error worsens. The corrected schema therefore
reports

```text
relative_sse_gain = 1 - SSE(posterior) / SSE(prior)
```

for each realised method, budget, seed, calibration policy, subset, and interaction order. Recovery
wording is not permitted when this quantity is undefined or non-positive. This is an implementation
rule, not a claim that any method passes it on either landscape.

## H-1 — tied structural scores

On the four-site benchmark universes, the loop-coverage score is constant within mutation order.
`structural` therefore means singles, then doubles, then triples, plus a within-order tie resolution;
it is not a complete biological ordering.

Tie handling is now seeded and invariant to candidate enumeration order for a fixed seed. A fixed
seed makes a run reproducible but does not estimate the method over its seed distribution. The
historical v1 comparator used enumeration-order ties, while each v2 diagnostic used only tie seed
zero. Agreement or disagreement between those two resolutions is not an estimate of tie variance.

A future promotional protocol that interprets `structural` as a stochastic acquisition method must
predeclare how seeds are sampled and aggregated. That empirical study is deferred; it is not needed
to retain the explicitly historical v1 table or to publish these implementation corrections.

## H-2 — method-specific calibration

Fitting a separate affine calibration to each selected plate changes both the data and the
transformation being compared. `per_method` calibration remains available as a descriptive
operational policy, but it is excluded from method-comparison claims.

Corrected recovery contrasts use only method-independent policies:

- `zero_prior`, which assigns zero to unmeasured terms;
- `fixed_unit`, which uses a fixed slope and intercept shared by all methods.

The policy is part of every record identity and every contrast identity.

## H-3 — trainability, score sign, and biological activity

These concepts are now separate:

- `trainable`: a finite fitness value in the domain required by the learner;
- `positive_score`: an aggregated score strictly above zero;
- `active`: an experimental classification supplied by the source assay.

The local TrpB mirror contains an aggregated score, not the two replicate-level activity calls
required by the published assay definition. The sign of that aggregate must therefore not be called
biological activity. Non-positive, trainable TrpB rows remain valid labels. Downstream schema fields
formerly named `active` or `live` are now named `positive_score`, and score-sign fractions remain
descriptive only.

NDCG uses a score-derived relevance transformation. It is not an activity-retrieval metric. No
count of positive TrpB scores may be presented as a count of active variants.

## H-4 — selection uncertainty and term leverage

The earlier emitter selected `draws[0]` as a representative plate and bootstrapped terms from that
single selection. That interval measured term leverage conditional on seed zero while being
presented as a method contrast.

Recovery schema v3 instead emits one method record per realised plate and one paired contrast per
realised plate pair. Pairing is defined before evaluation:

- deterministic versus seeded: broadcast the deterministic plate across realised seeds;
- the same seed mechanism on both sides: pair matching seeds;
- different stochastic mechanisms: use the Cartesian product of realised seeds.

Each record carries the seed kind, seed value, selected-plate identity hash, term count, and term-set
hash. The bootstrap interval is named `term_leverage_ci95` because it conditions on the selected
plates. Selection variability is summarized separately across realised plate pairs. No
`draws[0]` result is promoted to a method-level contrast.

## Cache identity and output publication

The recovery emitter validates cache identity against independently supplied expectations before
using any score. Validation covers model identifier, wild type, candidate-universe identity,
alphabet, maximum order, seed, and perturbation count. Same-size candidate permutations and extra
or missing entries fail closed.

Recovery output uses exclusive atomic publication. Reusing an existing destination fails without
altering the previous file.

## Provenance

The earlier downstream reruns captured repository state at process completion. Edits made during a
run could therefore change the recorded diff even though Python had loaded the original modules.
Their provenance is unsuitable for promotion.

Downstream v3 now captures at process start:

- the actual argument vector and a re-executable Windows command line;
- commit, repository state, scientific diff hash, and changed scientific files;
- input dataset and cache hashes.

The same repository and input states are captured again before publication. Any code drift or input
drift makes both decision gates ineligible and clears their support values. Recovery uses the same
start/end discipline and records the expected and observed cache identities.

## Protocol versions

| Version | Definition | Artifact status |
|---|---|---|
| `epibudget-downstream-v1` | registered profile with enumeration-order ties and legacy field names | tracked historical artifacts |
| `epibudget-downstream-v2` | seeded tie resolution and corrected label accounting | local seed-zero diagnostics only |
| `epibudget-downstream-v3` | truthful score-sign field names and start/end provenance gates | no artifact |

Version v3 is a schema and provenance correction. It does not itself add a tie-seed distribution,
change the frozen partition profile, or create a new empirical result.

## Withdrawn prospective amendment

`docs/specs/prospective-amendment-2.md` is withdrawn. It registered `fitness - random` while its
interpretation referred to `structural - fitness`, required a common evaluation set while defining
plate-dependent evaluability, and treated a ridge-regularized coefficient as identified from
insufficient observations. Ridge can return a unique fitted value without making a coefficient
identifiable from the data.

Contrast reconstruction remains diagnostic for the current method. A future method designed to
identify higher-order contexts needs a separate protocol whose estimand and design-matrix
identifiability conditions are registered before any result is observed.

## Public-claim boundary

The README may report the checksummed v1 table only as historical evidence, with its limitations
adjacent. It must not report the v2 reruns as promotable, corrected recovery numbers without tracked
artifacts, biological activity inferred from score sign, or robustness over tie seeds that were not
sampled.

A new promotional result would require a clean, fixed commit; stable input hashes; a preregistered
tie-seed sampling and aggregation rule; complete expected coverage; independent recomputation of the
decision gates; and artifact validation. Running such a study is deferred. It is not a prerequisite
for publishing the present code corrections and the narrowed historical wording.

## Verification required before release

The corrective change is releasable only after all of the following pass on the exact tree being
committed:

- the complete offline test suite;
- Ruff format and check on `src/`, `tests/`, and `scripts/`;
- strict mypy on `src/` and the changed recovery emitter;
- public-artifact and claim-map validation;
- `git diff --check`;
- a staged-diff review that excludes unrelated worktree changes.

Passing these software gates validates the implementation and documentation consistency. It does
not supply the deferred downstream v3 or corrected-recovery empirical evidence.
