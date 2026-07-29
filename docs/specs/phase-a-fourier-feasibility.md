# Fourier reconstruction feasibility

## Decision

The project will test the unverified sparsity premise before investing in another acquisition method or
an expensive benchmark. This work is diagnostic and cannot support a public reconstruction claim.

The first deliverable is a spectrum audit of the complete TrpB mirror. A recovery-versus-budget study is
conditional on that audit and has a separately fixed estimand. GB1 remains untouched as a possible
confirmation landscape.

## Data contract

- Dataset: `trpb_johnston2024` at the four registered sites.
- Signal: `labels.training_target(fitness)`, namely `log1p(fitness)`.
- Grid: all `20^4 = 160,000` redistributed rows.
- Caveat: 159,129 values were measured and 871 were imputed by the source analysis. The mirror does not
  identify the imputed rows. Reports must therefore call this the spectrum of the redistributed TrpB
  target, not a fully measured or true experimental spectrum.
- Every input file and scored cache is identified by SHA-256 before computation.

## Stage A0: spectrum audit

Compute the complete multiallelic Walsh transform without changing the existing basis or transform.
Report:

- Parseval residual between the target variance and non-constant coefficient energy;
- variance contribution for orders 1 through 4;
- coefficient-magnitude quantiles by order;
- the smallest coefficient counts carrying 90%, 95%, and 99% of non-constant energy, overall and by
  order;
- coefficient counts and finite-value checks.

All calculations are pure and deterministic. The JSON report is non-decision-eligible, uses create-only
atomic publication, and captures input and repository state at process start and end.

Stage A0 does not prove sparse recoverability. It answers only whether the premise is plausible enough to
justify Stage A1. A dense pairwise spectrum is a valid stop result.

## Stage A1: conditional recovery curve

Stage A1 must not use the withdrawn plate-dependent population of WT-referenced contrasts. A structural
plate contains every single mutant once the budget exceeds 76 and every double mutant once it exceeds
2,242; the proposed "entire loop unmeasured" population would therefore be empty at most registered
budgets.

The fixed primary estimand is instead recovery of the 2,166 order-2 coefficients in the same
background-averaged Fourier basis used by the estimator. Every method is scored on this identical
coefficient population. Fitted values are never overwritten with measured labels.

Primary metrics:

- Spearman correlation, unavailable when either vector is effectively constant;
- relative SSE gain against the all-zero coefficient prior;
- recovered support size and coefficient count.

The registered budgets are `48, 96, 192, 384, 768, 1536, 2242, 3072`. Acquisition sequences are built
once at the maximum budget and prefixes are reused. `structural` and `random` use seeds 0 through 19 and
retain every seed-level record. `info` and pairwise D-optimal are deterministic only if a preflight proves
that no unresolved score tie crosses a budget boundary.

A Stage B confirmation study is justified only if the same method and budget achieve all of:

- median pairwise Spearman at least 0.30;
- median relative SSE gain at least 0.10;
- positive relative SSE gain in at least 16 of 20 realised seeds for a stochastic method.

These are engineering thresholds for deciding whether to spend more compute, not public significance
claims. If no method passes by budget 3,072, this estimator family stops. Every attempted budget remains
in the diagnostic artifact.

## Deferred alternatives

Global-epistasis correction is not part of this phase. Choosing between isotonic and sigmoid links after
observing TrpB would add an unregistered modelling degree of freedom. Any latent-scale model requires a
new fixed algorithm, validation split, and registered attempt.

Mutual-coherence bounds are also outside the critical path. They are sufficient rather than necessary,
usually vacuous in an underdetermined sampled design, and do not establish failure when violated.

## Implementation boundary

The first implementation lot contains only:

- `src/epibudget/sparsity.py` for coefficient-energy summaries and effective sparsity;
- `src/epibudget/spectrum_diagnostic.py` for the validated report and start/end provenance;
- `tests/test_sparsity.py` with complete synthetic landscapes and Parseval oracles;
- `scripts/spectrum_diagnostic.py` as the thin command-line entry point.

It does not modify allocation, downstream, recovery, README claims, or tracked result artifacts.

## Acceptance

- Synthetic one-coefficient landscapes return the injected support and order exactly.
- Effective-sparsity counts are invariant to uniform rescaling and reject non-finite inputs.
- Per-order energy sums to the non-constant target variance within a documented floating-point tolerance.
- Real TrpB execution is explicit and data-dependent; default tests remain offline.
- Ruff, strict mypy, targeted tests, artifact validation, and `git diff --check` pass before completion.
