# Protocol amendment 2 — withdrawn contrast-recovery estimand

Status: **WITHDRAWN on 2026-07-28, before any run on a third landscape.**

This amendment does not govern any experiment and licenses no result. It was registered after the
independent audit, then withdrawn the same day when counter-review found that its primary estimand
was not coherent. The registered text remains recoverable from Git history; this file records the
current disposition so obsolete clauses cannot be mistaken for an active protocol.

## Why it was withdrawn

### 1. Regularized uniqueness was mistaken for identification

The proposal treated a pair coefficient as identified once that pair appeared inside one measured
higher-order variant. In a reference-coded main-effects-plus-pairwise model, one measured triple
contributes one equation involving several main and pair coefficients. It does not identify each
coefficient separately.

Ridge can return a unique fitted value that depends on the observed response, basis, and penalty.
That uniqueness under regularization is not identification from the data. A future protocol must
state an explicit design-matrix condition and verify it on every evaluated term.

### 2. The registered contrast and its interpretation disagreed

The primary cell compared `fitness` with `random`, while the claimed interpretation referred to
mutation-order coverage, represented by `structural`. A preregistration cannot define one contrast
and interpret another.

### 3. The evaluation population was contradictory

One clause required the compared methods to use an identical term set. Another defined whether a
term was evaluable from each selected plate, making the term set method-dependent. Both conditions
cannot hold simultaneously.

### 4. The error gate treated an exact null as recovery

The proposal allowed `relative_sse_gain >= 0`. An unchanged estimator has gain exactly zero and has
recovered nothing. Current corrective code permits recovery wording only for a strictly positive,
defined gain; this remains a diagnostic implementation rule, not a registered primary result.

## Current scientific boundary

Contrast reconstruction remains diagnostic for the present method. The historical downstream
variant-ranking benchmark asks a different question and retains its own tie-distribution and
provenance limitations. No replacement primary contrast-recovery estimand is registered.

A future protocol would need, before any result is observed:

- one unambiguous method contrast and matching interpretation;
- a common evaluation population or a predeclared method-specific estimand that is not compared as
  though it were common;
- term-level design-matrix identifiability checks;
- a strictly positive error-reduction gate;
- predeclared selection-seed sampling and aggregation;
- paired uncertainty with term leverage separated from selection variability;
- complete cache, input, code, and command provenance.

See [`../AUDIT_REMEDIATION_20260728.md`](../AUDIT_REMEDIATION_20260728.md) for the remediation record
and [`downstream.md`](downstream.md) for the separate downstream protocol.
