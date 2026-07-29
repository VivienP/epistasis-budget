# Evaluation methodology notes

This note explains why the original map-recovery interpretation is no longer used and how the
current validation artifacts should be read. The file path is retained because historical artifacts
refer to it.

## Evidence status

| Evidence | Status |
|---|---|
| Downstream v1 artifacts | Historical observations on particular tied plates |
| Downstream v2 reruns | Local diagnostics using tie seed 0 |
| Downstream v3 | Implemented schema; no empirical artifact |
| Corrected recovery | Implemented schema; no empirical artifact |

The v1 table may be reported as historical evidence with its limitations adjacent. The v2 reruns do
not estimate performance across tie seeds and are not public result artifacts.

## Why the original recovery metric is diagnostic

The inferred and measured contrasts shared the purchased part of each interaction loop. If `M` is
the set of measured loop members and `c_T` is the inclusion-exclusion sign, both quantities contain

```text
k(S) = sum_{T in M} c_T * fitness(T).
```

Correlation could therefore increase because the same measured values appeared on both sides, even
when prediction of the unmeasured terms did not improve. The corresponding fields are now named
`raw_*_with_skeleton` and are treated as diagnostics.

Correlation alone also does not establish error reduction. The corrected schema reports

```text
relative_sse_gain = 1 - SSE(posterior) / SSE(prior)
```

and permits recovery wording only when the gain is defined and strictly positive. No tracked
corrected-recovery artifact currently supplies this result.

Method-specific affine calibration remains available for descriptive use, but it is excluded from
method-comparison claims because each selected plate would otherwise be evaluated under a different
transformation. Corrected contrasts use only method-independent calibration policies.

## Selection variability

On the four-site benchmark universes, the loop-coverage score is constant within mutation order.
`structural` therefore orders singles, doubles and triples, then resolves ties within each order. A
fixed tie seed makes a plate reproducible; it does not estimate the acquisition method over its seed
distribution.

The v1 comparator used enumeration-order ties. Each v2 diagnostic used tie seed 0. Comparing those
two resolutions does not measure tie variance. A future result about `structural` as a stochastic
method must predeclare its seed sample and aggregation rule.

Recovery schema v3 stores every realised plate and every paired contrast rather than selecting one
representative seed. Each record includes the seed, selected-plate hash, term count and term-set hash.
Bootstrap intervals over terms are labelled `term_leverage_ci95`; selection variability is reported
separately across realised plates.

## TrpB label semantics

The following concepts are distinct:

- `trainable`: a finite value in the learner's `log1p` domain;
- `positive_score`: an aggregated score strictly above zero;
- `active`: an experimental classification supplied by the source assay.

The local TrpB mirror contains one aggregated score, not the replicate-level calls required to
reconstruct the assay's activity classification. Score sign is therefore not called biological
activity. Non-positive trainable values remain valid labels, and NDCG is described as score-derived
rather than activity retrieval.

## Cache and provenance requirements

Recovery validates the scored cache against independently supplied expectations for model, reference
sequence, candidate universe, alphabet, maximum order, scorer seed and perturbation count. Outputs use
create-only atomic publication.

Downstream v3 records the command, commit, scientific working-tree state and input hashes at process
start and again before publication. Code or input drift makes the decision gates ineligible. The v2
reruns captured provenance only at completion, so their records do not establish the code state that
was loaded at process start.

## Protocol versions

| Version | Definition | Artifact status |
|---|---|---|
| `epibudget-downstream-v1` | Enumeration-order ties and legacy field names | Tracked historical artifacts |
| `epibudget-downstream-v2` | Seeded ties and corrected label accounting | Local seed-0 diagnostics |
| `epibudget-downstream-v3` | Explicit score-sign fields and start/end provenance | No artifact |

The withdrawn contrast-recovery proposal in
[`specs/prospective-amendment-2.md`](specs/prospective-amendment-2.md) governs no experiment. A future
reconstruction protocol must define one estimand, one evaluation population and explicit
design-matrix identifiability conditions before examining results.

## Public result boundary

The README may retain the checksummed v1 downstream table only as historical evidence. It does not
present v2 as promotable, infer biological activity from score sign, claim robustness over unsampled
tie seeds or report corrected-recovery numbers without a tracked artifact.

A new promotional result requires a fixed clean commit, stable input hashes, complete expected
coverage, a preregistered seed rule, independent recomputation of the decision gates and artifact
validation.
