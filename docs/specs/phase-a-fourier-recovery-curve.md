# TrpB pairwise Fourier recovery curve

## Status and scope

This protocol is frozen before inspecting any TrpB recovery metric. It implements the narrow diagnostic
exception in `docs/SPEC.md` section 11 and does not change the v1 WT-referenced interaction model.

The report is always `public_claim_eligible=false`. A clean, stable run may be eligible only for the
internal architecture decision defined below. It cannot support a README result, a biological claim, or
promotion of background-averaged coefficients into the public API.

## Dataset and fixed estimand

- Dataset: the redistributed complete TrpB target, 160,000 rows over the four registered sites.
- Training signal: `y(a) = log1p(fitness(a))` for every genotype `a`.
- Caveat: the target contains 159,129 measured values and 871 source-imputed values whose identities are
  not exposed by the mirror.
- Basis: the existing per-site orthonormal contrast basis, with the WT residue at index zero.
- Normalized character: `psi_m(a) = sqrt(N) * product_s B_s[m_s, a_s]`, where `N = 20^4`.
- Coefficient: `c_m = mean_a y(a) * psi_m(a)`. With this convention,
  `Var(y) = sum_{m != 0} c_m^2`.
- Primary population: the canonical 2,166 modes with exactly two non-zero site indices.

The fitted model contains an unpenalized intercept plus all 76 order-1 and 2,166 order-2 characters.
Orders 3 and 4 are omitted residual structure and may alias into the estimated pairwise coefficients.
This reduced-model limitation is part of the result, not corrected after inspection.

## Candidate and cache contract

The selectable universe is the canonical set of all 29,678 non-WT variants of orders 1 through 3 over
the full 20-letter alphabet: 76 singles, 2,166 doubles, and 27,436 triples. The runner independently
enumerates this universe and validates the score cache with `validate_cache_against_universe`, including
model, WT sequence, alphabet, maximum order, scorer seed, perturbation count, candidate count, candidate
hash, and exact identity set.

The registered cache is `facebook/esm2_t33_650M_UR50D`, scorer seed 0, and 16 masking perturbations.
Cached scores must retain the conjoint-scoring non-additivity guard. No model forward pass, GPU, or network
access belongs to this phase.

## Acquisition methods

Every acquisition sequence is built to budget 3,072 before the landscape is accepted by any fitting or
scoring function. Smaller plates are exact prefixes at budgets
`(48, 96, 192, 384, 768, 1536, 2242, 3072)`.

- `info`: the existing dispersion-weighted loop-coverage score, with a fail-closed preflight if an exact
  score tie crosses a registered budget.
- `fitness`: descending conjoint ESM score, under the same tie preflight.
- `random`: uniform seeded permutations, seeds 0 through 19.
- `structural`: loop coverage with independently seeded within-stratum permutations, seeds 0 through 19.
- `doptimal_reduced_pairwise`: greedy posterior predictive variance for the reduced order-1-plus-order-2
  character model, coefficient prior `N(0, I)`, observation variance 1 after the registered character
  normalization. Exact acquisition ties use the lexicographically smallest SHA-256 of the canonical
  variant identity. This is a deterministic reduced-model comparator, not pairwise `D_s`-optimality and
  not protection against higher-order aliasing.

Selection functions cannot accept a measured landscape or measured label. Only after the complete
selection plan and every selected-identity hash are frozen may orchestration call
`reveal_measured_fitness` and apply `training_target`.

## Estimator

The sole registered primary estimator is Fourier LASSO:

`min_(alpha,c) mean((y - alpha - Zc)^2) + lambda * ||c||_1`.

The population-normalized characters are not rescaled from the selected plate. For each of five folds,
the response mean and every character mean are computed from training rows only. The intercept is
unpenalized by fitting centered training characters and applying those same training means to validation
rows. Then `lambda_max = 2 * max(abs(centered_Z_train.T @ centered_y_train)) / n_train`. The 20 fixed
ratios are `geomspace(1, 1e-3, 20)`. Each ratio therefore denotes `ratio * lambda_max` within its fold.
Select the ratio with the smallest summed held-out SSE, breaking an exact tie toward the larger penalty,
then refit the full plate with its own response and character means and `lambda_max` at that ratio.

Optimization uses warm-started FISTA with deterministic backtracking. Each lambda must reach a maximum
active/inactive KKT residual no greater than `1e-5 * max(1, lambda)` within 5,000 iterations; otherwise
the fit is non-converged and fails closed.

Fold identity is `variant_fold(variant, 5)`. A missing train or validation fold, non-convergence, a
constant response, or a non-finite intermediate fails the cell closed. No estimator may replace a
fitted value or coefficient with a measured truth value.

## Metrics and records

Each method, budget, and realized seed retains one raw record containing selected identity/hash, fold
hash, chosen lambda ratio and value, convergence status, pairwise support at absolute threshold `1e-12`,
and:

- Spearman correlation across the same 2,166 truth and estimated pairwise coefficients;
- relative SSE gain `1 - SSE(c_hat_2, c_2) / SSE(0, c_2)`;
- evaluated coefficient count, always 2,166.

Spearman is unavailable when either vector satisfies
`ptp(x) <= 1e-12 * max(1, max(abs(x)))`. Relative SSE gain is unavailable when the zero-prior SSE is
zero. Unavailable values remain JSON null and never become zero.

Stochastic methods retain all 20 records and aggregate by median with `n_valid`, minimum, maximum, and
the fraction of positive SSE gains. Deterministic methods retain their single record.

## Internal decision

A method-budget cell justifies a separate Stage B confirmation only if:

- median pairwise Spearman is at least 0.30;
- median relative SSE gain is at least 0.10; and
- a stochastic method has 20 valid SSE gains, at least 16 of them positive.

For a deterministic method, the point Spearman and SSE thresholds must hold and its SSE gain must be
positive. If no cell passes through budget 3,072, the result is `estimator_family_stopped`. Every attempted
cell remains in the report. This gate is an engineering resource decision, not statistical significance.

## Runtime preflight

Before any TrpB recovery metric is computed, a synthetic, label-independent benchmark times design
construction and one complete registered fit at all eight registered budgets. The sum of those eight
measurements is multiplied by the exact 43 fits per budget, so the projection covers all 344 registered
fits. It records design bytes, wall time, convergence, the candidate-universe hash, and clean matching
start/end repository snapshots. A D-optimal pilot builds the full-pool prefix through budget 48 and
projects its `O(N * B^2)` cost to budget 3,072. The real curve is not scheduled if projected wall time
exceeds eight hours or the dense D-optimal update matrix exceeds 2 GiB. A stale, dirty, incomplete, or
commit-mismatched preflight cannot authorize the curve. These are local engineering limits, not
scientific thresholds. No Colab or GPU run is required.

## Provenance and failure policy

The report uses create-only atomic publication and captures dataset, cache, sidecar, candidate-universe,
selection, fold, commit, code-diff, argv, and start/end hashes. Input or workspace drift, cache mismatch,
duplicate identity, missing row, wrong WT residue, off-target mutation, non-finite value, value outside the
`log1p` domain, incomplete Fourier grid, unresolved score tie, prefix failure, or missing seed makes the
architecture decision ineligible. The raw diagnostic may still be retained with the failure reason.

GB1 remains untouched. No recovery result from this phase updates README, public artifacts, or the v1
interaction semantics.
