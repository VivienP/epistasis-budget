# Downstream-impact benchmark — GB1 confirmatory result and TrpB generalization

**Status.** GB1: decision-eligible, `structural_downstream_supported = true`. TrpB: **exploratory,
non-decision-eligible** (scored at `n_perturbations = 0`), a direction-replication only. Both artifacts
are `status = provisional` and live under the git-ignored `report/`; neither is a registered public
artifact. See [`docs/specs/downstream.md`](../specs/downstream.md) for the frozen protocol.

## Question (the estimand)

At equal initial budget *B*, does a method's selected plate provide a better training set for a fixed
pairwise-ridge learner to rank **held-out** double/triple mutants? Primary statistic
`S_macro = ½(ρ_doubles + ρ_triples)`; the learning-curve AUC contrast is aggregated over the 20 salted
partitions. The predictor reads only revealed fitness labels — never the held-out variant's ESM score or
the prior-inclusive `infer_epistasis` output — so it cannot recover the ESM prior algebraically.

## GB1 — confirmatory (decision-eligible)

`report/20260715T111312Z/downstream.json`, R=20 × K=5 × 20 seeds, ESM-2 650M, `n_perturbations = 16`,
budgets {48, 96, 192}, target-blind / attempted-budget.

- **structural − fitness: 20/20 partitions positive, S_macro-AUC mean +0.342** → the 7-point robustness
  gate passes; `structural_downstream_supported = true`.
- **structural − random: 20/20 positive, mean +0.175.**
- **info − structural: 15/20 (below the 16/20 sign gate), mean +0.007 → not supported.** The ESM
  masking-variance prior adds nothing over the structural selection.

Per-method S_macro (B = 48 / 96 / 192): info 0.476 / 0.551 / 0.594; structural 0.423 / 0.572 / 0.587;
random 0.260 / 0.359 / 0.474; fitness 0.123 / 0.194 / 0.272; practice 0.058 / 0.141 / 0.244.
Fitness-greedy and practice are **worse than random** for training-set quality.

## TrpB — exploratory replication (non-decision-eligible)

`report/20260716T154715Z/downstream.json`, TrpB four-site landscape (Johnston 2024; enzyme catalysis,
biochemically independent of GB1's IgG-Fc binding), same protocol scale (R=20 × K=5 × 20 seeds, budgets
{48, 96, 192}) but scored at **`n_perturbations = 0`** to fit a CPU/GPU budget. The run is therefore
`status = nonconforming_protocol_profile`, `decision_eligible = false` (the only profile mismatch is
`n_perturbations`).

- **structural − random: 20/20 partitions positive, S_macro-AUC mean +0.135.**
- **structural − fitness: 20/20 positive, mean +0.286.**
- **practice − structural: 0/20** (structural wins every partition).

Per-method S_macro (B = 48 / 96 / 192): structural 0.337 / 0.426 / 0.443; random 0.197 / 0.271 / 0.354;
fitness 0.081 / 0.128 / 0.149; practice 0.098 / 0.144 / 0.266. Fitness-greedy is again worse than random.

**The `info` method is not interpretable here.** At `n_perturbations = 0` the masking variance
`var_delta_g` is exactly 0 for all 29,678 variants, so `info`'s weights are degenerate and its selection
reduces to an arbitrary tie-break, not the ESM uncertainty prior it represents. Its S_macro (0.381 / 0.427
/ 0.456) and the `info − structural` contrast are artifacts and are **not** claimed. This degeneracy is
exactly why an n≠16 run is pinned non-decision-eligible.

## Secondary — the "find winners" framing (TrpB, B = 192, exploratory)

`hit_rate@B` (fraction of the predicted top-B that is truly high-fitness): structural 0.439, practice
0.280, random 0.260, fitness 0.107. `ndcg@B`: structural 0.892, random 0.816, practice 0.757, fitness
0.616. `regret` (best available − best true in top-B; lower is better): practice 0.002, random 0.051,
info 0.055, structural 0.059, fitness 0.060. Reading: the structural plate covers and ranks the landscape
best; practice edges it only for finding the single highest-fitness variant. (`info` numbers degenerate,
as above.)

## Reading

Two biochemically independent combinatorial landscapes give the same result: at equal budget, a
structure-aware plate is a better training set for ranking held-out epistasis than a **random** plate
(GB1 +0.175, TrpB +0.135; 20/20 partitions each) and than **fitness-greedy** (GB1 +0.342, TrpB +0.286;
20/20 each), while fitness-greedy is worse than random on both. The GB1 finding is therefore not
GB1-specific; it replicates on a second landscape.

## Caveats (frozen)

- TrpB is **exploratory / non-decision-eligible** (n=0, not the confirmatory n=16 recipe). A publishable
  claim needs the n=16 confirmatory rerun, deferred on GPU cost.
- Two landscapes only (GB1, TrpB); retrospective; single primary learner (pairwise ridge); one-step, no
  sequential selector update.
- TrpB data: ~0.5% of the 160,000 values are imputed (not measured) and unflagged in the source mirror;
  inactivity is encoded as negative fitness (all labels > −1, so `log1p` is finite).
- Both `downstream.json` artifacts are `status = provisional` and not registered in `artifacts/`.

## Provenance

- GB1 cache `report/scored_650m.jsonl` (650M, n_perturbations=16). TrpB cache
  `report/scored_trpb_650m_n0.jsonl` (650M, n_perturbations=0, Colab-produced).
- TrpB run: `epibudget downstream --dataset trpb_johnston2024 --scored-cache
  report/scored_trpb_650m_n0.jsonl --n-perturbations 0 --partitions 20 --seeds 20 --budgets 48,96,192
  --max-order 3`.
