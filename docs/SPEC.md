# epibudget — technical specification

Source of truth for *what* to build. Pairs with
[`RESEARCH_EPISTASIS.md`](RESEARCH_EPISTASIS.md) (*why it is well-posed*) and
[`VALIDATION.md`](VALIDATION.md) (*how we prove it works*).

---

## 0. Problem statement

**Input:** a wild-type sequence `wt`, a set of candidate mutated positions `P` (and optionally an
allowed amino-acid set per position), an integer budget `B`, and an ESM-2 checkpoint.

**Output:** an ordered list of `B` variants (each a set of point mutations over `P`, of order 1–3),
ranked by expected reduction in epistasis uncertainty, plus the per-interaction uncertainty map and the
predicted information gain of the selected batch.

**Objective (informal):** pick the `B` variants whose measured fitness would most sharpen a model of
the landscape's **epistasis coefficients**, not the `B` with highest predicted fitness.

---

## 1. Architecture

```
                    ┌────────────────────────────────────────────────┐
  wt, positions ──► │ data.py        candidate enumeration + loaders  │
                    └───────────────┬────────────────────────────────┘
                                    │ candidate variants (order 1..3)
                    ┌───────────────▼────────────────────────────────┐
   ESM-2 ─────────► │ scoring.py     CONJOINT conditional log-likelihood
                    │                + masking-perturbation dispersion │
                    └───────────────┬────────────────────────────────┘
                       ΔG(v), var[ΔG(v)]  for each candidate variant
                    ┌───────────────▼────────────────────────────────┐
                    │ epistasis.py   ε terms (WT-referenced) via WHT   │
                    │                seed σ²(ε) from score dispersion  │
                    └───────────────┬────────────────────────────────┘
                    ┌───────────────▼────────────────────────────────┐
                    │ graph.py       factor graph: nodes/edges/hyper  │
                    │                Gaussian model over ε terms       │
                    └───────────────┬────────────────────────────────┘
                    ┌───────────────▼────────────────────────────────┐
                    │ acquisition.py greedy submodular selection of B  │
                    │                (variance reduction; λ slider)    │
                    └───────────────┬────────────────────────────────┘
             allocate │             │ validate
        ┌─────────────▼───┐  ┌──────▼───────────────────────────────┐
        │ cli.py allocate │  │ validate.py  GB1 harness vs baselines │
        └─────────────────┘  └───────────────────────────────────────┘
```

---

## 2. Data model (`src/epibudget/types.py`)

Typed, immutable where possible (pydantic v2 / dataclasses).

```python
Mutation = tuple[int, str, str]          # (position, wt_aa, mut_aa), 0-indexed into `wt`
Variant  = frozenset[Mutation]           # order = len(Variant); order 0 = wild type

class ScoredVariant(BaseModel):
    variant: Variant
    delta_g: float          # conjoint conditional log-likelihood ratio vs WT (higher = fitter)
    var_delta_g: float      # dispersion across masking perturbations (model uncertainty)

class Interaction(BaseModel):
    mutations: tuple[Mutation, ...]  # the identity: specific residues, e.g. ((39,'D','A'),(41,'G','W'))
    sites: tuple[int, ...]           # derived position summary, e.g. (39, 41)
    order: int                       # 2 or 3 for pairwise / third-order
    epsilon_hat: float               # ESM-2-predicted epistasis coefficient (WT-referenced)
    sigma2: float                    # current uncertainty (variance) about this coefficient
    # Keyed by `mutations`, not `sites`: ε depends on the residues, and there are 19² (resp. 19³)
    # amino-acid instances per position-pair (resp. -triple). Build via Interaction.of(...).

class Allocation(BaseModel):
    budget: int
    selected: list[Variant]                 # length B, in selection order
    expected_info_gain: list[float]         # marginal gain when each was added
    epistasis_map: list[Interaction]        # final uncertainty map
    seed: int
    model_id: str
```

---

## 3. Scoring — `scoring.py` (invariant #1 lives here)

We need `ΔG(v)` for every candidate variant `v`, computed **conjointly**.

### 3.1 Conjoint conditional score

For a variant `v` with mutated positions `S`:

1. Build the mutant sequence `x_v` = `wt` with every mutation in `v` applied.
2. For each position `p ∈ S`, mask `p` in `x_v` (all other mutations still present) and read the
   ESM-2 log-probabilities at `p`.
3. `ΔG(v) = Σ_{p∈S} [ log P(mut_aa_p | x_v \ p) − log P(wt_aa_p | x_v \ p) ]`
   — a pseudo-log-likelihood ratio of the mutant vs WT residues, **evaluated in the mutant context**.

The mutant *context* is what makes this non-additive: `ΔG(ij) ≠ ΔG(i) + ΔG(j)` in general, so the
epistasis terms in §4 are non-trivial.

> **Forbidden shortcut:** scoring each mutation on the WT background and summing. That yields exact
> additivity and ε ≡ 0. `tests/test_scoring.py::test_epsilon_not_identically_zero` asserts that, on a
> small real slice, `Var[ε] > 0`.

### 3.2 Uncertainty via masking perturbations

The model's uncertainty about `ΔG(v)` is estimated by `K` stochastic passes that perturb the scoring —
either MC-dropout at inference (dropout enabled) or randomised masking order for multi-site variants
(the masked-marginal order dependence documented for multi-mutants):

```
scores = [conjoint_score(v, perturbation=k) for k in range(K)]   # K ≈ 16–32
delta_g      = mean(scores)
var_delta_g  = var(scores)      # zero-shot proxy for "how unsure is the model here"
```

Batching, caching (per-sequence forward passes are reused across variants that share a context), and a
small-model fast path (`esm2_t12_35M`) keep the reduced-alphabet pass CPU-tractable. The full 20-letter
650M variance-inclusive pass is GPU-recommended; see [`LIMITATIONS.md`](LIMITATIONS.md) §1.

### 3.3 Public interface

```python
class ConjointScorer:
    def __init__(self, model_id: str, device: str = "cpu",
                 n_perturbations: int = 16, seed: int = 0) -> None: ...
    def score(self, wt: str, variant: Variant) -> ScoredVariant: ...
    def score_batch(self, wt: str, variants: Sequence[Variant]) -> list[ScoredVariant]: ...
```

---

## 4. Epistasis terms — `epistasis.py`

WT-referenced (biochemical) epistasis via the inclusion–exclusion form, which is the WT sub-sampling of
the (multiallelic) Walsh–Hadamard transform (see RESEARCH §3).

```python
def epsilon_pairwise(dg: Mapping[Variant, float], i: Mutation, j: Mutation) -> float:
    # ε(i,j) = ΔG(ij) − ΔG(i) − ΔG(j)      [ΔG(∅) = 0]

def epsilon_third(dg: Mapping[Variant, float], i, j, k) -> float:
    # ε(i,j,k) = ΔG(ijk) − ΔG(ij) − ΔG(ik) − ΔG(jk) + ΔG(i) + ΔG(j) + ΔG(k)
```

- `predicted_epistasis(scored) -> list[Interaction]` builds `ε_hat` from ESM-2 conjoint scores and
  propagates `var_delta_g` into a seed `σ²` per interaction (linear error propagation through the
  inclusion–exclusion sum, assuming independent score noise as a first approximation).
- `ground_truth_epistasis(dms_frame) -> list[Interaction]` computes the true ε terms from measured GB1
  fitness (for validation only). For completeness it can also compute the multiallelic WHT spectrum of
  the full landscape to report variance-by-order.

---

## 5. Factor graph & uncertainty model — `graph.py`

A Gaussian model over the interaction terms is enough to make selection tractable.

- **Model:** each candidate variant's `ΔG(v)` carries an independent Gaussian prior `N(ΔG_hat, τ²)`
  with `τ² = var_delta_g(v)` (the ESM masking-perturbation dispersion). Each `Interaction`'s coefficient
  is the fixed ±1 inclusion–exclusion combination of the `ΔG` of its loop, so `σ²(ε(S)) = Σ_{T∈loop} τ²_T`
  (independent noise, coefficient² = 1).
- **Observations:** measuring a variant `v` reveals `ΔG(v)` exactly, collapsing its `τ²` to 0 — standard
  linear-Gaussian conditioning at zero observation noise. Every loop that `v` braces loses that term.
- **Loop-closure structure:** a third-order interaction `ε(i,j,k)` couples 7 variants (`{i,j,k}`,
  `{ij,ik,jk}`, `{ijk}`). Measuring members of that loop is what "braces" the term (the geodetic
  collision made literal).
- **Submodularity (honest form):** under this independent-noise, exact-measurement model
  `info_gain(M, v) = τ²_v · n(v)` (n(v) = number of interactions whose loop contains v), independent
  of `M`. So `info_gain` is **modular** — a degenerate special case of submodular in which the
  diminishing-returns inequality holds with *equality*. Greedy is therefore not merely (1 − 1/e)-near
  optimal but *exactly* optimal for a fixed budget. This is not the general A-optimality result (which
  is not submodular once correlated priors or observation noise enter); it is a consequence of the
  independent-noise assumption. Correlated priors (strict submodularity) are out of scope for v1 (§11).

```python
class EpistasisFactorGraph:
    # var_delta_g must cover every order-1..max_order candidate (bare Variant carries no dispersion).
    def __init__(self, interactions: Sequence[Interaction],
                 var_delta_g: Mapping[Variant, float]) -> None: ...
    # keyed by the interaction's `mutations` tuple (unique), never by `sites` (which collides).
    def posterior_variance(self, measured: frozenset[Variant]) -> dict[tuple[Mutation, ...], float]: ...
    def total_uncertainty(self, measured: frozenset[Variant]) -> float:   # Σ σ² over interactions
        ...
    def info_gain(self, measured: frozenset[Variant], candidate: Variant) -> float:
        # reduction in total_uncertainty from additionally measuring `candidate`; depends only on
        # membership, never on any revealed fitness value (no label leakage).
        ...
```

---

## 6. Acquisition — `acquisition.py`

Maximisation of expected uncertainty reduction. Under the v1 independent-noise model `info_gain` is
modular (§5) — the per-candidate weight `info[v] = graph.info_gain(∅, v)` does not depend on what has
already been selected — so there is no iterative greedy loop: `allocate` is a single stable sort.

```
allocate(graph, candidates, B, lambda_) -> Allocation:
    info = {v: graph.info_gain(frozenset(), v) for v in candidates}   # fixed, modular
    if   lambda_ == 1.0: ranked = sort candidates by delta_g        desc   # == fitness_greedy exactly
    elif lambda_ == 0.0: ranked = sort candidates by info[v]        desc   # pure info-optimal
    else:                ranked = sort candidates by
                                  (1-lambda_)·minmax(info) + lambda_·minmax(delta_g) desc
    selected = ranked[:B]
    return Allocation(selected=selected,
                      expected_info_gain=[info[v] for v in selected],   # RAW info_gain, never blended
                      ...)
```

- `lambda_ = 0.0` → pure information-optimal (the thesis); `1.0` → fitness-greedy (the control).
- The λ∈{0,1} endpoints are special-cased to bypass the min-max normalisation (which is 0/0 when a
  score is constant across the pool), so `lambda_=1` reproduces `fitness_greedy` as an ordered list.
- `expected_info_gain` is always the raw `info_gain` of each selected variant, never the blended score.
- Because `info_gain` is modular, this exact single sort *is* greedy; the (1 − 1/e) submodular bound and
  the lazy-greedy priority queue are only relevant for a future correlated-prior model.

---

## 7. Baselines (mandatory) — `validate.py`

Every reported comparison includes all three, at each `B`:

1. **info-optimal** — `select(..., lambda_=0.0)`.
2. **fitness-greedy** — top-`B` by predicted `ΔG` (`select(..., lambda_=1.0)`).
3. **random** — uniform sample of `B` candidates (averaged over ≥ 20 seeds).
4. **practice heuristic** (v1.1, additional) — top beneficial singles then all their pairwise
   combinations (the real-world design, cf. MULTI-evolve). Reported as a fourth comparison; not part of the
   frozen decision rule, which stays on info vs fitness vs random. See `docs/VALIDATION.md`.

### Validation pipeline

```
validate(dataset, budgets, model_id, seeds) -> Report:
    load measured GB1 four-site rows (data.py)
    truth = ground_truth_epistasis(full landscape)          # via WHT / inclusion-exclusion
    for B in budgets:
        for method in {info, fitness, random}:
            selected = allocate(method, B)                  # zero-shot; never sees labels
            revealed = reveal_measured_fitness(selected)    # simulate wet-lab from GB1
            inferred = infer_epistasis(revealed)            # least-squares / MoCHI-style fit
            score[method, B] = map_recovery(inferred, truth)   # Spearman/Pearson over ε terms
    return Report(scores, figures)
```

**Primary metric** `map_recovery`: correlation between inferred and ground-truth ε coefficients over
all pairwise + third-order terms. **Secondary:** hit-rate@B (fraction of top-fitness variants captured)
to show we don't catastrophically sacrifice fitness discovery.

---

## 8. CLI — `cli.py` (typer)

```
epibudget allocate --fasta FILE --positions 39,40,41,54 --budget 96 \
                   [--model esm2_t33_650M] [--lambda 0.0] [--max-order 3] [--seed 0] \
                   [--out allocation.json]

epibudget validate --dataset gb1_wu2016 --budgets 48,96,192 \
                   [--model esm2_t12_35M] [--seeds 20] [--out report/]

epibudget robustness --scored-cache CACHE [--out report/]     # post-hoc robustness analyses (no GPU)

epibudget downstream --scored-cache CACHE [--out report/]     # downstream-impact benchmark (CPU-only)

epibudget score   --fasta FILE --variants variants.csv        # debug: dump conjoint ΔG + variance
```

`allocate` prints a rich table (rank, variant, order, ΔG, marginal info gain) and writes
`Allocation` JSON. `validate` writes `report/<run_id>/metrics.json` (one row per method × budget, with
per-order correlations, CIs, and coverage) and prints a rich summary; the figures are rendered
separately by `notebooks/gb1_demo.ipynb` from that JSON. Any empirical claim must trace to a
`metrics.json` written by this command.

---

## 9. Configuration (`pydantic`)

```python
class Config(BaseModel):
    model_id: str = "facebook/esm2_t33_650M_UR50D"
    device: str = "cpu"
    n_perturbations: int = 16
    max_order: int = 3            # cap interaction order (2 or 3)
    lambda_: float = 0.0          # exploitation weight
    seed: int = 0
    cache_dir: Path = Path("data/cache")
```

Deterministic given `(model_id, seed, config)`. Every output embeds the resolved config.

---

## 10. Module responsibilities & test map

| Module | Owns | Key tests |
|--------|------|-----------|
| `data.py` | GB1/ProteinGym loaders, candidate enumeration | `test_data`: GB1 loads to 160k rows; enumeration counts |
| `scoring.py` | conjoint scoring, dispersion | `test_scoring`: **ε not identically zero**; determinism under seed |
| `epistasis.py` | ε terms, WHT ground truth | `test_epistasis`: inclusion–exclusion identities; WHT round-trip |
| `graph.py` | Gaussian factor graph, variance updates | `test_graph`: variance is non-increasing in measurements; submodularity |
| `acquisition.py` | greedy selection, λ slider | `test_acquisition`: λ=1 ≡ fitness-greedy; gains monotone-nonincreasing |
| `validate.py` | GB1 harness, baselines | `test_validate` (slow/data): info ≥ random; report schema |
| `cli.py` | UX | `test_cli`: `allocate` on a tiny toy returns B variants |

## 11. Out of scope for v1

- Background-averaged (ensemble) epistasis (v1 is WT-referenced). It is the bridge to inference tools
  like MoCHI, and is a future extension only if that integration is pursued
  (see `docs/RESEARCH_EPISTASIS.md#3`).
- Orders > 3.
- Multi-round / sequential design (v1 is single-shot budget allocation at round 0).
- Any GPU-specific path.
- A second PLM or a learned surrogate — the uncertainty prior is ESM-2 zero-shot by design.
- Any web API / hosted service — v1 is a CPU-first, GPU-capable CLI + library.
