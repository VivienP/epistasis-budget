# Fourier Feasibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the redistributed TrpB training target has a sparse Fourier spectrum before
spending compute on a registered recovery-versus-budget curve.

**Architecture:** A pure library module summarizes coefficients produced by the existing multiallelic
Walsh transform. A separate script loads and hashes the real dataset, transforms `log1p(fitness)`, and
writes a validated diagnostic report atomically. Recovery-curve code remains conditional and is not part
of the first implementation lot.

**Tech Stack:** Python 3.12, NumPy, Pydantic v2, pytest, existing `epibudget.epistasis` transform helpers.

---

### Task 1: Pure spectrum summary

**Files:**
- Create: `src/epibudget/sparsity.py`
- Create: `tests/test_sparsity.py`

- [x] **Step 1: Write a failing one-coefficient oracle**

  Build a complete binary two-site landscape from the existing orthonormal basis. Assert that the
  summary reports one order-2 coefficient carrying 100% of non-constant energy and effective counts of
  one at 90%, 95%, and 99%.

- [x] **Step 2: Run the oracle and verify RED**

  Run `python -m pytest tests/test_sparsity.py -q`. Expected: import failure because
  `epibudget.sparsity` does not exist.

- [x] **Step 3: Implement the minimal summary API**

  Add immutable result models for per-order energy, magnitude quantiles, and effective counts. Reuse
  `_landscape_tensor` and `_wht_forward`; do not introduce a second transform. Exclude the constant
  coefficient from effective-sparsity totals.

- [x] **Step 4: Verify GREEN**

  Run `python -m pytest tests/test_sparsity.py -q`. Expected: all tests pass.

- [x] **Step 5: Add validation oracles**

  Add tests for Parseval, uniform-rescaling invariance of effective counts, zero-variance input, and
  rejection of non-finite values. Repeat RED then GREEN for each behavior.

### Task 2: Diagnostic report and CLI script

**Files:**
- Create: `src/epibudget/spectrum_diagnostic.py`
- Create: `scripts/spectrum_diagnostic.py`
- Modify: `tests/test_sparsity.py`

- [x] **Step 1: Write failing report tests**

  Assert schema version, `decision_eligible=false`, the imputation caveat, input hashes, start/end
  provenance, and exclusive atomic output behavior on a synthetic CSV.

- [x] **Step 2: Implement the script**

  Resolve `trpb_johnston2024`, hash the CSV, validate 160,000 unique variants, apply
  `labels.training_target`, call the pure summary, and publish the Pydantic report with exclusive atomic
  replacement semantics.

- [x] **Step 3: Verify script tests**

  Run `python -m pytest tests/test_sparsity.py -q`. Expected: all tests pass without network access.

### Task 3: Real-data diagnostic

**Files:**
- Create only after execution: `report/diagnostics/spectrum_trpb.json`
- Modify only after validated execution: `docs/ROADMAP.md`

- [x] **Step 1: Run from a clean scientific tree**

  Run the script against `data/proteingym/trpb_johnston2024.csv`. Do not run while source or scientific
  documentation is changing.

- [x] **Step 2: Independently verify the artifact**

  Recompute Parseval and effective counts from the CSV, compare the input hash, and confirm that all
  160,000 values are finite and in the `log1p` domain.

- [x] **Step 3: Record the go/no-go decision**

  Update the roadmap with the measured A0 result. Do not add a README number and do not start Stage A1
  unless the spectrum evidence justifies the compute.

### Task 4: Conditional Stage A1 specification

**Files:**
- Create only after the A0 decision: `docs/specs/phase-a-fourier-recovery-curve.md`

- [x] **Step 1: Freeze the coefficient estimand and seed identities**

  Specify the fixed 2,166 pairwise coefficient population, budgets, exact selection seeds, lambda path,
  fold assignment, constant-vector guard, and seed-level aggregation from the design specification.

- [ ] **Step 2: Benchmark runtime without labels**

  From a clean fixed commit, time design construction and one synthetic fit at all eight registered
  budgets. Project the exact 43 fits per budget, bind the candidate hash and matching start/end commit
  snapshots, and write the v3 preflight exclusively. Use the timings only for resource planning; do not
  inspect TrpB recovery metrics.

- [ ] **Step 3: Decide whether to schedule the curve**

  Proceed only if the validated v3 preflight covers all 344 fits and fits the available local CPU and
  memory budget. The v1 and v2 runtime files are superseded and cannot authorize the curve. A future GPU
  or Colab run is not required by this plan.

### Task 5: Verification

**Files:**
- Verify only; no new production scope.

- [x] **Step 1: Run targeted and full offline tests**

  Run `python -m pytest tests/test_sparsity.py -q`, then `python -m pytest tests/ -q`.

- [x] **Step 2: Run static checks**

  Run `ruff format --check src/ tests/ scripts/`, `ruff check src/ tests/ scripts/`, and
  `mypy --strict src/ scripts/spectrum_diagnostic.py`.

- [x] **Step 3: Run repository checks**

  Run `python scripts/validate_artifacts.py` and `git diff --check`. Review the scoped diff and preserve
  the unrelated `pyproject.toml` modification.
