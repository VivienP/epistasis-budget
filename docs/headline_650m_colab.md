# Running the frozen 650M headline on a free Colab T4

The frozen headline (`docs/VALIDATION.md`) scores the full 20-letter, four-site GB1 pool (29,678
variants) with `esm2_t33_650M` at `n_perturbations=16`. Its cost is dominated by the `var_delta_g`
pass — **~1.39M short forward passes** — which the info-optimal method needs on every candidate.

The package can execute this configuration on CPU, but no complete CPU duration is published for the
exact frozen run. A GPU is recommended. Colab is one available environment, not a throughput or runtime
guarantee; assigned hardware and load vary. Cell 4 measures the throughput of the *assigned* GPU and
extrapolates an honest ETA before Cell 5 commits to the full run. The resumable cache means an
interruption never discards completed variant scores — re-run the same cell to resume.

The same scoring algorithm supports CPU and GPU. Provenance records the resolved `device`; optimized and
reference CPU scoring are parity-tested, while cross-device floating-point identity is not assumed.

## Cells

**1. Preflight + install.** Runtime → Change runtime type → T4 GPU, then run. This clones the
`audit/scientific-hardening` branch (the frozen recipe and the `--scored-cache` flag live there, not on
`main`), records the checked-out commit for provenance, and downloads GB1:

```python
import sys, torch
print("python:", sys.version.split()[0])          # must be >= 3.12 (see pyproject requires-python)
print("cuda  :", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
assert sys.version_info >= (3, 12), "epibudget requires Python >= 3.12; resolve before continuing"
assert torch.cuda.is_available(), "no GPU assigned; set Runtime -> T4 GPU"

!git clone --branch audit/scientific-hardening https://github.com/VivienP/epistasis-budget.git
%cd epistasis-budget
!git rev-parse HEAD                                # pin: record this commit alongside the result
!pip install -q -e .
!python scripts/fetch_gb1.py                       # GB1 (Wu 2016) -> data/proteingym/gb1_wu2016.csv
```

**2. Persist the scored-variant cache to Drive** so a disconnect resumes instead of restarting:

```python
from google.colab import drive; drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/epibudget
```

**3. Smoke test (throwaway — NOT the headline).** Reduced alphabet and perturbations, so its numbers are
meaningless as science; it exists only to prove in ~1 minute that the GPU path, the CLI, and the
Drive-backed cache all work end to end before Cell 5 commits hours. Its cache and output are separate
files that never mix with the real run:

```python
!epibudget validate \
    --model esm2_t33_650M \
    --alphabet ADEF \
    --budgets 48 \
    --seeds 2 \
    --n-perturbations 2 \
    --device cuda \
    --batch-size 128 \
    --scored-cache /content/drive/MyDrive/epibudget/scored_smoke.jsonl \
    --out report_smoke/
```

**4. Honest ETA — measure the assigned GPU, don't guess.** `bench_scoring.py` scores a small slice at the
frozen `n_perturbations=16` and reports `optimized_variants_per_s`; extrapolate to the 29,678-candidate
pool. This is an estimate for the currently-assigned hardware, not a guarantee — it tells you whether the
free T4 is enough or a faster/paid GPU is worth it:

```python
import json
!python scripts/bench_scoring.py --model esm2_t33_650M --device cuda \
    --alphabet ACD --n-perturbations 16 --batch-size 128 --out report_smoke/bench_gpu.json
b = json.load(open("report_smoke/bench_gpu.json"))
vps = b["optimized_variants_per_s"]
eta_h = 29678 / vps / 3600
print(f"measured {vps:.3f} variants/s at n_perturbations=16  ->  full pool ETA ~ {eta_h:.1f} h")
```

**5. Run the frozen headline.** Full 20-letter alphabet, `n_perturbations=16`, B ∈ {48,96,192}, 20 seeds —
every frozen setting stated explicitly. **Re-run this exact cell after any Colab timeout**; `--scored-cache`
skips already-scored variants, so each re-run only advances. `--batch-size` is throughput-only (raise it on
a larger GPU):

```python
import time
t0 = time.perf_counter()
!epibudget validate \
    --model esm2_t33_650M \
    --alphabet ACDEFGHIKLMNPQRSTVWY \
    --budgets 48,96,192 \
    --seeds 20 \
    --n-perturbations 16 \
    --device cuda \
    --batch-size 128 \
    --scored-cache /content/drive/MyDrive/epibudget/scored_650m.jsonl \
    --out report/
print(f"elapsed this cell: {(time.perf_counter() - t0) / 3600:.2f} h")
```

**6. Retrieve and verify.** The run writes `report/<run_id>/metrics.json` (the headline recovery table
plus provenance: `model_id`, `device`, `n_perturbations`, `candidate_alphabet`, `n_candidates`, `seeds`,
`data_sha256`). Confirm the frozen configuration, read the pairwise-order Spearman/Pearson for info vs
fitness vs random, then apply the frozen decision rule in `docs/VALIDATION.md`:

```python
import json, glob
path = sorted(glob.glob("report/*/metrics.json"))[-1]
m = json.load(open(path))
print(path)
print("config:", {k: m.get(k) for k in
      ["device", "n_candidates", "n_perturbations", "candidate_alphabet", "seeds"]})
for r in m["results"]:
    pw = next((x for x in r["metrics"] if x["order"] == "pairwise"), None)
    if pw:
        print(f'  B={r["budget"]:>3} {r["method"]:<10} pairwise spearman={pw["spearman"]}')
from google.colab import files; files.download(path)
```

## Notes

- **Reproducibility.** Cell 1 clones a fixed branch and prints its commit SHA — record that SHA with the
  result. `n_candidates` in the output must be **29678** and `candidate_alphabet` the full 20 letters, or
  it is not the frozen run.
- **Resumability.** The cache is write-through per chunk (flushed and `fsync`-ed before the next chunk), so
  an interruption loses at most one chunk. The JSONL cache is bound to an immutable metadata sidecar
  (model, candidate-universe checksum, seed, perturbation count, device, scorer settings); a mismatched or
  legacy cache is rejected rather than silently reused.
- **Cells 3 and 4 are throwaway.** Their reduced alphabet/perturbations make their numbers non-scientific;
  they never touch the headline cache (`scored_650m.jsonl`) or `report/`.
- **Do not reduce `--n-perturbations` or the alphabet** on Cell 5 to go faster — that changes the numbers
  and breaks the frozen protocol (`docs/VALIDATION.md`, invariant #2). Speed comes from `--device cuda` and
  `--batch-size` only.
- **GPU choice.** The command is unchanged across compatible CUDA devices; runtime must be measured (Cell 4)
  and recorded for the assigned hardware rather than inferred from a device name.
