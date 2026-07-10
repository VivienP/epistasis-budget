# Running the frozen 650M headline on a free Colab GPU

The frozen headline (`docs/VALIDATION.md`) scores the full 20-letter, four-site GB1 pool (~29,678
variants) with `esm2_t33_650M` at `n_perturbations=16`. Its cost is dominated by the `var_delta_g`
pass — **~1.39M short forward passes** — which the info-optimal method needs on every candidate.

On this project's CPU-only host that pass is **~8–9 days** (measured 1.84 forward-rows/s; see
`docs/LIMITATIONS.md §1`). On a free Colab **T4** GPU the same pass runs in roughly **1–4 hours** at
FP32 (throughput varies with the assigned GPU and Colab load). This is not "minutes": budget an
afternoon, and use the resumable cache below so a session timeout never restarts the run.

The scoring is device-agnostic — `--device cuda` changes throughput only, not the numbers (the CPU
parity oracle guarantees the batched path; GPU FP32 output matches within float tolerance and does not
move any selection). Provenance records `device` in `metrics.json`.

## Cells

**1. GPU + install.** Runtime → Change runtime type → T4 GPU, then:

```python
import torch; print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
!git clone https://github.com/VivienP/epistasis-budget.git
%cd epistasis-budget
!pip install -q -e .
!python scripts/fetch_gb1.py            # GB1 (Wu 2016) -> data/proteingym/gb1_wu2016.csv
```

**2. Persist the scored-variant cache to Drive** so a disconnect resumes instead of restarting:

```python
from google.colab import drive; drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/epibudget
```

**3. Run the frozen headline** (full 20-letter alphabet is the `--alphabet` default; stated explicitly
here). Re-run the *same* cell after any timeout — `--scored-cache` skips already-scored variants:

```python
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
```

**4. Retrieve the result.** The run writes `report/<run_id>/metrics.json` (the headline recovery table
plus provenance: `model_id`, `device`, `n_perturbations`, `candidate_alphabet`, `n_candidates`,
`seeds`, `data_sha256`). Read the pairwise-order Spearman/Pearson for info vs fitness vs random and
apply the frozen decision rule in `docs/VALIDATION.md`:

```python
import json, glob
path = sorted(glob.glob("report/*/metrics.json"))[-1]
print(path); print(open(path).read()[:2000])
from google.colab import files; files.download(path)
```

## Notes

- **Resumability.** The cache is write-through per chunk; an interruption loses at most one chunk. The
  cached values are the exact scorer output, so a resumed run is identical to an uninterrupted one.
- **Do not reduce `--n-perturbations` or the alphabet** to go faster — that changes the numbers and
  breaks the frozen protocol (`docs/VALIDATION.md`, invariant #2). Speed comes from `--device cuda`
  and `--batch-size` only.
- **Larger GPU.** On an A100 (Colab Pro) the same run is well under an hour; the command is unchanged.
