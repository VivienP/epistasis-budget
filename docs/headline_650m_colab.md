# Running the frozen 650M headline on a free Colab GPU

The frozen headline (`docs/VALIDATION.md`) scores the full 20-letter, four-site GB1 pool (~29,678
variants) with `esm2_t33_650M` at `n_perturbations=16`. Its cost is dominated by the `var_delta_g`
pass — **~1.39M short forward passes** — which the info-optimal method needs on every candidate.

The package can execute this configuration on CPU, but no complete CPU duration is published for the
exact frozen run. A GPU is recommended. Colab is one available environment, not a throughput or runtime
guarantee; assigned hardware and load vary. Use the resumable cache below so an interruption does not
discard completed variant scores.

The same scoring algorithm supports CPU and GPU. Provenance records the resolved `device`; optimized and
reference CPU scoring are parity-tested, while cross-device floating-point identity is not assumed.

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
  JSONL cache is bound to an immutable metadata sidecar containing the model, candidate-universe
  checksum, seed, perturbation count, device and scorer settings. A mismatched or legacy cache is rejected.
- **Do not reduce `--n-perturbations` or the alphabet** to go faster — that changes the numbers and
  breaks the frozen protocol (`docs/VALIDATION.md`, invariant #2). Speed comes from `--device cuda`
  and `--batch-size` only.
- **GPU choice.** The command is unchanged across compatible CUDA devices; runtime must be measured and
  recorded for the assigned hardware rather than inferred from a device name.
