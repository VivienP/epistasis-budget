# TrpB Fourier recovery on Google Colab

This runbook executes the registered TrpB Fourier recovery diagnostic at commit
`89ac928601c203870e87dea753323246da4fa78f`. Google Drive stores the immutable inputs and durable run
store. The Colab-local `/content` filesystem stores the checkout and runtime preflight.

The score cache is selected by its complete scientific identity. File names are not sufficient: the
historical `scored_650m.jsonl` cache belongs to GB1 and must not be used for TrpB.

The registered TrpB cache identity is:

- model: `facebook/esm2_t33_650M_UR50D`;
- scorer seed: `0`;
- masking perturbations: `16`;
- candidate count: `29,678`;
- candidate SHA-256: `59c5ff5f50dc118adf14971100a77dc9ed322523493b4c4f28345a145333d2f5`;
- WT SHA-256: `c0964e6dbcd438545d5a28cf9bf743806e119e334bd5e21df3a5ba759a1af4cb`.

Run the cells below in order. After a Colab disconnect, rerun cells 1 and 2 to restore the mount,
environment, checkout, and dependencies. Then rerun cell 8: the engine verifies the Drive-backed journal
and resumes from its latest durable checkpoint.

## Cell 1 — Mount Drive and define the run

```python
from google.colab import drive
drive.mount("/content/drive")

import os

commit = "89ac928601c203870e87dea753323246da4fa78f"
short = commit[:7]
root = "/content/drive/MyDrive/epibudget"

os.environ.update(
    {
        "EPI_COMMIT": commit,
        "EPI_REPO": f"/content/epistasis-budget-{short}",
        "EPI_DATA": f"{root}/input/trpb_johnston2024.csv",
        "EPI_CACHE": f"{root}/input/scored_trpb_650m_n16.jsonl",
        "EPI_SIDECAR": f"{root}/input/scored_trpb_650m_n16.jsonl.meta.json",
        "EPI_PREFLIGHT": f"/content/fourier_recovery_runtime_{short}.json",
        "EPI_RUN_DIR": f"{root}/runs/fourier_recovery_{short}",
        "EPI_REPORT": f"{root}/fourier_recovery_trpb_{short}.json",
        "BLIS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "VECLIB_MAXIMUM_THREADS": "2",
    }
)

for name in (
    "EPI_COMMIT",
    "EPI_REPO",
    "EPI_DATA",
    "EPI_CACHE",
    "EPI_PREFLIGHT",
    "EPI_RUN_DIR",
    "EPI_REPORT",
):
    print(f"{name}={os.environ[name]}")
```

## Cell 2 — Restore the exact checkout and dependencies

```bash
%%bash
set -euo pipefail

if [ ! -d "$EPI_REPO/.git" ]; then
    git clone https://github.com/VivienP/epistasis-budget.git "$EPI_REPO"
fi

cd "$EPI_REPO"
git fetch origin "$EPI_COMMIT"
git checkout --detach "$EPI_COMMIT"
test "$(git rev-parse HEAD)" = "$EPI_COMMIT"

python -m pip install -q \
    "numpy==2.3.5" \
    "scipy==1.18.0" \
    "pandas==2.2.2" \
    "threadpoolctl==3.6.0"
python -m pip install -q -e ".[dev]"

test -z "$(git status --porcelain)"
python --version
python - <<'PY'
import mypy.version
import numpy
import pandas
import scipy
import threadpoolctl

assert numpy.__version__ == "2.3.5"
assert scipy.__version__ == "1.18.0"
assert pandas.__version__ == "2.2.2"
assert threadpoolctl.__version__ == "3.6.0"

print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("pandas:", pandas.__version__)
print("mypy:", mypy.version.__version__)
print("threadpoolctl:", threadpoolctl.__version__)
PY
```

Dependency-conflict warnings about packages preinstalled by Colab do not invalidate this cell. A nonzero
exit status does.

## Cell 3 — Validate the dataset and canonical TrpB cache

```bash
%%bash
set -euo pipefail
cd "$EPI_REPO"

python - <<'PY'
import hashlib
import os
from pathlib import Path

from epibudget.coeff_recovery import AA20
from epibudget.data import TRPB_SITES, TRPB_WT_AT_SITES, TRPB_WT_SEQUENCE, enumerate_candidates
from epibudget.fourier_recovery import validate_deterministic_selection_boundaries
from epibudget.recovery_protocol import REGISTERED_RECOVERY_PROTOCOL
from epibudget.scored_cache import validate_cache_against_universe

dataset = Path(os.environ["EPI_DATA"])
cache = Path(os.environ["EPI_CACHE"])
sidecar = Path(os.environ["EPI_SIDECAR"])

for path in (dataset, cache, sidecar):
    if not path.is_file():
        raise FileNotFoundError(path)

expected_sha256 = {
    dataset: REGISTERED_RECOVERY_PROTOCOL.dataset_sha256,
    cache: REGISTERED_RECOVERY_PROTOCOL.cache_sha256,
    sidecar: REGISTERED_RECOVERY_PROTOCOL.sidecar_sha256,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


for path, expected in expected_sha256.items():
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {observed} != {expected}")

candidates = enumerate_candidates(TRPB_SITES, TRPB_WT_AT_SITES, AA20, max_order=3)
loaded, metadata, identity = validate_cache_against_universe(
    cache,
    candidates,
    candidate_alphabet=AA20,
    max_order=3,
    model_id="facebook/esm2_t33_650M_UR50D",
    scorer_seed=0,
    n_perturbations=16,
    wt_sequence=TRPB_WT_SEQUENCE,
    sidecar_path=sidecar,
)

assert len(loaded) == 29_678
assert identity.candidate_sha256 == (
    "59c5ff5f50dc118adf14971100a77dc9ed322523493b4c4f28345a145333d2f5"
)
assert metadata.wt_sha256 == (
    "c0964e6dbcd438545d5a28cf9bf743806e119e334bd5e21df3a5ba759a1af4cb"
)
validate_deterministic_selection_boundaries(
    tuple(loaded.values()),
    budgets=REGISTERED_RECOVERY_PROTOCOL.budgets,
    max_order=REGISTERED_RECOVERY_PROTOCOL.selection_max_order,
)

print("Canonical TrpB cache validated:", cache)
print("Validated candidates:", len(loaded))
PY
```

## Cell 4 — Run the pre-run quality checks

```bash
%%bash
set -euo pipefail
cd "$EPI_REPO"

python -m pytest -q \
    tests/test_run_store.py \
    tests/test_recovery_runtime.py \
    tests/test_doptimal_checkpoint.py \
    tests/test_lasso_checkpoint.py \
    tests/test_recovery_state.py \
    tests/test_recovery_engine.py \
    tests/test_fourier_recovery_cli.py

python -m ruff check src/ tests/ scripts/
python -m ruff format --check src/ tests/ scripts/
python -m mypy --strict src/

test -z "$(git status --porcelain)"
```

## Cell 5 — Test the Drive-backed durable store

```bash
%%bash
set -euo pipefail
cd "$EPI_REPO"

python - <<'PY'
import os
from pathlib import Path

from epibudget.run_store import ContentAddressedRunStore

run_dir = Path(os.environ["EPI_RUN_DIR"])
run_dir.mkdir(parents=True, exist_ok=True)
store = ContentAddressedRunStore(run_dir)
store.initialise()
audit = store.verify()

assert not audit.has_errors, audit.problems()
print("Google Drive durable run store: OK")
PY
```

## Cell 6 — Generate and validate the local runtime preflight

`write_json_atomic` requires atomic hard-link publication, which the mounted Drive filesystem does not
provide. The preflight must remain under `/content`; `prepare` archives it into the durable store.

```bash
%%bash
set -euo pipefail
cd "$EPI_REPO"

test -z "$(git status --porcelain)"

if [ ! -f "$EPI_PREFLIGHT" ]; then
    python scripts/benchmark_fourier_recovery.py --out "$EPI_PREFLIGHT"
else
    echo "Existing preflight found: $EPI_PREFLIGHT"
fi

python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["EPI_PREFLIGHT"])
commit = os.environ["EPI_COMMIT"]
payload = json.loads(path.read_text(encoding="utf-8"))
provenance = payload["provenance"]

assert provenance["workspace_state_matches"] is True
assert provenance["workspace_start"]["code_state"] == "clean"
assert provenance["workspace_end"]["code_state"] == "clean"
assert provenance["workspace_start"]["execution_commit"] == commit
assert provenance["workspace_end"]["execution_commit"] == commit

print("Preflight valid:", path)
print("Projected seconds:", payload["projected_seconds"])
PY
```

## Cell 7 — Prepare the immutable run

```bash
%%bash
set -euo pipefail
cd "$EPI_REPO"

python scripts/fourier_recovery_curve.py prepare \
    --run-dir "$EPI_RUN_DIR" \
    --data "$EPI_DATA" \
    --cache "$EPI_CACHE" \
    --sidecar "$EPI_SIDECAR" \
    --runtime-preflight "$EPI_PREFLIGHT"

python scripts/fourier_recovery_curve.py status --run-dir "$EPI_RUN_DIR"
```

Preparation is successful when status reports `"prepared": true`. Repeating this cell with identical
inputs is idempotent.

## Cell 8 — Run or resume

```bash
%%bash
set -euo pipefail
cd "$EPI_REPO"

date -u
python scripts/fourier_recovery_curve.py run \
    --run-dir "$EPI_RUN_DIR"
date -u
```

The run publishes D-optimal checkpoints, LASSO-fold checkpoints, and completed cells to Drive. If the
runtime disconnects, rerun cells 1 and 2 and then this cell. Do not reuse the former `b0cc018` or
`c9c69e0` run stores: they are bound to different execution commits.

## Status during execution

The Colab terminal can inspect the process while cell 8 is active:

```bash
ps -eo pid,etime,time,pcpu,pmem,rss,stat,cmd | grep '[f]ourier_recovery_curve.py'
```

After a disconnect, or whenever cell 8 is not active, run:

```bash
cd /content/epistasis-budget-89ac928
python scripts/fourier_recovery_curve.py status \
    --run-dir /content/drive/MyDrive/epibudget/runs/fourier_recovery_89ac928
```

## Cell 9 — Verify and export the completed report

Run this cell only after cell 8 completes successfully.

```bash
%%bash
set -euo pipefail
cd "$EPI_REPO"

python scripts/fourier_recovery_curve.py verify --run-dir "$EPI_RUN_DIR"
python scripts/fourier_recovery_curve.py export \
    --run-dir "$EPI_RUN_DIR" \
    --out "$EPI_REPORT"

test -s "$EPI_REPORT"
ls -lh "$EPI_REPORT"
```
