# Notebooks

## `colab/` — GPU run provenance

The two notebooks that produced the ESM-2 score caches behind the downstream-impact benchmark, kept as a
record of what actually ran on Colab GPUs.

| Notebook | Landscape | GPU | `n_perturbations` | Cache produced |
|---|---|---|---|---|
| `gb1_650m_n16.ipynb` | GB1 (Wu 2016) | A100 | 16 | `scored_650m.jsonl` — confirmatory, decision-eligible |
| `trpb_650m_n0.ipynb` | TrpB (Johnston 2024) | T4 | 0 | `scored_trpb_650m_n0.jsonl` — exploratory, non-decision-eligible |

`n_perturbations` is the load-bearing difference. The confirmatory profile pins it at 16, so the TrpB run
at 0 is deliberately non-decision-eligible: masking variance is then identically zero, which makes the
`info` method degenerate. Full numbers and caveats in
[`../docs/experiments/trpb-downstream-generalization-20260716.md`](../docs/experiments/trpb-downstream-generalization-20260716.md).

Cell outputs are kept on purpose — they are the evidence of the runs (GPU model, timings, measured
Spearman). The score caches themselves live under the git-ignored `report/` and are not committed. The
GB1 notebook writes its cache to Google Drive so it survives a Colab session restart; substitute a local
path to run it elsewhere.

## The demo notebook

A reproducible demo that renders the headline figure straight from a saved report is not committed yet.
The live result is the downstream-impact benchmark — does a structure-aware budget build a better
training set for ranking held-out double and triple mutants? — decision-eligible on GB1 and replicated
(exploratory) on TrpB. See the **Result** section of the [root README](../README.md) and the spec in
[`../docs/specs/downstream.md`](../docs/specs/downstream.md).
