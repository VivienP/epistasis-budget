"""GB1 validation harness. Frozen protocol in docs/VALIDATION.md.

Compares three methods at each budget — info-optimal, fitness-greedy, random — on epistasis-map
recovery against the complete GB1 ground truth. All three baselines are mandatory in every
comparison.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel


class MethodResult(BaseModel):
    method: str  # "info" | "fitness" | "random"
    budget: int
    map_recovery_spearman: float
    map_recovery_pearson: float
    ci95: tuple[float, float]
    hit_rate: float


class Report(BaseModel):
    dataset: str
    model_id: str
    seeds: int
    results: list[MethodResult]
    var_epsilon: float  # invariant #1 sanity: must be > 0


def run_validation(
    dataset: str,
    budgets: Sequence[int],
    model_id: str,
    seeds: int,
    out_dir: Path,
) -> Report:
    """Execute the frozen protocol and write a report (metrics.json + figures). See VALIDATION.md.

    For each budget and method: allocate zero-shot, reveal true fitness of the selected variants
    only, infer epistasis from those measurements, and score recovery against the full-landscape
    ground truth. The same ``infer_epistasis`` is used for all methods so only the *selected set*
    differs.
    """
    raise NotImplementedError("Seedocs/ROADMAP.md")
