"""Resumable scored-variant cache for long runs (e.g. the 650M headline on a free Colab GPU).

A JSONL sidecar of already-computed ``ScoredVariant`` rows so a run interrupted by a session timeout
resumes without re-scoring. Purely a throughput/resilience aid: cached values are the scorer's exact
write-through output, so a resumed run is byte-identical to an uninterrupted one. No torch import —
the cache logic is offline-testable.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from epibudget.types import ScoredVariant, Variant


class _BatchScorer(Protocol):
    """The one method ``score_with_cache`` needs — structural, so no concrete scorer import here."""

    def score_batch(self, wt: str, variants: Sequence[Variant]) -> list[ScoredVariant]: ...


def load_cache(path: Path) -> dict[Variant, ScoredVariant]:
    """Load previously scored variants from a JSONL cache; empty dict if the file is absent."""
    if not path.exists():
        return {}
    out: dict[Variant, ScoredVariant] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sv = ScoredVariant.model_validate_json(line)
        out[sv.variant] = sv
    return out


def append_cache(path: Path, scored: Sequence[ScoredVariant]) -> None:
    """Append scored variants to the JSONL cache (write-through, one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for sv in scored:
            handle.write(sv.model_dump_json() + "\n")


def score_with_cache(
    scorer: _BatchScorer,
    wt: str,
    candidates: Sequence[Variant],
    path: Path,
    chunk_size: int = 512,
) -> list[ScoredVariant]:
    """Score ``candidates``, skipping any already in ``path`` and write-through appending the rest.

    Scores the missing candidates in ``chunk_size`` batches, flushing each to disk before the next,
    so an interruption loses at most one chunk. Returns a ScoredVariant per candidate, in order.
    """
    cache = load_cache(path)
    missing = [c for c in candidates if c not in cache]
    for start in range(0, len(missing), chunk_size):
        chunk = missing[start : start + chunk_size]
        scored = scorer.score_batch(wt, chunk)
        append_cache(path, scored)
        for sv in scored:
            cache[sv.variant] = sv
    return [cache[c] for c in candidates]
