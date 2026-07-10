"""Offline tests for the resumable scored-variant cache (no ESM-2)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from epibudget.scored_cache import append_cache, load_cache, score_with_cache
from epibudget.types import ScoredVariant, Variant


def _variant(pos: int) -> Variant:
    return frozenset({(pos, "A", "C")})


class _CountingScorer:
    """Records every variant it is asked to score, to prove cached ones are skipped on resume."""

    def __init__(self) -> None:
        self.scored: list[Variant] = []

    def score_batch(self, wt: str, variants: Sequence[Variant]) -> list[ScoredVariant]:
        self.scored.extend(variants)
        return [ScoredVariant(variant=v, delta_g=float(len(v)), var_delta_g=0.5) for v in variants]


def test_load_cache_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_cache(tmp_path / "absent.jsonl") == {}


def test_append_then_load_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    sv = ScoredVariant(variant=_variant(3), delta_g=1.25, var_delta_g=0.75)
    append_cache(path, [sv])
    loaded = load_cache(path)
    assert loaded[_variant(3)].delta_g == sv.delta_g
    assert loaded[_variant(3)].var_delta_g == sv.var_delta_g


def test_score_with_cache_resumes_without_rescoring(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    candidates = [_variant(p) for p in range(6)]

    first = _CountingScorer()
    out1 = score_with_cache(first, "WT", candidates[:3], path, chunk_size=2)
    assert [sv.variant for sv in out1] == candidates[:3]
    assert first.scored == candidates[:3]  # all three scored the first time

    # Resume over the full set: only the three new candidates are scored; cached ones are reused.
    second = _CountingScorer()
    out2 = score_with_cache(second, "WT", candidates, path, chunk_size=2)
    assert [sv.variant for sv in out2] == candidates  # returned in input order
    assert second.scored == candidates[3:]  # only the previously-unseen variants
