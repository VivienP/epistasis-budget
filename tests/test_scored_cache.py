"""Offline tests for the resumable scored-variant cache (no ESM-2)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from epibudget.scored_cache import (
    CacheMetadata,
    append_cache,
    build_cache_metadata,
    cache_metadata_path,
    candidate_sha256,
    load_cache,
    score_with_cache,
)
from epibudget.types import ScoredVariant, Variant


def _variant(pos: int) -> Variant:
    return frozenset({(pos, "A", "C")})


class _FakeConfiguredScorer:
    """A minimal stand-in satisfying the ``_ConfiguredBatchScorer`` protocol fields."""

    def __init__(self) -> None:
        self.model_id = "toy-model"
        self.device = "cuda"
        self.n_perturbations = 7
        self.seed = 42
        self.mask_fraction = 0.2
        self.batch_size = 64
        self.num_threads = 4

    def score_batch(self, wt: str, variants: Sequence[Variant]) -> list[ScoredVariant]:
        return [ScoredVariant(variant=v, delta_g=0.0, var_delta_g=0.0) for v in variants]


class _CountingScorer:
    """Records every variant it is asked to score, to prove cached ones are skipped on resume."""

    def __init__(self) -> None:
        self.scored: list[Variant] = []

    def score_batch(self, wt: str, variants: Sequence[Variant]) -> list[ScoredVariant]:
        self.scored.extend(variants)
        return [ScoredVariant(variant=v, delta_g=float(len(v)), var_delta_g=0.5) for v in variants]


def _metadata(candidates: Sequence[Variant], *, n_perturbations: int = 4) -> CacheMetadata:
    return CacheMetadata(
        model_id="toy-model",
        wt_sha256="wt-sha256",
        candidate_sha256=candidate_sha256(candidates),
        candidate_count=len(candidates),
        candidate_alphabet="AC",
        max_order=1,
        scorer_seed=0,
        n_perturbations=n_perturbations,
        device="cpu",
        mask_fraction=0.15,
        batch_size=32,
        num_threads=None,
    )


_MAX_ORDER = 2


def test_build_cache_metadata_maps_every_scorer_field() -> None:
    scorer = _FakeConfiguredScorer()
    candidates = [_variant(p) for p in range(3)]

    metadata = build_cache_metadata(
        scorer, "WT", candidates, candidate_alphabet="AC", max_order=_MAX_ORDER
    )

    assert metadata.model_id == scorer.model_id
    assert metadata.device == scorer.device
    assert metadata.n_perturbations == scorer.n_perturbations
    assert metadata.scorer_seed == scorer.seed  # the field is renamed on the way in
    assert metadata.mask_fraction == scorer.mask_fraction
    assert metadata.batch_size == scorer.batch_size
    assert metadata.num_threads == scorer.num_threads
    assert metadata.candidate_alphabet == "AC"
    assert metadata.max_order == _MAX_ORDER
    assert metadata.candidate_count == len(candidates)
    assert metadata.candidate_sha256 == candidate_sha256(candidates)
    assert metadata.wt_sha256 == hashlib.sha256(b"WT").hexdigest()


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
    out1 = score_with_cache(
        first, "WT", candidates[:3], path, metadata=_metadata(candidates), chunk_size=2
    )
    assert [sv.variant for sv in out1] == candidates[:3]
    assert first.scored == candidates[:3]  # all three scored the first time

    # Resume over the full set: only the three new candidates are scored; cached ones are reused.
    second = _CountingScorer()
    out2 = score_with_cache(
        second, "WT", candidates, path, metadata=_metadata(candidates), chunk_size=2
    )
    assert [sv.variant for sv in out2] == candidates  # returned in input order
    assert second.scored == candidates[3:]  # only the previously-unseen variants

    sidecar = json.loads(cache_metadata_path(path).read_text(encoding="utf-8"))
    assert sidecar["candidate_sha256"] == candidate_sha256(candidates)


def test_score_with_cache_rejects_metadata_mismatch_before_scoring(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    candidates = [_variant(p) for p in range(3)]
    first = _CountingScorer()
    score_with_cache(first, "WT", candidates, path, metadata=_metadata(candidates))

    second = _CountingScorer()
    with pytest.raises(ValueError, match="cache metadata mismatch"):
        score_with_cache(
            second,
            "WT",
            candidates,
            path,
            metadata=_metadata(candidates, n_perturbations=16),
        )
    assert second.scored == []


def test_score_with_cache_rejects_legacy_cache_without_metadata(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    candidates = [_variant(0)]
    append_cache(path, [ScoredVariant(variant=candidates[0], delta_g=1.0, var_delta_g=0.0)])

    with pytest.raises(ValueError, match="has no metadata sidecar"):
        score_with_cache(_CountingScorer(), "WT", candidates, path, metadata=_metadata(candidates))


def test_score_with_cache_repairs_only_a_truncated_final_line(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    candidates = [_variant(p) for p in range(3)]
    first = _CountingScorer()
    score_with_cache(first, "WT", candidates[:2], path, metadata=_metadata(candidates))
    with path.open("ab") as handle:
        handle.write(b'{"variant":')

    second = _CountingScorer()
    result = score_with_cache(
        second, "WT", candidates, path, metadata=_metadata(candidates), chunk_size=2
    )
    assert [sv.variant for sv in result] == candidates
    assert second.scored == [candidates[2]]
    assert load_cache(path).keys() == set(candidates)
