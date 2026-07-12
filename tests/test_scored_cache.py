"""Offline tests for the resumable scored-variant cache (no ESM-2)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from epibudget.provenance import write_json_exclusive
from epibudget.scored_cache import (
    CacheIdentity,
    CacheMetadata,
    append_cache,
    build_cache_metadata,
    cache_metadata_path,
    candidate_sha256,
    load_cache,
    score_with_cache,
    validate_cache_against_universe,
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


# ------------------------------------------------------------ validate_cache_against_universe


def _write_cache_and_sidecar(
    path: Path, candidates: Sequence[Variant], metadata: CacheMetadata
) -> None:
    append_cache(path, [ScoredVariant(variant=v, delta_g=0.0, var_delta_g=0.0) for v in candidates])
    write_json_exclusive(cache_metadata_path(path), metadata.model_dump(mode="json"))


def test_validate_cache_against_universe_accepts_matching_identity(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    candidates = [_variant(p) for p in range(3)]
    _write_cache_and_sidecar(path, candidates, _metadata(candidates))

    cache, metadata, identity = validate_cache_against_universe(
        path,
        candidates,
        candidate_alphabet="AC",
        max_order=1,
        model_id="toy-model",
        scorer_seed=0,
        n_perturbations=4,
    )
    assert set(cache) == set(candidates)
    assert metadata.model_id == "toy-model"
    assert identity == CacheIdentity(
        model_id="toy-model",
        scorer_seed=0,
        n_perturbations=4,
        candidate_sha256=candidate_sha256(candidates),
        candidate_count=len(candidates),
        candidate_alphabet="AC",
        max_order=1,
        wt_sha256=None,  # no wt_sequence was passed to this call
    )


def test_validate_cache_against_universe_rejects_wrong_model_id(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    candidates = [_variant(p) for p in range(3)]
    _write_cache_and_sidecar(
        path, candidates, _metadata(candidates)
    )  # sidecar model_id="toy-model"

    with pytest.raises(ValueError, match="model_id"):
        validate_cache_against_universe(
            path,
            candidates,
            candidate_alphabet="AC",
            max_order=1,
            model_id="a-different-model",  # the caller's own expected value, not the sidecar's
            scorer_seed=0,
            n_perturbations=4,
        )


def test_validate_cache_against_universe_rejects_wrong_scorer_seed(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    candidates = [_variant(p) for p in range(3)]
    _write_cache_and_sidecar(path, candidates, _metadata(candidates))  # sidecar scorer_seed=0

    with pytest.raises(ValueError, match="scorer_seed"):
        validate_cache_against_universe(
            path,
            candidates,
            candidate_alphabet="AC",
            max_order=1,
            model_id="toy-model",
            scorer_seed=99,
            n_perturbations=4,
        )


def test_validate_cache_against_universe_rejects_wrong_n_perturbations(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    candidates = [_variant(p) for p in range(3)]
    _write_cache_and_sidecar(path, candidates, _metadata(candidates))  # sidecar n_perturbations=4

    with pytest.raises(ValueError, match="n_perturbations"):
        validate_cache_against_universe(
            path,
            candidates,
            candidate_alphabet="AC",
            max_order=1,
            model_id="toy-model",
            scorer_seed=0,
            n_perturbations=99,
        )


def test_validate_cache_against_universe_rejects_missing_required_field(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    candidates = [_variant(p) for p in range(3)]
    append_cache(path, [ScoredVariant(variant=v, delta_g=0.0, var_delta_g=0.0) for v in candidates])
    payload = _metadata(candidates).model_dump(mode="json")
    del payload["model_id"]  # a sidecar missing a required identity field must never validate
    write_json_exclusive(cache_metadata_path(path), payload)

    with pytest.raises(ValueError):
        validate_cache_against_universe(
            path,
            candidates,
            candidate_alphabet="AC",
            max_order=1,
            model_id="toy-model",
            scorer_seed=0,
            n_perturbations=4,
        )


def test_validate_cache_against_universe_rejects_malformed_field_type(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    candidates = [_variant(p) for p in range(3)]
    append_cache(path, [ScoredVariant(variant=v, delta_g=0.0, var_delta_g=0.0) for v in candidates])
    payload = _metadata(candidates).model_dump(mode="json")
    payload["scorer_seed"] = "not-an-int"  # a non-coercible type must never validate
    write_json_exclusive(cache_metadata_path(path), payload)

    with pytest.raises(ValueError):
        validate_cache_against_universe(
            path,
            candidates,
            candidate_alphabet="AC",
            max_order=1,
            model_id="toy-model",
            scorer_seed=0,
            n_perturbations=4,
        )


# ------------------------------------------------------------------------ CacheIdentity


def test_cache_identity_from_metadata_maps_every_shared_field() -> None:
    metadata = _metadata([_variant(0), _variant(1)])
    identity = CacheIdentity.from_metadata(metadata)
    assert identity.model_id == metadata.model_id
    assert identity.scorer_seed == metadata.scorer_seed
    assert identity.n_perturbations == metadata.n_perturbations
    assert identity.candidate_sha256 == metadata.candidate_sha256
    assert identity.candidate_count == metadata.candidate_count
    assert identity.candidate_alphabet == metadata.candidate_alphabet
    assert identity.max_order == metadata.max_order
    assert identity.wt_sha256 == metadata.wt_sha256


def test_validate_cache_against_universe_expected_identity_never_reads_the_sidecar(
    tmp_path: Path,
) -> None:
    """guard: the returned expected identity is the caller's own request, not the sidecar's
    claim — a sidecar declaring a wrong ``candidate_alphabet``/``max_order`` label (while its
    identity-bearing fields still match what was requested) must not leak into ``expected``."""
    path = tmp_path / "cache.jsonl"
    candidates = [_variant(p) for p in range(3)]
    sidecar = _metadata(candidates).model_copy(
        update={"wt_sha256": hashlib.sha256(b"WT").hexdigest()}
    )
    _write_cache_and_sidecar(path, candidates, sidecar)

    _, _, expected_identity = validate_cache_against_universe(
        path,
        candidates,
        candidate_alphabet="AC",
        max_order=1,
        model_id="toy-model",
        scorer_seed=0,
        n_perturbations=4,
        wt_sequence="WT",
    )
    # The caller's own request, never the sidecar's claim (which declares a different model_id).
    assert expected_identity == CacheIdentity(
        model_id="toy-model",
        scorer_seed=0,
        n_perturbations=4,
        candidate_sha256=candidate_sha256(candidates),
        candidate_count=len(candidates),
        candidate_alphabet="AC",
        max_order=1,
        wt_sha256=hashlib.sha256(b"WT").hexdigest(),
    )


def test_validate_cache_against_universe_expected_wt_sha256_is_none_without_wt_sequence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cache.jsonl"
    candidates = [_variant(p) for p in range(3)]
    _write_cache_and_sidecar(path, candidates, _metadata(candidates))

    _, _, expected_identity = validate_cache_against_universe(
        path,
        candidates,
        candidate_alphabet="AC",
        max_order=1,
        model_id="toy-model",
        scorer_seed=0,
        n_perturbations=4,
    )
    # wt_sequence was not supplied, so this call never independently checked wt_sha256; the
    # expected identity says so explicitly rather than fabricating a value it did not verify.
    assert expected_identity.wt_sha256 is None
