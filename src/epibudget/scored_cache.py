"""Resumable scored-variant cache for long runs (e.g. the 650M headline on a free Colab GPU).

A JSONL sidecar of already-computed ``ScoredVariant`` rows so a run interrupted by a session timeout
resumes without re-scoring. Purely a throughput/resilience aid: cached values are the scorer's exact
write-through output, so a resumed run is byte-identical to an uninterrupted one. No torch import —
the cache logic is offline-testable.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from epibudget.provenance import write_json_exclusive
from epibudget.types import ScoredVariant, Variant


class _BatchScorer(Protocol):
    """The one method ``score_with_cache`` needs — structural, so no concrete scorer import here."""

    def score_batch(self, wt: str, variants: Sequence[Variant]) -> list[ScoredVariant]: ...


class _ConfiguredBatchScorer(_BatchScorer, Protocol):
    model_id: str
    device: str
    n_perturbations: int
    seed: int
    mask_fraction: float
    batch_size: int
    num_threads: int | None


class CacheMetadata(BaseModel):
    """Immutable identity of every scientific and execution setting bound to a score cache."""

    model_config = {"frozen": True}

    schema_version: int = 1
    model_id: str
    wt_sha256: str
    candidate_sha256: str
    candidate_count: int
    candidate_alphabet: str
    max_order: int
    scorer_seed: int
    n_perturbations: int
    device: str
    mask_fraction: float
    batch_size: int
    num_threads: int | None


def candidate_sha256(candidates: Sequence[Variant]) -> str:
    """Hash the candidate identities independently of their input order."""
    canonical = sorted([sorted([list(mutation) for mutation in variant]) for variant in candidates])
    payload = json.dumps(canonical, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def cache_metadata_path(path: Path) -> Path:
    """Return the immutable JSON sidecar path for a JSONL score cache."""
    return path.with_name(path.name + ".meta.json")


def build_cache_metadata(
    scorer: _ConfiguredBatchScorer,
    wt: str,
    candidates: Sequence[Variant],
    *,
    candidate_alphabet: str,
    max_order: int,
) -> CacheMetadata:
    """Bind a configured scorer and candidate universe to one immutable cache identity."""
    return CacheMetadata(
        model_id=scorer.model_id,
        wt_sha256=hashlib.sha256(wt.encode("ascii")).hexdigest(),
        candidate_sha256=candidate_sha256(candidates),
        candidate_count=len(candidates),
        candidate_alphabet=candidate_alphabet,
        max_order=max_order,
        scorer_seed=scorer.seed,
        n_perturbations=scorer.n_perturbations,
        device=scorer.device,
        mask_fraction=scorer.mask_fraction,
        batch_size=scorer.batch_size,
        num_threads=scorer.num_threads,
    )


def _ensure_metadata(path: Path, expected: CacheMetadata) -> None:
    sidecar = cache_metadata_path(path)
    if path.exists() and not sidecar.exists():
        raise ValueError(f"score cache {path} has no metadata sidecar; refuse unsafe legacy reuse")
    if not sidecar.exists():
        write_json_exclusive(sidecar, expected.model_dump(mode="json"))
        return
    actual = CacheMetadata.model_validate_json(sidecar.read_text(encoding="utf-8"))
    if actual != expected:
        differences = [
            name
            for name in CacheMetadata.model_fields
            if getattr(actual, name) != getattr(expected, name)
        ]
        raise ValueError(f"cache metadata mismatch for {path}: {', '.join(differences)}")


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


def _load_cache_repairing_truncated_tail(path: Path) -> dict[Variant, ScoredVariant]:
    """Load a cache, truncating only one malformed final unterminated line."""
    if not path.exists():
        return {}
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    out: dict[Variant, ScoredVariant] = {}
    valid_bytes = 0
    for index, encoded in enumerate(lines):
        try:
            line = encoded.decode("utf-8").strip()
            if line:
                sv = ScoredVariant.model_validate_json(line)
                out[sv.variant] = sv
            valid_bytes += len(encoded)
        except (UnicodeDecodeError, ValueError):
            is_truncated_tail = index == len(lines) - 1 and not raw.endswith(b"\n")
            if not is_truncated_tail:
                raise ValueError(
                    f"invalid score-cache record at line {index + 1} in {path}"
                ) from None
            with path.open("r+b") as handle:
                handle.truncate(valid_bytes)
            break
    return out


def append_cache(path: Path, scored: Sequence[ScoredVariant]) -> None:
    """Append scored variants to the JSONL cache (write-through, one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for sv in scored:
            handle.write(sv.model_dump_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def score_with_cache(
    scorer: _BatchScorer,
    wt: str,
    candidates: Sequence[Variant],
    path: Path,
    *,
    metadata: CacheMetadata,
    chunk_size: int = 512,
) -> list[ScoredVariant]:
    """Score ``candidates``, skipping any already in ``path`` and write-through appending the rest.

    Scores the missing candidates in ``chunk_size`` batches, flushing each to disk before the next,
    so an interruption loses at most one chunk. Returns a ScoredVariant per candidate, in order.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    _ensure_metadata(path, metadata)
    cache = _load_cache_repairing_truncated_tail(path)
    missing = [c for c in candidates if c not in cache]
    for start in range(0, len(missing), chunk_size):
        chunk = missing[start : start + chunk_size]
        scored = scorer.score_batch(wt, chunk)
        append_cache(path, scored)
        for sv in scored:
            cache[sv.variant] = sv
    return [cache[c] for c in candidates]
