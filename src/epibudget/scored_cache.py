"""Resumable scored-variant cache for long scoring runs.

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


class CacheIdentity(BaseModel):
    """The 8 identity fields ``validate_cache_against_universe`` independently checks.

    A caller builds one ``CacheIdentity`` from trusted, independently-computed values (the
    ``expected`` side) and one from an already-validated sidecar's :class:`CacheMetadata` via
    :meth:`from_metadata` (the ``observed`` side), so provenance can serialize both sides of every
    check that was actually performed rather than trusting the sidecar for a field it claims to
    have verified.
    """

    model_config = {"frozen": True}

    model_id: str
    scorer_seed: int
    n_perturbations: int
    candidate_sha256: str
    candidate_count: int
    candidate_alphabet: str
    max_order: int
    wt_sha256: str | None

    @classmethod
    def from_metadata(cls, metadata: CacheMetadata) -> CacheIdentity:
        """The observed identity read from an already cache-validated sidecar."""
        return cls(
            model_id=metadata.model_id,
            scorer_seed=metadata.scorer_seed,
            n_perturbations=metadata.n_perturbations,
            candidate_sha256=metadata.candidate_sha256,
            candidate_count=metadata.candidate_count,
            candidate_alphabet=metadata.candidate_alphabet,
            max_order=metadata.max_order,
            wt_sha256=metadata.wt_sha256,
        )


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
    """Load previously scored variants from a JSONL cache; empty dict if the file is absent.

    Rejects (never silently collapses) a duplicate candidate identity: two JSONL rows for the same
    variant would let insertion order pick one arbitrarily, exactly the kind of scientific-integrity
    mismatch a downstream analysis must never read past silently.
    """
    if not path.exists():
        return {}
    out: dict[Variant, ScoredVariant] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        sv = ScoredVariant.model_validate_json(line)
        if sv.variant in out:
            raise ValueError(
                f"duplicate candidate identity at line {line_number} in {path}: "
                f"{sorted(sv.variant)} was already loaded from an earlier line"
            )
        out[sv.variant] = sv
    return out


def validate_cache_against_universe(
    cache_path: Path,
    candidates: Sequence[Variant],
    *,
    candidate_alphabet: str,
    max_order: int,
    model_id: str,
    scorer_seed: int,
    n_perturbations: int,
    wt_sequence: str | None = None,
) -> tuple[dict[Variant, ScoredVariant], CacheMetadata, CacheIdentity]:
    """Load a scored cache and reject any mismatch against the exact requested candidate universe.

    ``model_id``, ``scorer_seed``, and ``n_perturbations`` are the caller's own expected values
    (e.g. a registered headline/protocol constant) — never derived from the sidecar under check,
    since comparing the sidecar against itself can never fail.

    Rejects (raises ``ValueError``, never warns) on: a missing sidecar; a sidecar whose
    ``candidate_sha256``, ``candidate_count``, ``candidate_alphabet``, ``max_order``, ``model_id``,
    ``scorer_seed``, ``n_perturbations``, or ``wt_sha256`` (when ``wt_sequence`` is given) does not
    match the requested configuration; duplicate or malformed JSONL rows (:func:`load_cache`); or a
    cache whose identity set is not exactly the requested ``candidates`` (missing or unexpected
    entries, including a same-count swap that a bare count comparison would miss).

    Returns the loaded cache, the sidecar's own parsed metadata, and the ``CacheIdentity`` this
    call actually checked the sidecar against — the same 8 independently-computed values, so a
    caller building provenance from the third element can never drift from what this function
    validated (``wt_sha256`` is ``None`` when ``wt_sequence`` was not supplied, since that field is
    then not independently checked here).
    """
    sidecar = cache_metadata_path(cache_path)
    if not sidecar.exists():
        raise ValueError(
            f"score cache {cache_path} has no metadata sidecar; refuse unsafe analysis"
        )
    metadata = CacheMetadata.model_validate_json(sidecar.read_text(encoding="utf-8"))
    expected_hash = candidate_sha256(candidates)
    expected_wt_hash = (
        hashlib.sha256(wt_sequence.encode("ascii")).hexdigest() if wt_sequence is not None else None
    )
    expected_identity = CacheIdentity(
        model_id=model_id,
        scorer_seed=scorer_seed,
        n_perturbations=n_perturbations,
        candidate_sha256=expected_hash,
        candidate_count=len(candidates),
        candidate_alphabet=candidate_alphabet,
        max_order=max_order,
        wt_sha256=expected_wt_hash,
    )
    mismatches: list[str] = []
    if metadata.candidate_sha256 != expected_hash:
        mismatches.append(
            f"candidate_sha256: sidecar={metadata.candidate_sha256} expected={expected_hash}"
        )
    if metadata.candidate_count != len(candidates):
        mismatches.append(
            f"candidate_count: sidecar={metadata.candidate_count} expected={len(candidates)}"
        )
    if metadata.candidate_alphabet != candidate_alphabet:
        mismatches.append(
            f"candidate_alphabet: sidecar={metadata.candidate_alphabet!r} "
            f"expected={candidate_alphabet!r}"
        )
    if metadata.max_order != max_order:
        mismatches.append(f"max_order: sidecar={metadata.max_order} expected={max_order}")
    if metadata.model_id != model_id:
        mismatches.append(f"model_id: sidecar={metadata.model_id!r} expected={model_id!r}")
    if metadata.scorer_seed != scorer_seed:
        mismatches.append(f"scorer_seed: sidecar={metadata.scorer_seed} expected={scorer_seed}")
    if metadata.n_perturbations != n_perturbations:
        mismatches.append(
            f"n_perturbations: sidecar={metadata.n_perturbations} expected={n_perturbations}"
        )
    if expected_wt_hash is not None and metadata.wt_sha256 != expected_wt_hash:
        mismatches.append(f"wt_sha256: sidecar={metadata.wt_sha256} expected={expected_wt_hash}")
    if mismatches:
        raise ValueError(
            f"score cache {cache_path} sidecar does not match the requested universe: "
            + "; ".join(mismatches)
        )

    cache = load_cache(cache_path)
    candidate_set = set(candidates)
    cache_set = set(cache)
    missing = candidate_set - cache_set
    unexpected = cache_set - candidate_set
    if missing or unexpected:
        raise ValueError(
            f"score cache {cache_path} does not match the exact requested candidate universe "
            f"of {len(candidates)} (alphabet={candidate_alphabet!r}, max_order={max_order}): "
            f"{len(missing)} missing, {len(unexpected)} unexpected. Refusing — a mismatched cache "
            "would silently analyse a different universe than scored."
        )
    return cache, metadata, expected_identity


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
