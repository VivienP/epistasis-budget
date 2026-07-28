"""Validation for small, provenance-rich public result artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# Markdown image/link targets, e.g. ![alt](figures/x.svg) or [text](figures/x.svg).
_FIGURE_REFERENCE = re.compile(r"!?\[[^\]]*\]\((?P<target>[^)\s]+\.(?:svg|png|jpg|jpeg|webp))\)")

Sha256 = str


class ArtifactValidationError(ValueError):
    """A public artifact or documented claim does not match its recorded provenance."""


class ArtifactEntry(BaseModel):
    """One checksummed result file listed by the public manifest."""

    path: str
    sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    source_run_id: str
    generation_command: str
    base_commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    code_state: Literal["clean", "dirty"]
    # None when there is no generating code diff to hash (a clean commit, or a hand-authored
    # asset). A sentinel of 64 zeros would be indistinguishable from a real hash.
    code_diff_sha256: Sha256 | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_id: str | None
    data_sha256: Sha256 | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    configuration: dict[str, object]
    status: Literal["primary", "supplementary", "smoke_test", "benchmark"]
    # Set when a finding invalidated the artifact's interpretation. The file itself is retained for
    # traceability and its checksum still verified; only its scientific standing changes.
    superseded_by: str | None = None
    # Required for any result figure a public document displays: the script that regenerates it.
    renderer: str | None = None
    # "result" figures carry numbers and need a renderer plus claim-map coverage; "illustration"
    # figures are hand-authored diagrams that assert no quantity and need only a manifest entry.
    figure_kind: Literal["result", "illustration"] | None = None
    evidence_classification: Literal[
        "reproduced",
        "traceable_not_rerun",
        "estimated",
        "session_only_uncommitted",
        "unsupported",
        "stale",
    ]


class ArtifactManifest(BaseModel):
    """Manifest for provisional or committed public artifacts."""

    schema_version: Literal[1]
    generated_at_utc: str
    provisional: bool
    requires_remanifest_after_commit: bool
    artifacts: list[ArtifactEntry]


class ClaimTransform(BaseModel):
    """Allowlisted deterministic rendering of one JSON value."""

    name: Literal["identity", "round", "ratio", "grouped"]
    digits: int = Field(default=0, ge=0, le=12)
    unicode_minus: bool = False
    show_plus: bool = False
    denominator_json_pointer: str | None = None


class ClaimEntry(BaseModel):
    """One numerical statement in a public Markdown document."""

    id: str
    document: str
    artifact: str
    json_pointer: str
    transform: ClaimTransform
    rendered: str
    anchor: str


class ClaimMap(BaseModel):
    """Machine-readable links from public prose to artifact fields."""

    schema_version: Literal[1]
    forbidden_literals: list[str]
    claims: list[ClaimEntry]


def _safe_path(repo: Path, relative: str) -> Path:
    candidate = (repo / relative).resolve()
    try:
        candidate.relative_to(repo.resolve())
    except ValueError as exc:
        raise ArtifactValidationError(f"artifact path escapes repository: {relative}") from exc
    return candidate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_artifact_payload(
    source: Path,
    *,
    source_path: str,
    source_run_id: str,
    evidence_classification: str,
) -> dict[str, object]:
    """Wrap one local JSON result without altering its numerical payload."""
    return {
        "schema_version": 1,
        "provenance": {
            "source_path": source_path,
            "source_run_id": source_run_id,
            "source_sha256": _sha256(source),
            "evidence_classification": evidence_classification,
        },
        "result": json.loads(source.read_text(encoding="utf-8")),
    }


def _resolve_json_pointer(document: object, pointer: str) -> object:
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ArtifactValidationError(f"invalid JSON pointer: {pointer}")
    current = document
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if part not in current:
                raise ArtifactValidationError(f"JSON pointer not found: {pointer}")
            current = current[part]
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError) as exc:
                raise ArtifactValidationError(f"JSON pointer not found: {pointer}") from exc
        else:
            raise ArtifactValidationError(f"JSON pointer not found: {pointer}")
    return current


def _format_number(value: float, transform: ClaimTransform) -> str:
    sign = "+" if transform.show_plus else ""
    rendered = format(value, f"{sign}.{transform.digits}f")
    return rendered.replace("-", "\N{MINUS SIGN}") if transform.unicode_minus else rendered


def _render_claim(artifact: object, claim: ClaimEntry) -> str:
    value = _resolve_json_pointer(artifact, claim.json_pointer)
    if claim.transform.name == "identity":
        return str(value)
    if not isinstance(value, int | float):
        raise ArtifactValidationError(f"claim {claim.id} does not resolve to a number")
    numeric = float(value)
    if claim.transform.name == "ratio":
        denominator_pointer = claim.transform.denominator_json_pointer
        if denominator_pointer is None:
            raise ArtifactValidationError(f"ratio claim {claim.id} has no denominator pointer")
        denominator = _resolve_json_pointer(artifact, denominator_pointer)
        if not isinstance(denominator, int | float) or float(denominator) == 0.0:
            raise ArtifactValidationError(f"ratio claim {claim.id} has invalid denominator")
        numeric /= float(denominator)
    if claim.transform.name == "grouped":
        return f"{int(numeric):,}"
    return _format_number(numeric, claim.transform)


def _validate_referenced_figures(  # noqa: PLR0912
    repo: Path,
    listed: dict[str, Path],
    manifest: ArtifactManifest,
    claim_map: ClaimMap,
) -> None:
    """Every figure a public document displays needs a manifest entry and a committed renderer.

    Audit L-4: the former headline figure had no renderer, no manifest entry and no claim-map
    coverage, so the most public artifact in the repository was the least traceable one.
    """
    by_path = {entry.path: entry for entry in manifest.artifacts}
    documents = [repo / "README.md", *(repo / "docs").rglob("*.md")]
    referenced: set[str] = set()
    for document in documents:
        if not document.is_file():
            continue
        for match in _FIGURE_REFERENCE.finditer(document.read_text(encoding="utf-8")):
            target = match.group("target").split("#")[0]
            if not target.startswith(("http://", "https://")):
                resolved = (document.parent / target).resolve()
                if resolved.is_relative_to(repo / "figures"):
                    referenced.add(resolved.relative_to(repo).as_posix())
    for figure in sorted(referenced):
        entry = by_path.get(figure)
        if entry is None:
            raise ArtifactValidationError(
                f"figure {figure} is displayed by a public document but is not in the manifest"
            )
        if entry.figure_kind is None:
            raise ArtifactValidationError(f"figure {figure} does not declare a figure_kind")
        if figure not in listed:
            raise ArtifactValidationError(f"figure {figure} is not checksum-verified")
        if entry.figure_kind == "illustration":
            continue  # asserts no quantity: a manifest entry and a checksum are the whole contract
        if not entry.renderer:
            raise ArtifactValidationError(
                f"result figure {figure} has no committed renderer recorded in the manifest"
            )
        if not (repo / entry.renderer).is_file():
            raise ArtifactValidationError(
                f"renderer {entry.renderer} for figure {figure} does not exist"
            )
        if not any(claim.artifact == figure for claim in claim_map.claims):
            raise ArtifactValidationError(f"result figure {figure} has no claim-map coverage")


def validate_public_artifacts(repo: Path) -> None:
    """Validate manifest checksums, claim rendering, and banned historical literals."""
    artifacts_dir = repo / "artifacts"
    manifest = ArtifactManifest.model_validate_json(
        (artifacts_dir / "manifest.json").read_text(encoding="utf-8")
    )
    listed: dict[str, Path] = {}
    for entry in manifest.artifacts:
        path = _safe_path(repo, entry.path)
        if not path.is_file():
            raise ArtifactValidationError(f"claimed artifact is missing: {entry.path}")
        actual_sha = _sha256(path)
        if actual_sha != entry.sha256:
            raise ArtifactValidationError(
                f"checksum mismatch for {entry.path}: expected {entry.sha256}, got {actual_sha}"
            )
        listed[entry.path] = path

    claim_map = ClaimMap.model_validate_json(
        (artifacts_dir / "claim_map.json").read_text(encoding="utf-8")
    )
    loaded_artifacts: dict[str, object] = {}
    for claim in claim_map.claims:
        if claim.artifact not in listed:
            raise ArtifactValidationError(f"claim {claim.id} uses an unlisted artifact")
        artifact = loaded_artifacts.setdefault(
            claim.artifact,
            json.loads(listed[claim.artifact].read_text(encoding="utf-8")),
        )
        expected = _render_claim(artifact, claim)
        if expected != claim.rendered:
            raise ArtifactValidationError(
                f"claim {claim.id} renders {expected!r}, claim map records {claim.rendered!r}"
            )
        document = _safe_path(repo, claim.document)
        text = document.read_text(encoding="utf-8")
        if claim.anchor not in text or claim.rendered not in claim.anchor:
            raise ArtifactValidationError(f"documented claim missing for {claim.id}")

    _validate_referenced_figures(repo, listed, manifest, claim_map)

    public_markdown = [repo / "README.md"]
    public_markdown.extend(
        path for path in (repo / "docs").rglob("*.md") if path.name != "ROADMAP.md"
    )
    for path in public_markdown:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for forbidden in claim_map.forbidden_literals:
            if forbidden in text:
                relative = path.relative_to(repo).as_posix()
                raise ArtifactValidationError(
                    f"forbidden historical literal {forbidden!r} in {relative}"
                )
