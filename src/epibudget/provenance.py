"""Deterministic provenance helpers for immutable scientific outputs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path

_EXCLUDED_DIFF_PREFIXES = (".agents/", ".codex/", "artifacts/", "report/")
_EXCLUDED_DIFF_PATHS = frozenset({"docs/ROADMAP.md"})


def code_diff_sha256(files: Mapping[str, bytes | None]) -> str:
    """Hash a path-to-content working-tree delta in stable path order.

    ``None`` represents a deleted file and is distinct from an empty file. Generated artifacts and
    reports are excluded by the collector so a manifest never hashes itself.
    """
    digest = hashlib.sha256()
    for path in sorted(files):
        encoded_path = path.replace("\\", "/").encode("utf-8")
        content = files[path]
        if content is None:
            digest.update(b"D\0")
            digest.update(encoded_path)
            digest.update(b"\0")
            continue
        digest.update(b"F\0")
        digest.update(encoded_path)
        digest.update(b"\0")
        digest.update(str(len(content)).encode("ascii"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def workspace_code_diff_sha256(repo: Path, base_commit_sha: str) -> str:
    """Hash tracked and untracked workspace changes relative to ``base_commit_sha``.

    The digest covers current file contents rather than Git's platform-sensitive textual rendering.
    Generated artifacts, reports, local agent tooling, and the ignored roadmap are excluded.
    """

    def git_lines(*args: str) -> list[str]:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return [line for line in result.stdout.splitlines() if line]

    tracked = git_lines("diff", "--name-only", "--diff-filter=ACDMRTUXB", base_commit_sha, "--")
    untracked = git_lines("ls-files", "--others", "--exclude-standard")
    paths = sorted(set(tracked) | set(untracked))
    files: dict[str, bytes | None] = {}
    for raw_path in paths:
        path = raw_path.replace("\\", "/")
        if path in _EXCLUDED_DIFF_PATHS or path.startswith(_EXCLUDED_DIFF_PREFIXES):
            continue
        absolute = repo / Path(path)
        files[path] = absolute.read_bytes() if absolute.is_file() else None
    return code_diff_sha256(files)


def write_json_exclusive(path: Path, payload: object) -> None:
    """Write one formatted JSON document and fail if ``path`` already exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False))
        handle.write("\n")
