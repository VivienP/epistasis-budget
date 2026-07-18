"""Deterministic provenance helpers for immutable scientific outputs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
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


def _git_diff_lines(repo: Path, *args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return [line for line in result.stdout.splitlines() if line]


def changed_scientific_files(repo: Path, base_commit_sha: str) -> list[str]:
    """Tracked+untracked workspace paths relative to ``base_commit_sha``, minus generated artifacts.

    The same path set :func:`workspace_code_diff_sha256` hashes, exposed on its own so provenance
    can record which files a code-diff hash actually covers, not only the opaque digest.
    """
    tracked = _git_diff_lines(
        repo, "diff", "--name-only", "--diff-filter=ACDMRTUXB", base_commit_sha, "--"
    )
    untracked = _git_diff_lines(repo, "ls-files", "--others", "--exclude-standard")
    paths = sorted(set(tracked) | set(untracked))
    return [
        path.replace("\\", "/")
        for path in paths
        if path.replace("\\", "/") not in _EXCLUDED_DIFF_PATHS
        and not path.replace("\\", "/").startswith(_EXCLUDED_DIFF_PREFIXES)
    ]


def workspace_code_diff_sha256(repo: Path, base_commit_sha: str) -> str:
    """Hash tracked and untracked workspace changes relative to ``base_commit_sha``.

    The digest covers current file contents rather than Git's platform-sensitive textual rendering.
    Generated artifacts, reports, local agent tooling, and the ignored roadmap are excluded.
    """
    files: dict[str, bytes | None] = {}
    for path in changed_scientific_files(repo, base_commit_sha):
        absolute = repo / Path(path)
        files[path] = absolute.read_bytes() if absolute.is_file() else None
    return code_diff_sha256(files)


def write_json_exclusive(path: Path, payload: object) -> None:
    """Write one formatted JSON document and fail if ``path`` already exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False))
        handle.write("\n")


def _fsync_directory_best_effort(directory: Path) -> None:
    """Fsync a directory entry for durability where the platform supports it.

    POSIX guarantees a directory fd is fsync-able; Windows has no equivalent and raises
    ``OSError`` when opening a directory with ``os.open``, or when fsyncing it. Both are treated
    as a no-op here rather than a failure, since this step is a durability best-effort, not part
    of the atomicity guarantee itself.
    """
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except (OSError, AttributeError, NotImplementedError):
        pass
    finally:
        os.close(dir_fd)


def write_json_atomic(path: Path, payload: object) -> None:
    """Write one formatted JSON document via a create-only hard-link publish.

    Writes and fsyncs a uniquely-named temporary file in the same directory as ``path`` (same
    filesystem), then publishes it to ``path`` with ``os.link``. ``os.link`` creates a hard link at
    ``path`` pointing at the temp file's inode and raises ``FileExistsError`` if ``path`` already
    exists -- on both POSIX and Windows, unlike ``os.rename``, whose POSIX behavior is to silently
    replace an existing destination. "does the destination exist" and "create the destination"
    are therefore a single atomic kernel operation, so two concurrent writers targeting the same
    ``path`` can never both publish: exactly one call to ``os.link`` succeeds, and the other raises
    ``FileExistsError``, matching :func:`write_json_exclusive`'s fail-on-exists contract. The final
    ``path`` never appears until the full payload has been written and flushed, so a crash before
    publication leaves only the orphaned temp file, never a truncated ``path``; that temp file is
    removed on any failure. On success, the redundant temp name is unlinked immediately -- this
    removes only that one name, since hard links are independent references to the same inode, so
    the published file at ``path`` is untouched. If the destination filesystem does not support
    hard links, this raises ``OSError`` rather than silently falling back to a non-atomic write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, indent=2, ensure_ascii=False))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp_path, path)
        except FileExistsError:
            raise
        except (OSError, NotImplementedError) as exc:
            raise OSError(
                f"{path.parent} does not support the atomic hard-link publish this function "
                "requires; refusing to fall back to a non-atomic write"
            ) from exc
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    else:
        tmp_path.unlink(missing_ok=True)
        _fsync_directory_best_effort(path.parent)
