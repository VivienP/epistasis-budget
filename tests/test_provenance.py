"""Offline tests for immutable outputs and deterministic provenance digests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from epibudget.provenance import code_diff_sha256, workspace_code_diff_sha256, write_json_exclusive

_SHA256_HEX_LENGTH = 64


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _init_repo_with_base_commit(repo: Path) -> str:
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    _git(repo, "config", "core.autocrlf", "false")
    (repo / "src.py").write_bytes(b"one\n")
    _git(repo, "add", "src.py")
    _git(repo, "commit", "-q", "-m", "base")
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def test_code_diff_sha256_is_order_independent_and_content_sensitive() -> None:
    first = code_diff_sha256({"src/b.py": b"two", "src/a.py": b"one"})
    reordered = code_diff_sha256({"src/a.py": b"one", "src/b.py": b"two"})
    changed = code_diff_sha256({"src/a.py": b"ONE", "src/b.py": b"two"})

    assert first == reordered
    assert first != changed
    assert len(first) == _SHA256_HEX_LENGTH


def test_code_diff_sha256_distinguishes_deleted_files() -> None:
    assert code_diff_sha256({"src/a.py": None}) != code_diff_sha256({"src/a.py": b""})


def test_write_json_exclusive_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    write_json_exclusive(path, {"value": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}

    with pytest.raises(FileExistsError):
        write_json_exclusive(path, {"value": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}


def test_workspace_code_diff_sha256_is_stable_with_no_changes(tmp_path: Path) -> None:
    repo = tmp_path
    base = _init_repo_with_base_commit(repo)

    first = workspace_code_diff_sha256(repo, base)
    second = workspace_code_diff_sha256(repo, base)

    assert first == second
    assert first == code_diff_sha256({})


def test_workspace_code_diff_sha256_detects_tracked_edit(tmp_path: Path) -> None:
    repo = tmp_path
    base = _init_repo_with_base_commit(repo)
    clean = workspace_code_diff_sha256(repo, base)

    (repo / "src.py").write_bytes(b"two\n")

    edited = workspace_code_diff_sha256(repo, base)
    assert edited != clean
    assert edited == code_diff_sha256({"src.py": b"two\n"})


def test_workspace_code_diff_sha256_detects_untracked_file(tmp_path: Path) -> None:
    repo = tmp_path
    base = _init_repo_with_base_commit(repo)
    clean = workspace_code_diff_sha256(repo, base)

    (repo / "new.py").write_bytes(b"new\n")

    with_untracked = workspace_code_diff_sha256(repo, base)
    assert with_untracked != clean
    assert with_untracked == code_diff_sha256({"new.py": b"new\n"})


def test_workspace_code_diff_sha256_excludes_generated_and_ignored_paths(tmp_path: Path) -> None:
    repo = tmp_path
    base = _init_repo_with_base_commit(repo)
    clean = workspace_code_diff_sha256(repo, base)

    for relative in (
        "artifacts/manifest.json",
        "report/run/metrics.json",
        ".agents/x.md",
        ".codex/y.toml",
    ):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ignored\n", encoding="utf-8")
    (repo / "docs").mkdir()
    (repo / "docs" / "ROADMAP.md").write_text("ignored\n", encoding="utf-8")

    assert workspace_code_diff_sha256(repo, base) == clean
