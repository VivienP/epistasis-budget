"""Offline tests for immutable outputs and deterministic provenance digests."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from epibudget import provenance
from epibudget.provenance import (
    code_diff_sha256,
    workspace_code_diff_sha256,
    write_json_atomic,
    write_json_exclusive,
)

_SHA256_HEX_LENGTH = 64
_GIT_SHA_LENGTH = 40


def _repository_local_git_env_vars() -> tuple[str, ...]:
    """Names of the git environment variables that bind git to one repository.

    ``git rev-parse --local-env-vars`` is git's own authoritative list (GIT_DIR, GIT_INDEX_FILE,
    GIT_WORK_TREE, ...). It prints only the variable names and needs no repository, so it is stable
    regardless of the surrounding environment.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--local-env-vars"], check=True, capture_output=True, text=True
    )
    return tuple(result.stdout.split())


_LOCAL_GIT_ENV_VARS = _repository_local_git_env_vars()


@contextlib.contextmanager
def _without_repository_local_git_env() -> Iterator[None]:
    """Temporarily drop every repository-local git variable from the process environment.

    A pre-commit hook runs with GIT_DIR / GIT_INDEX_FILE / GIT_WORK_TREE exported and pointing at
    the outer repository. Both the temp-repo helper below and the production
    ``workspace_code_diff_sha256`` it exercises spawn ``git`` inheriting ``os.environ``; stripping
    these variables is what makes every such child target the temp repo, in a hook or not.
    """
    saved = {name: os.environ[name] for name in _LOCAL_GIT_ENV_VARS if name in os.environ}
    for name in saved:
        del os.environ[name]
    try:
        yield
    finally:
        for name, value in saved.items():
            os.environ[name] = value


@pytest.fixture(autouse=True)
def _isolate_git_from_hook_env() -> Iterator[None]:
    """Run every test in this module with the repository-local git variables stripped, so the
    temp-repo tests pass whether invoked directly or from the pre-commit hook that exports them."""
    with _without_repository_local_git_env():
        yield


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


def test_temp_repo_tests_survive_a_simulated_hook_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a pre-commit hook exports GIT_DIR/GIT_INDEX_FILE/GIT_WORK_TREE pointing at the
    outer repo. The full temp-repo workflow — including the production diff — must still target the
    temp repo once the repository-local variables are stripped.

    The exported variables point at a throwaway directory under ``tmp_path`` (never the real repo),
    so nothing here can touch the outer repository even if the isolation regressed.
    """
    fake_outer = tmp_path / "outer"
    fake_outer.mkdir()
    monkeypatch.setenv("GIT_DIR", str(fake_outer / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(fake_outer / ".git" / "index"))
    monkeypatch.setenv("GIT_WORK_TREE", str(fake_outer))

    work = tmp_path / "work"
    work.mkdir()
    with _without_repository_local_git_env():
        base = _init_repo_with_base_commit(work)
        (work / "src.py").write_bytes(b"two\n")
        digest = workspace_code_diff_sha256(work, base)

    assert len(base) == _GIT_SHA_LENGTH
    # The diff reflects the temp repo's own edit — proof git did not follow the exported hook vars
    # (which would have yielded the empty/outer-repo diff instead).
    assert digest == code_diff_sha256({"src.py": b"two\n"})


def test_write_json_atomic_rejects_overwrite_and_leaves_no_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    write_json_atomic(path, {"which": "A"})

    with pytest.raises(FileExistsError):
        write_json_atomic(path, {"which": "B"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"which": "A"}
    assert [p.name for p in tmp_path.iterdir()] == ["result.json"]


def test_write_json_atomic_concurrent_writers_exactly_one_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """regression: two publishers racing on the same path via the real os.link.

    A barrier forces both threads to call the real, unwrapped ``os.link`` at (as close as
    possible to) the same instant; the wrapper only synchronizes, it never changes behavior. The
    OS serializes the two create-only calls, so exactly one publishes and the other observes
    ``FileExistsError`` -- the losing thread's own temp file is cleaned up, and the winner's
    payload is the only content ever visible at ``path``. Which thread wins is not deterministic;
    that there is exactly one winner and one loser is.
    """
    path = tmp_path / "result.json"
    barrier = threading.Barrier(2)
    real_link = provenance.os.link

    def synced_link(src: object, dst: object) -> None:
        barrier.wait()
        real_link(src, dst)

    monkeypatch.setattr(provenance.os, "link", synced_link)

    payload_a = {"which": "A", "note": "payload-A"}
    payload_b = {"which": "B", "note": "payload-B"}
    outcomes: dict[str, str] = {}
    outcomes_lock = threading.Lock()

    def writer(name: str, payload: dict[str, str]) -> None:
        try:
            write_json_atomic(path, payload)
            outcome = "success"
        except FileExistsError:
            outcome = "FileExistsError"
        with outcomes_lock:
            outcomes[name] = outcome

    thread_a = threading.Thread(target=writer, args=("A", payload_a))
    thread_b = threading.Thread(target=writer, args=("B", payload_b))
    thread_a.start()
    thread_b.start()
    thread_a.join(timeout=10)
    thread_b.join(timeout=10)

    assert not thread_a.is_alive()
    assert not thread_b.is_alive()
    assert sorted(outcomes.values()) == ["FileExistsError", "success"]

    final_payload = json.loads(path.read_text(encoding="utf-8"))
    assert final_payload in (payload_a, payload_b)
    assert [p.name for p in tmp_path.iterdir()] == ["result.json"]


def test_write_json_atomic_cleans_up_temp_file_on_serialization_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "result.json"
    call_count = 0

    def raising_dumps(*args: object, **kwargs: object) -> str:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("boom-before-publish")

    def failing_if_reached(src: object, dst: object) -> None:
        raise AssertionError("os.link must not be called when serialization fails first")

    monkeypatch.setattr(provenance.json, "dumps", raising_dumps)
    monkeypatch.setattr(provenance.os, "link", failing_if_reached)

    with pytest.raises(RuntimeError, match="boom-before-publish"):
        write_json_atomic(path, {"value": 1})

    assert call_count == 1
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_write_json_atomic_never_touches_preexisting_final_file(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    original_bytes = json.dumps({"which": "pre-existing"}, indent=2).encode("utf-8")
    path.write_bytes(original_bytes)
    original_mtime_ns = path.stat().st_mtime_ns

    with pytest.raises(FileExistsError):
        write_json_atomic(path, {"which": "new"})

    assert path.read_bytes() == original_bytes
    assert path.stat().st_mtime_ns == original_mtime_ns
    assert [p.name for p in tmp_path.iterdir()] == ["result.json"]
