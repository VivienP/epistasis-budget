from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _run_hook(tmp_path: Path, staged_path: str) -> tuple[str, list[str]]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Hook Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "hook@example.invalid"], cwd=repo, check=True)

    hooks = repo / "hooks"
    hooks.mkdir()
    source_hook = Path(__file__).resolve().parents[1] / "scripts" / "hooks" / "pre-commit"
    hook = hooks / "pre-commit"
    shutil.copyfile(source_hook, hook)
    hook.chmod(0o755)
    subprocess.run(["git", "config", "core.hooksPath", "hooks"], cwd=repo, check=True)

    tool_dir = tmp_path / "tools"
    tool_dir.mkdir()
    log = tmp_path / "hook.log"
    stub = '#!/usr/bin/env sh\nprintf "%s\\n" "$(basename "$0") $*" >> "$HOOK_LOG"\n'
    for name in ("ruff", "mypy", "pytest", "python"):
        tool = tool_dir / name
        tool.write_text(stub, encoding="utf-8", newline="\n")
        tool.chmod(0o755)

    staged = repo / staged_path
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_text("staged\n", encoding="utf-8")
    subprocess.run(["git", "add", staged_path], cwd=repo, check=True)

    env = os.environ.copy()
    env["PATH"] = str(tool_dir) + os.pathsep + env["PATH"]
    env["HOOK_LOG"] = log.as_posix()
    result = subprocess.run(
        ["git", "commit", "-m", "test"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    return output, log.read_text(encoding="utf-8").splitlines()


def test_docs_only_commit_skips_source_quality_checks(tmp_path: Path) -> None:
    output, invocations = _run_hook(tmp_path, "README.md")

    assert invocations == ["python scripts/validate_artifacts.py"]
    assert "[skip] source quality checks (no source/test/script file staged)" in output


def test_source_commit_runs_all_quality_checks(tmp_path: Path) -> None:
    _output, invocations = _run_hook(tmp_path, "src/epibudget/example.py")

    assert invocations == [
        "ruff format --check src/ tests/ scripts/",
        "ruff check src/ tests/ scripts/",
        "mypy --strict src/",
        "pytest -q -m not slow and not data",
        "python scripts/validate_artifacts.py",
    ]
