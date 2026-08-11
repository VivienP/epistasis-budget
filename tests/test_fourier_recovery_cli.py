from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "fourier_recovery_curve.py"


@pytest.fixture
def cli_module() -> ModuleType:
    name = "epibudget_test_fourier_recovery_cli"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load CLI module from {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_prepare_requires_and_forwards_the_explicit_sidecar(
    cli_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Path]] = []

    def prepare(run_dir: Path, **inputs: Path) -> dict[str, str]:
        calls.append({"run_dir": run_dir, **inputs})
        return {"stage": "doptimal"}

    monkeypatch.setattr(cli_module, "prepare_recovery_run", prepare)
    paths = {
        "run_dir": tmp_path / "run",
        "dataset": tmp_path / "dataset.csv",
        "cache": tmp_path / "cache.jsonl",
        "sidecar": tmp_path / "archived-sidecar.json",
        "runtime_preflight": tmp_path / "preflight.json",
    }

    exit_code = cli_module.main(
        [
            "prepare",
            "--run-dir",
            str(paths["run_dir"]),
            "--data",
            str(paths["dataset"]),
            "--cache",
            str(paths["cache"]),
            "--sidecar",
            str(paths["sidecar"]),
            "--runtime-preflight",
            str(paths["runtime_preflight"]),
        ]
    )

    assert exit_code == 0
    assert calls == [paths]


def test_run_exports_only_after_the_engine_returns(
    cli_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[tuple[str, Path]] = []
    run_dir = tmp_path / "run"
    out = tmp_path / "report.json"

    def run(path: Path, **_kwargs: object) -> dict[str, str]:
        events.append(("run", path))
        return {"stage": "complete"}

    def export(path: Path, destination: Path) -> dict[str, str]:
        events.append(("export", path))
        assert destination == out
        return {"sha256": "a" * 64}

    monkeypatch.setattr(cli_module, "run_recovery", run)
    monkeypatch.setattr(cli_module, "export_recovery_report", export)

    exit_code = cli_module.main(["run", "--run-dir", str(run_dir), "--out", str(out)])

    assert exit_code == 0
    assert events == [("run", run_dir), ("export", run_dir)]


def test_status_verify_and_export_never_call_the_runner(
    cli_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    out = tmp_path / "report.json"
    calls: list[str] = []
    monkeypatch.setattr(
        cli_module,
        "run_recovery",
        lambda *_args, **_kwargs: pytest.fail("read-only command called the runner"),
    )
    monkeypatch.setattr(
        cli_module,
        "recovery_status",
        lambda path: calls.append(f"status:{path}") or {"stage": "cells"},
    )
    monkeypatch.setattr(
        cli_module,
        "verify_recovery_run",
        lambda path: calls.append(f"verify:{path}") or {"valid": True},
    )
    monkeypatch.setattr(
        cli_module,
        "export_recovery_report",
        lambda path, destination: (
            calls.append(f"export:{path}:{destination}") or {"sha256": "a" * 64}
        ),
    )

    assert cli_module.main(["status", "--run-dir", str(run_dir)]) == 0
    assert cli_module.main(["verify", "--run-dir", str(run_dir)]) == 0
    assert cli_module.main(["export", "--run-dir", str(run_dir), "--out", str(out)]) == 0
    assert calls == [
        f"status:{run_dir}",
        f"verify:{run_dir}",
        f"export:{run_dir}:{out}",
    ]


def test_legacy_form_derives_adjacent_sidecar_and_runs_prepare_run_export(
    cli_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    dataset = tmp_path / "dataset.csv"
    cache = tmp_path / "cache.jsonl"
    preflight = tmp_path / "preflight.json"
    out = tmp_path / "report.json"
    adjacent = tmp_path / "cache.meta.json"
    events: list[str] = []
    monkeypatch.setattr(cli_module, "cache_metadata_path", lambda path: adjacent)

    def prepare(path: Path, **inputs: Path) -> dict[str, str]:
        assert path == run_dir
        assert inputs["sidecar"] == adjacent
        events.append("prepare")
        return {"stage": "doptimal"}

    monkeypatch.setattr(cli_module, "prepare_recovery_run", prepare)
    monkeypatch.setattr(
        cli_module,
        "run_recovery",
        lambda path, **_kwargs: events.append("run") or {"stage": "complete"},
    )
    monkeypatch.setattr(
        cli_module,
        "export_recovery_report",
        lambda path, destination: events.append("export") or {"sha256": "a" * 64},
    )

    exit_code = cli_module.main(
        [
            "--checkpoint-dir",
            str(run_dir),
            "--data",
            str(dataset),
            "--cache",
            str(cache),
            "--runtime-preflight",
            str(preflight),
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    assert events == ["prepare", "run", "export"]
