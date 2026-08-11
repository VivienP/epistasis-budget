"""Prepare, resume, inspect, verify, and export the registered recovery diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from epibudget.recovery_engine import (
    RecoveryProgress,
    export_recovery_report,
    prepare_recovery_run,
    recovery_status,
    run_recovery,
    verify_recovery_run,
)
from epibudget.scored_cache import cache_metadata_path

_COMMANDS = frozenset({"prepare", "run", "status", "verify", "export"})


def _add_run_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)


def _command_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="validate and archive the exact run inputs")
    _add_run_dir(prepare)
    prepare.add_argument("--data", type=Path, required=True)
    prepare.add_argument("--cache", type=Path, required=True)
    prepare.add_argument("--sidecar", type=Path, required=True)
    prepare.add_argument("--runtime-preflight", type=Path, required=True)

    run = commands.add_parser("run", help="resume computation from the durable journal")
    _add_run_dir(run)
    run.add_argument("--out", type=Path)

    status = commands.add_parser("status", help="show verified durable progress")
    _add_run_dir(status)

    verify = commands.add_parser("verify", help="audit store integrity and domain transitions")
    _add_run_dir(verify)

    export = commands.add_parser("export", help="write the published report outside the store")
    _add_run_dir(export)
    export.add_argument("--out", type=Path, required=True)
    return parser


def _legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/proteingym/trpb_johnston2024.csv"))
    parser.add_argument("--cache", type=Path, default=Path("report/scored_trpb_650m_n16.jsonl"))
    parser.add_argument(
        "--runtime-preflight",
        type=Path,
        default=Path("report/diagnostics/fourier_recovery_runtime_v4.json"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("report/diagnostics/fourier_recovery_trpb.json")
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    return parser


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    payload = getattr(value, "payload", None)
    if callable(payload):
        return payload()
    if isinstance(value, Path):
        return str(value)
    return value


def _emit(value: object) -> None:
    print(json.dumps(_jsonable(value), sort_keys=True, default=str), flush=True)


def _progress(event: RecoveryProgress) -> None:
    cell = ""
    if event.method is not None:
        cell = f" method={event.method} seed={event.seed} budget={event.budget}"
    print(
        f"{event.stage}: {event.completed} / {event.total}{cell}",
        flush=True,
    )


def _run_command(arguments: argparse.Namespace) -> object:
    command = arguments.command
    if command == "prepare":
        return prepare_recovery_run(
            arguments.run_dir,
            dataset=arguments.data,
            cache=arguments.cache,
            sidecar=arguments.sidecar,
            runtime_preflight=arguments.runtime_preflight,
        )
    if command == "run":
        status = run_recovery(arguments.run_dir, on_progress=_progress)
        if arguments.out is not None:
            export_recovery_report(arguments.run_dir, arguments.out)
        return status
    if command == "status":
        return recovery_status(arguments.run_dir)
    if command == "verify":
        return verify_recovery_run(arguments.run_dir)
    if command == "export":
        return export_recovery_report(arguments.run_dir, arguments.out)
    raise AssertionError(f"unhandled command: {command}")


def _run_legacy(arguments: argparse.Namespace) -> object:
    sidecar = cache_metadata_path(arguments.cache)
    prepare_recovery_run(
        arguments.checkpoint_dir,
        dataset=arguments.data,
        cache=arguments.cache,
        sidecar=sidecar,
        runtime_preflight=arguments.runtime_preflight,
    )
    status = run_recovery(arguments.checkpoint_dir, on_progress=_progress)
    export_recovery_report(arguments.checkpoint_dir, arguments.out)
    return status


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one thin CLI command and return its process exit code."""
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] in _COMMANDS:
        parsed = _command_parser().parse_args(raw)
        result = _run_command(parsed)
    else:
        parsed = _legacy_parser().parse_args(raw)
        result = _run_legacy(parsed)
    _emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
