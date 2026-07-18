"""Offline tests for standalone script help output."""

from __future__ import annotations

import sys
from collections.abc import Callable
from io import BytesIO, StringIO, TextIOWrapper
from pathlib import Path
from typing import TextIO

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from bench_scoring import main as bench_scoring_main
from calibrate_uncertainty import main as calibrate_uncertainty_main
from fetch_gb1 import main as fetch_gb1_main
from fetch_trpb import main as fetch_trpb_main
from gb1_epistasis_signal import main as gb1_epistasis_signal_main
from headline_650m_supplementary import main as headline_650m_supplementary_main

_HELP_COMMANDS: tuple[tuple[str, Callable[[], None]], ...] = (
    ("bench_scoring.py", bench_scoring_main),
    ("calibrate_uncertainty.py", calibrate_uncertainty_main),
    ("fetch_gb1.py", fetch_gb1_main),
    ("fetch_trpb.py", fetch_trpb_main),
    ("gb1_epistasis_signal.py", gb1_epistasis_signal_main),
    ("headline_650m_supplementary.py", headline_650m_supplementary_main),
)


def _run_help(
    monkeypatch: pytest.MonkeyPatch,
    script_name: str,
    main: Callable[[], None],
    stdout: TextIO,
) -> None:
    monkeypatch.setattr(sys, "argv", [script_name, "--help"])
    monkeypatch.setattr(sys, "stdout", stdout)

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0


@pytest.mark.parametrize(("script_name", "main"), _HELP_COMMANDS)
def test_help_reconfigures_cp1252_stdout_to_utf8(
    monkeypatch: pytest.MonkeyPatch,
    script_name: str,
    main: Callable[[], None],
) -> None:
    buffer = BytesIO()
    stdout = TextIOWrapper(buffer, encoding="cp1252")

    _run_help(monkeypatch, script_name, main, stdout)

    stdout.flush()
    assert stdout.encoding == "utf-8"
    assert b"usage:" in buffer.getvalue()


@pytest.mark.parametrize(("script_name", "main"), _HELP_COMMANDS)
def test_help_supports_stdout_without_reconfigure(
    monkeypatch: pytest.MonkeyPatch,
    script_name: str,
    main: Callable[[], None],
) -> None:
    stdout = StringIO()
    assert not hasattr(stdout, "reconfigure")

    _run_help(monkeypatch, script_name, main, stdout)

    assert "usage:" in stdout.getvalue()
