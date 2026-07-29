"""Measure Fourier-spectrum sparsity on the redistributed TrpB training target."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from epibudget.spectrum_diagnostic import main as run  # noqa: PLC0415

    run()


if __name__ == "__main__":
    main()
