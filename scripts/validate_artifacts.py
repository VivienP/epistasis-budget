"""Validate public result artifacts, checksums, and documented numerical claims."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from epibudget.artifacts import validate_public_artifacts


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    validate_public_artifacts(repo)


if __name__ == "__main__":
    main()
