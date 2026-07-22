# Contributing

Contributions should preserve the scientific contract: conjoint variant scoring, label-free selection,
artifact-backed quantitative claims, and honest null results.

## Development setup

```bash
git clone https://github.com/VivienP/epistasis-budget.git
cd epistasis-budget
python -m pip install -e ".[dev]"
```

Python 3.12 or later is required. The default development and test path must remain offline; model
downloads and public-data fetches belong behind the existing `slow` and `data` markers.

## Quality gate

Run the same checks as CI before opening a pull request:

```bash
ruff format --check src/ tests/ scripts/
ruff check src/ tests/ scripts/
mypy --strict src/
pytest -q -m "not slow and not data"
python scripts/validate_artifacts.py
```

## Pull requests

- Keep each pull request focused on one behavior or documentation change.
- Add or update a failing test before changing behavior.
- Never expose measured fitness labels to selection or acquisition code.
- Score multi-mutation variants in their conjoint background, never as a sum of single-site scores.
- Trace every quantitative statement to a reproducible artifact or citation.
- Preserve negative and inconclusive results without removing baselines or changing frozen decisions.
- Update the README or reference documentation when a public API or result changes.

Describe what changed, why it belongs in scope, and which verification commands passed.
