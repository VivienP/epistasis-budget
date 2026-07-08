"""epibudget command-line interface. See docs/SPEC.md#8."""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    add_completion=False,
    help="Information-optimal experimental budget allocation for mapping protein epistasis.",
)
console = Console()

_STUB = "[yellow]Not implemented yet[/] — this scaffold is initialised; see docs/ROADMAP.md."


@app.command()
def allocate(
    fasta: str = typer.Option(..., help="Path to the wild-type sequence (FASTA)."),
    positions: str = typer.Option(
        ..., help="Comma-separated 1-indexed positions, e.g. 39,40,41,54."
    ),
    budget: int = typer.Option(..., help="Number of variants to select (B wells)."),
    model: str = typer.Option("esm2_t33_650M", help="ESM-2 checkpoint short name."),
    lambda_: float = typer.Option(0.0, "--lambda", help="0=info-optimal, 1=fitness-greedy."),
    max_order: int = typer.Option(3, help="Max interaction order (2 or 3)."),
    seed: int = typer.Option(0),
    out: str = typer.Option("allocation.json"),
) -> None:
    """Rank the B most epistasis-informative variants for a target."""
    console.print(f"[bold]allocate[/] B={budget} λ={lambda_} model={model} positions={positions}")
    console.print(_STUB)


@app.command()
def validate(
    dataset: str = typer.Option("gb1_wu2016", help="Validation dataset id."),
    budgets: str = typer.Option("48,96,192", help="Comma-separated budgets."),
    model: str = typer.Option("esm2_t12_35M"),
    seeds: int = typer.Option(20),
    out: str = typer.Option("report/"),
) -> None:
    """Run the frozen GB1 benchmark (info vs fitness vs random). See docs/VALIDATION.md."""
    console.print(
        f"[bold]validate[/] dataset={dataset} budgets={budgets} model={model} seeds={seeds}"
    )
    console.print(_STUB)


@app.command()
def score(
    fasta: str = typer.Option(..., help="Path to the wild-type sequence (FASTA)."),
    variants: str = typer.Option(..., help="CSV of variants to score."),
) -> None:
    """Debug: dump conjoint ΔG + masking-perturbation variance for a variant list."""
    console.print("[bold]score[/]")
    console.print(_STUB)


if __name__ == "__main__":
    app()
