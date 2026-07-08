"""epibudget command-line interface. See docs/SPEC.md#8."""

from __future__ import annotations

import re
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from epibudget.types import Mutation, Variant

app = typer.Typer(
    add_completion=False,
    help="Information-optimal experimental budget allocation for mapping protein epistasis.",
)
console = Console()

_STUB = "[yellow]Not implemented yet[/] — this scaffold is initialised; see docs/ROADMAP.md."
_AA20 = "ACDEFGHIKLMNPQRSTVWY"
_TOKEN = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")
_SEPARATORS = re.compile(r"[\s,;+]+")


def read_fasta_sequence(path: Path) -> str:
    """Read a single-record FASTA and return its residue sequence (upper-case, header dropped)."""
    body = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(">")
    ]
    seq = "".join(body).upper()
    if not seq:
        raise ValueError(f"no sequence found in {path}")
    return seq


def parse_variant(spec: str, wt: str) -> Variant:
    """Parse DMS notation (e.g. ``"V39A D40C"``, 1-indexed) into a Variant against ``wt``.

    Each token is ``<wt_aa><position_1indexed><mut_aa>``; the WT letter is checked against ``wt``
    so a wrong position or off-by-one is caught here. Empty / ``"WT"`` means the wild type.
    """
    stripped = spec.strip()
    if not stripped or stripped.upper() == "WT":
        return frozenset()
    muts: set[Mutation] = set()
    for token in _SEPARATORS.split(stripped):
        if not token:
            continue
        m = _TOKEN.match(token)
        if m is None:
            raise ValueError(f"malformed mutation token {token!r} (expected e.g. 'V39A')")
        wt_aa, pos_1, mut_aa = m.group(1).upper(), int(m.group(2)), m.group(3).upper()
        pos = pos_1 - 1
        if not 0 <= pos < len(wt):
            raise ValueError(
                f"position {pos_1} in {token!r} is out of range for a length-{len(wt)} WT"
            )
        if wt[pos] != wt_aa:
            raise ValueError(
                f"WT mismatch in {token!r}: sequence has {wt[pos]!r} at position {pos_1}"
            )
        if mut_aa not in _AA20:
            raise ValueError(f"{mut_aa!r} in {token!r} is not a valid amino acid")
        if mut_aa == wt_aa:
            raise ValueError(f"synonymous mutation {token!r}: mutant residue equals WT")
        muts.add((pos, wt_aa, mut_aa))
    return frozenset(muts)


def read_variant_specs(path: Path) -> list[str]:
    """Read variant specs from ``path`` — one per line, first comma-field, optional header row."""
    specs: list[str] = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
        cell = raw.split(",")[0].strip()
        if not cell:
            continue
        if i == 0 and cell.lower() in {"variant", "variants", "mutations"}:
            continue
        specs.append(cell)
    return specs


def _order_label(variant: Variant) -> str:
    return {1: "single", 2: "double", 3: "triple"}.get(len(variant), str(len(variant)))


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
    variants: str = typer.Option(..., help="CSV of variants to score (DMS notation, e.g. V39A)."),
    model: str = typer.Option(
        "facebook/esm2_t12_35M_UR50D", help="ESM-2 checkpoint id (HuggingFace)."
    ),
    n_perturbations: int = typer.Option(16, help="Masking passes for var[ΔG]; 0 disables."),
    seed: int = typer.Option(0),
) -> None:
    """Debug: dump conjoint ΔG + masking-perturbation variance for a variant list."""
    from epibudget.scoring import ConjointScorer  # noqa: PLC0415  # heavy import, deferred

    wt = read_fasta_sequence(Path(fasta))
    specs = read_variant_specs(Path(variants))
    parsed = [parse_variant(spec, wt) for spec in specs]

    scorer = ConjointScorer(model, n_perturbations=n_perturbations, seed=seed)
    scored = scorer.score_batch(wt, parsed)

    table = Table(title=f"conjoint scores — {model}")
    table.add_column("variant")
    table.add_column("order")
    table.add_column("ΔG", justify="right")
    table.add_column("var[ΔG]", justify="right")
    for spec, sv in zip(specs, scored, strict=True):
        table.add_row(
            spec or "WT", _order_label(sv.variant), f"{sv.delta_g:+.4f}", f"{sv.var_delta_g:.4f}"
        )
    console.print(table)


if __name__ == "__main__":
    app()
